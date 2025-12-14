#!/usr/bin/env python3
"""
Training script that reads config YAML and runs lerobot training.
Installs policy extras based on config, then runs training.
Uploads trained model to Hugging Face Hub after training completes.
"""
import argparse
import os
import shutil
import subprocess
import sys
import yaml
from pathlib import Path


def install_policy_extras(policy_type: str):
    """Install lerobot extras based on policy type.
    
    Note: For groot, all dependencies (PyTorch, flash-attn, lerobot[groot])
    are installed in the Dockerfile.groot, so we just verify they're available.
    """
    print(f"Installing lerobot extras for policy type: {policy_type}")
    
    # For groot, dependencies are installed in Dockerfile.groot
    # Just verify flash-attn is available
    if policy_type == "groot":
        print("Groot: Verifying flash-attn is available (should be installed in Dockerfile)...")
        try:
            import flash_attn
            print(f"✓ Flash Attention {flash_attn.__version__} is available")
        except ImportError:
            print("✗ ERROR: Flash-attn not found!")
            print("Groot requires flash-attn. Make sure you're using Dockerfile.groot")
            print("and docker-compose.groot.yml for groot training.")
        return
    
    install_commands = {
        "xvla": ["pip", "install", "-e", "/opt/lerobot[xvla]"],
        "smolvla": ["pip", "install", "-e", "/opt/lerobot[smolvla]"],
        "pi": ["pip", "install", "-e", "/opt/lerobot[pi]"],
        "pi05": ["pip", "install", "-e", "/opt/lerobot[pi]"],
        "act": None,  # No extras needed
    }
    
    if policy_type not in install_commands:
        print(f"Warning: Unknown policy type '{policy_type}', skipping extras installation")
        print(f"Available policy types: groot, {', '.join(install_commands.keys())}")
        return
    
    cmd = install_commands[policy_type]
    if cmd is None:
        print(f"ACT policy: no extras needed (core lerobot is sufficient)")
        return
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Warning: Failed to install extras for {policy_type}, continuing anyway...")
    else:
        print("Policy extras installation complete.")


def get_dtype():
    """Detect optimal dtype based on GPU capabilities."""
    try:
        import torch
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return "bfloat16"
        return "float16"
    except ImportError:
        return "float16"


def load_config(config_path: str) -> dict:
    """Load config YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_train_command(config: dict) -> list:
    """Build lerobot_train.py command from config."""
    # Use lerobot_train from installed package (works with editable install)
    cmd = ["python3", "-m", "lerobot.scripts.lerobot_train"]
    
    # Required arguments
    if "dataset" in config and "repo_id" in config["dataset"]:
        cmd.extend(["--dataset.repo_id", config["dataset"]["repo_id"]])
    
    if "policy" in config and "type" in config["policy"]:
        cmd.extend(["--policy.type", config["policy"]["type"]])
    
    if "output_dir" in config:
        cmd.extend(["--output_dir", config["output_dir"]])
    
    if "job_name" in config:
        cmd.extend(["--job_name", config["job_name"]])
    
    # Optional arguments with defaults
    cmd.extend(["--steps", str(config.get("steps", 3000))])
    cmd.extend(["--batch_size", str(config.get("batch_size", 8))])
    
    # Handle dtype (only for policies that support it)
    # groot doesn't support dtype parameter
    policy_type = config.get("policy", {}).get("type", "")
    if policy_type not in ["groot"]:
        dtype = config.get("dtype", "auto")
        if dtype == "auto":
            dtype = get_dtype()
        cmd.extend(["--policy.dtype", dtype])
    
    # Device parameter (if supported by policy)
    device = config.get("device", "cuda")
    if device:
        cmd.extend(["--policy.device", device])
    
    # Wandb
    if config.get("wandb", {}).get("enable", True):
        cmd.append("--wandb.enable=true")
    else:
        cmd.append("--wandb.enable=false")
    
    # Policy-specific arguments
    if "policy" in config:
        policy = config["policy"]
        
        # repo_id is required for some policies (like groot)
        if "repo_id" in policy:
            cmd.extend(["--policy.repo_id", policy["repo_id"]])
        
        if policy.get("push_to_hub", False):
            cmd.append("--policy.push_to_hub=true")
        
        if "pretrained_path" in policy:
            cmd.extend(["--policy.pretrained_path", policy["pretrained_path"]])
        
        if policy.get("compile_model", False):
            cmd.append("--policy.compile_model=true")
        
        if policy.get("gradient_checkpointing", False):
            cmd.append("--policy.gradient_checkpointing=true")
    
    # Resume flag
    if config.get("resume", False):
        cmd.append("--resume=true")
    
    return cmd


def upload_to_hub(output_dir: str, repo_id: str, private: bool = True):
    """Upload trained model to Hugging Face Hub."""
    print(f"\n{'='*60}")
    print(f"Uploading model to Hugging Face Hub")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Repository ID: {repo_id}")
    print(f"Private: {private}")
    print()
    
    # Check if output directory exists
    if not os.path.exists(output_dir):
        print(f"Error: Output directory does not exist: {output_dir}")
        return False
    
    # Use huggingface_hub to upload
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        
        # Create repo if it doesn't exist (private by default)
        try:
            create_repo(repo_id=repo_id, private=private, exist_ok=True, token=os.getenv("HF_TOKEN"))
            print(f"Repository {repo_id} ready")
        except Exception as e:
            print(f"Note: Repository creation/check: {e}")
        
        # Upload the model
        print(f"Uploading files from {output_dir}...")
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="model",
            token=os.getenv("HF_TOKEN"),
            ignore_patterns=[".git*", "__pycache__*", "*.pyc"]
        )
        print(f"\n✓ Successfully uploaded model to https://huggingface.co/{repo_id}")
        return True
        
    except ImportError:
        print("Error: huggingface_hub not available, skipping upload")
        return False
    except Exception as e:
        print(f"Error uploading to Hub: {e}")
        return False


def delete_brev_instance(env_id: str, token: str):
    """Delete Brev instance after training completes."""
    print(f"\n{'='*60}")
    print(f"Auto-deleting Brev instance")
    print(f"{'='*60}")
    print(f"Environment ID: {env_id}")
    print()
    
    try:
        # Check if brev CLI is available
        brev_path = shutil.which("brev")
        
        # Install brev CLI if not available
        if not brev_path:
            print("Installing brev CLI...")
            install_result = subprocess.run(
                ["sudo", "bash", "-c", 
                 "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"],
                check=False
            )
            if install_result.returncode != 0:
                print("Warning: Failed to install brev CLI")
                return False
        
        # Login to brev
        print("Logging into Brev...")
        login_result = subprocess.run(
            ["brev", "login", "--token", token, "--skip-browser"],
            capture_output=True,
            text=True,
            check=False
        )
        if login_result.returncode != 0:
            print(f"Error: Failed to login to Brev: {login_result.stderr}")
            return False
        
        # Delete the instance
        print(f"Deleting workspace {env_id}...")
        delete_result = subprocess.run(
            ["brev", "delete", env_id],
            check=False
        )
        if delete_result.returncode != 0:
            print("Error: Failed to delete Brev instance")
            return False
        
        print(f"\n✓ Successfully deleted Brev instance {env_id}")
        return True
        
    except Exception as e:
        print(f"Error deleting Brev instance: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run lerobot training with config file")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join("/workspace", config_path)
    
    config = load_config(config_path)
    
    # Get policy type
    policy_type = config.get("policy", {}).get("type")
    if not policy_type:
        print("Error: policy.type must be specified in config")
        sys.exit(1)
    
    # Install policy extras
    install_policy_extras(policy_type)
    
    # Ensure output directory exists and is unique (unless resuming)
    output_dir = config.get("output_dir", "/workspace/outputs")
    job_name = config.get("job_name", "untitled")
    resume = config.get("resume", False)
    
    # If output directory exists and we're not resuming, make it unique
    if os.path.exists(output_dir) and not resume:
        # Create a timestamped subdirectory with counter to ensure uniqueness
        from datetime import datetime
        import time
        base_dir = output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counter = 0
        while True:
            if counter == 0:
                output_dir = os.path.join(base_dir, f"{job_name}_{timestamp}")
            else:
                output_dir = os.path.join(base_dir, f"{job_name}_{timestamp}_{counter}")
            
            if not os.path.exists(output_dir):
                break
            counter += 1
            # Small delay to ensure different timestamps if called rapidly
            time.sleep(0.1)
        
        print(f"Output directory already exists, using: {output_dir}")
        # Update config so the training command uses the new path
        config["output_dir"] = output_dir
    
    # Don't create the directory here - let lerobot create it
    # This ensures lerobot's validation passes
    
    # Build and run training command
    train_cmd = build_train_command(config)
    print(f"\nRunning training command:")
    print(" ".join(train_cmd))
    print()
    
    # Run training
    result = subprocess.run(train_cmd, check=False)
    
    # Check if training was successful
    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n{'='*60}")
    print(f"Training completed successfully!")
    print(f"{'='*60}\n")
    
    # Upload to Hugging Face Hub if configured
    upload_config = config.get("upload", {})
    if upload_config.get("enable", True):
        repo_id = upload_config.get("repo_id")
        if not repo_id:
            # Try to get from policy.repo_id or generate from job_name
            policy_repo_id = config.get("policy", {}).get("repo_id")
            if policy_repo_id and not policy_repo_id.startswith("local/"):
                repo_id = policy_repo_id
            else:
                job_name = config.get("job_name", "untitled")
                # Generate repo_id from job_name (sanitize)
                repo_id = job_name.replace("_", "-").lower()
                print(f"Warning: No upload.repo_id specified, using generated: {repo_id}")
        
        private = upload_config.get("private", True)
        upload_success = upload_to_hub(output_dir, repo_id, private)
        
        if not upload_success:
            print("Warning: Model upload failed, but training completed successfully")
    else:
        print("Upload disabled in config (upload.enable: false)")
    
    # Auto-delete Brev instance if configured
    auto_delete_config = config.get("auto_delete", {})
    if auto_delete_config.get("enable", False):
        env_id = os.getenv("BREV_ENV_ID")
        token = os.getenv("BREV_TOKEN")
        
        # Prompt for credentials if not set
        if not env_id:
            env_id = input("Brev Environment ID (BREV_ENV_ID): ").strip()
            if env_id:
                os.environ["BREV_ENV_ID"] = env_id
        
        if not token:
            import getpass
            token = getpass.getpass("Brev Token (BREV_TOKEN): ").strip()
            if token:
                os.environ["BREV_TOKEN"] = token
        
        if not env_id:
            print("Warning: auto_delete.enable is true but BREV_ENV_ID not provided")
            print("         Instance will not be deleted. Set BREV_ENV_ID to enable auto-delete.")
        elif not token:
            print("Warning: auto_delete.enable is true but BREV_TOKEN not provided")
            print("         Instance will not be deleted. Set BREV_TOKEN to enable auto-delete.")
        else:
            delete_success = delete_brev_instance(env_id, token)
            if not delete_success:
                print("Warning: Failed to delete Brev instance, but training completed successfully")
    
    sys.exit(0)


if __name__ == "__main__":
    main()

