#!/usr/bin/env python3
"""
Training script that reads config YAML and runs lerobot training.
Installs policy extras based on config, then runs training.
"""
import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path


def install_policy_extras(policy_type: str):
    """Install lerobot extras based on policy type."""
    print(f"Installing lerobot extras for policy type: {policy_type}")
    
    install_commands = {
        "groot": ["pip", "install", "lerobot[groot]"],
        "xvla": ["pip", "install", "-e", "/opt/lerobot[xvla]"],
        "smolvla": ["pip", "install", "-e", "/opt/lerobot[smolvla]"],
        "pi": ["pip", "install", "-e", "/opt/lerobot[pi]"],
        "pi05": ["pip", "install", "-e", "/opt/lerobot[pi]"],
        "act": None,  # No extras needed
    }
    
    if policy_type not in install_commands:
        print(f"Warning: Unknown policy type '{policy_type}', skipping extras installation")
        print(f"Available policy types: {', '.join(install_commands.keys())}")
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
    cmd = ["python3", "/opt/lerobot/src/lerobot/scripts/lerobot_train.py"]
    
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
    
    # Handle dtype (auto-detect if "auto")
    dtype = config.get("dtype", "auto")
    if dtype == "auto":
        dtype = get_dtype()
    cmd.extend(["--policy.dtype", dtype])
    
    cmd.extend(["--policy.device", config.get("device", "cuda")])
    
    # Wandb
    if config.get("wandb", {}).get("enable", True):
        cmd.append("--wandb.enable=true")
    else:
        cmd.append("--wandb.enable=false")
    
    # Policy-specific arguments
    if "policy" in config:
        policy = config["policy"]
        
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
    
    return cmd


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
    
    # Ensure output directory exists
    output_dir = config.get("output_dir", "/workspace/outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Build and run training command
    train_cmd = build_train_command(config)
    print(f"\nRunning training command:")
    print(" ".join(train_cmd))
    print()
    
    result = subprocess.run(train_cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

