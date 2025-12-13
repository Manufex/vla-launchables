# vla-train-lab

**vla-train-lab** offers a simplified approach to fine-tuning **Vision-Language-Action (VLA) models** on NVIDIA Brev or locally.

This project provides **one-click launchables** for different VLA models, making it as easy as possible to fine-tune models without extensive setup or configuration. Each model gets its own optimized Docker environment with preconfigured dependencies, so you can go from "deploy" to "training" in minutes.

Launchables are provided by [NVIDIA Brev](https://developer.nvidia.com/brev), using this repo as a template. Launchables are preconfigured, fully optimized compute and software environments that allow you to start training without wrestling with Python environments, CUDA versions, or dependency conflicts.

## What this project contains

The training setup for each VLA model is automated via Docker, such that it can be used locally, or be deployed on services such as NVIDIA Brev and run with cloud GPU resources.

Each model project includes:
- A Docker container with all dependencies preinstalled
- A training script that handles dataset loading, model configuration, and W&B logging
- Named cache volumes for Hugging Face, PyTorch, pip, and W&B (for fast reruns)
- Token-safe workflow that prompts for secrets at runtime (never commits or bakes secrets)

## Quickstart: Deploy on Brev

The fastest way to get started is to deploy a preconfigured launchable on Brev. Click one of the deploy buttons below to spin up an instance for your chosen VLA model:

### LeRobot π0.5 (pi05)

[![Deploy LeRobot π0.5 on Brev](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-36lFUACCCYMoYRnJQO2WISupFDU)

Fine-tune LeRobot's π0.5 policy model on your own dataset. Supports Hugging Face datasets and W&B logging.

### SmolVLA

[![Deploy SmolVLA on Brev](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy/now?launchableID=YOUR_SMOLVLA_LAUNCHABLE_ID)

Train SmolVLA, a compact vision-language-action model optimized for robotics tasks.

### GR00T-N1.5

[![Deploy GR00T-N1.5 on Brev](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy/now?launchableID=YOUR_GR00T_LAUNCHABLE_ID)

Fine-tune NVIDIA's GR00T-N1.5 foundation model for your specific robotics application.

> [!NOTE]
> More VLA models are coming soon (X-VLA, and others). This repo is actively maintained and new models are added regularly.

> [!IMPORTANT]
> Please note that Brev instances are pay-by-the-hour. To make the best use of credits, stop instances when they are not in use. Stopped instances have a smaller storage charge.

## What you'll need to provide

When you deploy a launchable or run training locally, you'll need to provide:

### Required credentials

- **Hugging Face token** (`HF_TOKEN`): Required to download datasets and models from Hugging Face Hub
  - Get one at: https://huggingface.co/settings/tokens
- **Weights & Biases API key** (`WANDB_API_KEY`): Required for experiment tracking and logging
  - Get one at: https://wandb.ai/authorize

> [!IMPORTANT]
> This repo **never stores tokens in git**. The training script prompts for them interactively if they are not already present in your environment. **Do not commit** a `.env` file containing secrets.

### Training parameters (optional, with defaults)

You can customize training via environment variables:

- `DATASET_REPO_ID`: Hugging Face dataset repository ID (e.g., `Zasha01/cube_transfer_final`)
- `OUTPUT_DIR`: Where to save trained models (default: `/workspace/outputs/{model_name}`)
- `JOB_NAME`: Name for this training run (used in W&B)
- `STEPS`: Number of training steps (default: `3000`)
- `BATCH_SIZE`: Training batch size (default: `8`; lower if you hit CUDA OOM)
- `WANDB_PROJECT`: W&B project name (default: `vla-train-lab`)

Example with custom parameters:

```bash
DATASET_REPO_ID=my-org/my-dataset STEPS=5000 BATCH_SIZE=4 just train
```

## How it works

This repo is structured like **isaac-launchable**: each VLA model lives in its own folder with its own `docker-compose.yml`, and a top-level `Justfile` provides a consistent interface across all models.

### Project structure

```text
vla-train-lab/
  Justfile                    # Top-level launcher (cd into project, run compose)
  common/
    prompt_tokens.sh          # Token prompt script (executable)
  lerobot-pi05/
    Dockerfile                # CUDA 12.1 + LeRobot setup
    docker-compose.yml        # Host networking + nvidia runtime + cache volumes
    train.sh                  # Training script (executable)
  smolvla/
    Dockerfile
    docker-compose.yml
    train.sh
  gr00t-n15/
    Dockerfile
    docker-compose.yml
    train.sh
  outputs/                    # Generated outputs directory
```

### Key features

- **Project-per-model layout**: Each VLA model is self-contained in its own directory
- **Docker + GPU runtime**: No host Python environment conflicts
- **Named cache volumes**: HF / torch / pip / wandb caches persist between runs for speed
- **Token-safe workflow**: Prompts for HF + W&B tokens at runtime (no secrets in images)
- **Brev-friendly defaults**: Host networking, NVIDIA runtime, large `/dev/shm`

## Using this project locally

This project can also be used to run training locally on your own NVIDIA GPU machine.

### Requirements

- Linux host with an NVIDIA GPU
- Docker + Docker Compose v2
- NVIDIA Container Toolkit installed (so containers can see the GPU)
- Optional but recommended: [`just`](https://github.com/casey/just)

Quick checks:
```bash
docker --version
docker compose version
nvidia-smi
```

### Local setup

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-org/vla-train-lab
   cd vla-train-lab
   ```

2. **Build the image** for your chosen model:

   With `just`:
   ```bash
   just build
   ```

   Without `just`:
   ```bash
   cd lerobot-pi05  # or smolvla, gr00t-n15
   docker compose build
   ```

3. **Run training** (prompts for HF + W&B tokens):

   With `just`:
   ```bash
   just train
   ```

   Without `just`:
   ```bash
   cd lerobot-pi05
   ./train.sh
   ```

   The script will prompt you for your Hugging Face token and W&B API key if they're not already in your environment.

### Useful commands

**Open a shell in the container**:
```bash
just shell
```

**Bring the service up/down** (useful for debugging):
```bash
just up
just down
```

**Use a different project**:
```bash
PROJECT=smolvla just shell
```

### Configuration

You can override training parameters via environment variables. See the [What you'll need to provide](#what-youll-need-to-provide) section above for the full list.

Example:
```bash
STEPS=1000 BATCH_SIZE=4 OUTPUT_DIR=/workspace/outputs/pi05_run2 just train
```

If you hit CUDA OOM errors, lower `BATCH_SIZE` first.

## Creating your own launchable

If you'd like to create a custom Brev launchable for one of these models (or your own), you can fork this repo and configure it on Brev.

### Configuring a custom Brev launchable

1. Log in to the [Brev](https://login.brev.nvidia.com/signin) website.
2. Go to the Launchables category.
3. Click the **Create Launchable** button.
4. Choose the "I don't have any code files" option.
5. Choose **VM Mode - Basic VM with Python installed**, then click Next.
6. On the next page, add a setup script. Under the *Paste Script* tab, add this code:
   ```bash
   #!/bin/bash
   export VSCODE_PASSWORD=your_password  # replace with a secure password
   git clone https://github.com/your-org/vla-train-lab
   cd vla-train-lab/lerobot-pi05  # or smolvla, gr00t-n15
   docker compose up -d
   ```
7. The VSCode container expects a password to be set via the `$VSCODE_PASSWORD` environment variable. Replace `your_password` with your desired password, or generate it securely.
8. Click Next.
9. Under "Do you want a Jupyter Notebook experience" select "No, I don't want Jupyter".
10. Select the TCP/UDP ports tab.
11. Expose port `80` (for Visual Studio Code Server) to a specific public IP address.
12. Click Next.
13. Choose your desired compute (GPU required for training).
14. Choose disk storage, then click Next.
15. Enter a name, then select **Create Launchable**.

## Troubleshooting

**Containers not starting?**
- Check that Docker and NVIDIA Container Toolkit are installed: `docker --version && nvidia-smi`
- Verify containers are running: `docker ps`
- Check logs: `docker compose logs`

**CUDA out of memory?**
- Lower `BATCH_SIZE` (try 4 or 2)
- Reduce `STEPS` if you're just testing

**Token prompts not working?**
- Make sure `common/prompt_tokens.sh` is executable: `chmod +x common/prompt_tokens.sh`
- You can also set tokens in your environment: `export HF_TOKEN=...` and `export WANDB_API_KEY=...`

## Roadmap

- [x] LeRobot π0.5 (pi05) fine-tuning
- [x] SmolVLA fine-tuning
- [x] GR00T-N1.5 fine-tuning
- [ ] X-VLA and other VLA models
- [ ] Pinned versions/commits per model for strict reproducibility
- [ ] CI "build images" workflow
- [ ] Optional dataset/model caching warm-up targets

## License

Choose a license that matches your intended usage (MIT/Apache-2.0 are common for tooling repos).
