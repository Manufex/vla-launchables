# LeRobot Launchable

Single Docker container for training all VLA models (pi05, smolvla, xvla, groot, act).

## Structure

```
lerobot-launchable/
├── Dockerfile              # Base image with core lerobot
├── docker-compose.yml      # Single compose file
├── run.sh                  # Training wrapper script
├── scripts/
│   ├── run_train.py        # Python training script (reads configs)
│   └── prompt_tokens.sh    # Token prompt helper
└── configs/
    ├── pi05_default.yaml
    ├── smolvla_default.yaml
    ├── xvla_default.yaml
    ├── groot_default.yaml
    └── act_default.yaml
```

## Quick Start

### Build the container
```bash
cd lerobot-launchable
just build
# or
docker compose build
```

### Run training
```bash
# Using just (recommended)
just train-pi          # Train pi05
just train-smolvla     # Train smolvla
just train-xvla        # Train xvla
just train-groot       # Train groot
just train-act         # Train act

# Or with custom config
just train CONFIG=configs/pi05_custom.yaml

# Or directly
./run.sh configs/pi05_default.yaml
```

## Configuration

Each model has a YAML config file in `configs/`. Example:

```yaml
# configs/pi05_default.yaml
dataset:
  repo_id: "Zasha01/cube_transfer_final"

policy:
  type: "pi05"
  repo_id: "local/pi05"
  push_to_hub: true
  pretrained_path: "lerobot/pi05_base"
  compile_model: true
  gradient_checkpointing: true

output_dir: "/workspace/outputs/pi05"
job_name: "pi05_cube_transfer_final"

steps: 3000
batch_size: 16
device: "cuda"
dtype: "auto"  # Auto-detects bfloat16/float16

wandb:
  enable: true
```

## How It Works

1. **Base Dockerfile**: Installs core lerobot (no extras)
2. **Config-driven**: Each model has a YAML config file
3. **Runtime installation**: `run_train.py` installs policy extras based on `policy.type` in config
4. **Single container**: All models use the same Docker image

## Customization

### Create a custom config
```bash
cp configs/pi05_default.yaml configs/pi05_custom.yaml
# Edit configs/pi05_custom.yaml
./run.sh configs/pi05_custom.yaml
```

### Override config values via environment
The config file is the source of truth, but you can override specific values by modifying the config before running.

## Outputs

Trained models are saved to `../outputs/` (relative to lerobot-launchable directory).

## Cache

Cache volumes are shared across all models:
- Hugging Face cache
- PyTorch cache
- pip cache
- W&B cache

