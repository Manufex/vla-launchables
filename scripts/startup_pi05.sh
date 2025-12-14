#!/bin/bash 

set -euo pipefail

# Use the current user's home 
REPO_DIR="${REPO_DIR:-$HOME/vla-launchables}"
PROJECT_DIR="${PROJECT_DIR:-$REPO_DIR/lerobot-launchable}"

# Fallback: try the ubuntu path if it exists AND is accessible
if [[ ! -d "$PROJECT_DIR" && -d "/home/ubuntu/vla-launchables/lerobot-launchable" ]]; then
  PROJECT_DIR="/home/ubuntu/vla-launchables/lerobot-launchable"
fi

cd "$PROJECT_DIR"

# Defaults (you can override later in your SSH session)
export ENV="${ENV:-brev}"
export CONFIG_FILE="${CONFIG_FILE:-configs/pi05_default.yaml}"

# Build once at boot so training starts faster later
docker compose build

echo "============================================================"
echo "VLA Launchable ready (SSH workflow) - LeRobot π0.5"
echo ""
echo "After you SSH into the instance, run:"
echo "  cd ${PROJECT_DIR}"
echo "  bash run.sh configs/pi05_default.yaml"
echo ""
echo "That script will PROMPT you for:"
echo "  - HF_TOKEN"
echo "  - WANDB_API_KEY"
echo ""
echo "To customize training parameters, edit the config file:"
echo "  nano configs/pi05_default.yaml"
echo "  # or"
echo "  vim configs/pi05_default.yaml"
echo ""
echo "You can modify: dataset.repo_id, steps, batch_size, output_dir, etc."
echo ""
echo "Optional: Create a custom config:"
echo "  cp configs/pi05_default.yaml configs/pi05_custom.yaml"
echo "  nano configs/pi05_custom.yaml"
echo "  bash run.sh configs/pi05_custom.yaml"
echo ""
echo "Auto-delete instance after training (saves costs):"
echo "  1. Edit configs/pi05_default.yaml and set auto_delete.enable: true"
echo "  2. Set environment variables (will be prompted if not set):"
echo "     export BREV_ENV_ID=your-env-id"
echo "     export BREV_TOKEN=your-token"
echo "============================================================"

