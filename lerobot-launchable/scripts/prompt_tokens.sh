#!/usr/bin/env bash
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  read -s -p "Hugging Face token (HF_TOKEN): " HF_TOKEN; echo
  export HF_TOKEN
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  read -s -p "Weights & Biases API key (WANDB_API_KEY): " WANDB_API_KEY; echo
  export WANDB_API_KEY
fi

# Note: Brev credentials (BREV_ENV_ID, BREV_TOKEN) are optional
# They will only be needed if auto_delete.enable is true in the config
# Users can set them via environment variables or they'll be prompted
# in run_train.py if auto_delete is enabled

