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

