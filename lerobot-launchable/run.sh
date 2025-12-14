#!/usr/bin/env bash
set -euo pipefail

# Wrapper script to run training with config files
# Usage: ./run.sh configs/pi05_default.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Prompt for tokens on the HOST
if [ -f "${SCRIPT_DIR}/scripts/prompt_tokens.sh" ]; then
  source "${SCRIPT_DIR}/scripts/prompt_tokens.sh"
elif [ -f "${SCRIPT_DIR}/../common/prompt_tokens.sh" ]; then
  source "${SCRIPT_DIR}/../common/prompt_tokens.sh"
fi

# Get config file from argument or environment
CONFIG_FILE="${1:-${CONFIG_FILE:-}}"

if [ -z "${CONFIG_FILE}" ]; then
  echo "Error: CONFIG_FILE must be provided as argument or environment variable"
  echo "Usage: ./run.sh configs/pi05_default.yaml"
  echo "   or: CONFIG_FILE=configs/pi05_default.yaml ./run.sh"
  exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Error: Config file not found: ${CONFIG_FILE}"
  exit 1
fi

# Run training via Python script
docker compose run --rm \
  -e HF_TOKEN \
  -e WANDB_API_KEY \
  -e CONFIG_FILE="${CONFIG_FILE}" \
  lerobot python3 scripts/run_train.py --config "${CONFIG_FILE}"

