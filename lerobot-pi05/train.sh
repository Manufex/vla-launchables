#!/usr/bin/env bash
set -euo pipefail

# Prompt for tokens on the HOST, then pass into container (no secrets baked into image).
# This matches the "env-driven" pattern in isaac-launchable style setups.
source ../common/prompt_tokens.sh

# Optional: let user override without editing files
: "${DATASET_REPO_ID:=Zasha01/cube_transfer_final}"
: "${OUTPUT_DIR:=/workspace/outputs/pi05}"
: "${JOB_NAME:=pi05_cube_transfer_final}"
: "${STEPS:=3000}"
: "${BATCH_SIZE:=16}"
: "${RESUME:=0}"

BASE_OUT="/workspace/outputs/pi05"

# If user sets OUTPUT_DIR to the base folder, make it unique unless resuming
if [[ "${OUTPUT_DIR}" == "${BASE_OUT}" && "${RESUME}" != "1" ]]; then
  OUTPUT_DIR="${BASE_OUT}/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
fi

docker compose run --rm \
  -e HF_TOKEN \
  -e WANDB_API_KEY \
  -e DATASET_REPO_ID \
  -e OUTPUT_DIR \
  -e JOB_NAME \
  -e STEPS \
  -e BATCH_SIZE \
  pi05 bash -lc '
    set -euo pipefail

    DTYPE="$(python3 - <<PY
import torch
if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
  print("bfloat16")
else:
  print("float16")
PY
)"

    mkdir -p "${OUTPUT_DIR}"

    python3 /opt/lerobot/src/lerobot/scripts/lerobot_train.py \
      --dataset.repo_id="${DATASET_REPO_ID}" \
      --policy.type=pi05 \
      --output_dir="${OUTPUT_DIR}" \
      --job_name="${JOB_NAME}" \
      --policy.repo_id="local/pi05" \
      --policy.push_to_hub=true \
      --policy.pretrained_path=lerobot/pi05_base \
      --policy.compile_model=true \
      --policy.gradient_checkpointing=true \
      --wandb.enable=true \
      --policy.dtype="${DTYPE}" \
      --steps="${STEPS}" \
      --policy.device=cuda \
      --batch_size="${BATCH_SIZE}"
  '

