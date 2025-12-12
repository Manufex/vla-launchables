FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG LEROBOT_REF=main

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git git-lfs ca-certificates curl \
    ffmpeg \
    cmake build-essential pkg-config \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN git lfs install --skip-repo || true

# PyTorch (CUDA 12.1)
RUN python3 -m pip install -U pip wheel setuptools && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio

# Tooling used by train.sh
RUN python3 -m pip install -U "huggingface_hub[cli]" wandb

# LeRobot + pi extras (pin via build arg when you want reproducibility)
RUN git clone --depth 1 --branch ${LEROBOT_REF} https://github.com/huggingface/lerobot.git /opt/lerobot && \
    python3 -m pip install -e "/opt/lerobot[pi]"

