# skillmix-n-skillsbench — vast.ai training image
#
# Built on PyTorch's official CUDA 12.1 image. ~7 GB base + ~3 GB deps.
# Designed to be launched on a vast.ai GPU instance and serve as the
# environment for v2.0+ runs against larger Qwen3.5 variants (4B / 9B).
#
# WHAT'S BAKED IN
#   - Python 3.11 + CUDA 12.1 + cuDNN 9 (from pytorch/pytorch base)
#   - All Python deps for training, eval, and probing
#   - The skillmix-n-skillsbench repo at /workspace/skillmix-n-skillsbench
#
# WHAT'S NOT BAKED IN (fetched at runtime / mounted)
#   - Qwen base weights (transformers downloads on first use → /root/.cache)
#   - The SFT corpus (committed in repo at data/pipeline-runs/default/)
#   - ANTHROPIC_API_KEY (pass via -e at container launch for the judge)
#
# BUILD
#   docker build -t <yourname>/skillmix-skillsbench:latest -f docker/Dockerfile .
#
# PUSH
#   docker push <yourname>/skillmix-skillsbench:latest
#
# LAUNCH ON VAST.AI
#   In "Edit Image & Config" set:
#     Image Path: docker.io/<yourname>/skillmix-skillsbench:latest
#     Launch Mode: SSH (or Jupyter)
#     On-start: bash -lc 'cd /workspace/skillmix-n-skillsbench && bash'
#   In "Environment Variables":
#     ANTHROPIC_API_KEY=...
#   GPU: pick by VRAM target (4B+LoRA: 24GB; 9B+LoRA: 40GB+)

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

LABEL maintainer="skillmix-n-skillsbench"
LABEL description="Training + eval environment for procedural-skill SFT experiments"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/hub \
    TOKENIZERS_PARALLELISM=false

# System tooling: git for repo ops, openssh for vast.ai SSH access,
# build-essential for any source-builds (flash-attn etc.), tmux for
# long-running training over flaky SSH.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        openssh-server \
        build-essential \
        ninja-build \
        curl \
        wget \
        ca-certificates \
        tmux \
        less \
        vim-tiny \
        htop \
    && rm -rf /var/lib/apt/lists/*

# Python deps — pinned floors known to work with the v1.x scripts.
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Bake the repo. Build context should be the repo root, not the docker/
# subdir, so Dockerfile path is referenced as -f docker/Dockerfile from
# the repo root (see BUILD comment above).
WORKDIR /workspace
COPY . /workspace/skillmix-n-skillsbench

# Drop heavy/unwanted artifacts that may have leaked in via the build
# context. .dockerignore should already filter these but defense-in-depth.
RUN cd /workspace/skillmix-n-skillsbench && \
    rm -rf qwen35-* probe-*.jsonl .git/objects/pack/*.pack 2>/dev/null || true

WORKDIR /workspace/skillmix-n-skillsbench

# Default to interactive shell; vast.ai's SSH launch mode handles the
# real entry. For one-shot jobs, override with `docker run ... <command>`.
CMD ["bash"]
