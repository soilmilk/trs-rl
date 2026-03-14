# TRS-RL Training Container
# Base: NVIDIA PyTorch image with CUDA 12.1 + cuDNN 8 (optimal for H100)
FROM nvcr.io/nvidia/pytorch:24.01-py3

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    tree \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /workspace/trs-rl

# ── Python dependencies (cached layer — copy requirements first) ───────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall \
        numpy==1.24.4 \
        pandas==2.0.3
# ── Copy codebase ──────────────────────────────────────────────────────────────
COPY . .

#add source to pythonpath because we want to be able to import from the root of the repo
ENV PYTHONPATH="/workspace/trs-rl:${PYTHONPATH}" 

# ── Pre-create output directories ─────────────────────────────────────────────
RUN mkdir -p runs data/eval

# ── Default command: run sanity checks (override for training) ─────────────────
CMD ["python3", "scripts/sanity_check.py"]
