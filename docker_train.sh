#!/bin/bash
# docker_train.sh
# Run this ON THE AWS INSTANCE after cloning the repo.
# It builds the image, runs checks, and starts training.

set -e

echo ""
echo "============================================================"
echo "  TRS-RL Docker Training Setup"
echo "============================================================"
echo ""

# ── Parse args ────────────────────────────────────────────────────────────────
MODEL="${1:-/workspace/models/Qwen3.5-2B}"
MAX_STEPS="${2:-8000}"
OUTPUT_DIR="runs/trs_rl_$(date +%Y%m%d_%H%M%S)"

echo "  Model:      $MODEL"
echo "  Max steps:  $MAX_STEPS"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# ── Ensure we're in the project root ──────────────────────────────────────────
cd "$(dirname "$0")"

# ── Pull latest code ──────────────────────────────────────────────────────────
echo "[1/6] Pulling latest code..."
git pull origin main 2>/dev/null || echo "  (not a git repo or no remote, skipping pull)"

# ── Build Docker image ────────────────────────────────────────────────────────
echo "[2/6] Building Docker image (trs-rl)..."
docker build -t trs-rl . --progress=plain 2>&1 | tail -20
echo "  Image built."

# ── Generate eval sets (idempotent) ───────────────────────────────────────────
echo "[3/6] Generating eval sets..."
mkdir -p data/eval
docker run --gpus all --rm \
    -v "$(pwd)/data:/workspace/trs-rl/data" \
    trs-rl \
    python3 data/generate_eval.py --seed 42
echo "  Eval sets ready."

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "[4/6] Running sanity checks..."
docker run --gpus all --rm trs-rl python3 scripts/sanity_check.py
echo "  All checks passed."

# ── Create output directories ─────────────────────────────────────────────────
echo "[5/6] Creating output directories..."
mkdir -p "$OUTPUT_DIR"

# ── Start training ────────────────────────────────────────────────────────────
echo "[6/6] Starting training..."
echo ""

# Remove any existing container with same name
docker rm -f trs-rl-training 2>/dev/null || true

# Use tmux so training persists after SSH disconnect
# The container name is trs-rl-training
tmux new-session -d -s training \
    "docker run --gpus all \
        --name trs-rl-training \
        -v $(pwd)/runs:/workspace/trs-rl/runs \
        -v $(pwd)/data:/workspace/trs-rl/data \
        --shm-size=16g \
        --restart unless-stopped \
        trs-rl \
        python3 scripts/train.py \
            --model '$MODEL' \
            --output-dir '$OUTPUT_DIR' \
            --max-steps $MAX_STEPS \
        2>&1 | tee /workspace/trs-rl/runs/train.log"

echo "============================================================"
echo "  Training started!"
echo ""
echo "  Watch GPU:     watch -n 2 nvidia-smi"
echo "  Watch logs:    docker logs -f trs-rl-training"
echo "  tmux attach:   tmux attach -t training"
echo "  Container:     docker ps"
echo ""
echo "  Output dir:    $OUTPUT_DIR/"
echo "  Log file:      runs/train.log"
echo "============================================================"
echo ""
