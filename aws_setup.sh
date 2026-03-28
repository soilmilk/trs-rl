#!/bin/bash
# =============================================================================
# aws_setup.sh
#
# Run this script ON YOUR LOCAL MACHINE.
# It will:
#   1. Fix SSH key permissions
#   2. SSH into your AWS H100 instance
#   3. Install all dependencies
#   4. Upload the codebase
#   5. Run sanity checks
#
# USAGE:
#   chmod +x aws_setup.sh
#   ./aws_setup.sh <path-to-your.pem> <ec2-ip-address>
#
# EXAMPLE:
#   ./aws_setup.sh ~/.ssh/my-key.pem 54.123.45.67
# =============================================================================

set -e  # Exit immediately on any error

PEM_FILE="${1}"
EC2_IP="${2}"
EC2_USER="${3:-ubuntu}"          # default user is 'ubuntu' for Ubuntu AMIs
REMOTE_DIR="${4:-~/trs-rl}"      # where to put the code on the instance

# ── Validate args ─────────────────────────────────────────────────────────────
if [ -z "$PEM_FILE" ] || [ -z "$EC2_IP" ]; then
    echo "Usage: ./aws_setup.sh <path-to-pem> <ec2-ip> [ec2-user] [remote-dir]"
    echo "  ec2-user defaults to 'ubuntu'"
    echo "  remote-dir defaults to ~/trs-rl"
    exit 1
fi

if [ ! -f "$PEM_FILE" ]; then
    echo "ERROR: PEM file not found: $PEM_FILE"
    exit 1
fi

echo ""
echo "============================================================"
echo "  TRS-RL AWS Setup"
echo "  PEM:  $PEM_FILE"
echo "  Host: $EC2_USER@$EC2_IP"
echo "  Dir:  $REMOTE_DIR"
echo "============================================================"
echo ""

# ── Step 1: Fix PEM permissions (AWS requires 600) ───────────────────────────
echo "[1/6] Fixing PEM key permissions..."
chmod 600 "$PEM_FILE"
echo "      Done."

# ── Step 2: Test connection ───────────────────────────────────────────────────
echo "[2/6] Testing SSH connection..."
ssh -i "$PEM_FILE" \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout=15 \
    "$EC2_USER@$EC2_IP" "echo '      SSH connection successful.'"

# ── Step 3: Upload codebase ───────────────────────────────────────────────────
echo "[3/6] Uploading codebase..."
# Get the directory this script lives in (the project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rsync -avz \
    --exclude ".git" \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude "runs/" \
    --exclude "data/eval/*.jsonl" \
    -e "ssh -i $PEM_FILE -o StrictHostKeyChecking=no" \
    "$SCRIPT_DIR/" \
    "$EC2_USER@$EC2_IP:$REMOTE_DIR/"

echo "      Upload complete."

# ── Step 4: Remote setup ──────────────────────────────────────────────────────
echo "[4/6] Setting up remote environment..."

ssh -i "$PEM_FILE" \
    -o StrictHostKeyChecking=no \
    "$EC2_USER@$EC2_IP" << 'REMOTE_SCRIPT'

set -e
echo "  Checking CUDA..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  WARNING: nvidia-smi not found"

echo "  Installing/upgrading pip..."
python3 -m pip install --upgrade pip --quiet

echo "  Installing project dependencies..."
cd ~/trs-rl
pip install -e ".[dev]" --quiet

echo "  Verifying torch + CUDA..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU count: {torch.cuda.device_count()}')
"

echo "  Remote setup complete."
REMOTE_SCRIPT

# ── Step 5: Run sanity checks remotely ───────────────────────────────────────
echo "[5/6] Running sanity checks on remote..."

ssh -i "$PEM_FILE" \
    -o StrictHostKeyChecking=no \
    "$EC2_USER@$EC2_IP" \
    "cd ~/trs-rl && python3 scripts/sanity_check.py"

# ── Step 6: Generate eval sets ────────────────────────────────────────────────
echo "[6/6] Generating eval sets (runs in background on remote)..."

ssh -i "$PEM_FILE" \
    -o StrictHostKeyChecking=no \
    "$EC2_USER@$EC2_IP" \
    "cd ~/trs-rl && nohup python3 data/generate_eval.py --seed 42 > data/eval_gen.log 2>&1 &
     echo 'Eval generation running in background. Check data/eval_gen.log'"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  To SSH in:     ssh -i $PEM_FILE $EC2_USER@$EC2_IP"
echo "  Check eval:    tail -f ~/trs-rl/data/eval_gen.log"
echo "  Start training:"
echo "    ssh -i $PEM_FILE $EC2_USER@$EC2_IP"
echo "    cd ~/trs-rl"
echo "    python3 scripts/train.py --model /workspace/models/Qwen3.5-2B"
echo "============================================================"
echo ""
