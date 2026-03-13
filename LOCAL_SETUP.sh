# LOCAL SETUP GUIDE
# Run these commands on YOUR LOCAL MACHINE (not AWS)
# Copy-paste one block at a time.

# ===========================================================================
# STEP 1: Install prerequisites
# ===========================================================================

# --- macOS ---
# Install Homebrew if you don't have it:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install git, AWS CLI, Python:
brew install git awscli python@3.11

# --- Windows (run in PowerShell as Administrator) ---
# winget install Git.Git
# winget install Amazon.AWSCLI
# winget install Python.Python.3.11

# --- Ubuntu/Debian Linux ---
# sudo apt-get update
# sudo apt-get install -y git python3.11 python3-pip unzip
# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
# unzip awscliv2.zip && sudo ./aws/install


# ===========================================================================
# STEP 2: Configure AWS credentials
# ===========================================================================
# You need: AWS Access Key ID + Secret Access Key
# Get these from: AWS Console → IAM → Users → Your User → Security credentials
# → Create access key → Application running outside AWS

aws configure
# It will ask for:
#   AWS Access Key ID:     [paste your key ID]
#   AWS Secret Access Key: [paste your secret]
#   Default region name:   us-east-1
#   Default output format: json

# Verify it worked:
aws sts get-caller-identity
# Should print your account ID and user ARN


# ===========================================================================
# STEP 3: Install Python dependencies for the launch script
# ===========================================================================
pip install boto3


# ===========================================================================
# STEP 4: Set up GitHub
# ===========================================================================

# Configure git identity (one-time):
git config --global user.email "your@email.com"
git config --global user.name "Your Name"

# Create a GitHub Personal Access Token (needed to push):
# GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
# → Generate new token → check "repo" scope → copy the token

# --- Option A: Create a NEW repo on GitHub ---
# 1. Go to github.com → New repository
# 2. Name it "trs-rl", set to Private, DON'T initialize with README
# 3. Copy the HTTPS URL: https://github.com/YOUR_USERNAME/trs-rl.git

# --- Then push the code: ---
cd /path/to/trs-rl          # wherever you downloaded the code
git init
git add .
git commit -m "Initial TRS-RL codebase"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/trs-rl.git
git push -u origin main

# If it asks for credentials:
#   Username: your GitHub username
#   Password: your Personal Access Token (NOT your GitHub password)


# ===========================================================================
# STEP 5: Edit aws_launch.py with your repo URL
# ===========================================================================
# Open aws_launch.py and update line ~52:
#   "github_repo": "https://github.com/YOUR_USERNAME/trs-rl.git",
# Change YOUR_USERNAME to your actual GitHub username.

# Also choose your instance type:
#   "instance_type": "p3.2xlarge"   # 1x V100,  $3/hr  — for testing
#   "instance_type": "p4d.24xlarge" # 8x A100, $32/hr  — for real training
#   "instance_type": "p5.48xlarge"  # 8x H100, $98/hr  — for fast training


# ===========================================================================
# STEP 6: Launch your AWS instance
# ===========================================================================
python3 aws_launch.py launch

# This will:
#   - Create an SSH key pair (saved to ~/.ssh/trs-rl-key.pem)
#   - Create a security group (allows SSH)
#   - Launch the instance with Deep Learning AMI
#   - Wait for it to boot
#   - Install Docker + NVIDIA Container Toolkit
#   - Print connection instructions


# ===========================================================================
# STEP 7: Connect and start training
# ===========================================================================
python3 aws_launch.py ssh
# You are now ON THE AWS INSTANCE

# On the instance:
git clone https://github.com/YOUR_USERNAME/trs-rl.git ~/trs-rl
cd ~/trs-rl
chmod +x docker_train.sh
./docker_train.sh    # builds image, runs checks, starts training


# ===========================================================================
# USEFUL COMMANDS (run locally)
# ===========================================================================

# Watch training logs live:
python3 aws_launch.py logs

# Check GPU usage + container status:
python3 aws_launch.py status

# Push code changes without committing to git (fast iteration):
python3 aws_launch.py push

# Stop instance when done (saves money — EBS preserved):
python3 aws_launch.py stop

# Permanently delete everything:
python3 aws_launch.py terminate


# ===========================================================================
# AWS COSTS REFERENCE
# ===========================================================================
# p3.2xlarge  (1x V100 16GB):  ~$3.06/hr   — start here, test the setup
# p3.8xlarge  (4x V100 64GB):  ~$12.24/hr  — good for 3B model
# p4d.24xlarge (8x A100 40GB): ~$32.77/hr  — production training
# p5.48xlarge  (8x H100 80GB): ~$98.32/hr  — fastest (overkill for 1.5B)
#
# COST ESTIMATE for full experiment (1.5B model, all 5 phases, ~3 days):
#   p3.2xlarge × 72 hrs = ~$220
#   p4d.24xlarge × 16 hrs = ~$524  (much faster per phase)
#
# TIP: Use spot instances for 60-70% discount:
#   python3 aws_launch.py launch --spot
#   Risk: can be interrupted. Save checkpoints frequently (already configured).


# ===========================================================================
# TROUBLESHOOTING
# ===========================================================================

# "Unable to locate credentials" from aws configure:
#   → Make sure you ran: aws configure
#   → Check: cat ~/.aws/credentials

# "InvalidClientTokenId":
#   → Your Access Key ID is wrong. Regenerate in AWS Console.

# "InsufficientInstanceCapacity" for p5 instances:
#   → H100s are scarce. Try a different region:
#     aws_launch.py  → change "region": "us-west-2" or "eu-west-1"
#   → Or use p4d.24xlarge instead

# "PermissionDenied" on SSH:
#   → chmod 600 ~/.ssh/trs-rl-key.pem

# Docker build fails on instance:
#   → Check disk space: df -h
#   → Prune Docker: docker system prune -a

# CUDA out of memory during training:
#   → Edit scripts/train.py → reduce --group-size to 4
#   → Or reduce --max-tokens to 512
