#!/usr/bin/env python3
"""
aws_launch.py

One-command setup: spins up an AWS H100 instance, configures it,
pulls the repo, builds the Docker container, and starts training.

PREREQUISITES (run once on your local machine):
  pip install boto3
  aws configure   (enter your Access Key ID, Secret, region)

USAGE:
  # Step 1: Launch instance + full setup
  python3 aws_launch.py launch

  # Step 2: SSH into the instance
  python3 aws_launch.py ssh

  # Step 3: Check training logs live
  python3 aws_launch.py logs

  # Step 4: Stop instance when done (saves money)
  python3 aws_launch.py stop

  # Terminate instance permanently
  python3 aws_launch.py terminate
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

CONFIG = {
    # Instance type — p3.2xlarge = 1x V100 (cheaper for testing)
    #                 p4d.24xlarge = 8x A100
    #                 p5.48xlarge  = 8x H100 (recommended)
    "instance_type": "p5.48xlarge",    # 8x H100 (~$98/hr) — use for real training
    # "instance_type": "p4d.24xlarge",   # 8x A100 (~$32/hr)
    # "instance_type": "p5.48xlarge",    # 8x H100 (~$98/hr) — use for real training

    # Deep Learning AMI (Ubuntu 22.04) with CUDA 12 pre-installed
    # This AMI ID is for us-east-1 — it changes by region (script auto-detects)
    "ami_id": "auto",                    # auto = look up latest DL AMI

    "region": "us-east-1",              # cheapest region for GPU instances
    "key_name": "trs-rl-key",           # SSH key pair name (created automatically)
    "security_group": "trs-rl-sg",      # security group (created automatically)
    "storage_gb": 200,                   # EBS volume (models + checkpoints need space)
    "spot": False,                       # True = spot instance (70% cheaper, can be interrupted)

    "github_repo": "https://github.com/soilmilk/trs-rl.git",

    # Where to store the SSH key locally
    "key_path": str(Path.home() / ".ssh" / "trs-rl-key.pem"),

    # State file — tracks instance ID across commands
    "state_file": ".aws_instance_state.json",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_state(data: dict):
    with open(CONFIG["state_file"], "w") as f:
        json.dump(data, f, indent=2)
    print(f"  State saved to {CONFIG['state_file']}")

def load_state() -> dict:
    if not Path(CONFIG["state_file"]).exists():
        return {}
    with open(CONFIG["state_file"]) as f:
        return json.load(f)

def run(cmd: str, check=True):
    """Run a shell command locally."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result


def ssh_run(ip: str, key: str, command: str, interactive=False):
    """Run a command on the remote instance via SSH."""
    flags = "-tt" if interactive else "-T"
    cmd = (
        f'ssh {flags} -i {key} '
        f'-o StrictHostKeyChecking=no '
        f'-o ServerAliveInterval=60 '
        f'ubuntu@{ip} "{command}"'
    )
    if interactive:
        os.execlp("ssh", "ssh",
                  "-i", key,
                  "-o", "StrictHostKeyChecking=no",
                  "-o", "ServerAliveInterval=60",
                  f"ubuntu@{ip}")
    else:
        return subprocess.run(cmd, shell=True, check=False)


def get_latest_dl_ami(ec2, region: str) -> str:
    """Find the latest AWS Deep Learning AMI (Ubuntu 22.04)."""
    print("  Looking up latest Deep Learning AMI...")
    response = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name",  "Values": ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError("No Deep Learning AMI found. Check region or AMI filters.")
    ami = images[0]
    print(f"  Found AMI: {ami['ImageId']} — {ami['Name'][:80]}")
    return ami["ImageId"]


def ensure_key_pair(ec2, key_name: str, key_path: str) -> str:
    """Create SSH key pair if it doesn't exist."""
    Path(key_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(key_path).exists():
        print(f"  SSH key already exists at {key_path}")
        return key_name

    try:
        # Check if key exists in AWS
        ec2.describe_key_pairs(KeyNames=[key_name])
        print(f"  Key pair {key_name!r} exists in AWS but .pem not found locally.")
        print(f"  If you lost the .pem, delete the key pair in AWS console and re-run.")
        sys.exit(1)
    except ClientError as e:
        if "InvalidKeyPair.NotFound" not in str(e):
            raise

    print(f"  Creating new key pair: {key_name}")
    response = ec2.create_key_pair(KeyName=key_name)
    pem_content = response["KeyMaterial"]
    with open(key_path, "w") as f:
        f.write(pem_content)
    os.chmod(key_path, 0o600)
    print(f"  Key saved to {key_path}")
    return key_name


def ensure_security_group(ec2, sg_name: str) -> str:
    """Create security group with SSH access if it doesn't exist."""
    try:
        response = ec2.describe_security_groups(GroupNames=[sg_name])
        sg_id = response["SecurityGroups"][0]["GroupId"]
        print(f"  Security group {sg_name!r} already exists: {sg_id}")
        return sg_id
    except ClientError:
        pass

    print(f"  Creating security group: {sg_name}")
    response = ec2.create_security_group(
        GroupName=sg_name,
        Description="TRS-RL training instance - SSH access",
    )
    sg_id = response["GroupId"]

    # Allow SSH from anywhere (you can restrict to your IP for security)
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp",
            "FromPort": 22,
            "ToPort": 22,
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}],
        }],
    )
    print(f"  Security group created: {sg_id}")
    return sg_id


# ─────────────────────────────────────────────────────────────────────────────
# Remote setup script (runs on the EC2 instance after boot)
# ─────────────────────────────────────────────────────────────────────────────

REMOTE_SETUP_SCRIPT = """#!/bin/bash
set -e

echo "============================================"
echo "  TRS-RL Remote Setup"
echo "============================================"

# Wait for cloud-init to finish
cloud-init status --wait 2>/dev/null || true

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "[1/5] Installing Docker..."
    curl -fsSL https://get.docker.com | bash
    usermod -aG docker ubuntu
    newgrp docker
else
    echo "[1/5] Docker already installed."
fi

# Install NVIDIA Container Toolkit
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "[2/5] Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -q
    apt-get install -y -q nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
else
    echo "[2/5] NVIDIA Container Toolkit already installed."
fi

# Install AWS CLI v2
if ! command -v aws &> /dev/null; then
    echo "[3/5] Installing AWS CLI..."
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp/
    /tmp/aws/install
    rm -rf /tmp/aws /tmp/awscliv2.zip
else
    echo "[3/5] AWS CLI already installed."
fi

# Verify GPU
echo "[4/5] GPU check:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Install git (usually present on DL AMIs)
apt-get install -y -q git tmux htop

echo "[5/5] Remote setup complete."
echo ""
echo "Next: clone your repo and build the Docker image."
echo "  git clone REPO_URL ~/trs-rl"
echo "  cd ~/trs-rl"
echo "  docker build -t trs-rl ."
echo "  docker run --gpus all trs-rl  # runs sanity checks"
"""

REMOTE_TRAIN_SCRIPT = """#!/bin/bash
# Quick-start training script — run this on the instance

set -e
cd ~/trs-rl

# Pull latest code
git pull origin main

# Build Docker image (uses cache — fast if no changes)
docker build -t trs-rl .

# Generate eval sets (idempotent — skips if already done)
docker run --gpus all -v $(pwd)/data:/workspace/trs-rl/data trs-rl \\
    python3 data/generate_eval.py --seed 42

# Run sanity checks
docker run --gpus all trs-rl python3 scripts/sanity_check.py

# Start training (persists runs/ to host volume)
docker run --gpus all \\
    -v $(pwd)/runs:/workspace/trs-rl/runs \\
    -v $(pwd)/data:/workspace/trs-rl/data \\
    --name trs-rl-training \\
    --restart unless-stopped \\
    -d trs-rl \\
    python3 scripts/train.py \\
        --model /workspace/models/Qwen3.5-2B \\
        --output-dir runs/trs_rl \\
        --max-steps 8000

echo ""
echo "Training started in background container: trs-rl-training"
echo "Watch logs: docker logs -f trs-rl-training"
echo "GPU usage:  watch -n 2 nvidia-smi"
"""


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_launch(args):
    """Launch a new EC2 instance and set everything up."""
    state = load_state()
    if state.get("instance_id"):
        print(f"Instance already exists: {state['instance_id']}")
        print(f"IP: {state.get('public_ip', 'unknown')}")
        print("Use 'python3 aws_launch.py ssh' to connect, or 'terminate' to destroy it.")
        return

    ec2 = boto3.client("ec2", region_name=CONFIG["region"])
    print(f"\n{'='*60}")
    print(f"  Launching {CONFIG['instance_type']} in {CONFIG['region']}")
    print(f"{'='*60}\n")

    # Setup infrastructure
    ami_id = get_latest_dl_ami(ec2, CONFIG["region"]) if CONFIG["ami_id"] == "auto" else CONFIG["ami_id"]
    key_name = ensure_key_pair(ec2, CONFIG["key_name"], CONFIG["key_path"])
    sg_id = ensure_security_group(ec2, CONFIG["security_group"])

    # Launch instance
    print(f"\n  Launching instance...")
    launch_kwargs = dict(
        ImageId           = ami_id,
        InstanceType      = CONFIG["instance_type"],
        KeyName           = key_name,
        SecurityGroupIds  = [sg_id],
        MinCount          = 1,
        MaxCount          = 1,
        BlockDeviceMappings = [{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": CONFIG["storage_gb"],
                "VolumeType": "gp3",
                "DeleteOnTermination": True,
            },
        }],
        TagSpecifications = [{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": "trs-rl-training"}],
        }],
    )

    if CONFIG["spot"]:
        launch_kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time"},
        }

    response = ec2.run_instances(**launch_kwargs)
    instance = response["Instances"][0]
    instance_id = instance["InstanceId"]
    print(f"  Instance launched: {instance_id}")

    # Wait for running state
    print("  Waiting for instance to be running", end="", flush=True)
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    print(" ✓")

    # Get public IP
    desc = ec2.describe_instances(InstanceIds=[instance_id])
    public_ip = desc["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")
    print(f"  Public IP: {public_ip}")

    # Save state
    save_state({
        "instance_id": instance_id,
        "public_ip":   public_ip,
        "region":      CONFIG["region"],
        "key_path":    CONFIG["key_path"],
        "instance_type": CONFIG["instance_type"],
    })

    # Wait for SSH to be ready
    print("\n  Waiting for SSH to be ready", end="", flush=True)
    for _ in range(40):
        result = subprocess.run(
            f'ssh -i {CONFIG["key_path"]} -o StrictHostKeyChecking=no '
            f'-o ConnectTimeout=5 ubuntu@{public_ip} "echo ok"',
            shell=True, capture_output=True,
        )
        if result.returncode == 0:
            print(" ✓")
            break
        print(".", end="", flush=True)
        time.sleep(10)
    else:
        print("\n  WARNING: SSH not ready after 400s. Try connecting manually.")
        return

    # Run remote setup
    print("\n  Running remote setup (Docker, NVIDIA toolkit, etc.)...")
    setup_cmd = REMOTE_SETUP_SCRIPT.replace("\n", "; ").replace("#!/bin/bash; ", "")
    result = subprocess.run(
        f'ssh -i {CONFIG["key_path"]} -o StrictHostKeyChecking=no '
        f'ubuntu@{public_ip} "sudo bash -s"',
        input=REMOTE_SETUP_SCRIPT.encode(),
        shell=True,
    )

    print(f"\n{'='*60}")
    print(f"  Instance ready!")
    print(f"  ID:  {instance_id}")
    print(f"  IP:  {public_ip}")
    print(f"  Key: {CONFIG['key_path']}")
    print(f"\n  Next steps:")
    print(f"  1. python3 aws_launch.py ssh      # connect")
    print(f"  2. git clone {CONFIG['github_repo']} ~/trs-rl")
    print(f"  3. cd ~/trs-rl && bash docker_train.sh")
    print(f"{'='*60}\n")


def cmd_ssh(args):
    """Open interactive SSH session."""
    state = load_state()
    if not state.get("instance_id"):
        print("No instance found. Run: python3 aws_launch.py launch")
        return

    ip  = state["public_ip"]
    key = state["key_path"]
    print(f"Connecting to ubuntu@{ip}...")
    os.execlp("ssh", "ssh",
              "-i", key,
              "-o", "StrictHostKeyChecking=no",
              "-o", "ServerAliveInterval=60",
              f"ubuntu@{ip}")


def cmd_logs(args):
    """Tail training logs from the Docker container."""
    state = load_state()
    if not state.get("public_ip"):
        print("No instance found.")
        return
    ip  = state["public_ip"]
    key = state["key_path"]
    print(f"Tailing logs from trs-rl-training container on {ip}...")
    print("(Ctrl+C to stop watching)\n")
    os.execlp("ssh", "ssh",
              "-i", key,
              "-o", "StrictHostKeyChecking=no",
              f"ubuntu@{ip}",
              "docker logs -f trs-rl-training")


def cmd_status(args):
    """Print instance status and current training progress."""
    state = load_state()
    if not state.get("instance_id"):
        print("No instance found.")
        return

    ec2 = boto3.client("ec2", region_name=CONFIG["region"])
    try:
        desc = ec2.describe_instances(InstanceIds=[state["instance_id"]])
        inst = desc["Reservations"][0]["Instances"][0]
        status = inst["State"]["Name"]
        ip = inst.get("PublicIpAddress", "none")
    except Exception as e:
        status = f"error: {e}"
        ip = state.get("public_ip", "unknown")

    print(f"\nInstance: {state['instance_id']}")
    print(f"Status:   {status}")
    print(f"IP:       {ip}")
    print(f"Type:     {state.get('instance_type', '?')}")

    if status == "running":
        key = state["key_path"]
        print("\nGPU status:")
        subprocess.run(
            f'ssh -i {key} -o StrictHostKeyChecking=no ubuntu@{ip} '
            f'"nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total '
            f'--format=csv,noheader"',
            shell=True, check=False,
        )
        print("\nContainer status:")
        subprocess.run(
            f'ssh -i {key} -o StrictHostKeyChecking=no ubuntu@{ip} '
            f'"docker ps --format \'table {{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.RunningFor}}}}\'"',
            shell=True, check=False,
        )


def cmd_stop(args):
    """Stop instance (preserves data, stops billing for compute)."""
    state = load_state()
    if not state.get("instance_id"):
        print("No instance found.")
        return
    ec2 = boto3.client("ec2", region_name=CONFIG["region"])
    print(f"Stopping {state['instance_id']}...")
    ec2.stop_instances(InstanceIds=[state["instance_id"]])
    print("Instance stopping. EBS data is preserved.")
    print("Use 'python3 aws_launch.py launch' logic to restart (start, not launch).")


def cmd_terminate(args):
    """Terminate instance permanently (deletes all data)."""
    state = load_state()
    if not state.get("instance_id"):
        print("No instance found.")
        return

    confirm = input(
        f"\nWARNING: This will PERMANENTLY delete {state['instance_id']} "
        f"and all its data.\nType the instance ID to confirm: "
    ).strip()

    if confirm != state["instance_id"]:
        print("Aborted.")
        return

    ec2 = boto3.client("ec2", region_name=CONFIG["region"])
    ec2.terminate_instances(InstanceIds=[state["instance_id"]])
    print("Instance terminated.")
    Path(CONFIG["state_file"]).unlink(missing_ok=True)


def cmd_push_code(args):
    """Sync local code changes to the running instance (bypasses git for quick iteration)."""
    state = load_state()
    if not state.get("public_ip"):
        print("No running instance found.")
        return
    ip  = state["public_ip"]
    key = state["key_path"]

    project_root = Path(__file__).parent
    print(f"Syncing {project_root} → {ip}:~/trs-rl/")

    run(
        f'rsync -avz --exclude ".git" --exclude "__pycache__" --exclude "runs/" '
        f'--exclude "data/eval/*.jsonl" '
        f'-e "ssh -i {key} -o StrictHostKeyChecking=no" '
        f'{project_root}/ ubuntu@{ip}:~/trs-rl/'
    )
    print("Sync complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

COMMANDS = {
    "launch":    (cmd_launch,    "Launch a new H100 instance and set up everything"),
    "ssh":       (cmd_ssh,       "Open an interactive SSH session"),
    "logs":      (cmd_logs,      "Tail training logs from Docker container"),
    "status":    (cmd_status,    "Show instance status and GPU utilization"),
    "stop":      (cmd_stop,      "Stop instance (save money, keep data)"),
    "terminate": (cmd_terminate, "Permanently delete instance"),
    "push":      (cmd_push_code, "Rsync local code to instance (fast iteration)"),
}


def main():
    parser = argparse.ArgumentParser(
        description="TRS-RL AWS Instance Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:12s} {v[1]}" for k, v in COMMANDS.items()),
    )
    parser.add_argument("command", choices=list(COMMANDS.keys()))
    parser.add_argument("--instance-type", help=f"Override instance type (default: {CONFIG['instance_type']})")
    parser.add_argument("--spot", action="store_true", help="Use spot instance (70% cheaper)")
    parser.add_argument("--repo", help="GitHub repo URL to clone")
    args = parser.parse_args()

    if args.instance_type:
        CONFIG["instance_type"] = args.instance_type
    if args.spot:
        CONFIG["spot"] = True
    if args.repo:
        CONFIG["github_repo"] = args.repo

    fn, _ = COMMANDS[args.command]
    fn(args)


if __name__ == "__main__":
    main()
