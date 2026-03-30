#!/bin/bash
# ONE-COMMAND SETUP: Creates all AWS resources from scratch and launches a ready-to-use GPU instance.
#
# Usage: bash infra/setup_all.sh
#
# What it does:
#   1. Creates SSH key pair
#   2. Creates security group (SSH locked to your IP)
#   3. Launches g6.xlarge with NVIDIA Deep Learning AMI
#   4. Waits for instance to be ready
#   5. Expands disk to 150GB
#   6. SSHs in and runs full setup (Docker, Nsight, vLLM, clone repo)
#   7. Starts vLLM server serving Phi-2
#   8. Prints SSH command — you're ready to go
#
# Prerequisites:
#   - AWS CLI configured with profile 'vllm-project'
#   - GPU quota of at least 4 vCPUs for G instances in us-east-1

set -e

PROFILE="vllm-project"
REGION="us-east-1"
AMI="ami-057b641f1539dc1c4"
INSTANCE_TYPE="g6.xlarge"
KEY_NAME="vllm-nsight-key"
EBS_SIZE=150
PROJECT_TAG="vllm-nsight-profiling"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KEY_FILE="$SCRIPT_DIR/../vllm-nsight-key.pem"
REPO_URL="https://github.com/Arnav-Purushotam-CUBoulder/vLLM---Nvidia-Nsight-LLM-Profiling.git"

echo "============================================"
echo "  vLLM + Nsight Project — Full Setup"
echo "============================================"
echo ""

# --------------------------------------------------
# Step 1: Create key pair
# --------------------------------------------------
echo "[1/7] Creating SSH key pair..."
aws ec2 delete-key-pair --key-name "$KEY_NAME" --region "$REGION" --profile "$PROFILE" 2>/dev/null || true
aws ec2 create-key-pair \
  --key-name "$KEY_NAME" \
  --key-type ed25519 \
  --query 'KeyMaterial' \
  --output text \
  --tag-specifications "ResourceType=key-pair,Tags=[{Key=Project,Value=$PROJECT_TAG}]" \
  --region "$REGION" \
  --profile "$PROFILE" > "$KEY_FILE"
chmod 400 "$KEY_FILE"
echo "  Key saved to $KEY_FILE"

# --------------------------------------------------
# Step 2: Create security group
# --------------------------------------------------
echo "[2/7] Creating security group..."
MY_IP=$(curl -s https://checkip.amazonaws.com)

# Get default VPC
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' --output text --region "$REGION" --profile "$PROFILE")

SG_ID=$(aws ec2 create-security-group \
  --group-name "vllm-nsight-sg-$(date +%s)" \
  --description "vLLM Nsight project SSH access" \
  --vpc-id "$VPC_ID" \
  --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=$PROJECT_TAG},{Key=Name,Value=vllm-nsight-sg}]" \
  --query 'GroupId' --output text \
  --region "$REGION" --profile "$PROFILE")

aws ec2 authorize-security-group-ingress \
  --group-id "$SG_ID" \
  --protocol tcp --port 22 \
  --cidr "${MY_IP}/32" \
  --region "$REGION" --profile "$PROFILE" > /dev/null

echo "  Security group: $SG_ID (SSH from $MY_IP)"

# --------------------------------------------------
# Step 3: Launch instance
# --------------------------------------------------
echo "[3/7] Launching $INSTANCE_TYPE instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$EBS_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=$PROJECT_TAG},{Key=Name,Value=vllm-nsight-gpu}]" "ResourceType=volume,Tags=[{Key=Project,Value=$PROJECT_TAG},{Key=Name,Value=vllm-nsight-ebs}]" \
  --query 'Instances[0].InstanceId' \
  --output text \
  --region "$REGION" --profile "$PROFILE")

echo "  Instance: $INSTANCE_ID"
echo "  Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION" --profile "$PROFILE"

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text --region "$REGION" --profile "$PROFILE")

echo "  Running at $PUBLIC_IP"

# --------------------------------------------------
# Step 4: Save config
# --------------------------------------------------
echo "[4/7] Saving config..."
cat > "$SCRIPT_DIR/config.json" << CONFIGEOF
{
  "project": "$PROJECT_TAG",
  "region": "$REGION",
  "instance_type": "$INSTANCE_TYPE",
  "ami_id": "$AMI",
  "key_name": "$KEY_NAME",
  "key_file": "../vllm-nsight-key.pem",
  "security_group_id": "$SG_ID",
  "vpc_id": "$VPC_ID",
  "ebs_size_gb": $EBS_SIZE,
  "instance_id": "$INSTANCE_ID",
  "public_ip": "$PUBLIC_IP"
}
CONFIGEOF

# --------------------------------------------------
# Step 5: Wait for SSH and setup instance
# --------------------------------------------------
echo "[5/7] Waiting for SSH to be ready..."
for i in $(seq 1 30); do
  if ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$PUBLIC_IP "echo ok" 2>/dev/null; then
    break
  fi
  sleep 5
done

echo "[6/7] Setting up instance (Docker, Nsight, vLLM, repo)..."
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP << 'SETUPEOF'
set -e

# Clone project
git clone https://github.com/Arnav-Purushotam-CUBoulder/vLLM---Nvidia-Nsight-LLM-Profiling.git ~/project 2>/dev/null || (cd ~/project && git pull)
cd ~/project

# Verify GPU
echo "--- GPU Check ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Docker should be pre-installed on Deep Learning AMI
echo "--- Docker Check ---"
docker --version

# Install NVIDIA Container Toolkit if needed
if ! dpkg -l | grep -q nvidia-container-toolkit 2>/dev/null; then
  echo "Installing NVIDIA Container Toolkit..."
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

# Install Nsight Systems
if ! command -v nsys &> /dev/null; then
  sudo apt-get install -y -qq nsight-systems-2025.6.3 2>/dev/null || true
  sudo chmod -R o+rx /opt/nvidia 2>/dev/null || true
  echo 'export PATH=/opt/nvidia/nsight-systems/2025.6.3/bin:$PATH' >> ~/.bashrc
fi

# Install vLLM for local profiling
pip install vllm --quiet 2>/dev/null || true

echo "--- Setup Complete ---"
SETUPEOF

# --------------------------------------------------
# Step 7: Start vLLM server
# --------------------------------------------------
echo "[7/7] Starting vLLM server with Phi-2..."
ssh -i "$KEY_FILE" ubuntu@$PUBLIC_IP "cd ~/project && docker compose up -d 2>&1 | tail -3"

echo ""
echo "============================================"
echo "  SETUP COMPLETE"
echo "============================================"
echo ""
echo "  Instance ID:  $INSTANCE_ID"
echo "  Public IP:    $PUBLIC_IP"
echo "  Security Grp: $SG_ID"
echo ""
echo "  SSH:"
echo "    ssh -i \"$KEY_FILE\" ubuntu@$PUBLIC_IP"
echo ""
echo "  The vLLM server is starting (wait ~2 min for model to load)."
echo "  Then run:"
echo "    cd ~/project"
echo "    bash scripts/test_server.sh"
echo "    python3 benchmark_stress.py"
echo ""
echo "  IMPORTANT: When done, run:"
echo "    bash infra/teardown_all.sh"
echo "============================================"
