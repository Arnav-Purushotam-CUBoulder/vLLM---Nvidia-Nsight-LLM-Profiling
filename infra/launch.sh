#!/bin/bash
set -e

PROFILE="vllm-project"
REGION="us-east-1"
AMI="ami-057b641f1539dc1c4"
INSTANCE_TYPE="g6.xlarge"
KEY_NAME="vllm-nsight-key"
SG_ID="sg-024b9365bc14014c9"
EBS_SIZE=75
PROJECT_TAG="vllm-nsight-profiling"

echo "Launching $INSTANCE_TYPE instance..."

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$EBS_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=$PROJECT_TAG},{Key=Name,Value=vllm-nsight-gpu}]" "ResourceType=volume,Tags=[{Key=Project,Value=$PROJECT_TAG},{Key=Name,Value=vllm-nsight-ebs}]" \
  --query 'Instances[0].InstanceId' \
  --output text \
  --region "$REGION" \
  --profile "$PROFILE")

echo "Instance launched: $INSTANCE_ID"
echo "Waiting for instance to be running..."

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION" --profile "$PROFILE"

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region "$REGION" \
  --profile "$PROFILE")

echo "Instance running!"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP:   $PUBLIC_IP"
echo ""
echo "SSH command:"
echo "  ssh -i ../vllm-nsight-key.pem ubuntu@$PUBLIC_IP"

# Save state
cd "$(dirname "$0")"
python3 -c "
import json
with open('config.json', 'r') as f:
    cfg = json.load(f)
cfg['instance_id'] = '$INSTANCE_ID'
cfg['public_ip'] = '$PUBLIC_IP'
with open('config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"
echo "Config updated with instance ID and IP."
