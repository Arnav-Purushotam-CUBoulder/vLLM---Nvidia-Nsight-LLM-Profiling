#!/bin/bash
set -e
PROFILE="vllm-project"
REGION="us-east-1"

INSTANCE_ID=$(python3 -c "import json; print(json.load(open('$(dirname $0)/config.json'))['instance_id'])")

if [ "$INSTANCE_ID" = "null" ] || [ -z "$INSTANCE_ID" ]; then
  echo "No instance ID found in config.json"
  exit 1
fi

echo "Stopping instance $INSTANCE_ID..."
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" --profile "$PROFILE"
echo "Instance stopping. EBS volume preserved. Compute billing will stop shortly."
echo "Run launch-stopped.sh to restart it later."
