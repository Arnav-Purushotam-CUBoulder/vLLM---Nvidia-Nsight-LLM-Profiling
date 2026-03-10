#!/bin/bash
set -e
PROFILE="vllm-project"
REGION="us-east-1"

INSTANCE_ID=$(python3 -c "import json; print(json.load(open('$(dirname $0)/config.json'))['instance_id'])")

if [ "$INSTANCE_ID" = "null" ] || [ -z "$INSTANCE_ID" ]; then
  echo "No instance ID found in config.json"
  exit 1
fi

echo "WARNING: This will permanently terminate instance $INSTANCE_ID and delete its EBS volume."
read -p "Are you sure? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
  echo "Aborted."
  exit 0
fi

aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" --profile "$PROFILE"
echo "Instance terminating. All data on the instance will be lost."
