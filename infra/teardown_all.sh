#!/bin/bash
# TEARDOWN: Deletes ALL AWS resources for this project. Cost goes to $0.
#
# Usage: bash infra/teardown_all.sh

set -e

PROFILE="vllm-project"
REGION="us-east-1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/config.json"

if [ ! -f "$CONFIG" ]; then
  echo "No config.json found. Nothing to tear down."
  exit 0
fi

INSTANCE_ID=$(python3 -c "import json; print(json.load(open('$CONFIG')).get('instance_id', 'null'))")
SG_ID=$(python3 -c "import json; print(json.load(open('$CONFIG')).get('security_group_id', 'null'))")
KEY_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG')).get('key_name', 'null'))")

echo "============================================"
echo "  Tearing down all AWS resources"
echo "============================================"
echo ""

# Terminate instance
if [ "$INSTANCE_ID" != "null" ] && [ -n "$INSTANCE_ID" ]; then
  echo "[1/3] Terminating instance $INSTANCE_ID..."
  aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" \
    --region "$REGION" --profile "$PROFILE" > /dev/null 2>&1 || echo "  (already terminated)"
  echo "  Waiting for termination..."
  aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID" \
    --region "$REGION" --profile "$PROFILE" 2>/dev/null || true
  echo "  Instance terminated."
else
  echo "[1/3] No instance to terminate."
fi

# Delete security group (may need to wait for instance to fully terminate)
if [ "$SG_ID" != "null" ] && [ -n "$SG_ID" ]; then
  echo "[2/3] Deleting security group $SG_ID..."
  sleep 5
  aws ec2 delete-security-group --group-id "$SG_ID" \
    --region "$REGION" --profile "$PROFILE" 2>/dev/null || echo "  (already deleted or still in use, retry in 30s)"
  echo "  Security group deleted."
else
  echo "[2/3] No security group to delete."
fi

# Delete key pair
if [ "$KEY_NAME" != "null" ] && [ -n "$KEY_NAME" ]; then
  echo "[3/3] Deleting key pair $KEY_NAME..."
  aws ec2 delete-key-pair --key-name "$KEY_NAME" \
    --region "$REGION" --profile "$PROFILE" 2>/dev/null || echo "  (already deleted)"
  rm -f "$SCRIPT_DIR/../vllm-nsight-key.pem"
  echo "  Key pair deleted."
else
  echo "[3/3] No key pair to delete."
fi

# Reset config
cat > "$CONFIG" << 'EOF'
{
  "project": "vllm-nsight-profiling",
  "region": "us-east-1",
  "instance_type": "g6.xlarge",
  "instance_id": null,
  "public_ip": null,
  "security_group_id": null,
  "key_name": "vllm-nsight-key"
}
EOF

echo ""
echo "============================================"
echo "  ALL RESOURCES DELETED — Cost is now $0"
echo "============================================"
echo ""
echo "  To set up again later: bash infra/setup_all.sh"
