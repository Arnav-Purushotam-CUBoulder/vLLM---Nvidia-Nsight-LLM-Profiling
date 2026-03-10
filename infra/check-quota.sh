#!/bin/bash
PROFILE="vllm-project"
REQUEST_ID="0a874824d1914f6b892f3ed2c12a0befSWWG9GWh"

echo "Checking GPU quota request status..."
aws service-quotas get-requested-service-quota-change \
  --request-id "$REQUEST_ID" \
  --profile "$PROFILE" \
  --query 'RequestedQuota.[Status,DesiredValue,QuotaName]' \
  --output text

echo ""
echo "Current G-instance vCPU limit:"
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --profile "$PROFILE" \
  --query 'Quota.Value' \
  --output text
