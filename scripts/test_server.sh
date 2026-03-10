#!/bin/bash
# Quick test to verify the vLLM server is responding
# Usage: bash scripts/test_server.sh

echo "Checking if server is up..."
echo ""

# Check model list
echo "=== Available Models ==="
curl -s http://localhost:8000/v1/models | python3 -m json.tool 2>/dev/null
echo ""

# Send a test completion
echo "=== Test Completion ==="
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/phi-2",
    "prompt": "The capital of France is",
    "max_tokens": 30,
    "temperature": 0.0
  }' | python3 -m json.tool 2>/dev/null

echo ""
echo "If you see a response above, the server is working."
