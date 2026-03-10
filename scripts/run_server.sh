#!/bin/bash
# Start the vLLM server serving Phi-2
# Usage: bash scripts/run_server.sh

set -e

echo "Starting vLLM server with Phi-2..."
echo "This will download the model on first run (~5GB). Be patient."
echo ""

docker compose up -d

echo ""
echo "Server starting in background."
echo "Wait ~60-90 seconds for model to load, then test with:"
echo "  curl http://localhost:8000/v1/models"
echo ""
echo "To view logs:"
echo "  docker compose logs -f"
echo ""
echo "To stop:"
echo "  docker compose down"
