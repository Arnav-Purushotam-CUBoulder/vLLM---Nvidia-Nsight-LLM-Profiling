#!/bin/bash
# Profile vLLM inference with NVIDIA Nsight Systems
#
# This script:
#   1. Starts vLLM serving Phi-2 directly (not in Docker) so Nsight can trace GPU activity
#   2. Sends a few requests while Nsight is recording
#   3. Saves the .nsys-rep trace file for analysis
#
# Prerequisites:
#   - nsys (Nsight Systems CLI) installed
#   - vLLM installed: pip install vllm
#   - Model already downloaded (or will download on first run)
#
# Usage: bash scripts/profile_inference.sh

set -e

REPORT_NAME="vllm_phi2_profile"
MODEL="microsoft/phi-2"
PORT=8001  # different port to avoid conflict with Docker server
DURATION=60  # seconds to profile

echo "=== Nsight Systems Profiling for vLLM ==="
echo ""

# Check nsys is available
if ! command -v nsys &> /dev/null; then
    echo "ERROR: nsys not found. Install Nsight Systems first."
    echo "  sudo apt install nsight-systems-cli"
    echo "  OR download from: https://developer.nvidia.com/nsight-systems"
    exit 1
fi

echo "Step 1: Starting vLLM server under Nsight profiler..."
echo "  Model:    $MODEL"
echo "  Port:     $PORT"
echo "  Duration: ${DURATION}s capture"
echo ""

# Start vLLM under nsys profiling in background
nsys profile \
    --output "$REPORT_NAME" \
    --trace cuda,nvtx,osrt \
    --duration "$DURATION" \
    --force-overwrite true \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --max-model-len 2048 &

NSYS_PID=$!

echo "Waiting for server to load model (~30-60s)..."
sleep 45

# Check if server is ready
echo "Step 2: Sending test requests for profiling..."
for i in $(seq 1 5); do
    echo "  Request $i/5..."
    curl -s http://localhost:$PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"prompt\": \"Explain the concept of parallel computing and why GPUs excel at it.\",
            \"max_tokens\": 100,
            \"temperature\": 0.0
        }" > /dev/null 2>&1 || echo "  (request $i may have failed, continuing...)"
    sleep 2
done

echo ""
echo "Step 3: Waiting for Nsight capture to complete..."
wait $NSYS_PID 2>/dev/null || true

echo ""
echo "=== Profiling complete ==="
echo ""
echo "Output file: ${REPORT_NAME}.nsys-rep"
echo ""
echo "To view on your Mac:"
echo "  1. Copy the file: scp -i vllm-nsight-key.pem ubuntu@<IP>:~/project/${REPORT_NAME}.nsys-rep ."
echo "  2. Open with Nsight Systems GUI (free download from NVIDIA)"
echo ""
echo "What to look for in the timeline:"
echo "  - GPU kernel activity during inference"
echo "  - CPU-GPU overlap patterns"
echo "  - Idle gaps between requests"
echo "  - Kernel launch patterns during token generation"
