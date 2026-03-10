#!/bin/bash
# Run this ONCE after SSH-ing into a fresh EC2 instance.
# It installs Docker, NVIDIA container toolkit, vLLM, and Nsight Systems.
#
# The Deep Learning AMI should already have NVIDIA drivers + Docker.
# This script ensures everything else is ready.
#
# Usage: bash scripts/setup_instance.sh

set -e

echo "=== EC2 Instance Setup for vLLM + Nsight Project ==="
echo ""

# 1. Verify GPU
echo "Step 1: Checking GPU..."
nvidia-smi
echo ""

# 2. Verify Docker
echo "Step 2: Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose-plugin
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to log out and back in for group changes."
else
    echo "Docker is installed: $(docker --version)"
fi
echo ""

# 3. Verify NVIDIA Container Toolkit
echo "Step 3: Checking NVIDIA Container Toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "NVIDIA Container Toolkit installed."
else
    echo "NVIDIA Container Toolkit is installed."
fi
echo ""

# 4. Test Docker GPU access
echo "Step 4: Testing Docker GPU access..."
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
echo ""

# 5. Install Nsight Systems CLI
echo "Step 5: Checking Nsight Systems..."
if ! command -v nsys &> /dev/null; then
    echo "Installing Nsight Systems CLI..."
    sudo apt-get install -y nsight-systems-cli 2>/dev/null || {
        echo "nsight-systems-cli not in apt. Trying nsight-systems..."
        sudo apt-get install -y nsight-systems 2>/dev/null || {
            echo "WARNING: Could not install Nsight Systems via apt."
            echo "You may need to install it manually from:"
            echo "  https://developer.nvidia.com/nsight-systems"
        }
    }
else
    echo "Nsight Systems is installed: $(nsys --version)"
fi
echo ""

# 6. Install Python dependencies (for running vLLM outside Docker for profiling)
echo "Step 6: Installing Python dependencies..."
pip install vllm --quiet 2>/dev/null || pip3 install vllm --quiet 2>/dev/null || {
    echo "WARNING: Could not install vLLM via pip."
    echo "You can still use the Docker setup for benchmarking."
    echo "For Nsight profiling, you'll need vLLM installed locally."
}
echo ""

echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Clone/copy your project files to this instance"
echo "  2. Start the server:    bash scripts/run_server.sh"
echo "  3. Test the server:     bash scripts/test_server.sh"
echo "  4. Run benchmarks:      python3 benchmark.py"
echo "  5. Run profiling:       bash scripts/profile_inference.sh"
