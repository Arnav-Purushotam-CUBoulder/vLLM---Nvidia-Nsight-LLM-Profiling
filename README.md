# GPU LLM Inference Profiling with vLLM and NVIDIA Nsight

Serve a small LLM (Phi-2) with vLLM on an NVIDIA GPU, benchmark inference performance under varying loads, and profile GPU execution with NVIDIA Nsight Systems.

## Setup

### Infrastructure

- **Instance**: AWS EC2 g6.xlarge (NVIDIA L4, 24GB VRAM)
- **AMI**: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
- **Model**: Microsoft Phi-2 (2.7B parameters, ~6GB VRAM)

### First-Time Instance Setup

SSH into your EC2 instance and run:

```bash
git clone <your-repo-url> ~/project && cd ~/project
bash scripts/setup_instance.sh
```

This verifies GPU/Docker/Nsight and installs dependencies.

## Usage

### 1. Start the vLLM Server

```bash
bash scripts/run_server.sh
```

Wait ~60-90 seconds for the model to load, then verify:

```bash
bash scripts/test_server.sh
```

### 2. Run Benchmarks

```bash
python3 benchmark.py
```

This runs 6 test cases (short/long prompts x concurrency 1/4/8) and saves results to `results.json`.

Options:
```bash
python3 benchmark.py --url http://localhost:8000 --output results.json
```

### 3. Profile with Nsight Systems

Stop the Docker server first, then run vLLM under Nsight:

```bash
docker compose down
bash scripts/profile_inference.sh
```

This captures a ~60s trace of GPU activity during inference. Copy the `.nsys-rep` file to your local machine and open in Nsight Systems GUI.

## Benchmark Cases

| Case | Prompt | Concurrency | Requests |
|------|--------|-------------|----------|
| short_prompt_c1 | Short | 1 | 10 |
| short_prompt_c4 | Short | 4 | 16 |
| short_prompt_c8 | Short | 8 | 16 |
| long_prompt_c1 | Long | 1 | 10 |
| long_prompt_c4 | Long | 4 | 16 |
| long_prompt_c8 | Long | 8 | 16 |

## What to Look For

### Benchmarks
- How does latency change with concurrency?
- How does throughput improve (or stop improving) with more concurrent requests?
- How does prompt length affect latency and throughput?

### Nsight Profiling
- Is the GPU busy or idle during inference?
- Are there gaps between kernel launches?
- Is there CPU-GPU overlap during token generation?
- How does the timeline differ between short and long prompts?

## Results

_(To be filled after running experiments)_

## Project Structure

```
.
├── benchmark.py                 # Main benchmark script
├── docker-compose.yml           # vLLM server config
├── scripts/
│   ├── run_server.sh            # Start vLLM in Docker
│   ├── test_server.sh           # Quick server health check
│   ├── setup_instance.sh        # One-time EC2 setup
│   ├── profile_inference.sh     # Nsight Systems profiling
│   └── profile_requests.py      # Send requests during profiling
├── infra/
│   ├── launch.sh                # Launch EC2 instance
│   ├── stop.sh                  # Stop instance (preserve disk)
│   ├── terminate.sh             # Terminate instance
│   └── check-quota.sh           # Check GPU quota status
├── results.json                 # Benchmark output (generated)
└── README.md
```

## Future Extensions

- Compare across serving frameworks (vLLM, TensorRT-LLM, SGLang)
- Separate prefill vs decode latency
- Profile KV cache memory growth under load
- Use Nsight Compute for kernel-level analysis
- Test quantized vs fp16 models
- Multi-GPU / tensor parallel serving
