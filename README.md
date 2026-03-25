# GPU LLM Inference Profiling with vLLM and NVIDIA Nsight

Serve a small LLM (Phi-2) with vLLM on an NVIDIA GPU, benchmark inference performance under varying loads, and profile GPU execution with NVIDIA Nsight Systems.

## Results

### Baseline Benchmark Results

**Environment:** AWS EC2 g6.xlarge | NVIDIA L4 (24GB) | vLLM 0.19.0 | Phi-2 (2.7B, fp16) | max_tokens=100

| Case | Prompt | Concurrency | Avg Latency | P95 Latency | Req/s | Tok/s |
|------|--------|-------------|-------------|-------------|-------|-------|
| short_prompt_c1 | Short | 1 | 2.180s | 2.181s | 0.46 | 45.9 |
| short_prompt_c4 | Short | 4 | 2.280s | 2.287s | 1.75 | 175.2 |
| short_prompt_c8 | Short | 8 | 2.332s | 2.343s | 3.42 | 342.5 |
| long_prompt_c1 | Long | 1 | 2.199s | 2.202s | 0.46 | 45.5 |
| long_prompt_c4 | Long | 4 | 2.311s | 2.319s | 1.73 | 172.9 |
| long_prompt_c8 | Long | 8 | 2.374s | 2.384s | 3.36 | 336.4 |

### Key Observations

**1. Throughput scales nearly linearly with concurrency.**
Throughput went from 46 tok/s (c=1) to 342 tok/s (c=8) — a 7.4x increase for 8x concurrency. At low concurrency the GPU is underutilized. vLLM's continuous batching groups multiple requests into each decode step, using more of the GPU's parallel capacity without significantly slowing individual requests.

**2. Latency barely increased under load.**
Per-request latency only grew from 2.18s to 2.33s (+7%) going from concurrency 1 to 8, while the system handled 7.4x more total work. This demonstrates that vLLM's iteration-level batching generates one token per active request per decode step in parallel, so adding requests costs very little per-request overhead.

**3. P95 latency stayed within 11ms of the average.**
At concurrency 8: avg = 2.332s, p95 = 2.343s. This means virtually no tail latency — all requests complete in consistent time. The system is not overloaded, there is no queuing, and vLLM's scheduling is uniform. In production, this consistency matters as much as raw speed.

**4. Short vs long prompt showed almost no difference.**
Short prompt c=1: 2.180s vs long prompt c=1: 2.199s (only 19ms difference). This is because even the "long" prompt (~80 tokens) is small relative to the model's context window (2048). The decode phase (generating 100 output tokens sequentially) dominates total latency at ~95% of the time. Prefill (processing the input prompt in parallel) takes only ~20-50ms for short and ~50-100ms for long, so the difference is barely visible.

### Nsight Systems Profiling

Captured a GPU execution trace using NVIDIA Nsight Systems while serving a mixed workload of short and long prompts. The trace file (`vllm_phi2_profile.nsys-rep`) shows kernel launch timelines, GPU utilization, and CPU-GPU overlap during inference.

**Profiling workload results:**

| Request | Type | Latency | Tokens |
|---------|------|---------|--------|
| short_1 | Short (cold) | 3.17s | 13 |
| short_2 | Short | 0.44s | 20 |
| long_1 | Long | 3.28s | 150 |
| short_3 | Short | 0.44s | 20 |
| long_2 | Long | 3.28s | 150 |

The first request was slower (3.17s vs 0.44s for subsequent short requests) due to CUDA graph warmup and JIT compilation on the first inference pass. Subsequent requests of the same type show consistent latency.

### GPU Kernel Breakdown

Analysis of GPU compute time from the Nsight trace:

| % GPU Time | Kernel | Role |
|------------|--------|------|
| 18.7% | `vectorized_elementwise_kernel` | Memory initialization, tensor zeroing |
| 10.2% | `ampere_fp16_s16816gemm_256x128` | Matrix multiplication (GEMM) — core of transformer layers |
| 8.1% | `gemvx::kernel` | Matrix-vector multiply — the decode step (one token at a time) |
| 8.0% | `ampere_fp16_s1688gemm_128x128` | GEMM variant (different matrix dimensions) |
| 7.5% | `flash_fwd_splitkv_kernel` | Flash Attention — attention computation over KV cache |
| 5.6% | `ampere_fp16_s1688gemm_256x64` | GEMM variant |
| 3.8% | `cutlass_tensorop_gemm_relu_256x128` | GEMM with fused ReLU activation |
| 3.6% | `triton_` | Triton-compiled custom kernels (layer norm, fused ops) |
| 3.5% | `cutlass_tensorop_gemm_relu_64x256` | GEMM+ReLU variant |

GEMM (matrix multiplication) kernels collectively account for ~45% of GPU compute time. Flash Attention accounts for 7.5%. The decode-specific matrix-vector kernel (`gemvx`) ran 546 times across the 5 requests, confirming that single-token generation uses a fundamentally different (and less parallelizable) operation than batch processing.

### Memory Transfer Analysis

| Direction | % Transfer Time | Count | Purpose |
|-----------|----------------|-------|---------|
| Host to Device (CPU → GPU) | 97.4% | 3,591 | Model weights + input tokens |
| Device to Device (GPU → GPU) | 2.5% | 1,757 | Internal tensor operations |
| Device to Host (GPU → CPU) | 0.01% | 354 | Output tokens returned |

Memory transfers are heavily asymmetric. Loading model weights to GPU dominates transfer time. Returning generated tokens to CPU is essentially free (354 transfers totaling 0.4ms).

### CPU-GPU Interaction

| CPU Activity | % Time | Meaning |
|-------------|--------|---------|
| `cudaEventSynchronize` | 36.5% | CPU waiting for GPU to finish |
| `cudaLaunchKernel` | 27.8% | CPU dispatching work to GPU (44,131 kernel launches) |
| `cudaMemcpyAsync` | 19.1% | CPU initiating data transfers |
| `cudaDeviceSynchronize` | 12.7% | CPU waiting for all GPU work |

The CPU spent 49.2% of its time waiting for the GPU, confirming the system is GPU-bound. The CPU can dispatch work faster than the GPU can execute it.

### Nsight Profiling Observations

**1. GEMM dominates GPU compute (~45%).**
Matrix multiplication across feed-forward layers and attention projections is the most expensive operation, as expected for transformer inference.

**2. Flash Attention is efficient at 7.5% of GPU time.**
The optimized Flash Attention kernel keeps attention cost low relative to total compute. For larger models or longer sequences, this percentage would grow.

**3. Decode uses matrix-vector multiply, not matrix-matrix.**
The `gemvx` kernel (8.1% of GPU time, 546 calls) confirms that single-token generation processes one vector at a time rather than batched matrix operations. This is why decode is inherently sequential and why higher concurrency helps — it gives the GPU more vectors to process in parallel.

**4. The system is GPU-bound, not CPU-bound.**
CPU spent 49.2% of time waiting for the GPU. The CPU can launch 44,131 kernels (~2,200/sec) faster than the GPU can execute them.

**5. Memory transfers are asymmetric — output is free.**
97.4% of transfer time is loading data TO the GPU. Getting output tokens back costs 0.01% of transfer time. The bottleneck is never returning results.

**6. vLLM uses CUDA graphs aggressively.**
1,786 CUDA graphs were compiled during warmup, and 514 graph launches occurred during inference. CUDA graphs bundle many kernel launches into a single dispatch, reducing CPU-GPU round-trip overhead significantly.

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
