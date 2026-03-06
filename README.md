# GPU LLM Inference Profiling with vLLM and NVIDIA Nsight

This project documents a small, reproducible workflow for serving an LLM with `vLLM`, benchmarking inference performance, and profiling execution with `NVIDIA Nsight Systems`.

## Planned Scope

- Run a small LLM through a Dockerized `vLLM` serving setup
- Expose an OpenAI-compatible local or cloud-hosted inference endpoint
- Benchmark throughput and latency across different prompt lengths and load levels
- Capture and inspect Nsight Systems traces to study GPU activity and CPU-GPU overlap
- Document the setup and experiment workflow for repeatable runs

## Why This Project

This repository is intentionally scoped as a compact performance engineering project rather than a full inference framework contribution. The initial goal is to build a clean baseline for LLM serving, benchmarking, and profiling on an NVIDIA GPU, then use that baseline as a foundation for deeper optimization work.

## Future Extensions

The current plan covers a small slice of LLM inference profiling. The following extensions would push the project closer to the kind of work described in NVIDIA Deep Learning Software Engineer, LLM Performance roles.

### 1. Broaden Beyond a Single Serving Stack

- Compare the same model and workload across `vLLM`, `TensorRT-LLM`, `SGLang`, and `Triton`
- Document framework-level tradeoffs in setup complexity, latency, throughput, memory usage, and batching behavior
- Build a common benchmark harness so all frameworks are evaluated under the same test conditions

### 2. Study Prefill, Decode, and KV Cache Behavior

- Separate prompt processing (`prefill`) from token generation (`decode`) in benchmark results
- Analyze how prompt length, output length, and concurrency affect `KV cache` growth and GPU memory pressure
- Relate observed timeline behavior in Nsight Systems to concepts like cache reuse, request scheduling, and GPU occupancy

### 3. Optimize for Multiple Performance Targets

- Measure max throughput, minimum latency, and throughput under latency constraints
- Add sweeps over concurrency, batch size, and generation length
- Report not just average latency, but also tail metrics such as `p95` and `p99`
- Explore simple service-level objective style questions such as "what is the highest throughput achievable while keeping latency below a chosen threshold?"

### 4. Compare Across GPU Architectures

- Repeat the same experiment on multiple NVIDIA accelerator types when available
- Compare behavior on datacenter-style GPUs such as `L4`, `A10G`, or higher-end instances
- Document how memory capacity, compute capability, and hardware generation influence LLM serving behavior

### 5. Add Deeper Profiling and Performance Analysis

- Use `Nsight Compute` for a more kernel-focused view in addition to `Nsight Systems`
- Correlate high-level serving metrics with lower-level GPU execution details
- Study GPU utilization gaps, CPU launch overhead, kernel launch frequency, and CPU-GPU overlap
- Add lightweight performance modeling to explain where the system is likely compute-bound versus memory-bound

### 6. Extend the Benchmarking Workflow

- Save benchmark results in structured formats such as `CSV` or `JSON`
- Generate plots for throughput-latency tradeoffs and prompt-length sensitivity
- Add reproducible experiment configs so workloads can be rerun exactly
- Include automated scripts for launching, profiling, parsing results, and exporting artifacts

### 7. Expand Toward Real Inference Engineering

- Profile and compare quantized versus non-quantized models
- Evaluate tensor parallel or multi-GPU serving if larger infrastructure is available
- Add experiments for streaming responses, dynamic batching, and long-context inference
- Test additional open models and compare how architecture choices affect serving efficiency

## Status

Initial repository scaffold. Implementation and experiment setup are in progress.
