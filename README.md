# GPU LLM Inference Profiling with vLLM and NVIDIA Nsight

This project documents a small, reproducible workflow for serving an LLM with `vLLM`, benchmarking inference performance, and profiling execution with `NVIDIA Nsight Systems`.

## Planned Scope

- Run a small LLM through a Dockerized `vLLM` serving setup
- Expose an OpenAI-compatible local or cloud-hosted inference endpoint
- Benchmark throughput and latency across different prompt lengths and load levels
- Capture and inspect Nsight Systems traces to study GPU activity and CPU-GPU overlap
- Document the setup and experiment workflow for repeatable runs

## Status

Initial repository scaffold. Implementation and experiment setup are in progress.
