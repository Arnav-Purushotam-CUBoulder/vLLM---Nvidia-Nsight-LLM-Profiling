"""
Stress benchmark for vLLM inference serving.

These tests push the system toward degradation and failure to find the limits.
Run AFTER the baseline benchmark to compare.

Tests:
  1. Concurrency ramp    — increase concurrency until latency spikes
  2. Long prompt stress   — very long prompts to stress prefill + KV cache
  3. Max output length    — generate as many tokens as possible per request
  4. Memory pressure      — long prompts + high concurrency (KV cache pressure)
  5. Burst load           — dump many requests at once (queue overload)

Usage:
    python3 benchmark_stress.py
    python3 benchmark_stress.py --test all
    python3 benchmark_stress.py --test concurrency_ramp
"""

import argparse
import json
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import URLError


MODEL = "microsoft/phi-2"
SERVER = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Prompts of varying sizes
# ---------------------------------------------------------------------------

SHORT_PROMPT = "What is 2+2?"

# ~200 tokens
MEDIUM_PROMPT = (
    "You are a computer science professor teaching an undergraduate systems class. "
    "Explain the following concepts in detail with examples: "
    "1) Virtual memory and page tables. "
    "2) Context switching between processes. "
    "3) The difference between user space and kernel space. "
    "4) How system calls work. "
    "5) The role of the TLB in address translation."
)

# ~500 tokens — fills a significant chunk of the 2048 context window
LONG_PROMPT = (
    "You are writing a comprehensive textbook chapter. Cover ALL of the following topics "
    "in extreme detail with concrete examples, code snippets, and diagrams described in text: "
    "1) The complete history of CPU architecture from Von Neumann to modern out-of-order "
    "superscalar processors, including pipelining, branch prediction, speculative execution, "
    "and simultaneous multithreading. Explain each concept with a specific real-world processor example. "
    "2) The evolution of GPU computing from fixed-function graphics pipelines to general-purpose "
    "GPU computing with CUDA, including the SIMT execution model, warp scheduling, shared memory "
    "bank conflicts, and occupancy optimization. Provide CUDA code examples. "
    "3) Memory hierarchy design including SRAM vs DRAM physics, cache coherence protocols like "
    "MESI and MOESI, directory-based coherence for multi-socket systems, NUMA architectures, "
    "and the impact of memory access patterns on performance. Include bandwidth and latency numbers. "
    "4) Interconnect design including PCIe, NVLink, NVSwitch, CXL, and UCIe. Compare their "
    "bandwidth, latency, and use cases. Explain how NVLink enables multi-GPU training. "
    "5) Deep learning accelerator design including systolic arrays, dataflow architectures, "
    "mixed-precision compute units, sparsity support, and the roofline model for analyzing "
    "compute vs memory boundedness. Compare TPU, GPU, and custom ASIC approaches. "
    "6) Distributed systems fundamentals including consensus algorithms, CAP theorem, "
    "vector clocks, Paxos, Raft, and how they apply to distributed training with parameter "
    "servers versus all-reduce collectives. "
    "7) Operating system internals including process scheduling algorithms, virtual memory "
    "implementation, file system design, I/O scheduling, and how containers and namespaces "
    "work at the kernel level. "
    "8) Network programming including TCP congestion control, RDMA, kernel bypass networking, "
    "DPDK, and how high-performance inference servers handle thousands of concurrent connections."
)

# ~1500 tokens — pushes close to the 2048 context limit, leaving little room for output
HUGE_PROMPT = (LONG_PROMPT + " " + LONG_PROMPT + " " + MEDIUM_PROMPT +
    " Now synthesize everything above into a unified framework for understanding "
    "modern computing systems from transistor to distributed application. "
    "Draw connections between every topic. Be extremely thorough.")


# ---------------------------------------------------------------------------
# Core request function
# ---------------------------------------------------------------------------

def send_request(url, prompt, max_tokens, model, timeout=180):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")

    req = Request(
        f"{url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        end = time.perf_counter()
    except Exception as e:
        return {"error": str(e), "latency": None, "output_tokens": 0, "prompt_tokens": 0}

    return {
        "latency": end - start,
        "output_tokens": body["usage"]["completion_tokens"],
        "prompt_tokens": body["usage"]["prompt_tokens"],
    }


def run_batch(url, prompt, max_tokens, model, concurrency, num_requests):
    results = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(send_request, url, prompt, max_tokens, model)
            for _ in range(num_requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    wall_time = time.perf_counter() - wall_start
    successful = [r for r in results if r["latency"] is not None]
    errors = [r for r in results if r["latency"] is None]

    if not successful:
        return {"error": "all failed", "errors": len(errors)}

    latencies = [r["latency"] for r in successful]
    total_tokens = sum(r["output_tokens"] for r in successful)

    return {
        "successful": len(successful),
        "errors": len(errors),
        "wall_time": round(wall_time, 3),
        "avg_latency": round(statistics.mean(latencies), 3),
        "p95_latency": round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
        "max_latency": round(max(latencies), 3),
        "throughput_req_s": round(len(successful) / wall_time, 3),
        "throughput_tok_s": round(total_tokens / wall_time, 3),
        "total_tokens": total_tokens,
        "avg_prompt_tokens": round(statistics.mean([r["prompt_tokens"] for r in successful]), 1),
    }


def print_result(label, result):
    if "error" in result and result["error"] == "all failed":
        print(f"  {label}: ALL REQUESTS FAILED ({result['errors']} errors)")
        return
    print(f"  {label}: avg={result['avg_latency']}s  p95={result['p95_latency']}s  "
          f"max={result['max_latency']}s  tok/s={result['throughput_tok_s']}  "
          f"errors={result['errors']}  prompt_tok={result['avg_prompt_tokens']}")


# ---------------------------------------------------------------------------
# Test 1: Concurrency Ramp
# ---------------------------------------------------------------------------

def test_concurrency_ramp(url, model):
    """
    WHY: Find where the system saturates.

    In the baseline test, concurrency 8 still scaled well. Here we push to
    16, 32, 64 to find where latency spikes and throughput plateaus.

    WHAT TO EXPECT:
    - At some point, KV cache fills up and vLLM starts rejecting or queuing
    - Latency will spike sharply (not gradually)
    - Throughput will plateau then possibly drop
    - Errors may start appearing (out of memory, timeouts)
    """
    print("\n" + "=" * 70)
    print("TEST 1: CONCURRENCY RAMP")
    print("=" * 70)
    print("Goal: Find where latency spikes and throughput stops scaling")
    print("Prompt: short  |  Max tokens: 100  |  Requests per level: 16")
    print("-" * 70)

    concurrency_levels = [1, 2, 4, 8, 16, 32, 64]
    results = []

    for c in concurrency_levels:
        print(f"\n  Concurrency {c}...", flush=True)
        result = run_batch(url, SHORT_PROMPT, 100, model, c, max(16, c))
        result["concurrency"] = c
        results.append(result)
        print_result(f"c={c:>2}", result)

    return {"test": "concurrency_ramp", "results": results}


# ---------------------------------------------------------------------------
# Test 2: Long Prompt Stress
# ---------------------------------------------------------------------------

def test_long_prompts(url, model):
    """
    WHY: Stress the prefill phase and KV cache memory.

    In the baseline, short vs long prompt barely mattered because both were
    small. Here we use prompts of ~10, ~200, ~500, and ~1500 tokens.

    WHAT TO EXPECT:
    - Prefill time increases significantly with prompt length
    - KV cache per request grows, limiting concurrency
    - The huge prompt (~1500 tokens) leaves only ~500 tokens for output
      in a 2048 context window
    - May see out-of-memory at high prompt lengths + concurrency
    """
    print("\n" + "=" * 70)
    print("TEST 2: LONG PROMPT STRESS")
    print("=" * 70)
    print("Goal: See how prompt length affects latency and memory")
    print("Concurrency: 4  |  Max tokens: 100  |  Requests: 12")
    print("-" * 70)

    prompts = [
        ("short (~10 tok)", SHORT_PROMPT),
        ("medium (~200 tok)", MEDIUM_PROMPT),
        ("long (~500 tok)", LONG_PROMPT),
        ("huge (~1500 tok)", HUGE_PROMPT),
    ]
    results = []

    for label, prompt in prompts:
        print(f"\n  {label}...", flush=True)
        result = run_batch(url, prompt, 100, model, 4, 12)
        result["prompt_type"] = label
        results.append(result)
        print_result(f"  {label}", result)

    return {"test": "long_prompts", "results": results}


# ---------------------------------------------------------------------------
# Test 3: Max Output Length
# ---------------------------------------------------------------------------

def test_max_output(url, model):
    """
    WHY: See how output length affects latency and throughput.

    Decode is sequential — each token depends on the previous. More output
    tokens = linearly more decode time. This test confirms that relationship.

    WHAT TO EXPECT:
    - Latency should scale roughly linearly with max_tokens
    - 100 tokens ≈ 2s means 500 tokens ≈ 10s, 1000 tokens ≈ 20s
    - Throughput (tok/s) should stay roughly constant since the GPU does
      the same work per token regardless of total count
    """
    print("\n" + "=" * 70)
    print("TEST 3: MAX OUTPUT LENGTH")
    print("=" * 70)
    print("Goal: Confirm decode time scales linearly with output tokens")
    print("Prompt: short  |  Concurrency: 1  |  Requests: 5")
    print("-" * 70)

    token_counts = [10, 50, 100, 200, 500, 1000]
    results = []

    for tokens in token_counts:
        print(f"\n  max_tokens={tokens}...", flush=True)
        result = run_batch(url, SHORT_PROMPT, tokens, model, 1, 5)
        result["max_tokens"] = tokens
        results.append(result)
        print_result(f"  {tokens:>5} tokens", result)

    return {"test": "max_output", "results": results}


# ---------------------------------------------------------------------------
# Test 4: Memory Pressure (long prompt + high concurrency)
# ---------------------------------------------------------------------------

def test_memory_pressure(url, model):
    """
    WHY: Force KV cache to fill up.

    Each concurrent request with a long prompt needs a large KV cache.
    Long prompt (500 tok) + 100 output tokens = 600 tokens of KV cache per request.
    At concurrency 16, that's 9600 tokens of KV cache simultaneously.

    WHAT TO EXPECT:
    - Low concurrency with long prompts: fine
    - High concurrency with long prompts: possible OOM, request failures,
      or severe latency spikes as vLLM starts swapping or queuing
    - This is where PagedAttention's limits become visible
    """
    print("\n" + "=" * 70)
    print("TEST 4: MEMORY PRESSURE (long prompt + high concurrency)")
    print("=" * 70)
    print("Goal: Push KV cache toward capacity limits")
    print("Prompt: long (~500 tok)  |  Max tokens: 200")
    print("-" * 70)

    concurrency_levels = [1, 4, 8, 16, 32]
    results = []

    for c in concurrency_levels:
        num_req = max(8, c)
        print(f"\n  Concurrency {c} ({num_req} requests)...", flush=True)
        result = run_batch(url, LONG_PROMPT, 200, model, c, num_req)
        result["concurrency"] = c
        results.append(result)
        print_result(f"  c={c:>2}", result)

    return {"test": "memory_pressure", "results": results}


# ---------------------------------------------------------------------------
# Test 5: Burst Load
# ---------------------------------------------------------------------------

def test_burst_load(url, model):
    """
    WHY: Simulate a traffic spike.

    Instead of steady concurrent requests, dump a large number of requests
    all at once and see how the server handles the queue.

    WHAT TO EXPECT:
    - The first few requests complete at normal speed
    - Later requests wait in queue, so their latency is much higher
    - Max latency >> average latency (unlike baseline where they were close)
    - This shows the difference between sustained load and burst load
    - P95 latency will be much higher than average
    """
    print("\n" + "=" * 70)
    print("TEST 5: BURST LOAD")
    print("=" * 70)
    print("Goal: See queuing behavior under sudden traffic spike")
    print("Prompt: short  |  Max tokens: 100")
    print("-" * 70)

    burst_sizes = [8, 16, 32, 64]
    results = []

    for burst in burst_sizes:
        # Send ALL requests at once (concurrency = burst size)
        print(f"\n  Burst of {burst} simultaneous requests...", flush=True)
        result = run_batch(url, SHORT_PROMPT, 100, model, burst, burst)
        result["burst_size"] = burst
        results.append(result)
        print_result(f"  burst={burst:>2}", result)
        # Gap between bursts so server recovers
        time.sleep(2)

    return {"test": "burst_load", "results": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = {
    "concurrency_ramp": test_concurrency_ramp,
    "long_prompts": test_long_prompts,
    "max_output": test_max_output,
    "memory_pressure": test_memory_pressure,
    "burst_load": test_burst_load,
}


def main():
    parser = argparse.ArgumentParser(description="Stress benchmark for vLLM")
    parser.add_argument("--url", default=SERVER)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", default="results_stress.json")
    parser.add_argument("--test", default="all",
                        choices=["all"] + list(TESTS.keys()),
                        help="Which test to run (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("vLLM STRESS BENCHMARK")
    print("=" * 70)
    print(f"Server: {args.url}")
    print(f"Model:  {args.model}")

    # Verify server
    try:
        req = Request(f"{args.url}/v1/models")
        with urlopen(req, timeout=10):
            pass
        print("Server is up.")
    except Exception as e:
        print(f"ERROR: Cannot reach server: {e}")
        return

    # Warmup
    print("Warmup...", flush=True)
    send_request(args.url, "Hello", 10, args.model)
    print("Ready.\n")

    # Run tests
    all_results = []
    tests_to_run = TESTS if args.test == "all" else {args.test: TESTS[args.test]}

    for name, test_fn in tests_to_run.items():
        result = test_fn(args.url, args.model)
        all_results.append(result)

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"All results saved to {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
