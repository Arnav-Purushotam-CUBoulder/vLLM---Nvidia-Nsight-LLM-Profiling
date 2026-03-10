"""
Benchmark script for vLLM inference serving.

Measures latency and throughput under different prompt lengths and concurrency levels.
Sends requests to a vLLM OpenAI-compatible endpoint and records timing data.

Usage:
    python3 benchmark.py
    python3 benchmark.py --url http://localhost:8000 --output results.json
"""

import argparse
import json
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import URLError


# ---------------------------------------------------------------------------
# Prompts: one short, one long
# ---------------------------------------------------------------------------

SHORT_PROMPT = "Explain what a GPU is in one sentence."

LONG_PROMPT = (
    "You are a computer science professor. Explain the following topics in detail: "
    "1) How CPU and GPU architectures differ fundamentally in their design philosophy. "
    "2) Why GPUs are better suited for parallel workloads like matrix multiplication. "
    "3) The concept of CUDA cores and how they relate to GPU computing. "
    "4) How modern deep learning frameworks leverage GPU parallelism for training. "
    "5) The memory hierarchy in GPUs including global memory, shared memory, and registers. "
    "Provide concrete examples for each topic."
)


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

BENCHMARK_CASES = [
    {"name": "short_prompt_c1",  "prompt": SHORT_PROMPT, "concurrency": 1,  "num_requests": 10},
    {"name": "short_prompt_c4",  "prompt": SHORT_PROMPT, "concurrency": 4,  "num_requests": 16},
    {"name": "short_prompt_c8",  "prompt": SHORT_PROMPT, "concurrency": 8,  "num_requests": 16},
    {"name": "long_prompt_c1",   "prompt": LONG_PROMPT,  "concurrency": 1,  "num_requests": 10},
    {"name": "long_prompt_c4",   "prompt": LONG_PROMPT,  "concurrency": 4,  "num_requests": 16},
    {"name": "long_prompt_c8",   "prompt": LONG_PROMPT,  "concurrency": 8,  "num_requests": 16},
]

MAX_TOKENS = 100
MODEL = "microsoft/phi-2"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def send_request(url: str, prompt: str, max_tokens: int, model: str) -> dict:
    """Send a single completion request and return timing + token info."""

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
        with urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        end = time.perf_counter()
    except URLError as e:
        return {"error": str(e), "latency": None, "output_tokens": 0}

    output_tokens = body["usage"]["completion_tokens"]
    latency = end - start

    return {
        "latency": latency,
        "output_tokens": output_tokens,
    }


def run_case(url: str, case: dict, model: str, max_tokens: int) -> dict:
    """Run a single benchmark case with the given concurrency."""

    name = case["name"]
    prompt = case["prompt"]
    concurrency = case["concurrency"]
    num_requests = case["num_requests"]

    print(f"\n--- {name} ---")
    print(f"  Prompt length: {'short' if len(prompt) < 100 else 'long'}")
    print(f"  Concurrency:   {concurrency}")
    print(f"  Requests:      {num_requests}")

    results = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(send_request, url, prompt, max_tokens, model)
            for _ in range(num_requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    wall_end = time.perf_counter()
    wall_time = wall_end - wall_start

    # Filter out errors
    successful = [r for r in results if r["latency"] is not None]
    errors = [r for r in results if r["latency"] is None]

    if not successful:
        print("  All requests failed!")
        return {"name": name, "error": "all requests failed"}

    latencies = [r["latency"] for r in successful]
    total_output_tokens = sum(r["output_tokens"] for r in successful)

    summary = {
        "name": name,
        "prompt_type": "short" if len(prompt) < 100 else "long",
        "concurrency": concurrency,
        "num_requests": num_requests,
        "successful": len(successful),
        "errors": len(errors),
        "wall_time_sec": round(wall_time, 3),
        "avg_latency_sec": round(statistics.mean(latencies), 3),
        "p50_latency_sec": round(statistics.median(latencies), 3),
        "p95_latency_sec": round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
        "min_latency_sec": round(min(latencies), 3),
        "max_latency_sec": round(max(latencies), 3),
        "throughput_req_per_sec": round(len(successful) / wall_time, 3),
        "throughput_tokens_per_sec": round(total_output_tokens / wall_time, 3),
        "total_output_tokens": total_output_tokens,
    }

    print(f"  Avg latency:   {summary['avg_latency_sec']}s")
    print(f"  P95 latency:   {summary['p95_latency_sec']}s")
    print(f"  Throughput:    {summary['throughput_req_per_sec']} req/s, "
          f"{summary['throughput_tokens_per_sec']} tok/s")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM inference serving")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--output", default="results.json", help="Output file for results")
    parser.add_argument("--model", default=MODEL, help="Model name")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Max output tokens")
    args = parser.parse_args()

    print("=" * 60)
    print("vLLM Inference Benchmark")
    print("=" * 60)
    print(f"Server:     {args.url}")
    print(f"Model:      {args.model}")
    print(f"Max tokens: {args.max_tokens}")

    # Verify server is up
    print("\nChecking server...")
    try:
        req = Request(f"{args.url}/v1/models")
        with urlopen(req, timeout=10) as resp:
            models = json.loads(resp.read().decode("utf-8"))
        print(f"Server is up. Models: {[m['id'] for m in models['data']]}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.url}: {e}")
        print("Make sure the vLLM server is running first.")
        return

    # Warmup: send one request to make sure model is ready
    print("\nWarmup request...")
    warmup = send_request(args.url, "Hello", 10, args.model)
    if warmup["latency"] is None:
        print(f"ERROR: Warmup failed: {warmup['error']}")
        return
    print(f"Warmup done in {warmup['latency']:.2f}s")

    # Run all benchmark cases
    all_results = []
    for case in BENCHMARK_CASES:
        result = run_case(args.url, case, args.model, args.max_tokens)
        all_results.append(result)

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Case':<22} {'Avg Lat':>8} {'P95 Lat':>8} {'Req/s':>8} {'Tok/s':>8}")
    print("-" * 60)
    for r in all_results:
        if "error" in r:
            continue
        print(f"{r['name']:<22} {r['avg_latency_sec']:>7.3f}s {r['p95_latency_sec']:>7.3f}s "
              f"{r['throughput_req_per_sec']:>7.2f} {r['throughput_tokens_per_sec']:>7.1f}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
