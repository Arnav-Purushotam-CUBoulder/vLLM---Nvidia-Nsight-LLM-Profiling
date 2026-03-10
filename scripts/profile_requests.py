"""
Send profiling workload requests to vLLM server.

This script is meant to be run WHILE Nsight Systems is capturing a trace.
It sends a controlled set of requests to generate interesting GPU activity.

Usage:
    python3 scripts/profile_requests.py
    python3 scripts/profile_requests.py --url http://localhost:8001
"""

import argparse
import json
import time
from urllib.request import Request, urlopen


MODEL = "microsoft/phi-2"

# Workload: a mix of short and long prompts to create visible
# differences in the Nsight timeline
WORKLOAD = [
    {"label": "short_1", "prompt": "What is 2+2?", "max_tokens": 20},
    {"label": "short_2", "prompt": "Name three colors.", "max_tokens": 20},
    {"label": "long_1",  "prompt": "Explain in detail how a CPU executes instructions, covering the fetch-decode-execute cycle, pipelining, and branch prediction.", "max_tokens": 150},
    {"label": "short_3", "prompt": "What is Python?", "max_tokens": 20},
    {"label": "long_2",  "prompt": "Describe the differences between SRAM and DRAM, including their use in CPU caches and main memory. Explain why cache hierarchies exist and how they improve performance.", "max_tokens": 150},
]


def send_request(url, prompt, max_tokens, model):
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
    with urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - start

    tokens = body["usage"]["completion_tokens"]
    return elapsed, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    print("Sending profiling workload requests...")
    print(f"Server: {args.url}")
    print()

    for item in WORKLOAD:
        label = item["label"]
        print(f"  [{label}] Sending...", end=" ", flush=True)
        try:
            elapsed, tokens = send_request(args.url, item["prompt"], item["max_tokens"], args.model)
            print(f"done in {elapsed:.2f}s ({tokens} tokens)")
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(1)  # gap between requests so timeline shows clear separation

    print()
    print("Workload complete. Check Nsight trace for GPU activity.")


if __name__ == "__main__":
    main()
