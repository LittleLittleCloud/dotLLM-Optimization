"""
Benchmark script for dotLLM.
Runs natively on Windows.

Prerequisites:
  - dotLLM built and running with OpenAI-compatible API
  - Model: llama 3.1 8B int4
  - TODO: Add specific setup instructions and commit id
"""

import time
import requests
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.common import (
    BenchmarkResult, save_results, print_result, get_timestamp,
    get_prompts, DEFAULT_MAX_TOKENS, DEFAULT_REPEAT, MODEL_NAME, QUANTIZATION
)

ENGINE_NAME = "dotllm"

# dotLLM server endpoint
DOTLLM_BASE_URL = os.environ.get("DOTLLM_BASE_URL", "http://localhost:8004")
DOTLLM_MODEL = os.environ.get("DOTLLM_MODEL", "Meta-Llama-3.1-8B-Instruct-Q4_K_M")


def check_server_running():
    """Check if dotLLM server is running."""
    try:
        resp = requests.get(f"{DOTLLM_BASE_URL}/v1/models", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def benchmark_single(prompt: str, prompt_name: str, run_index: int) -> BenchmarkResult:
    """Run a single benchmark with streaming to measure TTFT."""
    start_time = time.perf_counter()
    first_token_time = None
    output_text = ""
    output_tokens = 0
    input_tokens = 0

    resp = requests.post(
        f"{DOTLLM_BASE_URL}/v1/completions",
        json={
            "model": DOTLLM_MODEL,
            "prompt": prompt,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.0,
            "stream": True,
        },
        stream=True,
        timeout=120,
    )

    for line in resp.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("text", "")
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            output_text += delta
                    if "usage" in chunk and chunk["usage"]:
                        input_tokens = chunk["usage"].get("prompt_tokens", 0)
                        output_tokens = chunk["usage"].get("completion_tokens", 0)
                except json.JSONDecodeError:
                    pass

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time

    if input_tokens == 0:
        input_tokens = max(1, len(prompt) // 4)
    if output_tokens == 0:
        output_tokens = max(1, len(output_text) // 4)

    ttft_ms = (first_token_time - start_time) * 1000
    total_time = end_time - start_time
    decode_time = end_time - first_token_time

    prefill_tps = input_tokens / (ttft_ms / 1000) if ttft_ms > 0 else 0
    decode_tps = output_tokens / decode_time if decode_time > 0 else 0

    return BenchmarkResult(
        engine=ENGINE_NAME,
        model=MODEL_NAME,
        quantization=QUANTIZATION,
        prompt_name=prompt_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        time_to_first_token_ms=ttft_ms,
        prefill_tokens_per_sec=prefill_tps,
        decode_tokens_per_sec=decode_tps,
        total_time_sec=total_time,
        run_index=run_index,
        timestamp=get_timestamp(),
    )


def run_benchmark():
    """Run full benchmark suite for dotLLM."""
    print(f"=== Benchmarking {ENGINE_NAME} ===\n")
    print(f"Server: {DOTLLM_BASE_URL}")
    print(f"Model:  {DOTLLM_MODEL}\n")

    if not check_server_running():
        print("ERROR: dotLLM server is not running.")
        print(f"Start it and ensure it's listening on {DOTLLM_BASE_URL}")
        sys.exit(1)

    # Warmup
    print("Warming up...")
    requests.post(
        f"{DOTLLM_BASE_URL}/v1/completions",
        json={"model": DOTLLM_MODEL, "prompt": "Hi", "max_tokens": 10, "temperature": 0.0},
        timeout=60,
    )

    results = []
    prompts = get_prompts()

    for prompt_name, prompt_text in prompts.items():
        print(f"\nPrompt: {prompt_name}")
        for i in range(DEFAULT_REPEAT):
            result = benchmark_single(prompt_text, prompt_name, i)
            results.append(result)
            print_result(result)

    save_results(results, ENGINE_NAME)
    print(f"\nResults saved to benchmarks/results/{ENGINE_NAME}.jsonl")


if __name__ == "__main__":
    run_benchmark()
