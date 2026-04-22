"""
Benchmark script for Ollama (llama.cpp backend).
Runs natively on Windows.

Prerequisites:
  - Install Ollama: https://ollama.com/download
  - Pull model: ollama pull llama3.1:8b-instruct-q4_0
"""

import time
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.common import (
    BenchmarkResult, save_results, print_result, get_timestamp,
    get_prompts, DEFAULT_MAX_TOKENS, DEFAULT_REPEAT, MODEL_NAME, QUANTIZATION
)

ENGINE_NAME = "ollama"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_0"


def check_ollama_running():
    """Check if Ollama server is running."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def count_tokens_ollama(text: str) -> int:
    """Use Ollama's tokenize endpoint to count tokens."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_MODEL, "input": text},
            timeout=30,
        )
        # Fallback: estimate ~4 chars per token
        return max(1, len(text) // 4)
    except Exception:
        return max(1, len(text) // 4)


def benchmark_single(prompt: str, prompt_name: str, run_index: int) -> BenchmarkResult:
    """Run a single benchmark iteration with streaming to measure TTFT."""
    input_tokens = count_tokens_ollama(prompt)

    # Use streaming to measure time to first token
    start_time = time.perf_counter()
    first_token_time = None
    output_text = ""
    output_tokens = 0

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": DEFAULT_MAX_TOKENS,
                "temperature": 0.0,
            },
        },
        stream=True,
        timeout=120,
    )

    for line in resp.iter_lines():
        if line:
            import json
            chunk = json.loads(line)
            if "response" in chunk and chunk["response"]:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                output_text += chunk["response"]
            if chunk.get("done", False):
                # Ollama provides token counts in the final response
                if "eval_count" in chunk:
                    output_tokens = chunk["eval_count"]
                if "prompt_eval_count" in chunk:
                    input_tokens = chunk["prompt_eval_count"]
                break

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time

    # Calculate metrics
    ttft_ms = (first_token_time - start_time) * 1000
    total_time = end_time - start_time
    decode_time = end_time - first_token_time

    if output_tokens == 0:
        output_tokens = max(1, len(output_text) // 4)

    prefill_tps = input_tokens / (ttft_ms / 1000) if ttft_ms > 0 else 0
    decode_tps = output_tokens / decode_time if decode_time > 0 else 0

    result = BenchmarkResult(
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
    return result


def run_benchmark():
    """Run full benchmark suite for Ollama."""
    print(f"=== Benchmarking {ENGINE_NAME} ({OLLAMA_MODEL}) ===\n")

    if not check_ollama_running():
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    # Warmup
    print("Warming up...")
    requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": "Hi", "stream": False,
              "options": {"num_predict": 10}},
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
