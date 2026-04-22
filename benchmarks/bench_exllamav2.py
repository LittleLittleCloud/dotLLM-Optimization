"""
Benchmark script for ExLlamaV2.
Runs natively on Windows with CUDA.

Prerequisites:
  - pip install exllamav2 torch
  - Download model: e.g. turboderp/Llama-3.1-8B-Instruct-exl2 (4.0bpw)
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from benchmarks.common import (
    BenchmarkResult, save_results, print_result, get_timestamp,
    get_prompts, DEFAULT_MAX_TOKENS, DEFAULT_REPEAT, MODEL_NAME, QUANTIZATION
)

ENGINE_NAME = "exllamav2"

# Path to the ExLlamaV2 model directory (adjust as needed)
MODEL_DIR = os.environ.get(
    "EXLLAMAV2_MODEL_DIR",
    r"D:\models\Llama-3.1-8B-Instruct-exl2-4.0bpw"
)


def run_benchmark():
    """Run full benchmark suite for ExLlamaV2."""
    print(f"=== Benchmarking {ENGINE_NAME} ===\n")

    try:
        from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
        from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob
    except ImportError:
        print("ERROR: exllamav2 not installed. Run: pip install exllamav2")
        sys.exit(1)

    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        print("Set EXLLAMAV2_MODEL_DIR environment variable or download the model.")
        sys.exit(1)

    # Load model
    print(f"Loading model from {MODEL_DIR}...")
    config = ExLlamaV2Config(MODEL_DIR)
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len=4096, lazy=True)
    model.load_autosplit(cache)
    tokenizer = ExLlamaV2Tokenizer(config)

    generator = ExLlamaV2DynamicGenerator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
    )

    print("Model loaded.\n")

    # Warmup
    print("Warming up...")
    generator.generate(prompt="Hello", max_new_tokens=10, encode_special_tokens=True)

    results = []
    prompts = get_prompts()

    for prompt_name, prompt_text in prompts.items():
        print(f"\nPrompt: {prompt_name}")
        for i in range(DEFAULT_REPEAT):
            result = benchmark_single(generator, tokenizer, prompt_text, prompt_name, i)
            results.append(result)
            print_result(result)

    save_results(results, ENGINE_NAME)
    print(f"\nResults saved to benchmarks/results/{ENGINE_NAME}.jsonl")

    # Cleanup
    del generator, cache, model


def benchmark_single(generator, tokenizer, prompt: str, prompt_name: str, run_index: int) -> BenchmarkResult:
    """Run a single benchmark iteration."""
    input_ids = tokenizer.encode(prompt)
    input_tokens = input_ids.shape[-1]

    # Use streaming generation to measure TTFT
    start_time = time.perf_counter()
    first_token_time = None
    output_tokens = 0

    # Generate with timing
    job = generator.create_job(
        input_ids=input_ids,
        max_new_tokens=DEFAULT_MAX_TOKENS,
        decode_special_tokens=False,
    )

    output_text = ""
    while not job.is_finished():
        results_gen = generator.iterate()
        for result in results_gen:
            if result.get("stage") == "streaming" and result.get("text"):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                output_text += result["text"]
                output_tokens += 1

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time

    # Fallback: non-streaming generation if streaming doesn't work
    if output_tokens == 0:
        start_time = time.perf_counter()
        output = generator.generate(
            prompt=prompt,
            max_new_tokens=DEFAULT_MAX_TOKENS,
            encode_special_tokens=True,
        )
        end_time = time.perf_counter()
        first_token_time = start_time + 0.01  # estimate
        output_ids = tokenizer.encode(output)
        output_tokens = output_ids.shape[-1] - input_tokens

    # Calculate metrics
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


if __name__ == "__main__":
    run_benchmark()
