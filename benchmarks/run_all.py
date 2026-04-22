"""
Run all available benchmarks sequentially.
Skips engines whose servers are not running.
"""

import subprocess
import sys
import os

BENCHMARKS_DIR = os.path.dirname(__file__)

BENCH_SCRIPTS = [
    ("ollama", "bench_ollama.py"),
    ("exllamav2", "bench_exllamav2.py"),
    ("vllm", "bench_vllm.py"),
    ("tensorrt_llm", "bench_tensorrt_llm.py"),
    ("sglang", "bench_sglang.py"),
    ("mlc_llm", "bench_mlc_llm.py"),
    ("dotllm", "bench_dotllm.py"),
]


def main():
    print("=" * 60)
    print("  LLM Inference Benchmark Suite")
    print("  Model: llama 3.1 8B int4 | Hardware: RTX 4090")
    print("=" * 60)
    print()

    results_summary = []

    for engine_name, script_file in BENCH_SCRIPTS:
        script_path = os.path.join(BENCHMARKS_DIR, script_file)
        print(f"\n{'─' * 60}")
        print(f"  Running: {engine_name}")
        print(f"{'─' * 60}\n")

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=os.path.dirname(BENCHMARKS_DIR),
                timeout=600,  # 10 min timeout per engine
            )
            if result.returncode == 0:
                results_summary.append((engine_name, "SUCCESS"))
            else:
                results_summary.append((engine_name, "FAILED"))
        except subprocess.TimeoutExpired:
            results_summary.append((engine_name, "TIMEOUT"))
            print(f"  TIMEOUT: {engine_name} exceeded 10 minutes")
        except Exception as e:
            results_summary.append((engine_name, f"ERROR: {e}"))
            print(f"  ERROR running {engine_name}: {e}")

    # Print summary
    print(f"\n{'═' * 60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'═' * 60}")
    for engine, status in results_summary:
        icon = "✓" if status == "SUCCESS" else "✗"
        print(f"  {icon} {engine:20s} {status}")
    print(f"{'═' * 60}")
    print(f"\nResults stored in: benchmarks/results/")
    print("Run 'python benchmarks/analyze.py' to compare results.")


if __name__ == "__main__":
    main()
