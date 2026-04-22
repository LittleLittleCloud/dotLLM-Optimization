"""
Analyze and compare benchmark results across all engines.
Reads .jsonl files from benchmarks/results/ and produces comparison tables.
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_all_results():
    """Load all benchmark results from .jsonl files."""
    all_results = []
    if not os.path.exists(RESULTS_DIR):
        return all_results

    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(RESULTS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_results.append(json.loads(line))
    return all_results


def compute_averages(results):
    """Group results by (engine, prompt_name) and compute averages."""
    groups = defaultdict(list)
    for r in results:
        key = (r["engine"], r["prompt_name"])
        groups[key].append(r)

    averages = {}
    for key, runs in groups.items():
        n = len(runs)
        averages[key] = {
            "engine": key[0],
            "prompt_name": key[1],
            "num_runs": n,
            "avg_input_tokens": sum(r["input_tokens"] for r in runs) / n,
            "avg_output_tokens": sum(r["output_tokens"] for r in runs) / n,
            "avg_ttft_ms": sum(r["time_to_first_token_ms"] for r in runs) / n,
            "avg_prefill_tps": sum(r["prefill_tokens_per_sec"] for r in runs) / n,
            "avg_decode_tps": sum(r["decode_tokens_per_sec"] for r in runs) / n,
            "avg_total_time": sum(r["total_time_sec"] for r in runs) / n,
        }
    return averages


def print_comparison_table(averages):
    """Print a formatted comparison table."""
    if not averages:
        print("No results found. Run benchmarks first.")
        return

    # Get unique engines and prompts
    engines = sorted(set(k[0] for k in averages.keys()))
    prompts = sorted(set(k[1] for k in averages.keys()))

    # Print decode speed comparison
    print("\n" + "=" * 80)
    print("  DECODE SPEED (tokens/s) - higher is better")
    print("=" * 80)
    header = f"{'Engine':<20s}"
    for p in prompts:
        header += f"{'  ' + p:<15s}"
    header += f"{'  avg':<12s}"
    print(header)
    print("-" * 80)

    for engine in engines:
        row = f"{engine:<20s}"
        values = []
        for p in prompts:
            key = (engine, p)
            if key in averages:
                val = averages[key]["avg_decode_tps"]
                values.append(val)
                row += f"  {val:<13.1f}"
            else:
                row += f"  {'N/A':<13s}"
        if values:
            row += f"  {sum(values)/len(values):<10.1f}"
        print(row)

    # Print prefill speed comparison
    print("\n" + "=" * 80)
    print("  PREFILL SPEED (tokens/s) - higher is better")
    print("=" * 80)
    header = f"{'Engine':<20s}"
    for p in prompts:
        header += f"{'  ' + p:<15s}"
    header += f"{'  avg':<12s}"
    print(header)
    print("-" * 80)

    for engine in engines:
        row = f"{engine:<20s}"
        values = []
        for p in prompts:
            key = (engine, p)
            if key in averages:
                val = averages[key]["avg_prefill_tps"]
                values.append(val)
                row += f"  {val:<13.1f}"
            else:
                row += f"  {'N/A':<13s}"
        if values:
            row += f"  {sum(values)/len(values):<10.1f}"
        print(row)

    # Print TTFT comparison
    print("\n" + "=" * 80)
    print("  TIME TO FIRST TOKEN (ms) - lower is better")
    print("=" * 80)
    header = f"{'Engine':<20s}"
    for p in prompts:
        header += f"{'  ' + p:<15s}"
    header += f"{'  avg':<12s}"
    print(header)
    print("-" * 80)

    for engine in engines:
        row = f"{engine:<20s}"
        values = []
        for p in prompts:
            key = (engine, p)
            if key in averages:
                val = averages[key]["avg_ttft_ms"]
                values.append(val)
                row += f"  {val:<13.1f}"
            else:
                row += f"  {'N/A':<13s}"
        if values:
            row += f"  {sum(values)/len(values):<10.1f}"
        print(row)

    print()


def export_csv(averages):
    """Export results to CSV for further analysis."""
    csv_path = os.path.join(RESULTS_DIR, "comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("engine,prompt_name,num_runs,avg_input_tokens,avg_output_tokens,"
                "avg_ttft_ms,avg_prefill_tps,avg_decode_tps,avg_total_time_sec\n")
        for key, avg in sorted(averages.items()):
            f.write(f"{avg['engine']},{avg['prompt_name']},{avg['num_runs']},"
                    f"{avg['avg_input_tokens']:.0f},{avg['avg_output_tokens']:.0f},"
                    f"{avg['avg_ttft_ms']:.2f},{avg['avg_prefill_tps']:.1f},"
                    f"{avg['avg_decode_tps']:.1f},{avg['avg_total_time']:.3f}\n")
    print(f"CSV exported to: {csv_path}")


def main():
    print("Loading benchmark results...")
    results = load_all_results()

    if not results:
        print(f"No results found in {RESULTS_DIR}/")
        print("Run benchmarks first: python benchmarks/run_all.py")
        sys.exit(1)

    print(f"Loaded {len(results)} result entries.")

    averages = compute_averages(results)
    print_comparison_table(averages)
    export_csv(averages)

    # Summary stats
    engines = set(r["engine"] for r in results)
    print(f"\nEngines benchmarked: {', '.join(sorted(engines))}")
    print(f"Total runs: {len(results)}")


if __name__ == "__main__":
    main()
