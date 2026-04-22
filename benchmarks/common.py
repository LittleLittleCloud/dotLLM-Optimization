"""
Common utilities for LLM inference benchmarks.
Unified output format and shared configurations.
"""

import json
import time
import os
from dataclasses import dataclass, asdict
from typing import Optional

# ============ Benchmark Configuration ============
# Model: llama 3.1 8B int4
MODEL_NAME = "llama-3.1-8b"
QUANTIZATION = "int4"

# Standard test prompt (~4K tokens input)
PROMPT_LONG = """You are a senior GPU systems engineer writing an internal technical document for your team. Please write an extremely detailed and comprehensive guide covering ALL of the following topics. For each topic, provide specific numbers, code examples, hardware specifications, and real-world performance data. Do not summarize — expand each section fully with sub-sections, detailed explanations, and concrete examples.

## Part 1: Modern GPU Architecture Deep Dive

### 1.1 Memory Hierarchy
Provide a complete breakdown of the NVIDIA Ada Lovelace (RTX 4090) memory hierarchy:
- Register file: size per SM, number of registers per thread, register pressure and occupancy tradeoffs
- Shared memory: size per SM (configurable vs L1), bank organization (32 banks, 4-byte width), bank conflict scenarios with specific access patterns, strategies to avoid conflicts (padding, swizzling)
- L1 cache: unified with shared memory, configurable split ratios, cache line size (128 bytes), sector size (32 bytes)
- L2 cache: total size (72MB on RTX 4090), partitioning across memory controllers, cache policies
- Global memory (GDDR6X): capacity (24GB), bus width (384-bit), effective bandwidth calculation (21 Gbps × 384 / 8 = 1008 GB/s), memory controller architecture
- Texture and constant memory: use cases in inference workloads

### 1.2 Compute Architecture
- Streaming Multiprocessors (SMs): count (128 on RTX 4090), internal structure
- CUDA cores per SM: 128 FP32 cores, dual-issue capability
- Tensor Cores: 4th generation, supported data types and throughput for each (FP16: 165.2 TFLOPS, BF16: 165.2 TFLOPS, INT8: 330.3 TOPS, FP8: 330.3 TFLOPS, INT4: 660.6 TOPS)
- Warp schedulers: 4 per SM, instruction dispatch, warp context switching (zero-cost)
- Special Function Units (SFUs): transcendental operations, throughput

### 1.3 Execution Model
- Thread hierarchy: threads → warps (32 threads) → thread blocks → grid
- Warp execution: SIMT model, predicated execution, divergence costs
- Occupancy: definition, calculation methodology, relationship to register usage and shared memory
- Instruction-Level Parallelism (ILP): hiding arithmetic latency
- Memory-Level Parallelism (MLP): hiding memory latency with multiple outstanding requests
- Cooperative groups and synchronization primitives

## Part 2: LLM Inference on GPUs — Detailed Analysis

### 2.1 Transformer Architecture Computation Breakdown
For Llama 3.1 8B specifically (32 layers, hidden dim 4096, 32 attention heads, GQA with 8 KV heads):
- Attention computation: QKV projection (4096 × 4096 × 3 per head group), attention scores, softmax, value aggregation
- FFN computation: gate projection (4096 → 14336), up projection (4096 → 14336), SiLU activation, down projection (14336 → 4096)
- RMSNorm: computation pattern, memory access pattern
- RoPE: rotary position embedding computation
- Exact FLOP count per layer and total for forward pass
- Parameter count breakdown: embedding, attention, FFN, norms

### 2.2 Prefill Phase Analysis
- Compute-bound nature: arithmetic intensity analysis
- Matrix multiplication: GEMM shapes for each layer (batch × seq_len × hidden vs hidden × hidden)
- Achievable FLOPS utilization for different sequence lengths
- Tensor Core utilization: tile sizes (16×16×16 for FP16, 8×8×32 for INT8), occupancy requirements
- Flash Attention: algorithm description, memory savings, IO complexity analysis O(N²d/M) where M is SRAM size
- Paged attention vs Flash attention tradeoffs in prefill

### 2.3 Decode Phase Analysis
- Memory-bound nature: why batch=1 decoding is bandwidth-limited
- Roofline model analysis: operational intensity = 2 FLOPS / (2 bytes for FP16 weight) = 1 FLOP/byte, vs machine balance of 165.2 TFLOPS / 1008 GB/s = 163.9 FLOPS/byte
- KV cache: memory layout, size calculation (2 × 32_layers × 8_kv_heads × head_dim × seq_len × 2_bytes_fp16)
- Token generation latency breakdown: attention (reading KV cache) vs FFN (reading weights)
- Strategies to improve decode: batching (converting to compute-bound), speculative decoding, quantization

### 2.4 Quantization Deep Dive
- Weight-only quantization (W4A16): dequantization during GEMV, fused vs separate dequant kernels
- Group quantization: group size impact (32, 64, 128), scales and zero-points storage overhead
- GPTQ algorithm: layer-wise quantization, Hessian-based rounding, calibration data requirements
- AWQ (Activation-Aware Weight Quantization): salient weight channels, scaling factors
- ExL2: variable bits-per-weight, mixed precision per layer
- Performance impact: memory reduction (8B × 4bit = 4GB vs 8B × 16bit = 16GB), bandwidth savings, dequantization overhead

## Part 3: CUDA Kernel Optimization for LLM Inference

### 3.1 GEMM/GEMV Optimization
- CUTLASS library: template-based GEMM, tile sizes, software pipelining stages
- For decode (GEMV): memory access patterns, vectorized loads (float4/int4), thread coarsening
- Warp-level matrix operations (WMMA/MMA): PTX instructions, fragment types
- Quantized GEMV: fused dequantization, INT4 unpacking (2 weights per byte), scale application
- Register blocking and shared memory staging
- Autotuning: tile size selection based on problem dimensions

### 3.2 Attention Kernel Optimization
- Multi-head attention: parallelization over heads and batch dimension
- Flash Attention kernel implementation details: block sizes, online softmax, shared memory usage
- PagedAttention: virtual memory for KV cache, block tables, memory pool management
- Fused attention kernels: QKV projection + RoPE + attention in single kernel launch

### 3.3 Fusion and Launch Overhead
- Kernel fusion opportunities: RMSNorm + residual, GEMM + activation (SiLU), dequant + GEMV
- CUDA graphs: capturing kernel sequences, replay overhead reduction
- Persistent kernels: staying resident on SMs, reducing launch overhead
- Custom allocators: memory pool design, avoiding cudaMalloc during inference
- Stream management: overlapping compute and memory operations

### 3.4 Performance Analysis Methodology
- Nsight Compute: metrics to examine (sm__throughput, dram__throughput, achieved_occupancy)
- Roofline analysis: plotting operational intensity vs achieved performance
- Memory throughput analysis: L1 hit rates, sector utilization, memory divergence
- Compute throughput analysis: pipe utilization (FMA, Tensor Core, ALU)
- Common bottlenecks and their signatures in profiling data

## Part 4: Framework-Level Optimizations

### 4.1 Runtime Scheduling
- Continuous batching: iteration-level scheduling vs request-level batching
- Priority queues for prefill vs decode
- Memory management: KV cache eviction, preemption strategies
- Dynamic batching: padding-free attention, variable sequence lengths

### 4.2 Model Parallelism (Multi-GPU)
- Tensor parallelism: splitting attention heads and FFN across GPUs, all-reduce communication
- Pipeline parallelism: micro-batching, bubble fraction
- Expert parallelism (for MoE models): all-to-all communication
- NVLink bandwidth (900 GB/s bidirectional on NVLink 4.0) vs PCIe (64 GB/s PCIe 5.0)

### 4.3 Compilation and Code Generation
- TensorRT-LLM: layer fusion, INT4 GEMM kernels, in-flight batching
- Triton: tile-based programming model, auto-tuning
- CUDA ahead-of-time compilation: PTX generation, cubin caching
- JIT compilation overhead and mitigation strategies

## Part 5: Benchmarking Methodology

### 5.1 Metrics Definition
- Time to First Token (TTFT): includes prompt processing and first decode step
- Inter-token Latency (ITL): time between consecutive tokens during decode
- Throughput: tokens per second for decode, tokens per second for prefill
- Normalized throughput: tokens/s/dollar, tokens/s/watt

### 5.2 Measurement Best Practices
- Warmup runs: GPU clock stabilization, CUDA context initialization, JIT compilation
- Statistical rigor: number of runs, confidence intervals, outlier handling
- Controlling variables: GPU clock lock, thermal throttling monitoring, memory fragmentation
- Prompt design: varying lengths to test different operational regimes
- Reporting: hardware specs, driver versions, framework versions, model exact checkpoint

### 5.3 Common Pitfalls
- Measuring wall-clock time vs GPU time (CUDA events)
- Not accounting for tokenization overhead
- Comparing different quantization methods unfairly
- Batch size conflation: batch=1 latency vs throughput-optimized batching
- KV cache warming effects on multi-turn benchmarks

Please provide all of the above content with maximum detail. Use code snippets (CUDA C++, Python) where appropriate. Include specific numeric calculations and hardware specifications throughout. This document will serve as the definitive reference for our inference optimization team."""

# Default generation parameters
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0  # deterministic for reproducibility
DEFAULT_REPEAT = 3  # number of runs for averaging

# Output directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


@dataclass
class BenchmarkResult:
    """Unified benchmark result format."""
    engine: str
    model: str
    quantization: str
    prompt_name: str  # "short", "medium", "long"
    input_tokens: int
    output_tokens: int
    time_to_first_token_ms: float  # TTFT in milliseconds
    prefill_tokens_per_sec: float  # input_tokens / TTFT
    decode_tokens_per_sec: float  # output_tokens / decode_time
    total_time_sec: float  # total generation time
    run_index: int  # which run (for averaging)
    timestamp: str
    notes: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def save_result(result: BenchmarkResult, engine_name: str):
    """Save a single benchmark result to the engine's result file."""
    filepath = os.path.join(RESULTS_DIR, f"{engine_name}.jsonl")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def save_results(results: list[BenchmarkResult], engine_name: str):
    """Save multiple benchmark results."""
    for r in results:
        save_result(r, engine_name)


def get_timestamp():
    """Get current timestamp string."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_result(result: BenchmarkResult):
    """Pretty print a benchmark result."""
    print(f"  [{result.engine}] prompt={result.prompt_name} run={result.run_index}")
    print(f"    Input tokens:  {result.input_tokens}")
    print(f"    Output tokens: {result.output_tokens}")
    print(f"    TTFT:          {result.time_to_first_token_ms:.1f} ms")
    print(f"    Prefill:       {result.prefill_tokens_per_sec:.1f} tokens/s")
    print(f"    Decode:        {result.decode_tokens_per_sec:.1f} tokens/s")
    print(f"    Total time:    {result.total_time_sec:.3f} s")
    print()


def get_prompts():
    """Return dict of prompt_name -> prompt_text."""
    return {
        "long": PROMPT_LONG,
    }
