# dotLLM Prefill Profiling 完整过程

本文档记录了对 dotLLM CUDA backend prefill 阶段进行 NSight Systems profiling 的完整过程，包括所有使用的命令行、遇到的问题、以及最终的分析结果。

**硬件环境**: RTX 4090 + i9 14900k + 64GB  
**模型**: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf (约4.92GB)  
**NSight Systems**: 2024.6.2  
**dotLLM**: CUDA backend, 全部层 GPU offload

---

## 1. 背景：benchmark 基线数据

在 profiling 之前，我们已经通过 benchmark 得到了 dotLLM 的 prefill 基线。

> **注意**：最初的 benchmark 实际上使用的是 `flash_attention_prefill_f16` kernel（而非预期的 naive attention）。后来发现后，将代码回退到 naive `attention_f16` kernel 并重新跑了 benchmark 和 profiling。本文档第 2-6 节记录了 flash attention 版本的 profiling 过程（历史参考），第 9 节记录了 naive attention 版本的 profiling 结果（**当前基线**）。

### 当前基线（naive attention）

| 指标 | 值 |
|------|------|
| Prompt tokens | ~2091 |
| TTFT (3次平均) | 1,584 ms |
| Prefill tok/s | 1,388 |
| 理论上限 (W4A16) | 10,325 tok/s |
| 效率 | **13.4%** |

benchmark 详细数据（3次运行）：
```
run0: TTFT=1786ms, prefill=1219 tok/s
run1: TTFT=1556ms, prefill=1399 tok/s
run2: TTFT=1410ms, prefill=1545 tok/s
```

有明显的 warm-up 效应（1786→1556→1410），run 1-2 趋于稳定。

### 历史基线（flash attention，已弃用）

| 指标 | 值 |
|------|------|
| Prompt tokens | ~2177 |
| TTFT (3次平均) | 2,013 ms |
| Prefill tok/s | 1,081 |
| 效率 | **10.5%** |

```
run0: TTFT=2042ms, prefill=1066 tok/s
run1: TTFT=2015ms, prefill=1081 tok/s
run2: TTFT=1983ms, prefill=1098 tok/s
```

---

## 2. 尝试一：nsys profile server 模式（失败）

### 2.1 启动 nsys profiling dotLLM server

```powershell
# 设置 nsys 路径
$env:Path = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64;$env:Path"

# 尝试 profile server 模式
nsys profile --trace=cuda,nvtx --output=dotllm_nsys_server `
    dotnet run --project C:\Users\xiaoyuz\source\repos\dotLLM\src\DotLLM.Cli -- serve `
    C:\Users\xiaoyuz\source\repos\dotLLM\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf `
    -d gpu --gpu-layers 99
```

### 2.2 发送请求

在另一个终端发送请求：
```powershell
curl http://localhost:5000/v1/chat/completions `
    -H "Content-Type: application/json" `
    -d '{"model":"llama","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
```

### 2.3 问题：只捕获到 warmup 的 kernel

nsys 只捕获到了server启动时warmup阶段的CUDA kernel，用户请求触发的inference kernel完全没有被捕获。

**根因分析**：dotLLM server 使用 ASP.NET 的 ThreadPool 线程处理 HTTP 请求。在 Windows 上，nsys 的 CUDA injection 无法追踪到 .NET ThreadPool 子线程上的 CUDA 调用。warmup 代码在主线程执行所以能被捕获，但真正的 inference 在 ThreadPool 线程上，nsys 看不到。

### 2.4 尝试 --no-warmup

```powershell
nsys profile --trace=cuda,nvtx --output=dotllm_nsys_no_warmup `
    dotnet run --project C:\Users\xiaoyuz\source\repos\dotLLM\src\DotLLM.Cli -- serve `
    C:\Users\xiaoyuz\source\repos\dotLLM\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf `
    -d gpu --gpu-layers 99 --no-warmup
```

**结果**：零 CUDA 数据。因为没有 warmup，没有任何早期 CUDA 活动来触发 CUPTI 初始化，加上 ThreadPool 线程的问题，完全没有捕获到任何 kernel。

**结论**：在 Windows 上 nsys profile server 模式不可靠。需要改用 CLI `run` 命令。

---

## 3. 尝试二：nsys profile CLI run 模式（成功）

### 3.1 执行 nsys profile

```powershell
$env:Path = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64;$env:Path"

nsys profile --trace=cuda,nvtx --output=dotllm_nsys_run `
    dotnet run --project C:\Users\xiaoyuz\source\repos\dotLLM\src\DotLLM.Cli -- run `
    C:\Users\xiaoyuz\source\repos\dotLLM\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf `
    -d gpu --gpu-layers 99 -n 1 `
    -p "You are a senior GPU systems engineer. Please write a detailed guide covering modern GPU architecture, LLM inference optimization, CUDA kernel optimization, and benchmarking methodology."
```

> `-n 1` 表示只生成1个token（我们只关心prefill阶段）

**成功**：CLI `run` 命令在主线程执行所有 CUDA 调用，nsys 完整捕获了所有 kernel。

生成的文件：
- `dotllm_nsys_run.nsys-rep` — NSight Systems 原始报告
- `dotllm_nsys_run.sqlite` — 导出的 SQLite 数据库

---

## 4. nsys 数据提取与分析

### 4.1 GPU Kernel 总览

```powershell
nsys stats --report cuda_gpu_kern_sum dotllm_nsys_run.nsys-rep --format csv
```

输出：

```
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
50.1,99385786,32,3105805.8,3041179.0,2987340,3777790,216832.4,flash_attention_prefill_f16
19.0,37727461,193,195479.1,93278.0,22207,3320037,266509.9,dequant_q4_k_f16
16.2,32179988,128,251406.2,249467.5,113790,390073,135318.3,ampere_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_tn
6.3,12416202,32,388006.3,388185.0,381882,390201,1848.4,ampere_fp16_s1688gemm_fp16_128x64_sliced1x2_ldg8_f2f_tn
4.9,9815058,32,306720.6,304234.5,38559,577142,272017.9,dequant_q6_k_f16
1.4,2682577,1,2682577.0,2682577.0,2682577,2682577,0.0,quantized_gemv_q6_k
1.0,2049307,64,32020.4,32047.0,31455,32703,233.8,ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn
0.5,992557,63,15754.9,16480.0,12863,20544,2378.5,fused_add_rmsnorm_f16
0.4,775922,32,24247.6,24224.0,24127,24736,126.5,swiglu_f16
0.2,385787,32,12055.8,12032.0,11840,12255,117.4,rope_f16
0.0,70302,65,1081.6,1088.0,992,1344,56.3,convert_f32_to_f16
0.0,15328,2,7664.0,7664.0,7424,7904,339.4,rmsnorm_f16
0.0,7168,1,7168.0,7168.0,7168,7168,0.0,embedding_lookup_f16
0.0,5952,1,5952.0,5952.0,5952,5952,0.0,add_f16
0.0,1632,1,1632.0,1632.0,1632,1632,0.0,convert_f16_to_f32
```

整理后的 kernel 时间分布：

| Kernel | 总时间 (ms) | 调用次数 | 平均 (ms) | 占比 |
|--------|---:|---:|---:|---:|
| `flash_attention_prefill_f16` | 99.4 | 32 | 3.11 | **50.1%** |
| `dequant_q4_k_f16` | 37.7 | 193 | 0.20 | 19.0% |
| ampere GEMM (64×128) | 32.2 | 128 | 0.25 | 16.2% |
| ampere GEMM (128×64) | 12.4 | 32 | 0.39 | 6.3% |
| `dequant_q6_k_f16` | 9.8 | 32 | 0.31 | 4.9% |
| `quantized_gemv_q6_k` | 2.7 | 1 | 2.68 | 1.4% |
| ampere GEMM (64×64) | 2.0 | 64 | 0.03 | 1.0% |
| `fused_add_rmsnorm_f16` | 1.0 | 63 | 0.016 | 0.5% |
| `swiglu_f16` | 0.8 | 32 | 0.024 | 0.4% |
| `rope_f16` | 0.4 | 32 | 0.012 | 0.2% |
| 其他 (convert, embed, etc.) | 0.1 | - | - | <0.1% |
| **GPU Kernel 合计** | **198.5** | **676** | - | **100%** |

### 4.2 Host 端 CUDA API 总览

```powershell
nsys stats --report cuda_api_sum dotllm_nsys_run.nsys-rep --format csv
```

输出：

```
Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
65.1,1417580893,293,4838160.0,2514360.0,6311,125393723,9731274.4,cuMemcpyHtoD_v2
12.0,262416874,25,10496675.0,7899384.0,7301945,62304798,10913461.1,cuModuleLoadData
8.7,188683340,455,414688.7,99978.0,21691,23570807,1346051.5,cuLaunchKernel
5.9,129261717,450,287248.3,50235.0,813,56856173,3507653.0,cuMemAlloc_v2
2.4,53283433,1,53283433.0,53283433.0,53283433,53283433,0.0,cuCtxCreate_v2
2.3,49911538,224,222819.4,130603.0,44934,497929,159112.5,cudaLaunchKernel
1.9,40534249,450,90076.1,68150.0,724,4197864,226834.5,cuMemFree_v2
1.0,20937944,1,20937944.0,20937944.0,20937944,20937944,0.0,cuCtxDestroy_v2
0.6,12311920,14,879422.9,904019.5,337175,1737540,441732.4,cuLibraryLoadData
0.0,1011019,65,15554.1,14250.0,9188,36205,5433.0,cuMemcpyDtoDAsync_v2
0.0,545103,1,545103.0,545103.0,545103,545103,0.0,cuCtxSynchronize
...
```

整理：

| CUDA API | 总时间 (ms) | 调用次数 | 平均 (μs) | 说明 |
|----------|---:|---:|---:|------|
| `cuMemcpyHtoD` | **1,418** | 293 | 4,838 | 模型权重上传 |
| `cuModuleLoadData` | 262 | 25 | 10,497 | PTX → cubin JIT编译 |
| `cuLaunchKernel` | 189 | 455 | 415 | kernel 启动 |
| `cuMemAlloc` | 129 | 450 | 287 | 显存分配 |
| `cuCtxCreate` | 53 | 1 | 53,000 | CUDA context 创建 |
| `cudaLaunchKernel` | 50 | 224 | 223 | cuBLAS kernel 启动 |
| `cuMemFree` | 41 | 450 | 90 | 显存释放 |
| `cuCtxDestroy` | 21 | 1 | 21,000 | context 销毁 |
| `cuLibraryLoadData` | 12 | 14 | 879 | cuBLAS 库加载 |

### 4.3 GPU Trace Timeline 分析

导出 GPU trace 到 CSV：

```powershell
nsys stats --report cuda_gpu_trace dotllm_nsys_run.nsys-rep --format csv --output .
# 生成 dotllm_nsys_run_cuda_gpu_trace.csv
```

查看 trace 的整体结构：

```powershell
$csv = Import-Csv dotllm_nsys_run_cuda_gpu_trace.csv
"Total events: $($csv.Count)"
$csv | Group-Object Name | Sort-Object Count -Descending | Select-Object Count, Name | Format-Table -AutoSize
```

输出：
```
Total events: 1134

Count Name
  293 [CUDA memcpy Host-to-Device]
  193 dequant_q4_k_f16
  128 ampere_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_tn
   96 [CUDA memset]
   65 [CUDA memcpy Device-to-Device]
   65 convert_f32_to_f16
   64 ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn
   63 fused_add_rmsnorm_f16
   32 ampere_fp16_s1688gemm_fp16_128x64_sliced1x2_ldg8_f2f_tn
   32 rope_f16
   32 dequant_q6_k_f16
   32 flash_attention_prefill_f16
   32 swiglu_f16
    2 rmsnorm_f16
    1 add_f16
    1 convert_f16_to_f32
    1 [CUDA memcpy Device-to-Host]
    1 embedding_lookup_f16
    1 quantized_gemv_q6_k
```

### 4.4 确认 H2D 拷贝全在 load 阶段

通过 GPU trace 时间戳确认推理阶段是否有 H2D 拷贝：

```powershell
$csv = Import-Csv dotllm_nsys_run_cuda_gpu_trace.csv

# 找到推理开始点（embedding kernel）
$first_embed = $csv | Where-Object { $_.Name -eq 'embedding_lookup_f16' }
"Inference start: $($first_embed.'Start (ns)')"

# 检查推理阶段是否有 H2D 拷贝
$inference_start = [long]$first_embed.'Start (ns)'
$h2d_during_inference = $csv | Where-Object {
    $_.Name -eq '[CUDA memcpy Host-to-Device]' -and [long]$_.'Start (ns)' -ge $inference_start
}
"H2D copies during inference: $($h2d_during_inference.Count)"
```

输出：
```
Inference start: 4264916060
H2D copies during inference: 0
```

**确认：推理期间零次 H2D 拷贝。** 全部 1418ms 的 H2D 都发生在模型加载阶段。

### 4.5 确定 nsys trace 的 prompt 长度

```powershell
# Flash attention grid dimensions 揭示了 prompt 长度
$attn = $csv | Where-Object { $_.Name -eq 'flash_attention_prefill_f16' } | Select-Object -First 3
$attn | Select-Object GrdX, GrdY, GrdZ, BlkX | Format-Table -AutoSize
```

输出：
```
GrdX GrdY GrdZ BlkX
119  32   1    128
119  32   1    128
119  32   1    128
```

- GrdX = 119 = ceil(seq_len / TILE_Q) = ceil(476 / 4)
- GrdY = 32 = num_heads
- **nsys trace 的 prompt 长度 ≈ 476 tokens**

这比 benchmark 的 ~2177 tokens 短很多，后续需要做缩放计算。

### 4.6 GPU Timeline 关键时间点

```powershell
$csv = Import-Csv dotllm_nsys_run_cuda_gpu_trace.csv

# 各阶段起止时间
$first_event = $csv[0]
$first_embed = $csv | Where-Object { $_.Name -eq 'embedding_lookup_f16' }
$first_attn  = $csv | Where-Object { $_.Name -eq 'flash_attention_prefill_f16' } | Select-Object -First 1
$last_event  = $csv | Select-Object -Last 1

"First event:     $($first_event.'Start (ns)') - $($first_event.Name)"
"First embedding: $($first_embed.'Start (ns)') - $($first_embed.Name)"
"First attention: $($first_attn.'Start (ns)') - $($first_attn.Name)"
"Last event:      $($last_event.'Start (ns)') + $($last_event.'Duration (ns)') - $($last_event.Name)"
```

输出：
```
First event:     2674434356  - [CUDA memcpy Host-to-Device]
First embedding: 4264916060  - embedding_lookup_f16
First attention: 4335643882  - flash_attention_prefill_f16
Last event:      4546316527  + 21247 - [CUDA memcpy Device-to-Host]
```

### 4.7 Host 端 CUDA API Trace 分析

导出 CUDA API trace：
```powershell
nsys stats --report cuda_api_trace dotllm_nsys_run.nsys-rep --format csv --output .
# 生成 dotllm_nsys_run_cuda_api_trace.csv
```

匹配 GPU event 到 host API call（通过 CorrId）：

```powershell
$gpu = Import-Csv dotllm_nsys_run_cuda_gpu_trace.csv
$api = Import-Csv dotllm_nsys_run_cuda_api_trace.csv

# Embedding kernel 的 host API
$embed_gpu = $gpu | Where-Object { $_.Name -eq 'embedding_lookup_f16' }
$embed_host = $api | Where-Object { $_.CorrID -eq $embed_gpu.CorrId }
"Embedding Host: Start=$($embed_host.'Start (ns)'), Duration=$($embed_host.'Duration (ns)'), Name=$($embed_host.Name)"

# cuCtxSynchronize（inference结束标志）
$sync = $api | Where-Object { $_.Name -eq 'cuCtxSynchronize' }
"cuCtxSynchronize: Start=$($sync.'Start (ns)'), Duration=$($sync.'Duration (ns)')"
```

输出：
```
Embedding Host: Start=4264869317, Duration=62808, Name=cuLaunchKernel
cuCtxSynchronize: Start=5089683786, Duration=545103
```

**Host 端推理时间跨度**：从 embedding launch (4264ms) 到 sync 完成 (5090ms) = **825ms**

### 4.8 PTX JIT 编译时间

```powershell
$module_loads = $api | Where-Object { $_.Name -eq 'cuModuleLoadData' }
$module_loads | Select-Object 'Start (ns)', 'Duration (ns)' | Format-Table -AutoSize
```

输出（25 次 cuModuleLoadData 调用）：
```
Start (ns)  Duration (ns)
2201713278  14153125      # 第一次（含 PTX JIT）
2224111345  7889462
2240178486  8155317
2256060750  7806444
...
2575368482  62304798      # 最大的一次（62ms）
2652975688  12923999
```

25 次 cuModuleLoadData 总计 262ms（PTX → cubin JIT 编译）。全部在模型加载阶段完成。

### 4.9 推理期间 Host 端开销分析

```powershell
# 推理阶段（4100ms → 4600ms）的 host CUDA API 调用
$api = Import-Csv dotllm_nsys_run_cuda_api_trace.csv
$inf_calls = $api | Where-Object {
    [long]$_.'Start (ns)' -ge 4100000000 -and [long]$_.'Start (ns)' -le 4600000000
}

$inf_calls | Group-Object Name | ForEach-Object {
    $sum = ($_.Group | ForEach-Object { [long]$_.'Duration (ns)' } | Measure-Object -Sum).Sum
    [PSCustomObject]@{ Count=$_.Count; TotalMs=[math]::Round($sum/1e6,1); Name=$_.Name }
} | Sort-Object TotalMs -Descending | Format-Table -AutoSize
```

输出：
```
Count TotalMs Name
  391  157.00 cuLaunchKernel
    6  141.60 cuMemcpyHtoD_v2
  224   49.90 cudaLaunchKernel
  288   25.20 cuMemFree_v2
   14   12.30 cuLibraryLoadData
   99    2.50 cuMemAlloc_v2
   65    1.00 cuMemcpyDtoDAsync_v2
   96    0.50 cudaMemsetAsync
   96    0.30 cudaEventRecord
    1    0.20 cuMemcpyDtoH_v2
   14    0.10 cuLibraryGetKernel
  480    0.00 cuStreamGetCaptureInfo_v2
  224    0.00 cuKernelGetName
    1    0.00 cudaEventQuery
    2    0.00 cuCtxSetCurrent
    2    0.00 cuStreamSynchronize
```

推理阶段 host 端耗时最大的是 `cuLaunchKernel`（157ms, 391 calls）和 `cudaLaunchKernel`（50ms, 224 calls）。

### 4.10 推理后清理开销

```powershell
$post_kern = $api | Where-Object { [long]$_.'Start (ns)' -ge 4550000000 }
$post_kern | Group-Object Name | ForEach-Object {
    $sum = ($_.Group | ForEach-Object { [long]$_.'Duration (ns)' } | Measure-Object -Sum).Sum
    [PSCustomObject]@{ Count=$_.Count; TotalMs=[math]::Round($sum/1e6,1); Name=$_.Name }
} | Sort-Object TotalMs -Descending | Format-Table -AutoSize
```

输出：
```
Count TotalMs Name
  437   40.50 cuMemFree_v2
    1   20.90 cuCtxDestroy_v2
    1    0.50 cuCtxSynchronize
   25    0.40 cuModuleUnload
    4    0.20 cudaDeviceSynchronize
   18    0.20 cudaEventDestroy
    3    0.20 cudaFree
    1    0.20 cuStreamDestroy_v2
```

437 次 `cuMemFree` 合计 40ms — 这些是推理完成后释放 scratch buffer 和 KV cache 的开销。

---

## 5. Kernel 时间缩放分析

nsys trace 的 prompt 只有 ~476 tokens，而 benchmark 使用 ~2177 tokens。不同 kernel 的时间复杂度不同，需要做缩放：

```python
seq_nsys, seq_long = 476, 2083
r = seq_long / seq_nsys  # 4.38x

attn_476 = 99.4   # ms, O(n^2) — attention score 矩阵是 n×n
dequant  = 47.5    # ms, O(1)  — 反量化权重矩阵, 大小不随 seq_len 变化
gemm_476 = 46.6    # ms, O(n)  — GEMM 的 M 维度 = seq_len, 线性增长
other_476 = 4.5    # ms, O(n)  — norms/rope/swiglu 都是 O(n)

attn_2083 = attn_476 * r * r    # O(n²): 99.4 * 19.1 = 1903 ms
gemm_2083 = gemm_476 * r        # O(n):  46.6 * 4.38 = 204 ms
other_2083 = other_476 * r      # O(n):  4.5 * 4.38 = 20 ms
total = attn_2083 + dequant + gemm_2083 + other_2083  # 2175 ms
```

输出：
```
Scaling from 476 -> 2083 tokens (ratio=4.38x)

ESTIMATED kernel time for 2083 tokens:
  Flash Attention:     1903 ms  (88%)  O(n²)
  Dequant:               48 ms  (2%)   O(1)
  cuBLAS GEMM:          204 ms  (9%)   O(n)
  Other:                 20 ms  (1%)   O(n)
  TOTAL kernel:        2175 ms

Measured wall clock: 2052 ms
Estimated GPU util:  106%
```

**估算 2175ms vs 实测 2052ms，误差仅 6%**。

> 误差原因：(1) O(n²) 假设是精确的二次缩放, 实际 kernel 可能有常数项; (2) 476 vs 2083 不是精确的整数倍; (3) cuBLAS 对不同 shape 有不同的 kernel 选择。但 6% 的误差验证了分析的正确性。

---

## 6. 使用 benchmark 相同 prompt 的 nsys 验证

上面的缩放分析是从 476 tokens 外推到 ~2083 tokens 的估算。为了**直接验证**，我们使用与 benchmark 相同的 PROMPT_LONG 重新跑了一次 nsys profile。

### 6.1 准备工作

首先需要解决两个 nsys 命令行问题：
1. **`dotnet run` 子进程问题**：nsys 只追踪直接启动的进程，而 `dotnet run` 会 spawn 一个新进程执行 .NET 应用，CUDA 调用发生在子进程中，nsys 捕获不到
2. **Unicode 字符问题**：PROMPT_LONG 包含 Unicode 字符（`—`, `→`, `×`, `²`, `+`），nsys 的命令行解析器抛出 `std::range_error: bad conversion`

**解决方案**：

```powershell
# 1. 先 build CLI 项目为独立 exe，避免 dotnet run 的子进程
dotnet build C:\Users\xiaoyuz\source\repos\dotLLM\src\DotLLM.Cli `
    --configuration Release -o cli_build

# 2. 将 PROMPT_LONG 中的 Unicode 替换为 ASCII 等价字符
python -c "
from benchmarks.common import PROMPT_LONG
clean = PROMPT_LONG.replace('\u2014', '--').replace('\u2192', '->') \
    .replace('\u00D7', 'x').replace('\u00B2', '^2').replace('\u002B', '+')
with open('prompt_long_ascii.txt', 'w', encoding='ascii', errors='replace') as f:
    f.write(clean)
"
```

### 6.2 验证 ASCII prompt 的 token 数量

```powershell
python -c "
import subprocess, json
f = open('prompt_long_ascii.txt','r')
prompt = f.read()
f.close()
r = subprocess.run(['cli_build\\DotLLM.Cli.exe','run',
    'C:\\Users\\xiaoyuz\\models\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
    '-d','gpu','--gpu-layers','99','-n','1','--json','-p',prompt],
    capture_output=True, text=True, timeout=120)
d = json.loads(r.stdout)
print('tokens:', d['usage']['prompt_tokens'])
print('prefill_ms:', d['timings']['prefill_ms'])
print('tok_s:', d['timings']['prefill_tok_s'])
"
```

输出：
```
tokens: 2091
prefill_ms: 2083
tok_s: 1003.84
```

ASCII 版本产生 **2091 tokens**（与原始 PROMPT_LONG 的 2083 tokens 几乎相同），prefill 时间 **2083ms**，与 benchmark 平均 2013ms 吻合。

### 6.3 nsys profile 执行

```powershell
$env:Path = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64;$env:Path"

# 通过 PowerShell 变量传递 prompt，直接 profile 已编译的 exe
$p = Get-Content prompt_long_ascii.txt -Raw
nsys profile --trace=cuda --output=dotllm_nsys_long_real -f true `
    -- cli_build\DotLLM.Cli.exe run `
    C:\Users\xiaoyuz\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf `
    -d gpu --gpu-layers 99 -n 1 --json -p $p
```

输出：
```
WARNING: CPU context switches trace requires administrative privileges, disabling.
WARNING: CPU sampling requires administrative privileges, disabling.
Collecting data...
[1/1] [========================100%] dotllm_nsys_long_real.nsys-rep
Generated: dotllm_nsys_long_real.nsys-rep
```

### 6.4 Kernel 时间结果（2091 tokens, 实测）

```powershell
nsys stats --report cuda_gpu_kern_sum dotllm_nsys_long_real.nsys-rep --format csv --force-export=true
```

```
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
87.2,1707204307,32,53350134.6,53000722.0,51807255,57816302,1418419.4,flash_attention_prefill_f16
9.7,189673553,224,846756.9,444281.0,121821,2408374,667942.0,ampere_fp16_s1688gemm_fp16_128x64_sliced1x2_ldg8_f2f_tn
2.0,38453787,193,199242.4,98750.0,21183,3318182,269270.3,dequant_q4_k_f16
0.5,9616635,32,300519.8,297834.5,37504,583990,266604.3,dequant_q6_k_f16
0.3,5568895,32,174028.0,173293.0,172317,196061,4043.2,swiglu_f16
0.2,4122836,63,65441.8,81055.0,45695,83934,17725.5,fused_add_rmsnorm_f16
0.1,2513940,1,2513940.0,2513940.0,2513940,2513940,0.0,quantized_gemv_q6_k
0.1,1543458,32,48233.1,48031.0,46975,50431,739.0,rope_f16
0.0,70141,65,1079.1,1119.0,992,1344,63.1,convert_f32_to_f16
0.0,32704,2,16352.0,16352.0,7488,25216,12535.6,rmsnorm_f16
0.0,29280,1,29280.0,29280.0,29280,29280,0.0,add_f16
0.0,16959,1,16959.0,16959.0,16959,16959,0.0,embedding_lookup_f16
0.0,1792,1,1792.0,1792.0,1792,1792,0.0,convert_f16_to_f32
```

整理后的 kernel 时间分布（2091 tokens）：

| Kernel | 总时间 (ms) | 调用次数 | 平均 (ms) | 占比 |
|--------|---:|---:|---:|---:|
| `flash_attention_prefill_f16` | **1,707.2** | 32 | **53.4** | **87.2%** |
| ampere GEMM (128×64) | 189.7 | 224 | 0.85 | 9.7% |
| `dequant_q4_k_f16` | 38.5 | 193 | 0.20 | 2.0% |
| `dequant_q6_k_f16` | 9.6 | 32 | 0.30 | 0.5% |
| `swiglu_f16` | 5.6 | 32 | 0.17 | 0.3% |
| `fused_add_rmsnorm_f16` | 4.1 | 63 | 0.065 | 0.2% |
| `quantized_gemv_q6_k` | 2.5 | 1 | 2.5 | 0.1% |
| `rope_f16` | 1.5 | 32 | 0.048 | 0.1% |
| 其他 | 0.2 | - | - | <0.1% |
| **GPU Kernel 合计** | **1,958.8** | - | - | **100%** |

通过 flash attention 的 grid 配置确认 token 数：
```powershell
$csv = Import-Csv dotllm_nsys_long_real_cuda_gpu_trace.csv
$attn = $csv | Where-Object { $_.Name -eq 'flash_attention_prefill_f16' } | Select-Object -First 1
# GrdX=523, TILE_Q=4 → 523 × 4 = 2092 tokens ✓
```

### 6.5 外推估算 vs 实测对比

| 组件 | 476tok nsys | 外推估算 | **2091tok nsys** | 误差 |
|------|---:|---:|---:|---:|
| Flash Attention | 99.4ms | 1,918ms | **1,707ms** | +12.4% |
| cuBLAS GEMM | 46.6ms | 205ms | **190ms** | +7.9% |
| Dequant (Q4K+Q6K) | 47.5ms | 48ms | **48ms** | -1.2% |
| Other | 4.5ms | 20ms | **14ms** | +43.0% |
| **TOTAL** | **198ms** | **2,190ms** | **1,959ms** | **+11.8%** |

| 指标 | 值 |
|------|------|
| CLI wall clock `prefill_ms` | 2,083 ms |
| nsys 实测 kernel 总时间 | 1,959 ms |
| Host overhead | 124 ms (6.0%) |
| **Attention 占比** | **87.2%** |
| Attention vs Tensor Core 理论 | **246× slower** (1,707ms vs 6.9ms) |

**关键结论**：

1. **外推分析误差 ~12%** — 可接受的精度，主要因为 Attention 的 O(n²) 实际系数比纯二次略低（register/SMEM reuse 有一定效果）
2. **GPU utilization 94%** — 2091 tokens 下 host overhead 仅 6%，GPU 几乎满负载
3. **Attention 占 87%** — 实测确认 attention 是绝对瓶颈
4. **246× gap** — 与 Tensor Core 理论峰值的差距实测为 246×（外推估计 276×，因为外推的 attention 偏高）
5. **cuBLAS 选择了不同 kernel** — 476 tokens 使用 `64×128` tile，2091 tokens 使用 `128×64` tile（cuBLAS autotuning 根据 GEMM shape 选择最优 kernel）

---

## 7. Wall Clock 验证：不同 prompt 长度

为了验证 prefill 时间，直接使用 CLI `run` 命令测量不同 prompt 长度：

```powershell
# 启动 Python 子进程调用 dotLLM CLI（可以传递长 prompt）
python -c "
import subprocess, json
prompt = 'Hello, please explain GPU architecture'  # ~10 tokens
result = subprocess.run([
    'dotnet', 'run', '--project', r'C:\Users\xiaoyuz\source\repos\dotLLM\src\DotLLM.Cli',
    '--', 'run',
    r'C:\Users\xiaoyuz\source\repos\dotLLM\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
    '-d', 'gpu', '--gpu-layers', '99', '-n', '1', '--json',
    '-p', prompt
], capture_output=True, text=True)
print(result.stdout)
"
```

三组不同长度的测试结果：

| Prompt tokens | Load ms | Prefill ms | Prefill tok/s |
|---:|---:|---:|---:|
| 179 | 1,954 | 172 | 1,039 |
| 722 | 2,089 | 435 | 1,658 |
| 2,083 | 2,136 | 2,052 | 1,015 |

**关键观察**：
- Load time (~2000ms) 一致：模型加载 + PTX JIT 是固定开销
- Prefill 时间与 benchmark 吻合（2052ms ≈ benchmark 的 2013ms）
- 722 tokens 的 prefill tok/s (1658) 高于 2083 tokens (1015)，因为 attention 是 O(n²)，更长的序列 attention 占比更大

---

## 8. 关键发现总结

> **注意**：以下是基于 flash attention profiling 的历史总结。当前基线使用 naive attention，详见第 9 节。

### 8.1 时间线全景

对于 **476 tokens** 的 nsys trace：

```
Time (ms)    0                     2674        4265   4547  5090
             |                      |           |      |     |
             [cuCtxCreate]          [H2D Model]  [Inference]  [Sync]
             |-- 53ms --| ....      |-- ~1590ms --| -- 282ms --| -- 543ms --|
                                    (load phase)   (GPU compute) (cleanup)
```

- **Load 阶段** (2674→4265): H2D 拷贝 (1418ms) + PTX JIT (262ms) + cuMemAlloc (129ms)
- **Inference** (4265→4547): 282ms GPU span, 其中 198ms 是 kernel compute
- **Cleanup** (4547→5090): cuMemFree (41ms) + cuCtxSynchronize (0.5ms) + cuCtxDestroy (21ms)

### 8.2 Kernel 时间分析 (2091 tokens, **实测**)

```
Attention ██████████████████████████████████████████████ 1707ms (87%)
cuBLAS    █████ 190ms (10%)
Dequant   █ 48ms (2%)
Other     ▏ 14ms (1%)
          ─────────────────────────────────────────────── 1959ms total
```

### 8.3 Attention kernel 与理论峰值

```python
# Attention 计算量
attn_flops = 2 * 2091**2 * 128 * 32 * 32 = 1.15 TFLOP

# RTX 4090 FP16 Tensor Core
theoretical_time = 1.15 TFLOP / 165.2 TFLOPS = 6.9 ms

# 实测
current_time = 1707 ms

# 差距
ratio = 1707 / 6.9 = 246×
```

当前 attention kernel 使用 CUDA core 而不是 Tensor Core，效率只有理论峰值的 **0.40%**。

### 8.4 问题定位

| 编号 | 问题 | 证据 | 影响 |
|:---:|------|------|------|
| 1 | **Attention 不用 Tensor Core** | kernel 使用 `__shfl_down_sync` 做 warp reduction 而非 `mma.sync` | **246× gap** (实测) |
| 2 | **TILE_Q=4 太小** | GrdX=523 for 2091 tok, 意味着每个 block 只处理 4 个 Q row | 低 arithmetic intensity (~4 FLOP/byte) |
| 3 | **12.5% occupancy** | 37.2KB SMEM/block, 128 threads/block | 无法隐藏 memory latency |
| 4 | **Dequant + GEMM 分离** | 193 次 dequant + 224 次 GEMM = 独立 kernel | 额外 4B/param GMEM 流量 |
| 5 | **推理中 cuMemAlloc/Free** | 450 次 alloc + 450 次 free | ~170ms overhead |
| 6 | **高 launch overhead** | 455 次 cuLaunchKernel, 平均 415μs | 短序列下成为瓶颈 |

### 8.5 优化优先级

| 优先级 | 优化 | 当前耗时 (实测) | 预期耗时 | 加速比 |
|:---:|------|---:|---:|---:|
| **P0** | Attention 使用 Tensor Core + 优化 tiling | **1,097ms** (naive) / **1,707ms** (flash) | 30-100ms | 11-57× |
| P1 | Fused dequant + GEMM (Marlin-style) | 238ms* | ~47ms | 5× |
| P2 | CUDA Graph / memory pool | ~236ms† (naive) / ~124ms (flash) | ~0ms | - |

> *238ms = dequant 48ms + cuBLAS GEMM 190ms (包含因分离导致的额外显存带宽)  
> †naive attention 的 host overhead 更大（cuLaunchKernel 1,207ms），但与 kernel 重叠

**理论优化后的 prefill 总时间：~200-350ms (4-7× speedup), 预计 ~6,000-10,000 tok/s**。

---

---

## 9. Naive Attention Profiling（当前基线）

发现原始 benchmark 实际上使用的是 flash attention kernel 后，将代码回退到 naive `attention_f16` kernel（`CudaTransformerModel.LaunchAttentionBest()` 中注释掉 flash attention 分支），重新跑了 benchmark 和 nsys profiling。

### 9.1 重新构建 CLI exe

```powershell
dotnet build C:\Users\xiaoyuz\source\repos\dotLLM\src\DotLLM.Cli `
    --configuration Release -o cli_build
```

### 9.2 Benchmark 结果（naive attention）

```powershell
python benchmarks\bench_dotllm.py
```

3 次运行结果：
```
run0: TTFT=1786ms, prefill=1219 tok/s, decode=20.2 tok/s, total=109.8s
run1: TTFT=1556ms, prefill=1399 tok/s, decode=20.4 tok/s, total=108.1s
run2: TTFT=1410ms, prefill=1545 tok/s, decode=20.3 tok/s, total=108.7s
```

平均: TTFT=1584ms, prefill=1388 tok/s, decode=20.3 tok/s

### 9.3 nsys Profile 执行

```powershell
$p = Get-Content prompt_long_ascii.txt -Raw
nsys profile --trace=cuda --output=dotllm_nsys_naive -f true `
    -- cli_build\DotLLM.Cli.exe run `
    C:\Users\xiaoyuz\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf `
    -d gpu --gpu-layers 99 -n 1 --json -p $p
```

### 9.4 Kernel 时间结果（2091 tokens, naive attention）

```powershell
nsys stats --report cuda_gpu_kern_sum dotllm_nsys_naive.nsys-rep --format csv --force-export=true
```

| Kernel | 总时间 (ms) | 调用次数 | 平均 (ms) | 占比 |
|--------|---:|---:|---:|---:|
| `attention_f16` (naive) | **1,097.4** | 32 | **34.3** | **81.4%** |
| ampere GEMM (128×64) | 188.5 | 224 | 0.84 | 14.0% |
| `dequant_q4_k_f16` | 37.9 | 193 | 0.20 | 2.8% |
| `dequant_q6_k_f16` | 9.9 | 32 | 0.31 | 0.7% |
| `swiglu_f16` | 5.6 | 32 | 0.17 | 0.4% |
| `fused_add_rmsnorm_f16` | 4.1 | 63 | 0.065 | 0.3% |
| 其他 (rope, gemv, convert) | 5.8 | - | - | 0.4% |
| **GPU Kernel 合计** | **1,348** | **676** | - | **100%** |

Grid 配置确认：`attention_f16` grid = 66,912×1×1, block = 256×1（2091 seq × 32 heads = 66,912 blocks/layer）

### 9.5 Host 端 CUDA API

```powershell
nsys stats --report cuda_api_sum dotllm_nsys_naive.nsys-rep --format csv --force-export=true
```

| CUDA API | 总时间 (ms) | 调用次数 | 说明 |
|----------|---:|---:|------|
| cuMemcpyHtoD | 1,823 | 293 | 模型权重上传（load阶段） |
| **cuLaunchKernel** | **1,207** | 455 | kernel启动开销（**巨大**） |
| cuModuleLoadData | 202 | 25 | PTX JIT |
| cudaLaunchKernel | 197 | 224 | cuBLAS 内部 |
| cuMemAlloc | 187 | 450 | 显存分配 |
| cuMemFree | 56 | 450 | 显存释放 |

**关键发现**：`cuLaunchKernel` host 时间 **1,207ms**（naive attention 的 66,912 blocks 导致每次 launch 平均 2.7ms，远高于 flash attention 的 ~0.4ms）。

### 9.6 Flash vs Naive 对比

| | Flash `flash_attention_prefill_f16` | Naive `attention_f16` | 变化 |
|--|---:|---:|---:|
| Attention kernel time | 1,707ms | **1,097ms** | **-35.7%** |
| 总 kernel time | 1,959ms | **1,348ms** | **-31.2%** |
| cuLaunchKernel host | 189ms | 1,207ms | +539% |
| Benchmark TTFT | 2,013ms | 1,584ms | -21.3% |
| cuBLAS GEMM | 190ms | 189ms | ≈相同 |
| Dequant | 48ms | 48ms | ≈相同 |

**Naive attention 在 GPU kernel 时间上快 36%**，因为：
- 100% occupancy（256 threads, ~2.1KB SMEM）vs flash attention 的 12.5%（128 threads, 37KB SMEM）
- 大量 warp 隐藏了 GMEM latency，弥补了不使用 SMEM tiling 的劣势
- flash attention 的 TILE_Q=4 太小，SMEM 太大，occupancy 太低

但 naive attention 的 host 端 cuLaunchKernel 开销巨大（1,207ms vs 189ms），因为 66,912 blocks/layer vs flash 的 16,672 blocks/layer。不过由于 kernel 异步执行，host overhead 与 GPU execution 重叠，对总 wall clock 影响有限。

### 9.7 Naive Attention 效率分析

| | 当前 (实测) | 理论 (TC FP16) | 差距 |
|--|---:|---:|---:|
| Attention 时间 | **1,097ms** | **6.9ms** | **158×** |
| 利用的算力 | 1.05 TFLOPS | 165.2 TFLOPS | |

Naive kernel 虽然比 flash 快，但仍然与 Tensor Core 理论峰值有 **158× 差距**。核心优化方向不变：使用 Tensor Core + 合理的 FlashAttention2 tiling（更大的 TILE_Q，更小的 SMEM，更高的 occupancy）。

---

## 附录 A：nsys 常用命令

```powershell
# 设置 PATH
$env:Path = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64;$env:Path"

# Profile CUDA 应用
nsys profile --trace=cuda,nvtx --output=<name> <command>

# 查看 kernel 汇总
nsys stats --report cuda_gpu_kern_sum <name>.nsys-rep --format csv

# 查看 CUDA API 汇总
nsys stats --report cuda_api_sum <name>.nsys-rep --format csv

# 导出 GPU trace timeline
nsys stats --report cuda_gpu_trace <name>.nsys-rep --format csv --output .

# 导出 API trace
nsys stats --report cuda_api_trace <name>.nsys-rep --format csv --output .

# 强制重新导出（如果修改了 nsys-rep）
nsys stats --report cuda_gpu_kern_sum <name>.nsys-rep --format csv --force-export=true
```

## 附录 B：PowerShell 分析脚本

```powershell
# 加载 GPU trace CSV
$csv = Import-Csv dotllm_nsys_run_cuda_gpu_trace.csv

# 查看所有 event 类型及数量
$csv | Group-Object Name | Sort-Object Count -Descending | Select-Object Count, Name | Format-Table

# 查看某类 kernel 的 grid 配置（用于推断 sequence length）
$csv | Where-Object { $_.Name -eq 'flash_attention_prefill_f16' } |
    Select-Object -First 3 GrdX, GrdY, GrdZ, BlkX | Format-Table

# 计算推理阶段的 H2D 拷贝
$inference_start = [long]($csv | Where-Object { $_.Name -eq 'embedding_lookup_f16' }).'Start (ns)'
$csv | Where-Object {
    $_.Name -eq '[CUDA memcpy Host-to-Device]' -and [long]$_.'Start (ns)' -ge $inference_start
} | Measure-Object

# 加载 API trace 并匹配 GPU event
$api = Import-Csv dotllm_nsys_run_cuda_api_trace.csv
$gpu_event = $csv | Where-Object { $_.Name -eq 'embedding_lookup_f16' }
$host_call = $api | Where-Object { $_.CorrID -eq $gpu_event.CorrId }
$host_call | Format-List
```

## 附录 C：Kernel 时间缩放计算脚本

```python
"""Extrapolate nsys kernel times from short prompt to benchmark prompt length."""

seq_nsys, seq_long = 476, 2083
r = seq_long / seq_nsys  # 4.38x

# nsys measured kernel times (476 tokens)
attn_476  = 99.4   # ms, O(n²) — Q·K^T is n×n for each head
dequant   = 47.5   # ms, O(1)  — weight dequantization, fixed size
gemm_476  = 46.6   # ms, O(n)  — GEMM M-dim = seq_len
other_476 = 4.5    # ms, O(n)  — norms, rope, swiglu

# Extrapolate to benchmark prompt length
attn_2083  = attn_476 * r * r    # O(n²)
gemm_2083  = gemm_476 * r        # O(n)
other_2083 = other_476 * r       # O(n)
total = attn_2083 + dequant + gemm_2083 + other_2083

print(f"Scaling from {seq_nsys} -> {seq_long} tokens (ratio={r:.2f}x)")
print(f"  Flash Attention:  {attn_2083:>7.0f} ms  ({attn_2083/total*100:.0f}%)  O(n²)")
print(f"  Dequant:          {dequant:>7.0f} ms  ({dequant/total*100:.0f}%)  O(1)")
print(f"  cuBLAS GEMM:      {gemm_2083:>7.0f} ms  ({gemm_2083/total*100:.0f}%)  O(n)")
print(f"  Other:            {other_2083:>7.0f} ms  ({other_2083/total*100:.0f}%)  O(n)")
print(f"  TOTAL kernel:     {total:>7.0f} ms")
print(f"\nMeasured wall clock: 2052 ms")
print(f"Estimated GPU util:  {total/2052*100:.0f}%")

# Theoretical attention time with Tensor Cores
attn_flops = 2 * seq_long**2 * 128 * 32 * 32  # FLOPs
tc_tflops = 165.2e12  # RTX 4090 FP16 TC
theoretical_ms = attn_flops / tc_tflops * 1000
print(f"\nAttention TFLOP: {attn_flops/1e12:.2f}")
print(f"Theoretical attention time (TC): {theoretical_ms:.1f} ms")
print(f"Current vs theoretical: {attn_2083/theoretical_ms:.0f}x slower")
```
