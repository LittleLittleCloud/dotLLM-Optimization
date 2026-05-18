# llm-inference-perf-guide: use llama 3.1 && dotLLM as an example

## 背景
llama 3.1大家都很熟了，以下重点介绍dotLLM
dotLLM是一个基于C#实现的大模型推理引擎，支持纯CPU和cuda backend。
dotLLM的cuda编译路径
- .cu -> nvcc (.ptx) -> cuda module (load .ptx) -> inference

本文主要针对提高llama在dotLLM以及cuda上的推理速度。会包含如下章节
- 理论上限
- benchmark 比较dotLLM在推理llama上和主流推理框架的差距
- 性能及瓶颈分析，包括dotLLM本身的瓶颈，kernel实现的瓶颈
- 改进1
- 改进2
...
- 最终改进后的结果
- 结尾

硬件: RTX 4090 + i9 14900k + 64GB

## 理论上限分析
推理速度在prefill和decoding阶段受到不同的boundary限制，prefill是compute bound的，而decoding是memory bound的
因此需要分别分析llama 3.1在该硬件上的推理速度上限

### decoding
4090的显存位宽为384bit，等效频率为21Gbps，所以显存带宽为
$$\text{memory bandwidth} = \frac{\text{显存位宽} \times \text{等效频率}}{8} = 1000 \text{GB.s}$$

而计算每个token，需要从显存中读取全部模型权重
$$
memory per token = 参数量 \times 每参数字节数 = 8 \times 10^9 \times \frac{4bits}{8} = 4GB
$$

所以decoding阶段的理论速度上限为
$$
token per second = \frac{1008}{4} = 252 tokens/s
$$

以上没有考虑compute以及从KV cache所占的开销，所以实际情况会略低

### prefilling
prefilling阶段是compute heavy的，4090的算力如下表所示


| 精度 | 算力 |
|------|------|
| FP32 | 82.6 TFLOPS |
| FP16 (Tensor Core) | 165.2 TFLOPS |
| BF16 (Tensor Core) | 165.2 TFLOPS |
| INT8 (Tensor Core) | 330.3 TOPS |
| **INT4 (Tensor Core)** | **660.6 TOPS** |
| FP8 (Tensor Core) | 330.3 TFLOPS |

在transformer模型中，每个参数大致对应一次乘法和一次加法 （为什么？思考一下）
所以对于llama 3.1 8B的模型，其每个token所需的计算量近似为
$$
OPs per token = 2 \times 8 \times 10^9 = 16 GOPS
$$

在int4精度下，如果实际计算为**W4A16** （权重为INT4，计算的时候将INT4反量化为FP16）（反量化本身是compute bound还是memory bound的？思考一下）（反量化需要单独执行吗，思考一下），则prefilling的理论上限为
$$
tokens/s = \frac{165.2 TFLOPS}{16GOPS} \approximate 10325 tokens/s
$$

如果实际计算为**W4A4** （权重和计算都为INT4），理论上限为
$$
tokens/s = \frac{660.6 TFLOPS}{16GOPS} \approximate 41,288 tokens/s
$$

## 实际推理速度 benchmark （batch = 1的情况下）
以上为理论上限分析，接下来对主流的推理引擎进行基准线分析，选取以下推理引擎

| 引擎 | Windows 原生 | WSL2 | 备注 |
|------|:---:|:---:|------|
| **llama.cpp / Ollama** | ✅ | ✅ | 原生 Windows 支持最好，CUDA 直接可用 |
| **ExLlamaV2** | ✅ | ✅ | pip 安装即可，Windows 原生 CUDA 支持（本次跳过） |
| **vLLM** | ❌ | ✅ | 仅 Linux，需通过 WSL2 使用 |
| **TensorRT-LLM** | ❌ | ✅ | 仅 Linux，**跳过**（见下方说明） |
| **SGLang** | ❌ | ✅ | 仅 Linux，需通过 WSL2 使用 |
| **MLC-LLM** | ✅ | ✅ | 有 Windows 预编译包（本次跳过） |
| **dotLLM** (TODO 包含commit id) |

> **TensorRT-LLM 跳过说明**: TRT-LLM v1.2.1 无法直接加载 GPTQ INT4 预量化 checkpoint（`hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4`）。
> - `pytorch` backend: fused QKV weight loading AssertionError（GPTQ权重格式不兼容）
> - `tensorrt` backend: 不支持 `desc_act=True` 的 GPTQ quantization_config
> - `_autodeploy` backend: `torch.export` dimension specialization 冲突
>
> 需要从 FP16 模型手动构建 TRT engine（FP16模型16GB + KV cache 在 24GB 4090 上空间紧张），故本次跳过。

推理参数如下
- batch size = 1
- prefill length ≈ 2k tokens
- decode length = 2k tokens (max_tokens=2048)
- temperature = 0 (deterministic)
- 每引擎跑3次取平均

### Benchmark 结果

| 引擎 | Decode (tokens/s) | Prefill (tokens/s) | TTFT (ms) | 总时间 (s) | 备注 |
|------|---:|---:|---:|---:|------|
| **SGLang** | **143.6** | **9,669** | **225** | **15.1** | gptq_marlin kernel, `--disable-radix-cache` |
| **Ollama** | 127.3 | 6,062 | 337 | 16.4 | llama.cpp backend, `llama3.1:8b-instruct-q4_0` |
| **vLLM** | 115.6 | 5,759 | 356 | 18.1 | run 1-2 avg (run 0 有 torch.compile JIT 开销) |
| **dotLLM** | 20.3 | 1,388 | 1,584 | 108.9 | CUDA backend, Q4_K_M GGUF, naive attention kernel |
| *理论上限* | *252* | *10,325* | — | — | *RTX 4090, W4A16* |

> **注**: SGLang/vLLM 使用 `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4`（GPTQ 4bit, group_size=128）；Ollama 使用 `llama3.1:8b-instruct-q4_0`（llama.cpp Q4_0）；dotLLM 使用 `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`。量化方式略有差异但均为 ~4bit。
>
> vLLM 的 run 0 因 torch.compile / CUDA graph JIT 编译导致 TTFT ~11s，故仅取 run 1-2 的稳定结果。

### 与理论上限对比

- **Decode**: SGLang 达到理论上限的 **57%**，Ollama **50%**，dotLLM 仅 **8.1%**
- **Prefill**: SGLang 达到 W4A16 理论上限的 **94%**（！），Ollama **59%**，dotLLM **13.4%**

dotLLM 与 Ollama 相比有 **~6.3× 的 decode 差距**，与最快的 SGLang 相比有 **~7.1× 差距**。这就是本文要优化的目标。

### 为什么SGLang的prefill速度接近理论上限？
因为SGLang使用了高效的Marlin kernel，带来三个关键优化点
- fused dequant：将权重的反量化操作与矩阵乘法融合，减少内存访问和计算开销
- 使用cp.async异步加载next tile，同时当前tile在tensor core上计算，减少显存访问等待时间
- 高效tile mapping：根据GEMM维度自动选择最优tile size

### 为什么接近的是FP16理论上限？
- W4A16计算模式下，权重为INT4，计算为FP16，计算量等价于FP16 GEMM
- 非matmul开销很小（例如RMSNorm，ROPE， SiLU等）

## dotLLM性能分析
本章节主要分析dotLLM与主流推理框架的性能差距

我们可以看到dotLLM与主流性能框架间有较大的差距
- prefill部分比SGLang慢7x，为理论上限的13.4%
- decode部分比vLLM慢5.7x

因此有较大的优化空间

我们先使用nsight来profiling 各个kernel的开销，然后分析瓶颈

### Prefill瓶颈分析

prefill的TTFT平均为1584ms（3次: 1786/1556/1410ms，有warm-up效应）

#### dequant和GEMM分开
在计算GEMM的时候，dotLLM的dequant和计算是在两个kernel内完成的，其中dequant调用dequant.cu，GEMM调用cublas。
这导致了两次显存读写，dequant的结果需要写回GMEM，然后GEMM再重新从显存里面读取反量化后的数据

而主流的推理引擎采用fused kernel，反量化工作直接在寄存器上完成，这样就减少了一次现存访问。

dequant和GEMM分开每个权重参数的GMEM访问
步骤	操作	字节/参数
Dequant	读 Q4_K	0.5625
Dequant	写 FP16 到 scratch	2.0
cuBLAS	读 FP16 从 scratch	2.0
cuBLAS	读 activation + 写 output	(与 fused 相同)
合计权重流量		4.5625

fusedkernel每个权重参数的GMEM访问为0.5625

所以额外的流量为4bytes/param, 放在Llama 3.1 8B这个模型中，每层7个linear的参数量为
投影	Shape	参数量	Dequant 流量	Dequant 耗时 （理论峰值1008GB/s)
Q	4096×4096	16.7M	43 MB	~0.04ms
K (GQA)	4096×1024	4.2M	11 MB	~0.01ms
V (GQA)	4096×1024	4.2M	11 MB	~0.01ms
O	4096×4096	16.7M	43 MB	~0.04ms
gate	4096×14336	58.7M	150 MB	~0.15ms
up	4096×14336	58.7M	150 MB	~0.15ms
down	14336×4096	58.7M	150 MB	~0.15ms
层合计		217.9M	558 MB	~0.55ms

32层所带来的额外开销理论上合计17.6ms,实际开销可能为35-70ms

#### NSight Systems Profiling

使用NSight Systems对dotLLM CLI `run`命令进行profiling，使用与benchmark相同的PROMPT_LONG（2091 tokens），得到以下kernel时间分布：

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

> `attention_f16` grid = 66,912 blocks (2091 seq × 32 heads)，每个block只处理1个query×1个head。

同时查看Host端CUDA API调用时间：

| CUDA API | 总时间 (ms) | 调用次数 | 说明 |
|----------|---:|---:|------|
| cuMemcpyHtoD | 1,823 | 293 | 模型权重上传（load阶段） |
| **cuLaunchKernel** | **1,207** | 455 | kernel启动开销（**巨大**） |
| cuModuleLoadData (PTX JIT) | 202 | 25 | PTX → cubin JIT编译 |
| cudaLaunchKernel | 197 | 224 | cuBLAS 内部kernel启动 |
| cuMemAlloc | 187 | 450 | 显存分配 |
| cuMemFree | 56 | 450 | 显存释放 |

**关键发现**：
- H2D拷贝全部发生在模型加载阶段（inference期间零次H2D拷贝），权重是一次性预加载到GPU的
- **cuLaunchKernel host时间高达1,207ms**（naive attention 每次launch平均2.7ms，因为66,912 blocks导致driver端调度开销巨大）
- Benchmark TTFT (avg 1,584ms) ≈ kernel时间 (1,348ms) + host overhead (~236ms, 15%)

#### attention_f16 naive attention kernel 效率分析

Naive attention kernel 参数（`attention.cu`）：

```
TILE_KV = 256       // 每个KV tile 256个位置
Block size = 256    // 256 threads，__launch_bounds__(256)
SMEM = ~2.1KB       // 仅Q vector (FP32) + score_tile + warp_scratch
Grid = seq_len × num_heads = 2091 × 32 = 66,912 blocks
```

**问题1: 不使用Tensor Core**

所有Q·K点积和V累加使用CUDA core串行计算，不使用Tensor Core。

$$
\text{Attention TFLOP (2091 tokens)} = 2 \times 2091^2 \times 128 \times 32 \times 32 = 1.15 \text{ TFLOP}
$$

| | 当前 (实测) | 理论 (TC FP16) | 差距 |
|--|---:|---:|---:|
| Attention时间 | **1,097ms** | **6.9ms** | **158×** |
| 利用的算力 | 1.05 TFLOPS | 165.2 TFLOPS | |

> 合理的预期（使用FlashAttention2 + Tensor Core）应在30-100ms范围内。

**问题2: 每个block只处理1个query位置**

每个block处理1个query × 1个head，导致：
- 66,912 blocks/layer → 32层 = GPU端无问题（RTX 4090可以调度），但driver端dispatch开销巨大
- K/V直接从GMEM读取（不缓存在SMEM中），每个query都要重新读全部KV
- 低arithmetic intensity：每次GMEM读取只做1个query的点积

**问题3: K/V从GMEM直接读取**

- flash attention 将K/V tile加载到SMEM后被TILE_Q个query复用
- naive attention 每个thread直接从GMEM逐元素读取K和V，无SMEM复用
- 但由于100% occupancy，GPU通过大量warp隐藏了memory latency，实际比低occupancy的flash attention更快

**问题4: cuLaunchKernel host overhead**

- 66,912 blocks × 32 layers = 每次forward pass 32次cuLaunchKernel（每次66,912 blocks）
- 每次cuLaunchKernel平均2.7ms，总计1,207ms host端开销
- 但由于kernel异步执行，host overhead与GPU kernel执行重叠，实际影响小于1,207ms

**有意思的发现: Naive比Flash更快**

| | Naive `attention_f16` | Flash `flash_attention_prefill_f16` |
|--|---:|---:|
| Attention kernel time | **1,097ms** | 1,707ms |
| Occupancy | ~100% | ~12.5% |
| K/V from | GMEM | SMEM |
| 总体 | **快36%** | 慢 |

Naive kernel虽然从GMEM读K/V效率低，但100% occupancy意味着大量warp可以隐藏memory latency。Flash attention的SMEM tiling策略是对的，但TILE_Q=4 + 37KB SMEM导致occupancy过低（12.5%），反而更慢。

### Prefill瓶颈总结

| 瓶颈 | 影响 | 当前开销 (实测) | 优化后预期 | 优先级 |
|------|------|---------|-----------|--------|
| Attention无Tensor Core | 158× slower than peak | **1,097ms (81%)** | ~30-100ms | **P0** |
| cuLaunchKernel host overhead | 由naive attention的66,912 blocks导致 | 1,207ms host (与kernel重叠) | 随P0一并解决（block数大幅减少） | **P0附属** |
| Dequant与GEMM分离 | 额外4B/param GMEM流量 | ~48ms (4%) + extra GEMM读取 | fused后接近0 | P1 |
| cuMemAlloc/Free during inference | 每次forward分配释放 | ~236ms (15% host overhead) | memory pool可消除 | P2 |

**核心结论: Attention kernel是prefill的绝对瓶颈（81% kernel time）。使用Tensor Core + FlashAttention2 tiling可以将attention从1,097ms降到~30-100ms，将prefill总时间从~1,400ms降到~200-350ms (约4-7× speedup, ~6,000-10,000 tok/s)。**

### Prefill改进 - Flash Attention

所有改进在 `dev/flash_attention` branch 上进行，kernel 文件为 `flash_attention.cu`（`flash_attention_prefill_f16`）。

Prefill kernel 仅用于 seqQ > 1（prefill阶段），decode（seqQ=1）保持使用原有 naive attention kernel。

#### Commit 1: Flash Attention tiling — TILE_Q=16, BLOCK=256 (`8d7c71a`)

**改动**:
- TILE_Q 从 4 扩大至 16：每个 block 处理 16 行 Q，K/V tile 在 SMEM 中被 16 行 Q 复用（4× 复用提升）
- BLOCK 从 128 增至 256（8 warps）：更多 warp 隐藏 memory latency
- Q tile 以 FP16 存储在 SMEM（节省 4KB）
- 添加 `cuFuncSetAttribute` 支持 >48KB dynamic SMEM
- Grid: `ceil(2091/16) × 32 = 4,192 blocks`（原 naive: 66,912 blocks，减少 **16×**）

**SMEM layout（49.4 KB, 2 blocks/SM, 25% occupancy）**:

| 区域 | 大小 | 说明 |
|------|------|------|
| q_tile[16][128] | 4 KB | FP16, Q tile |
| k_tile[64][128] | 16 KB | FP16, K tile（后续 aliased 为 score_f16） |
| v_tile[64][128] | 16 KB | FP16, V tile |
| score_f32[16][64] | 4 KB | FP32, attention scores |
| out_accum[16][128] | 8 KB | FP32, output accumulator |
| running_max/sum[16] | 128 B | online softmax state |

此版本仍使用 CUDA core（scalar FMA）计算 Q·K^T 和 score·V。

**NSight Profiling**:

| Kernel | Time | Instances | % |
|--------|------|-----------|---|
| flash_attention_prefill_f16 | **999ms** | 32 | 44.5% |
| ampere GEMM | 185ms | 224 | 8.2% |
| 其他 | 62ms | — | — |
| **Kernel 合计** | **1,246ms** | — | **100%** |

**结果（vs naive baseline）**:

| 指标 | Naive | Commit 1 | 变化 |
|------|-------|----------|------|
| Attention kernel | 1,097ms | 999ms | -9% |
| Prefill tok/s | 1,388 | 1,696 | **+22%** |
| TTFT | 1,584ms | 1,284ms | -19% |

> 虽然 attention kernel 仅快了 9%（从 CUDA core 角度已接近极限），但 block 数从 66,912 降到 4,192 大幅减少了 cuLaunchKernel host overhead（从 1,207ms 降至几十 ms），使 TTFT 降低了 19%。

#### Commit 2: Tensor Core wmma m16n16k16 (`245a6e0`)

**改动**:
将 Q·K^T 和 score·V 两个 GEMM 从 CUDA core scalar FMA 替换为 `wmma::mma_sync`（m16n16k16 FP16→FP32）。

Kernel 4-phase 设计：

```
Phase 1: S[16,64] = Q[16,128] · K^T[128,64]    ← wmma (4 warps × 8 K-iters)
Phase 2: Online softmax + causal mask            ← CUDA core (串行 per-query)
Phase 3: score FP32 → FP16 转换                  ← 复用 k_tile area (aliased)
Phase 4: O[16,128] += score[16,64] · V[64,128]  ← wmma (8 warps × 4 K-iters)
```

- Phase 1 使用 4 个 warp：4 个 N-tile（64/16=4），每个 warp 8 次 K 迭代（128/16=8）
- Phase 4 使用 8 个 warp：8 个 N-tile（128/16=8），每个 warp 4 次 K 迭代（64/16=4）
- K tile 区域在不同 phase 被 alias 复用（score_f16、delta_O）
- Partial tile（Q 行数 < 16）zero-pad 以满足 wmma tile 尺寸

SMEM layout、occupancy 不变（49.4 KB, 25%）。

**NSight Profiling**:

| Kernel | Time | % |
|--------|------|---|
| flash_attention_prefill_f16 | **410ms** | 62.4% |
| ampere GEMM | 185ms | 28.2% |
| 其他 | 62ms | 9.4% |
| **Kernel 合计** | **657ms** | **100%** |

**结果（vs Commit 1）**:

| 指标 | Commit 1 | Commit 2 | 变化 |
|------|----------|----------|------|
| Attention kernel | 999ms | 410ms | **2.4× faster** |
| Prefill tok/s | 1,696 | 2,725 | **+61%** |
| TTFT | 1,284ms | 767ms | -40% |

> Tensor Core 带来了最大的单步提升。Attention 从 999ms 直降至 410ms，但此时 Phase 2（online softmax）的串行设计成为了新瓶颈：每个 query 行串行执行 block-wide `__syncthreads` reduction，16 行 × ~8 次 sync per query = ~128 次 `__syncthreads` per KV tile。

#### Commit 3: Warp-parallel softmax (`ac09810`)

**改动**:
将 Phase 2 从串行 per-query softmax 替换为 warp-parallel 处理。

**Before (串行)**:
```cuda
for (int qi = 0; qi < num_q; qi++) {
    // block-wide reduction: __syncthreads × 2 (max + sum)
    // 所有 256 threads 参与, 但只处理 1 行 query
    // ~8 __syncthreads per query × 16 queries = ~128 syncs per KV tile
}
```

**After (warp-parallel)**:
```cuda
// 8 warps × 2 queries each = 16 queries 并行处理
int warp_id = threadIdx.x / 32;
int lane = threadIdx.x % 32;
for (int w = 0; w < 2; w++) {
    int qi = warp_id * 2 + w;
    // warp-level reduction: __shfl_down_sync + __shfl_sync (无 __syncthreads)
    // 每个 warp 独立处理, 只需 lane 0-31 做 reduction
}
__syncthreads();  // 仅 1 次, Phase 2 结束时
```

关键改进：
- 每个 warp 独立处理 2 行 query（32 lanes 遍历 64 KV 位置）
- Reduction 使用 `__shfl_down_sync` (warp-level)，不需要 `__syncthreads` (block-level)
- 每个 KV tile 从 ~128 次 `__syncthreads` 降到 **1 次**
- 移除了不再需要的 `warp_scratch` shared memory（节省 128B）

**NSight Profiling**:

| Kernel | Time | % |
|--------|------|---|
| flash_attention_prefill_f16 | **194ms** | 43.4% |
| ampere GEMM | 187ms | 41.9% |
| dequant_q4_k_f16 | 40ms | 8.9% |
| 其他 | 26ms | 5.8% |
| **Kernel 合计** | **447ms** | **100%** |

**结果（vs Commit 2）**:

| 指标 | Commit 2 | Commit 3 | 变化 |
|------|----------|----------|------|
| Attention kernel | 410ms | 194ms | **2.1× faster** |
| Prefill tok/s | 2,725 | 3,821 | **+40%** |
| TTFT | 767ms | 589ms | -23% |

> Attention 与 GEMM 现在基本持平（43% vs 42%），瓶颈不再集中于单一 kernel。

#### Commit 4: cp.async K/V prefetch (`d8c4def`)

**改动**:
将 K/V tile 的同步 GMEM→SMEM 加载替换为 `cp.async.cg.shared.global`（16B per copy，硬件直传绕过 L1 + 寄存器），V tile 加载与 Phase 1+2+3 overlap。

**Before (同步加载)**:
```cuda
// K tile: 同步加载, 所有线程等待完成
for (int i = tid; i < tile_kv_len * head_dim; i += PREFILL_BLOCK)
    k_tile[...] = k_base[...];
// V tile: 也是同步加载, K 和 V 全部加载完才能开始 Phase 1
for (int i = tid; i < tile_kv_len * head_dim; i += PREFILL_BLOCK)
    v_tile[...] = v_base[...];
__syncthreads();  // K+V 都 ready
```

**After (cp.async pipeline)**:
```cuda
// K tile: cp.async 16B copies → commit → wait (Phase 1 需要 K)
cp_async_cg_16(&k_tile[...], &k_base[...]);  // 硬件执行, 绕过 L1
cp_async_commit();
cp_async_wait_all();
__syncthreads();  // K ready → Phase 1 开始

// V tile: cp.async 16B copies → commit (不 wait!)
cp_async_cg_16(&v_tile[...], &v_base[...]);
cp_async_commit();
// V 加载与 Phase 1+2+3 重叠执行...

// Phase 3 结束后:
cp_async_wait_all();  // V ready → Phase 4 开始
__syncthreads();
```

关键改进：
- `cp.async.cg.shared.global`: GMEM → SMEM 硬件直传，不经过寄存器和 L1 cache
- V tile 16KB 的加载延迟完全隐藏在 Phase 1（Q·K^T wmma）+ Phase 2（softmax）+ Phase 3（FP32→FP16）的计算之后
- 每个 KV tile 迭代节省一次显存 stall
- SMEM layout 不变，occupancy 不变（25%）
- PTX target 从 `compute_75` 升级到 `compute_80`（cp.async 需要 sm_80+）

**NSight Profiling**:

| Kernel | Time | % |
|--------|------|---|
| ampere GEMM | **193ms** | 52.4% |
| flash_attention_prefill_f16 | **113ms** | 30.7% |
| dequant_q4_k_f16 | 39ms | 10.5% |
| 其他 | 24ms | 6.4% |
| **Kernel 合计** | **369ms** | **100%** |

**结果（vs Commit 3）**:

| 指标 | Commit 3 | Commit 4 | 变化 |
|------|----------|----------|------|
| Attention kernel | 194ms | 113ms | **-42%** |
| Prefill tok/s | 3,821 | 4,423 | **+16%** |
| TTFT | 589ms | 522ms | -11% |

> GEMM 现在成为绝对瓶颈（52%），attention 降到 31%。进一步优化 attention 的收益递减，瓶颈已经转移到 dequant+GEMM。

#### Commit 5: Persistent O accumulator in wmma registers (`fea2e7b`)

**改动**:
将 Phase 4 的输出 accumulator 从 SMEM（delta_O → out_accum 逐 tile 累加）替换为 persistent wmma fragment（`o_frag`）保持在寄存器中跨 KV tile 存活。

**Before (delta_O → SMEM 累加)**:
```cuda
// 每个 KV tile:
wmma::fragment<..., float> o_frag;           // 每 tile 新建、清零
wmma::fill_fragment(o_frag, 0.0f);
wmma::mma_sync(o_frag, w_frag, v_frag, o_frag);
__syncthreads();                              // wmma 完成
wmma::store_matrix_sync(delta_O, o_frag, ...); // 写 SMEM (8KB)
__syncthreads();                              // delta_O ready
for (i) out_accum[i] += delta_O[i];          // 读+写 SMEM (16KB)
__syncthreads();                              // out_accum updated
```

**After (persistent register accumulator)**:
```cuda
// KV tile 循环之前:
wmma::fragment<..., float> o_frag;            // 一次分配，跨 tile 存活
wmma::fill_fragment(o_frag, 0.0f);

// 每个 KV tile:
// 1. 用 correction_s[] 缩放 o_frag（online softmax 的 prev→new max 修正）
o_frag.x[i] *= correction_s[row_of_element_i];
// 2. 直接累加
wmma::mma_sync(o_frag, w_frag, v_frag, o_frag);
__syncthreads();  // 仅 1 次，准备下一 tile

// 循环结束后:
wmma::store_matrix_sync(out_accum, o_frag, ...);  // 仅最终 1 次 SMEM 写
```

关键改进：
- **消除 3 个 `__syncthreads` per KV tile**：delta_O store sync、delta_O load sync、out_accum update sync
- **消除 ~32KB SMEM 读写 per KV tile**：delta_O write (8KB) + read (8KB) + out_accum read (8KB) + write (8KB)
- 对 33 个 KV tile（2091 tokens）：节省 **99 次 `__syncthreads`** 和 **~1056KB SMEM I/O** per block
- correction_s[16] array (64B) 存储在 SMEM，由 Phase 2 softmax 写入供 Phase 4 preamble 读取
- wmma fragment 的 element→row 映射（sm_80）：elements [0-3] → row = lane/4, elements [4-7] → row = lane/4 + 8
- SMEM layout 新增 correction_s（+64B），CudaKernels.cs sharedBytes 同步更新

**NSight Profiling**:

| Kernel | Time | % |
|--------|------|---|
| ampere GEMM | **180ms** | 49.4% |
| flash_attention_prefill_f16 | **93ms** | 25.6% |
| quantized_gemv_q6_k | 51ms | 14.0% |
| dequant_q4_k_f16 | 39ms | 10.7% |
| 其他 | ~1ms | 0.3% |
| **Kernel 合计** | **364ms** | **100%** |

**结果（vs Commit 4）**:

| 指标 | Commit 4 | Commit 5 | 变化 |
|------|----------|----------|------|
| Attention kernel | 113ms | 93ms | **-17.7%** |
| Prefill tok/s | 4,423 | 4,442 (avg) / 4,575 (best) | **+0.4%** / **+3.4%** |
| TTFT | 522ms | 471ms (avg) / 457ms (best) | **-9.8%** / **-12.5%** |

> attention kernel 节省 20ms（17.7%），但 GEMM+dequant 仍占 60% kernel time，所以 prefill 总吞吐提升有限。TTFT 因 attention 耗时减少而改善更显著。

#### Prefill改进总览

**Attention kernel 进化**:

| 版本 | Attention | 核心变化 |
|------|-----------|---------|
| Naive baseline | 1,097ms | CUDA core, 1 query/block, GMEM K/V |
| Commit 1: Flash tiling | 999ms | TILE_Q=16, SMEM K/V, 25% occupancy |
| Commit 2: Tensor Core | 410ms | wmma m16n16k16, 4-phase pipeline |
| Commit 3: Warp softmax | 194ms | warp-parallel reductions, 1 sync/tile |
| Commit 4: cp.async | 113ms | V prefetch overlapped with Phase 1+2+3 |
| **Commit 5: Persistent O** | **93ms** | **Register accumulator, -3 syncs/tile** |

**端到端 Prefill 性能**:

| 版本 | Prefill tok/s | TTFT | vs Naive | vs SGLang |
|------|--------------|------|----------|----------|
| Naive baseline | 1,388 | 1,584ms | 1.0× | 0.14× |
| Commit 1 | 1,696 | 1,284ms | 1.2× | 0.18× |
| Commit 2 | 2,725 | 767ms | 2.0× | 0.28× |
| Commit 3 | 3,821 | 589ms | 2.8× | 0.40× |
| Commit 4 | 4,423 | 522ms | 3.2× | 0.46× |
| **Commit 5** | **4,442** | **471ms** | **3.2×** | **0.46×** |
| *SGLang* | *9,669* | *225ms* | *7.0×* | *1.0×* |
| *理论上限 (W4A16)* | *10,325* | — | *7.4×* | — |

#### 更新后的 Benchmark 结果

| 引擎 | Decode (tokens/s) | Prefill (tokens/s) | TTFT (ms) | 备注 |
|------|---:|---:|---:|------|
| **SGLang** | **143.6** | **9,669** | **225** | gptq_marlin kernel |
| **Ollama** | 127.3 | 6,062 | 337 | llama.cpp backend |
| **vLLM** | 115.6 | 5,759 | 356 | torch.compile |
| **dotLLM (优化后)** | 21.2 | **4,442** | **471** | **flash attn + TC + warp softmax + cp.async + persistent O** |
| dotLLM (baseline) | 20.3 | 1,388 | 1,584 | naive attention |
| *理论上限* | *252* | *10,325* | — | *RTX 4090, W4A16* |

- **Prefill**: dotLLM 从理论上限的 13.4% 提升到 **43.0%**，与 SGLang 差距从 7.0× 缩小到 **2.2×**
- **Decode**: 尚未优化（21.2 tok/s, 理论上限的 8.4%）

#### 剩余差距分析

dotLLM（4,442 tok/s）vs SGLang（9,669 tok/s）仍有 **2.2×** 差距，主要来自：

| 瓶颈 | dotLLM 当前 | SGLang | 差距原因 |
|------|-----------|--------|--------|
| GEMM+Dequant | 180ms + 39ms = 219ms (60%) | ~150ms (Marlin fused) | dotLLM dequant 与 GEMM 分离，多一次 GMEM 读写 |
| Attention | 93ms (26%) | ~30ms (est.) | occupancy 仅 25%, Phase 1 仅 4/8 warp |
| 其他 kernel | 52ms (14%) | ~20ms | gemv_q6 + misc |

下一步可优化方向：
1. **Fused dequant GEMM** — 消除分离的 dequant kernel（39ms）+ 减少 GEMM 的 GMEM 读取，当前最大瓶颈
2. **Phase 1 全 8 warp 参与 Q·K^T** — 当前仅 4 warp 活跃，另 4 warp idle（已尝试，+4ms 因额外 sync 开销，需更高效分工策略）
3. **Double-buffer KV tiles** — 当前仅 V 与 compute overlap，K 仍同步加载；用 2× SMEM 做 ping-pong 可重叠下一 tile 的 K 加载

## 附录：如何实测 SGLang attention kernel 耗时

下面给出可直接执行的最小流程。核心思路是：
- prefill 场景：长 prompt + `max_tokens=1`
- decode 场景：短 prompt + `max_tokens=2048`
- 用 Nsight Systems 统计 GPU kernel 时间，再筛选 attention 相关 kernel

### 1) 启动 SGLang 服务（WSL2/Linux）

```bash
python -m sglang.launch_server \
    --model-path hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 \
    --quantization gptq \
    --port 8002 \
    --disable-radix-cache
```

### 2) Prefill 场景 profiling（attention prefill kernel）

```bash
cd /mnt/c/Users/xiaoyuz/source/repos/dotLLM-Optimization

nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --force-overwrite=true \
    --output sglang_nsys_prefill \
    python benchmarks/profile_sglang_attention.py \
        --mode prefill \
        --repeat 3 \
        --warmup 2
```

导出统计：

```bash
nsys stats --report gpukernsum,cuapisum sglang_nsys_prefill.nsys-rep > sglang_nsys_prefill_stats.txt
```

### 3) Decode 场景 profiling（attention decode kernel）

```bash
cd /mnt/c/Users/xiaoyuz/source/repos/dotLLM-Optimization

nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --force-overwrite=true \
    --output sglang_nsys_decode \
    python benchmarks/profile_sglang_attention.py \
        --mode decode \
        --repeat 3 \
        --warmup 2 \
        --decode-max-tokens 2048
```

导出统计：

```bash
nsys stats --report gpukernsum,cuapisum sglang_nsys_decode.nsys-rep > sglang_nsys_decode_stats.txt
```

### 4) 聚焦单个 attention kernel（Nsight Compute）

```bash
ncu \
    --set full \
    --kernel-name-base demangled \
    --kernel-name ".*attn.*|.*flash.*|.*paged.*" \
    --launch-skip 10 \
    --launch-count 20 \
    -o sglang_attn_ncu \
    python benchmarks/profile_sglang_attention.py --mode prefill --repeat 1 --warmup 1
```

### 5) 结果判读建议

- 在 `gpukernsum` 中按 kernel 名称过滤 `attn|flash|paged|triton`，累计 `Total Time` 作为 attention 总耗时。
- prefill 与 decode 分开汇报，不要混成一个均值。
- 首轮常有 JIT/graph 开销，建议丢弃 run0 或固定 warmup>=2。
- 若 attention 占比下降但端到端收益有限，瓶颈通常已转移到 GEMM/dequant 或调度。

