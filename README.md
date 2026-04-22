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

