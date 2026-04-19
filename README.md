# 提高llm的推理速度 - 以llama 3.1 8b int4 和dotLLM 为例

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
| **ExLlamaV2** | ✅ | ✅ | pip 安装即可，Windows 原生 CUDA 支持 |
| **vLLM** | ❌ | ✅ | 仅 Linux，需通过 WSL2 使用 |
| **TensorRT-LLM** | ❌ | ✅ | 仅 Linux，需通过 WSL2/Docker 使用 |
| **SGLang** | ❌ | ✅ | 仅 Linux，需通过 WSL2 使用 |
| **MLC-LLM** | ✅ | ✅ | 有 Windows 预编译包 |
| **dotLLM** (TODO 包含commit id) |

## dotLLM性能分析
