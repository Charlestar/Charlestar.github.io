---
layout: post
title: "NVFP4 KV Cache：Blackwell 上的 4-bit 缓存量化"
subtitle: "从 E2M1 与双层缩放到容量、内核和精度验证"
date: 2026-06-02 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
mathjax: true
tags: [AI Infra, LLM推理, KV Cache, NVFP4, Blackwell, 量化]
---

KV Cache 是用显存换计算：每生成一个 token，模型把它在各层产生的 Key 和 Value 保存下来；下一步只计算新 token，再读取历史 K/V 完成 attention。上下文越长、并发越高，省下的重复计算越多，驻留在显存中的状态也越大。

把 KV 从 BF16/FP16 压到 FP8 已经能显著减少容量。NVFP4 继续把主要数值压到 4 bit，并用细粒度缩放维持动态范围。它带来的直接变化不是“模型以 4 bit 计算所有东西”，而是：

- KV block 更小，相同 HBM 能放更多上下文或请求；
- decode 读取历史 KV 的字节数减少；
- 写入时要量化，读取时要反量化；
- attention 接收到的是带量化误差的 K/V。

所以判断 NVFP4 是否合适，需要同时理解数值格式、缓存布局、attention 内核和业务精度，而不能只看 `4 bit` 这一个数字。

## KV Cache 为什么会成为显存主体

以标准 multi-head/GQA attention 为例，每个 token、每一层要保存 \(H_{kv}\) 个 K heads 和 \(H_{kv}\) 个 V heads，每个 head 维度为 \(D\)。忽略页对齐、缩放因子与 allocator metadata，缓存大小为：

\[
M_{KV}=2\cdot L\cdot N\cdot H_{kv}\cdot D\cdot B
\]

其中：

- \(L\)：attention 层数；
- \(N\)：所有在途请求当前持有的 token 总数；
- \(H_{kv}\)：KV head 数量；
- \(D\)：head dimension；
- \(B\)：每个 K/V 元素的字节数；
- 系数 2：分别保存 Key 与 Value。

单 token 的裸缓存量则是：

\[
M_{token}=2LH_{kv}DB
\]

假设一个模型有 80 层、8 个 KV heads、head dimension 为 128：

| 存储精度 | 裸字节/token | 128K token 裸容量 |
| --- | ---: | ---: |
| BF16/FP16 | 320 KiB | 40 GiB |
| FP8 | 160 KiB | 20 GiB |
| 裸 FP4 payload | 80 KiB | 10 GiB |

这个例子不是某个具体模型的配置，而是为了展示线性关系。实际容量还会受 page/block size、tensor parallel 切分、GQA/MLA、prefix cache、scale metadata、滑动窗口和混合 attention 层影响。

它也解释了为什么权重量化后，KV 更容易成为下一项瓶颈：权重大小对一个实例大体固定，KV 却随活跃 token 总数持续增长。

## Prefill 写一次，Decode 反复读

KV Cache 在两个阶段中的数据流不同。

### Prefill

模型并行处理 prompt 中的大量 token，计算每层 K/V 并写入 cache：

```text
prompt hidden states
  -> K/V projection
  -> attention
  -> write all prompt K/V blocks
```

prefill 通常有较大的矩阵乘法，更容易偏向 compute-bound。KV 写入和容量仍重要，但不是所有场景的主导时间。

### Decode

每一步只有少量新 token，却要读取整个历史上下文的 K/V：

```text
new token
  -> new Q/K/V
  -> read historical K
  -> QKᵀ + softmax
  -> read historical V
  -> attention output
  -> append new K/V
```

上下文较长时，每一步读取量随序列长度增长，decode attention 更容易受 HBM 带宽限制。把历史 K/V 压缩，既减少容量，也可能减少每个 decode step 的字节流量。

但减少存储字节不等于 attention 时间按同一比例下降。反量化、scale 读取、kernel launch、softmax、Q/K/V projection、MLP 和 model parallel collective 仍然存在。只有 profile 显示 KV 访问确实处于关键路径，带宽节省才容易转化为 TPOT 改善。

## NVFP4 中的 4 bit 表示什么

NVFP4 的数值 payload 使用 E2M1：

```text
1 sign bit | 2 exponent bits | 1 mantissa bit
```

忽略特殊编码细节，可以把非负可表示值直观地看作：

```text
0, 0.5, 1, 1.5, 2, 3, 4, 6
```

负数使用对应符号。因此单看 E2M1，它能表示的点非常少，范围也只有约 \([-6,6]\)。若直接把真实 K/V 四舍五入到这些点，较小值会被压成 0，较大值会被截断到 6，误差很难接受。

缩放因子负责把不同范围的真实数据映射到这组离散值上。量化的关键并不是“4 bit 能表示多少绝对数值”，而是每组数值能否找到合适的局部尺度。

## 双层缩放怎样恢复动态范围

NVFP4 对每 16 个值使用一个 E4M3 FP8 block scale，并为更大张量使用一个 FP32 global scale。设某个 micro-block 的全局尺度为 \(s_t\)，局部尺度为 \(s_b\)，可以用下式理解量化与反量化：

\[
q_i=Q_{E2M1}\left(\frac{x_i}{s_t s_b}\right)
\]

\[
\hat{x}_i=s_t s_b q_i
\]

其中 \(Q_{E2M1}\) 表示舍入到最近 E2M1 值并处理超出范围的输入。

数据组织可以画成：

```text
tensor global scale: FP32
│
├── values  0..15  -> one E4M3 scale -> sixteen E2M1 values
├── values 16..31  -> one E4M3 scale -> sixteen E2M1 values
├── values 32..47  -> one E4M3 scale -> sixteen E2M1 values
└── ...
```

这种设计有两层意义：

1. block size 只有 16，局部异常值只影响附近一小组数；
2. E4M3 scale 不局限于 2 的整数次幂，能用小数精度贴近 block 的真实范围。

相比之下，MXFP4 使用更大的 32-value block 和 E8M0 power-of-two scale。NVFP4 付出更密集、精度更高的 scale metadata，换取更低的量化误差。

## 用一组数直观看缩放的作用

假设某个 micro-block 的值主要落在 \([-0.7,0.7]\)，另一个 block 落在 \([-5.5,5.5]\)。若二者共享同一尺度：

- 尺度按第二组选择时，第一组大量小值可能被舍入到 0 或少数几个点；
- 尺度按第一组选择时，第二组的极值会被 clipping。

让它们各自拥有 block scale 后，两个 block 都可以把自己的最大幅度映射到 E2M1 的 6 附近，充分使用有限的离散值。

不过 max-based scale 也未必总是重建误差最小。一个孤立 outlier 可能把尺度拉大，让其余 15 个值精度降低。当前 vLLM 源码还出现了 `nvfp4_4over6` 模式，它会在 `max/6` 与 `max/4` 两类尺度中按重建误差选择，正说明量化策略不只由存储格式决定。

文章讨论的 NVFP4 是格式与常见路径；具体 rounding、scale selection、stochastic rounding 或校准算法应以所用 Model Optimizer 和 backend 版本为准。

## “4 bit”为什么不是严格的 4 bits/value

一个 16-value micro-block 包含：

- 16 个 E2M1：\(16\times4=64\) bits，即 8 bytes；
- 1 个 E4M3 block scale：8 bits，即 1 byte；
- 另有 amortized FP32 tensor scale、padding 和布局元数据。

暂不计 global scale 与对齐，仅 payload + block scale 就是：

\[
\frac{64+8}{16}=4.5\text{ bits/value}
\]

相对于 BF16 的 16 bits/value，理想压缩比约为：

\[
\frac{16}{4.5}\approx3.56\times
\]

相对于 FP8 的 8 bits/value，理想压缩比约为：

\[
\frac{8}{4.5}\approx1.78\times
\]

因此 NVIDIA 对通用 NVFP4 格式使用过约 3.5 倍于 FP16、1.8 倍于 FP8 的内存缩减表述；KV Cache 博文则将实现效果概括为相对 FP8 约/最高 50% 的 footprint 降低。二者并不意味着可以忽略 scale，而是统计口径和具体布局不同。

容量规划不应手算 `4/8` 后直接把并发翻倍。应读取框架报告的真实 page size、可分配 block 数与 scale buffer；vLLM 当前的 KV cache interface 也为 NVFP4 使用专门的 real page size 计算，而不是把 dtype 大小简单乘到 head dimension 上。

## KV 的写入与读取路径

NVFP4 KV Cache 并不要求 attention 全程以 E2M1 做乘加。NVIDIA 公布的路径是：缓存以 NVFP4 保存，使用时先反量化到 FP8，再进入 attention/context math。

### 写入新 token

```text
K/V projection output
    │
    ├── compute/select tensor + block scales
    ├── round and clip to E2M1
    ├── pack two 4-bit values per byte
    └── write FP4 payload + scale metadata to KV page
```

### 读取历史 token

```text
KV page payload + scales
    │
    ├── unpack E2M1
    ├── apply block and tensor scales
    └── produce FP8 values for attention kernel
```

高性能实现会尽量把 unpack/dequant 与 attention 的 tile load 融合，避免先生成一份完整 FP8 临时 KV。若实现先反量化整个 cache 到额外 buffer，容量节省仍可能存在，但带宽和 workspace 成本会显著改变。

所以“框架接受 `nvfp4` 参数”只证明配置入口存在，不能证明当前模型、head dimension 与 GPU 上走了最优 fused kernel。启动日志、backend 选择和 profiler trace 都需要检查。

## K 与 V 的量化误差进入 attention 的位置不同

attention 可写为：

\[
A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}+M\right),
\qquad O=AV
\]

量化后使用 \(K+\Delta K\) 与 \(V+\Delta V\)：

- \(\Delta K\) 先影响 attention logits，再经过 softmax；当两个候选位置分数接近时，小误差可能改变注意力分配；
- \(\Delta V\) 在既定权重下进入加权和，误差会被 attention weights 组合；
- 多层误差继续通过 residual stream 传递，最终可能改变 token logits。

因此只测 KV 张量的平均 MSE 不足以代表生成质量。相同 MSE 若集中在关键 head、长距离检索位置或极少数 outlier 上，影响可能完全不同。

RoPE 也要纳入理解。实现通常缓存已经应用位置变换的 K；位置越长，数值分布和 attention pattern 可能变化。校准集只有短上下文时，不能保证 64K/128K 位置上的 scale 与误差仍具有代表性。

## Blackwell 支持不是一个单一布尔值

NVFP4 是为 NVIDIA Blackwell 引入的格式，但 Blackwell 包含不同 compute capability，kernel 支持也分 attention backend、head dimension、模型架构和精度组合。

截至 2026-08-09：

- TensorRT-LLM 当前硬件矩阵把 NVFP4 KV Cache 标为支持 Blackwell SM100/SM103，未把 SM120 标为支持；
- vLLM 当前源码已包含 `nvfp4` KV cache dtype 和专门的 page-size 路径；
- vLLM 的实际执行仍依赖 FlashInfer/TRT-LLM attention kernel 与 GPU 架构，SM120 等路径有单独的支持缺口和演进记录。

因此不能只检查 `torch.cuda.get_device_capability()` 属于 Blackwell，就假设任何 NVFP4 checkpoint 都可运行。兼容矩阵至少包含：

```text
GPU compute capability
CUDA / driver
framework revision
attention backend revision
model architecture
head dimension / GQA or MLA
prefill and decode kernel
prefix caching / chunked prefill
```

不支持的组合应在服务初始化时 fail fast，而不是启动成功后在第一个请求上崩溃，更不能静默回退到 FP8 后继续宣称使用 NVFP4。

## TensorRT-LLM 的交付路径包含离线量化

当前 TensorRT-LLM 文档要求先用 NVIDIA Model Optimizer 生成带 NVFP4 KV 配置的 checkpoint，再在运行时选择 `nvfp4` cache：

```bash
# 以当前官方示例的约束为准：FP8 weights/activations + NVFP4 KV
scripts/huggingface_example.sh \
  --model <model> \
  --quant fp8 \
  --kv_cache_quant nvfp4
```

运行时概念如下：

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig

llm = LLM(
    model="/path/to/model",
    kv_cache_config=KvCacheConfig(dtype="nvfp4"),
)
```

当前文档还说明，TensorRT-LLM 在 NVFP4 KV Cache 下要求 FP8 weight/activation quantization，因此不能把这一配置随意套到其他权重精度上。模型支持矩阵也不是全绿；每次升级都应查与所用 revision 对应的 matrix。

Model Optimizer 可以执行 PTQ，也支持在需要时继续 QAT。对于 KV Cache，校准 forward 不只是决定权重量化尺度，还要覆盖运行时 K/V 的统计分布。选错 chat template、截断所有长样本或只使用单一领域，都会让 calibration 与服务流量脱节。

## vLLM 中一个参数背后仍有多项约束

当前 vLLM 的主要入口是：

```bash
vllm serve <model> \
  --kv-cache-dtype nvfp4
```

但在使用前仍应确认：

1. 当前稳定版而非某个 issue 中的开发分支确实包含所需实现；
2. attention backend 同时支持 NVFP4 cache write、prefill 和 decode；
3. GPU 架构存在对应 cubin/kernel；
4. 模型的 head dimension、GQA/MLA 或 hybrid layer 已在支持范围；
5. page-size 计算包含 block scale，调度器没有高估 token capacity；
6. prefix caching、chunked prefill、speculative decoding 和 KV transfer 组合经过测试。

只跑一个 32-token prompt 不能覆盖这些分支。长 prompt 会进入 chunked prefill，第二次相同 prompt 才会命中 prefix cache，多并发才会触发 block pressure，P/D 才会序列化并传输 cache layout。

## 权重量化与 KV 量化是两个独立轴

模型名中带 `NVFP4` 常表示权重/激活使用 NVFP4，不自动代表 KV Cache 也以 NVFP4 存储。反过来，也可能在 FP8 weight/activation checkpoint 上单独启用 NVFP4 KV。

需要分别记录：

```text
weight dtype / quantization recipe
activation dtype
KV cache storage dtype
attention compute dtype
scale granularity and calibration
```

否则实验表中的“NVFP4”无法复现。速度变化可能来自 FP4 Tensor Core GEMM，也可能来自 KV 容量，二者对 prefill/decode 的影响完全不同。

一个清晰的对照矩阵可以是：

| 组别 | Weight/Activation | KV Cache | 目的 |
| --- | --- | --- | --- |
| A | BF16 | BF16 | 高精度参考 |
| B | FP8 | FP8 | 当前低精度基线 |
| C | FP8 | NVFP4 | 隔离 KV 量化影响 |
| D | NVFP4/其他 | NVFP4 | 评估完整低精度栈 |

并非所有框架/模型都支持四组，但应尽量避免同时改变权重与 KV 后把差异全部归因于 cache。

## GQA、MLA 与滑动窗口会改变收益基线

### GQA

Grouped-Query Attention 让多个 Q heads 共享较少的 KV heads，公式中的 \(H_{kv}\) 已经降低。NVFP4 仍按比例压缩 K/V，但 KV 在总显存中的占比可能比 MHA 小。

### MLA

Multi-head Latent Attention 通过低维 latent 和特定投影减少缓存表示。量化的是 latent cache 还是展开后的 K/V，取决于 backend 实现；不能直接套用标准 `2 * H_kv * D` 公式。

### Sliding-window attention

窗口外 token 不再为该层保留，cache 随长度达到平台。若模型只有部分 full-attention 层，NVFP4 的长期容量收益主要来自这些层。

### Hybrid attention / state-space layers

Mamba、Gated DeltaNet 等层维护的状态结构不同，不属于标准 KV page。框架可能需要让不同 cache groups 使用不同 dtype 或对齐到统一 page size。全局设置 `nvfp4` 时，必须确认哪些层实际量化、哪些层保持原精度。

优化可以叠加，但边际收益会变化。GQA/MLA 已经把 KV 压得很小后，scale、反量化和其他模型层在端到端时间中的占比会升高。

## Prefix Cache、P/D 与 Offload 要共享同一种物理契约

KV Cache 一旦离开单进程，dtype 就不仅是 attention backend 的内部实现。

### Prefix caching

前缀键通常由 token 与模型配置决定，不应因为 dtype 相同就跨不兼容实例复用。模型 revision、RoPE、KV layout、quantization scale contract 或 backend 不同，都可能让同一 token prefix 对应不同字节表示。

### Prefill/Decode 解耦

prefiller 写出的 packed FP4 payload、block scales 和 global scale，decoder 必须按完全相同的 layout 读取。传输 metadata 还要描述 dtype、shape、stride、block 版本与目标地址。只传主 payload 而漏掉 scale，会得到 shape 正确但数值错误的 cache。

### CPU/NVMe offload

更小 KV 也减少 offload/restore 字节数，但序列化格式要保留 scale 与对齐。若 offload connector 先反量化到 FP8 再写 CPU，容量与网络收益会缩水；这要从实际传输字节而不是配置名判断。

### 滚动升级

同一服务池的新旧 Pod 若 cache layout 版本不同，不应直接共享 KV。升级时可让旧连接排空、缓存自然失效，或提供显式转换；不能只依靠相同模型 ID 判断兼容。

## 量化误差该怎样测

质量验证需要从张量误差逐步走到业务输出。

### 张量层

对代表性 K/V 记录：

```text
MSE / normalized MSE
max absolute error
cosine similarity
zero ratio
clipping / saturation ratio
scale distribution
error by layer / head / position
```

K 和 V 应分开统计。平均到所有层后，一个异常的长程检索 head 很容易被淹没。

### Attention 与 logits

固定同一 prompt，对比 BF16/FP8/NVFP4 的：

```text
attention score / probability divergence
layer output error
final logits KL divergence
top-1 / top-k token agreement
first divergence position in greedy decode
```

greedy 的首次分叉很敏感，但分叉后的文本不可逐 token 直接比较，因为上下文已经不同。logits 级对比应使用相同 teacher-forced prefix。

### 任务层

覆盖量化可能放大的场景：

- 长上下文 needle retrieval 和多文档问答；
- 需要精确符号的代码生成与执行测试；
- 数学/推理任务及答案一致性；
- 多轮对话中很久以前的事实回忆；
- 不同语言、结构化输出和 tool calling；
- 业务真实 prompt/response 长度分布。

对 sampling 输出，应比较成功率、任务得分和分布统计，而不是要求文本逐字相同。

## 校准数据要覆盖“缓存会看到的分布”

KV 值由模型层、token 内容、位置和上下文共同决定。一个有效校准集至少应跨越：

| 维度 | 缺失时的风险 |
| --- | --- |
| 短、中、长位置 | 长上下文 scale 不代表 |
| 多种领域与语言 | token/激活分布偏窄 |
| chat / code / reasoning | K/V outlier 模式不同 |
| prefill 与 decode | 只覆盖批量 prompt 写入 |
| 模板和特殊 token | 与线上序列不一致 |
| 不同 batch/packing | kernel 与对齐路径未覆盖 |

PTQ 后若个别层质量敏感，可以考虑 QAT、跳过特定层的 KV 量化，或回退到 FP8。全模型统一追求最低 bit width，并不是唯一可行方案。

## 性能实验要区分容量收益与带宽收益

NVFP4 可以通过两条不同路径改善性能，实验应分别设计。

### 固定 token 工作集

保持请求数、上下文和 KV budget 足够，比较 FP8 与 NVFP4。此时主要观察反量化后的 decode kernel 是否因读取更少字节而降低 TPOT：

```text
same prompts
same concurrency
no eviction in either mode
compare attention time / HBM bytes / TPOT
```

### 固定 HBM budget

让工作集逐渐超过 FP8 容量，观察 NVFP4 是否容纳更多 block、减少 eviction/recompute 或提高最大并发：

```text
same KV memory budget
increase active tokens / repeated prefixes
compare block capacity / hit rate / preemption / TTFT
```

NVIDIA 博文中“更高 hit rate 和更低 TTFT”的路径主要属于第二类：更小 cache 在相同 HBM 下保留了更多前缀。它不等价于单个 attention kernel 变快三倍。

两类实验混在一起，就无法判断收益来自 fused dequant kernel，还是因为 FP8 baseline 已开始频繁 eviction。

## 需要记录的系统指标

### 容量

```text
real bytes per KV page
payload bytes vs scale bytes
GPU blocks / tokens capacity
max concurrency at target context
eviction / preemption / recompute tokens
```

### Kernel 与带宽

```text
KV quantize/write time
unpack/dequant time
attention kernel time
HBM read/write bytes
workspace bytes
achieved memory bandwidth
```

### 请求体验

```text
TTFT P50/P95/P99
TPOT P50/P95/P99
request throughput / output tokens per second
SLO-qualified goodput
peak and steady-state HBM
```

### 正确性

```text
load-time capability checks
actual selected attention backend
no silent dtype fallback
prefix cache hit correctness
chunked prefill parity
P/D transfer parity
```

把 `nvidia-smi` 的剩余显存作为唯一容量证据并不可靠；CUDA context、graph capture、allocator fragmentation、temporary workspace 和通信 buffer 都会占用 HBM。

## 如何读 NVIDIA 公布的结果

NVIDIA 的 NVFP4 KV Cache 博文在 Qwen3-Coder-480B-A35B 上比较了 FP16、FP8 与 NVFP4，并报告代码、知识和 RULER 64K 等基准上的差异小于约 1%；同一文章也展示了相对 FP8 的缓存容量、命中率和 TTFT 改善。

这些结果回答的是“该实现和模型在对应测试中能否保持质量并改善缓存效率”，不是所有模型的误差上界。复现时至少对齐：

- checkpoint 与 weight/activation quantization；
- Blackwell 型号与互联；
- Model Optimizer、TensorRT-LLM/其他 backend 版本；
- KV memory budget 和并行度；
- prefix 重复率、上下文长度与并发；
- benchmark 版本、采样和评分方法。

博文还比较了 Llama 3.3 70B 上的 NVFP4 与 MXFP4，结果支持细粒度 E4M3 scale 的精度优势；这仍是特定模型的实验，不应转写为任意任务固定高出 5 个点。

## 从异常现象定位到具体层

| 现象 | 优先检查 | 可能原因 |
| --- | --- | --- |
| 配置成功但显存几乎不变 | actual dtype、page size、fallback log | backend 未走 NVFP4 或 scale/workspace 被忽略 |
| 启动成功、首请求崩溃 | compute capability、kernel cubin、head dimension | capability check 太晚或架构不支持 |
| 短上下文正常、长上下文退化 | error by position、RoPE、clipping | 校准长度不足或长位置 outlier |
| 只有代码/数学明显变差 | logits divergence、关键 head、任务分桶 | 少量数值误差改变精确 token 选择 |
| TPOT 没改善但容量翻升 | dequant time、attention 占比、MLP/collective | 负载不是 KV-bandwidth-bound |
| 高并发才加速 | eviction、preemption、cache hit | 收益来自容量而非单步 kernel |
| prefix hit 后输出异常 | scale metadata、block reuse、revision | cache key 命中但物理布局不兼容 |
| P/D 单机正常、跨节点错误 | transfer bytes、dtype metadata、scale buffer | 只传 payload 或 stride/layout 不一致 |
| OOM 早于容量估算 | auxiliary scales、padding、graphs、workspace | 用裸 4 bit 低估真实 page 成本 |
| 量化/反量化占比过高 | fused path、tile shape、batch/context | kernel 没有融合或 shape 不适合 |

这种定位方式比直接“换一个校准集再试”更有效，因为数值、内核与缓存管理问题会表现出不同的故障边界。

## 一条完整的上线验证路径

可以把部署分成六个逐步收紧的关卡：

1. **格式关**：确认 checkpoint 中记录 scale contract，离线 quant/dequant 的张量误差合理；
2. **兼容关**：目标 GPU、driver、框架、backend、模型结构与 head dimension 明确受支持；
3. **功能关**：短/长 prompt、chunked prefill、prefix cache、sampling、EOS 都正确；
4. **质量关**：代表性长上下文与业务任务达到预设回归阈值；
5. **性能关**：分别证明固定工作集的带宽收益和固定 HBM 的容量收益；
6. **系统关**：P/D、offload、滚动升级、故障恢复和监控能识别 dtype/layout。

每一关都保留 FP8 或 BF16 fallback。低精度路径出现 backend 缺口时，显式回退并标注指标，比静默改变 dtype 更安全。

## 回到最初的问题：NVFP4 实际优化了什么

NVFP4 KV Cache 的本质，是用 16-value 微块、E4M3 scale 和全局 FP32 scale，把 K/V 的存储精度降到接近 4.5 bits/value 的量级。它首先改变可驻留状态的规模，其次在合适 fused attention kernel 上减少 decode 的 HBM 流量。

它没有消除 attention，也没有让 MLP、collective 与调度自动变快。模型结构已经通过 GQA、MLA 或 sliding window 压缩 KV 时，边际收益会减小；校准不匹配或 backend 不完整时，数值与系统风险会增大。

因此最可靠的判断不是“Blackwell 支持 FP4，所以应该开启”，而是：当前模型的 KV 是否真的限制容量或带宽，所用内核是否完整支持写入与读取，业务质量是否通过长上下文验证。三者同时成立，4-bit cache 才会从格式优势变成可用的推理收益。

## 参考资料

- [NVIDIA：Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
- [NVIDIA：Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [TensorRT-LLM Quantization 与 NVFP4 KV Cache](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)
- [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- [vLLM KV Cache Interface](https://docs.vllm.ai/en/latest/api/vllm/v1/kv_cache_interface/)
- [vLLM NVFP4 KV Cache 支持记录](https://github.com/vllm-project/vllm/issues/32220)
- [vLLM CacheConfig 源码](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py)
