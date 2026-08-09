---
layout: post
title: "FP8 推理：E4M3、E5M2、Scale 与 Tensor Core 到底怎样配合"
subtitle: "分清权重存储、Activation 量化、GEMM、KV Cache 与通信的五条低精度路径"
date: 2026-08-09 17:35:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 30
technology_year: 2022
mathjax: true
tags: [AI Infra, FP8, E4M3, Tensor Core, LLM Serving]
---

“这个模型是 FP8”可能表达五种不同事实：checkpoint 权重以 FP8 保存、加载时把 BF16 权重转成 FP8、线性层同时使用 FP8 权重和 activation、KV Cache 以 FP8 存储，或者跨 GPU activation 以 FP8 传输。它们节省的资源、需要的 scale、数值风险和硬件 kernel 完全不同。

FP8 真正的执行链不是把 dtype 名字从 `bf16` 改成 `float8`，而是：

```text
high-precision tensor
  → choose scale from its distribution
  → scale and cast to an FP8 encoding
  → run a supported low-precision operation
  → accumulate / output in a chosen precision
  → preserve or update scale metadata
```

任何一步退回 BF16、在热路径重复转换，或者选择了过粗的 scale，都可能让“FP8 模型”只有容量收益，没有计算收益，甚至产生质量回退。

## 8 个 Bit 怎样分配

常见 FP8 有两个编码：E4M3 和 E5M2。名称中的 `E` 是 exponent bits，`M` 是 mantissa/fraction bits；另有一个 sign bit。

```text
E4M3: sign 1 + exponent 4 + mantissa 3 = 8 bits
E5M2: sign 1 + exponent 5 + mantissa 2 = 8 bits
```

NVIDIA Transformer Engine 当前文档给出的范围是：

| 格式 | 最大有限绝对值 | 特点 |
| --- | ---: | --- |
| E4M3 | 448 | 尾数更多，相对精度更好，范围较小 |
| E5M2 | 57,344 | 指数更多，动态范围更大，相对精度更低 |

BF16 有 8-bit exponent、7-bit mantissa；FP16 有 5-bit exponent、10-bit mantissa。FP8 同时减少范围或有效精度，不能在没有 scale 的情况下覆盖任意 Transformer tensor。

原始 FP8 论文提出 E4M3/E5M2 两种格式，是因为深度学习不同张量的需求不同：forward 权重/activation 更在意精度，gradient 更需要动态范围。Transformer Engine 的 hybrid training recipe 因此默认 forward 使用 E4M3，backward gradient 使用 E5M2。

纯推理没有 backward，E5M2 并不会因为“更安全”就自动更好；通常要根据 tensor 分布和硬件 recipe 选择。

## FP8 与 INT8 的根本差异

对称 INT8 的整数 levels 等间距：

```text
..., -2s, -s, 0, s, 2s, ...
```

FP8 有指数，绝对值越大，相邻可表示值的间距也越大。它能在一个编码中覆盖多个数量级，对 LLM 中分布跨度较大的权重/activation 更友好，但小 mantissa 仍会带来明显 rounding error。

两者都通常需要外部 scale：

```text
INT8 code + scale
FP8 code + scale
```

“FP8 本身有指数”不等于可以省略 scale。外部 scale 负责把当前 tensor 的有效范围移动到 FP8 最合适的编码区间，内部 exponent 再表达 tensor 内部的数量级差异。

## Scale 怎样把 Tensor 塞进 E4M3

设原 tensor 为 $X$，选择 scale $s>0$。一种一致写法是：

$$
X_{fp8}=\operatorname{cast}_{fp8}
\left(\operatorname{clip}\left(\frac{X}{s}\right)ight)
$$

反量化近似为：

$$
\hat X=sX_{fp8}
$$

若使用 E4M3，最大表示值记为 $F_{max}=448$，基于当前绝对最大值：

$$
a_{max}=\max |X|
$$

最直接的 scale 为：

$$
s=\frac{a_{max}}{F_{max}}
$$

这让最大值刚好落在格式边界。实际 recipe 还可能加入 margin、power-of-two rounding、历史统计、clipping 和 distributed amax reduction。

scale 太小会让大值溢出/饱和，太大则让多数小值落入稀疏的低精度区间甚至下溢为零。

## Per-tensor Scaling 的优点与问题

整张 tensor 共用一个 FP32 scale：

```text
X_fp8[n, h] + one scale
```

优点是元数据少、GEMM/通信接口简单。问题是一个 outlier 可以决定全局 scale，使其他值损失分辨率。

设 tensor 的主体绝对值约 1，单个 outlier 为 400。scale 为 $400/448$ 后，主体被压到 FP8 编码约 1 的区域；若没有 outlier，主体本可使用大得多的编码范围。

Per-tensor 适合分布相对稳定、outlier 可控的张量，也常是硬件/库最先支持的方案。模型越大、层间差异越明显，就越需要逐层独立 scale，而不是整个模型共用一份。

## Per-channel 与 Block Scaling

更细粒度的 scale 让每组数值只需覆盖局部分布：

```text
per-tensor:  one scale for X
per-channel: one scale for each row/column
per-block:   one scale for each fixed-size block
```

若 block $g$ 的绝对最大值为 $a_g$：

$$
s_g=\frac{a_g}{F_{max}}
$$

精度通常随 granularity 变细而提高，代价是：

- 更多 scale metadata；
- scale load/broadcast/indexing；
- tensor layout 与 block axis 必须匹配 kernel；
- distributed shard 边界更复杂；
- checkpoint 兼容性更严格。

Transformer Engine 当前支持 current/delayed per-tensor scaling，也支持 FP8 block scaling；Blackwell 的 MXFP8 则为连续 32 个值提供 block scale，并可更广泛使用 E4M3。普通 FP8、block-scaled FP8 与 MXFP8 不能只看“都是 8 bit”就混用。

## Current Scaling

Current scaling 在使用 tensor 的同一步计算当前 $a_{max}$ 和 scale：

```text
X
  → reduce amax(X)
  → compute scale
  → cast FP8
  → GEMM/communication
```

好处是 scale 跟随当前数据，突发分布变化不易使用过期范围。代价是 amax reduction 位于关键路径，distributed 场景还可能需要跨 ranks 同步，使所有 shards 使用一致 scale。

Transformer Engine 将它称为最简单的 FP8 recipe，但“概念简单”不等于运行成本为零。

## Delayed Scaling

Delayed scaling 使用历史 amax 估计下一步 scale：

```text
step t:
  quantize with scale derived from history up to t-1
  collect amax_t asynchronously/for next update

step t+1:
  use updated scale
```

历史可以取最近值、窗口最大值或其他算法：

$$
s_{t+1}=f(a_t,a_{t-1},\ldots,a_{t-h+1})/F_{max}
$$

它把部分 amax/scale 工作移出当前关键路径，更容易做高吞吐训练。风险是分布突然变大时，旧 scale 可能饱和；窗口最大值又可能因一个旧 outlier 长期过于保守。

在线推理输入分布可因租户、模态和 prompt 改变，不能无条件把训练时稳定的 delayed recipe 套到所有 activation。

## Weight Scale 与 Activation Scale 生命周期不同

模型权重固定。PTQ 后的 weight scale 可以离线计算并随 checkpoint 发布：

```text
W_fp8 + scale_W
```

Activation 每个请求/批次变化。选择有三类：

1. 静态 calibration scale：部署前从代表数据估计；
2. 在线 current scale：每批计算 amax；
3. delayed/history scale：运行中维护统计。

静态 scale 没有在线 reduction 开销，却依赖 calibration 覆盖；动态 scale 更适应流量，但带来 kernel、同步和 CUDA Graph 状态管理。

所以“checkpoint 已经是 FP8”只证明 weight storage/scale 存在，不能证明 activation 也以 FP8 进入 Tensor Core。

## 一次 FP8 GEMM 的完整精度契约

设：

$$
Y=XW
$$

使用 FP8 输入时：

$$
X\approx s_xX_8,
\qquad
W\approx s_wW_8
$$

于是：

$$
Y\approx s_xs_w(X_8W_8)
$$

需要明确：

```text
X format and scale granularity
W format and scale granularity
Tensor Core input instruction format
accumulator precision
output dtype
output scale / requantization if next op is FP8
bias/residual/activation fusion precision
```

FP8 multiply 通常使用更高精度累加，输出也可为 BF16/FP16/FP32。若下一层仍需 FP8，可能在 epilogue 中重新计算/应用 output scale，避免先完整写 BF16 再另起 cast kernel。

仅说“FP8 accumulate”或“FP8 output”容易误导；要查看具体库和硬件支持。

## 哪些算子适合 FP8

最主要收益来自大矩阵乘：

- Attention Q/K/V/O projections；
- MLP gate/up/down；
- MoE expert grouped GEMM；
- multimodal encoder linear layers。

以下算子常保留较高精度，或只在经过验证的 fused kernel 内使用低精度：

- normalization statistics；
- softmax/reduction；
- residual accumulation；
- logits 与部分 sampling；
- 小 shape、不满足对齐的 linear；
- 对 outlier 极敏感的模块。

Transformer Engine 的 autocast 只让被认定为 FP8-safe 的模块进入低精度，而不是把 context 内所有运算盲目 cast。生产 PTQ 同样需要 per-module policy。

## Weight-only FP8 与 W8A8 FP8

### Weight-only FP8

权重以 FP8 存储，activation 仍 BF16。Kernel 可能读取 FP8 后转换，或者使用混合输入能力。主要收益是 weight memory/bandwidth，计算路径取决于硬件。

### FP8 W8A8

权重和 activation 都以 FP8 进入 GEMM，更可能使用完整 FP8 Tensor Core 吞吐，并同时降低两侧 operand traffic。代价是 activation scale/calibration 与更高数值风险。

### Dynamic FP8

权重静态 FP8，activation 每批在线量化。它在适应数据与 amax/cast 开销之间权衡。

同一个模型仓库的 “FP8” 标签可能对应其中任一种。Benchmark 前要从 runtime 日志/kernel trace 确认真实路径。

## FP8 KV Cache 是另一套误差传播

KV Cache 保存每层历史 keys/values，被未来每个 Decode step 反复读取。FP8 KV 可以近似减半 BF16 KV 容量和带宽，但与 FP8 linear weights 不同：

- KV 是请求动态产生的；
- scale 必须在线生成或按预定义粒度计算；
- 同一 block 被多次使用；
- 长上下文误差可能持续影响 Attention；
- block/paged layout 决定 scale metadata 放置。

一个 runtime 支持 FP8 GEMM 不代表支持 FP8 KV kernel，反之亦然。应独立验证 attention quality、KV bytes/token、dequant cost 与长上下文任务。

博客已有 NVFP4 KV Cache 专文；普通 E4M3 KV、MXFP8/NVFP4 也不是可直接互换的格式。

## FP8 通信也独立于计算

MoE dispatch、TP/CP collective 或 activation transfer 可以将 BF16 payload 转为 FP8，网络 bytes 近似减半：

$$
V_{fp8}\approx\frac{1}{2}V_{bf16}+V_{scale}
$$

但路径增加：

```text
amax/scale
  → cast/pack
  → communication
  → scale-aware consumer/dequant
```

如果网络主导，收益明显；小消息 Decode 中，quantization kernel 和 scale synchronization 可能抵消带宽节省。DeepEP 等库支持 FP8 dispatch，却可使用 BF16 combine，正是因为两个方向的精度/带宽目标可以不同。

## Prefill 与 Decode 的收益不同

### Prefill

大 GEMM 更容易发挥 FP8 Tensor Core 峰值吞吐。若矩阵维度满足对齐、kernel 成熟，compute reduction 可能显著；Attention 和非 GEMM 部分仍限制端到端收益。

### Decode

小 batch 时线性层更偏 weight-bandwidth-bound，FP8 权重减少读取量；但 GEMM shape 太小时可能无法达到 Tensor Core 峰值。动态 activation amax/cast 的固定开销也更突出。

Roofline 仍适用：

$$
T\approx\max
\left(\frac{FLOPs}{P_{effective}},
\frac{Bytes}{BW_{effective}}
\right)
+T_{scale/cast}
$$

需要分别测 TTFT 与 TPOT，不能用一组大 matrix microbenchmark 代表在线 Decode。

## Hardware Support 决定是不是“真 FP8”

Hopper Tensor Cores 原生引入 FP8 计算能力，Ada 及之后部分路径也有支持；具体 recipe、shape 和库能力要看架构与软件版本。更老 GPU 即使框架能保存 FP8 tensor，也可能：

- 软件模拟；
- 先转换 BF16/FP16 再算；
- 缺少目标 shape kernel；
- 只获得容量收益；
- 运行得比 BF16 更慢。

要从 profiler/kernel 名称、instruction throughput 与硬件文档确认，不要从 checkpoint dtype 推断。

Shape alignment 也重要。Transformer Engine 文档当前对部分 FP8 Linear 要求维度可被 16 整除；其他库/硬件限制可能不同。Padding 能满足 alignment，却增加无效计算。

## PTQ、QAT 与原生 FP8 Training

### PTQ

从 BF16/FP16 checkpoint 收集 calibration statistics，生成 FP8 weights/scales。成本低，质量取决于模型与 granularity。

### QAT

训练/微调时模拟 FP8 quantization，让模型适应误差。成本高，但可能改善敏感模型。

### FP8 Training

从训练过程就使用 Transformer Engine recipe 管理 forward/backward formats、amax 和 scales。最终 checkpoint 的 master weights、保存格式与推理 artifact 仍需明确，不是训练使用 FP8 就自动产出任意 runtime 可加载的 FP8 inference checkpoint。

三者必须记录来源。相同 E4M3 codes，在不同 scale/granularity 下代表完全不同数值。

## Checkpoint Manifest 至少包含什么

```text
model revision and tokenizer
quantized modules / excluded modules
FP8 encoding: E4M3 or E5M2 variant
weight scale granularity and axis
scale dtype/layout
block size if block-scaled
calibration/recipe metadata
original/master weight policy
packing/layout version
target runtime/backend and minimum architecture
```

加载器验证 tensor shape、scale shape、block axis、NaN/Inf 和 checksum。不能把 E4M3 数据按 E5M2 解码，也不能把 per-channel scales 当 per-tensor scalar 广播。

## 数值验证看什么

### 量化统计

```text
amax distribution
saturation/clipping rate
underflow/zero rate
scale range and outliers
per-layer reconstruction error
```

### 模型输出

- hidden/logits cosine 与 error；
- greedy token divergence；
- perplexity；
- reasoning/code/math/多语言；
- long-context 与 structured output；
- MoE routing/负载变化；
- speculative decoding acceptance rate。

### 非有限值

逐层监控 NaN/Inf，尤其是 reduction、softmax input、residual 和 dynamic scale。E4M3/E5M2 对 Inf/NaN 的编码约定不同，错误处理不能照搬 FP16。

质量验证要使用真实 kernel。Fake cast/dequant 正确不能证明 scale layout、Tensor Core input 或 fused epilogue 正确。

## 性能验收拆成四层

### 单 GEMM

扫实际 $M,N,K$、dtype、scale granularity，比较 BF16/FP8 kernel latency 与数值。

### 单层

包含 amax、cast、GEMM、bias/residual/output cast，避免只测核心 matmul。

### 模型迭代

分解 Attention、MLP/MoE、通信、scale/cast；检查 CUDA Graph/compile coverage。

### Serving

```text
TTFT / TPOT / goodput
weight and KV memory
max concurrency
HBM/NVLink/NIC bytes
startup conversion/capture time
quality and fallback rate
```

若加载 BF16 后在线转 FP8，还要把转换时间与峰值 host/GPU memory 计入 worker 冷启动。

## 常见误区

### “8-bit 所以一定是 BF16 两倍快”

端到端还有 Attention、memory、scale/cast、小 shape 和调度；硬件峰值不是应用 speedup。

### “FP8 有指数，所以不需要 calibration”

外部 scale 仍决定有效范围。Per-tensor outlier 仍会浪费 precision。

### “权重 FP8 就是 W8A8”

Activation 可能仍 BF16，kernel 也可能在线 dequant。

### “模型支持 FP8，KV 和通信自然也支持”

三者有不同 kernel、layout 和 scale 生命周期。

### “E4M3 总比 E5M2 好”

E4M3 精度更高但范围更小；格式要匹配 tensor 和 scaling recipe。

### “Block 越小精度越好，所以总是更优”

Scale metadata、layout、kernel support 和 load overhead 也随之增加。

## 一条生产落地路径

1. **建立 BF16 基线**：保存质量、分层时间、显存与真实 traffic；
2. **只量化权重**：验证 checkpoint/layout 和容量收益；
3. **确认 kernel**：目标 GPU 上是否真正走 FP8 Tensor Core；
4. **加入 activation FP8**：选择 static/current/delayed/block scaling；
5. **逐层排除敏感模块**：通过消融决定，而非固定名单；
6. **分别评测 P/D**：关注大 GEMM 与小 batch 固定开销；
7. **再评估 KV/通信**：每条路径独立做质量与性能实验；
8. **集成 compile/Graph**：确认 scale buffers、amax history 和地址生命周期；
9. **发布 manifest/cache**：固定硬件、runtime、recipe 与 artifact revision；
10. **监控饱和与回退**：流量分布变化后 scale 可能失效。

每步只增加一种低精度边界，出现问题才能判断是 weight、activation、KV、通信还是 kernel integration。

## 小结

FP8 的核心不是一个 8-bit dtype，而是“编码格式 + scale granularity + scale 更新 recipe + 计算/累加/output 精度”的完整契约。

可以记住八点：

1. E4M3 用更多 mantissa 换精度，E5M2 用更多 exponent 换范围；
2. 外部 scale 仍必不可少，负责把 tensor 分布映射进 FP8；
3. Current、delayed、per-tensor、per-channel、block scaling 有不同成本；
4. Weight FP8、W8A8 GEMM、FP8 KV 与 FP8 communication 是四条独立路径；
5. GEMM 输入、accumulator、output 和 epilogue 精度都要明确；
6. Prefill 更可能利用计算吞吐，Decode 更关注权重带宽与 scale 固定开销；
7. 真正收益依赖原生硬件、匹配 shape 的 kernel 与完整 runtime coverage；
8. Saturation、underflow、质量、显存和 SLO goodput 必须一起验收。

FP8 提供了浮点低精度的硬件基础。接下来的 SmoothQuant 会处理另一条 8-bit 路径：当 activation outliers 让 INT8 W8A8 难以量化时，怎样通过等价缩放把量化难度从 activation 迁移到更容易处理的 weights。

## 参考资料

- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [NVIDIA Transformer Engine: FP8 Current Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html)
- [NVIDIA Transformer Engine: FP8 and FP4 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [NVIDIA Transformer Engine: FP8 Blockwise Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html)
- [NVIDIA TensorRT: Quantization Schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
