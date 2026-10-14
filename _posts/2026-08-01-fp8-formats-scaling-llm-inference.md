---
layout: post
title: "FP8 推理：E4M3、E5M2、Scale 与 Tensor Core 到底怎样配合"
subtitle: "分清权重存储、Activation 量化、GEMM、KV Cache 与通信的五条低精度路径"
date: 2026-08-01 09:00:00 +0800
last_modified_at: 2026-10-15
author: iStar
catalog: true
series: model-quantization
series_order: 30
technology_year: 2022
mathjax: true
tags: [模型量化, GPU优化, LLM推理]
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

若原定的 FP8 GEMM 实际回退为 BF16 GEMM，或者在热路径重复转换，可能只剩容量收益、没有预期计算收益；过粗的 scale 还可能导致质量回退。这与主动让累加、输出或敏感算子保留较高精度不同，后者本来就是混合精度计算的一部分。

格式部分采用原始 FP8 论文中的 E4M3、E5M2 定义；其中 E4M3 对应常见类型名 `E4M3FN`。软件 recipe 则以 Transformer Engine **v2.3** 的固定实现为例，后文的默认值与支持条件都限定在这个版本，不代表所有低精度库的统一约定。

## 8 个 Bit 怎样分配

常见 FP8 有两个编码：E4M3 和 E5M2。名称中的 `E` 是 exponent bits，`M` 是 fraction bits；另有一个 sign bit。正常数有效数的隐含前导 1 不包含在这三位或两位 fraction 中。

```text
E4M3: sign 1 + exponent 4 + mantissa 3 = 8 bits
E5M2: sign 1 + exponent 5 + mantissa 2 = 8 bits
```

两种格式的关键数值如下：

| 格式 | Exponent bias | 最大有限绝对值 | 最小正常数 | 最小正 subnormal | 特殊值 |
| --- | ---: | ---: | --- | --- | --- |
| E4M3FN | 7 | 448 | $$2^{-6}$$ | $$2^{-9}$$ | 正负零、NaN，没有 Inf |
| E5M2 | 15 | 57,344 | $$2^{-14}$$ | $$2^{-16}$$ | 正负零、NaN、正负 Inf |

表中的 subnormal 是非正规数：它们不使用正常数的隐含前导 1，因而能继续表示更接近零的值，但相对精度逐步下降。以 exponent 字段 $$e$$、fraction 整数 $$f$$、fraction 位数 $$m$$ 和 exponent bias $$b$$ 表示，正常有限数可写成 $$(-1)^S2^{e-b}(1+f/2^m)$$；exponent 为零时改用 $$(-1)^S2^{1-b}(f/2^m)$$。因此，E4M3FN 最小正 subnormal 恰为 $$2^{-6}/8=2^{-9}$$。[FP8 原论文 §3、表 1](https://arxiv.org/html/2209.05433v2#S3)

格式后缀不能省略。`E4M3FNUZ` 的 bias 为 8，没有负零，与这里的 `E4M3FN` 并非相同的字节解释；E5M2 也有 FNUZ 变体。即使指数位、fraction 位的数量相同，也不能只改 dtype 标签就完成转换。[ONNX float8 定义](https://onnx.ai/onnx/technical/float8.html#e4m3fnuz-and-e5m2fnuz)

BF16 有 8-bit exponent、7-bit mantissa；FP16 有 5-bit exponent、10-bit mantissa。FP8 同时减少范围或有效精度，不能在没有 scale 的情况下覆盖任意 Transformer tensor。

E4M3 的 fraction 更多，E5M2 的指数范围更宽，对应不同张量的需求。Transformer Engine v2.3 的 `DelayedScaling` 与 `Float8CurrentScaling` 默认使用 hybrid：forward 的相关 FP8 张量采用 E4M3，backward 采用 E5M2。但不能把这个默认值扩大成所有 recipe：同版本的 `MXFP8BlockScaling` 默认使用 E4M3。[固定版本 recipe 定义](https://github.com/NVIDIA/TransformerEngine/blob/5de3e1481cc1f5d6f7da6ac8938870b791b796db/transformer_engine/common/recipe/__init__.py)

纯推理没有 backward，E5M2 并不会因为“更安全”就自动更好；通常要根据 tensor 分布和硬件 recipe 选择。

## FP8 与 INT8 的根本差异

对称 INT8 的整数 levels 等间距：

```text
..., -2s, -s, 0, s, 2s, ...
```

FP8 有指数，绝对值越大，相邻可表示值的间距也越大。它能在一个编码中覆盖多个数量级，不必给整个范围使用相同绝对间距，但较少的 fraction 位仍会带来明显 rounding error。

两者都通常需要外部 scale：

```text
INT8 code + scale
FP8 code + scale
```

“FP8 本身有指数”不等于可以省略 scale。外部 scale 负责把当前 tensor 的有效范围移动到选定的 FP8 区间，内部 exponent 再表达 tensor 内部的数量级差异。关于整数等距格点与舍入，可以对照 [量化数值基础]({% post_url 2026-10-15-quantization-numerical-foundations %})。

## Scale 怎样把 Tensor 塞进 E4M3

设原 tensor 为 $$X$$，目标格式的最大有限绝对值为 $$F_{max}$$，选择 scale $$s>0$$。一种一致写法是：

$$
X_{fp8}=\operatorname{cast}_{fp8}
\left(\operatorname{clip}\left(\frac{X}{s},-F_{max},F_{max}\right)\right)
$$

反量化近似为：

$$
\hat X=sX_{fp8}
$$

若使用 E4M3，最大表示值记为 $$F_{max}=448$$，基于当前绝对最大值：

$$
a_{max}=\max\lvert X\rvert
$$

最直接的 scale 为：

$$
s=\frac{a_{max}}{F_{max}}
$$

这个式子要求输入有限且 $$a_{max}>0$$，让最大值刚好落在格式边界。全零 tensor/block 应显式返回零 codes 并使用合法正 scale（例如 1），不能令 $$s=0$$ 后再计算 $$X/s$$；NaN/Inf 则需按 recipe 报错或处理。实际 recipe 还可能加入 margin、power-of-two rounding、历史统计、clipping 和 distributed amax reduction。本文的 $$s$$ 是反量化乘子，有些库把其倒数称为 `scale`，接入时必须核对方向。

本文的算例选择最近偶数舍入，并将有限越界值饱和到 $$\pm F_{max}$$。这是明确的转换策略，不是“E4M3 没有 Inf，所以所有 cast 一定自动饱和”。例如 CUDA Math API 12.8 的 FP8 转换就有显式 saturation 参数，调用方必须选择对应行为。[CUDA 12.8 转换接口](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html)

scale 太小会让大值溢出/饱和；太大则可能让小值进入 subnormal（非正规数）范围，丢失有效精度，甚至舍入为零。只要数值仍处于正常数范围，就不能仅凭缩放后的数值变小，断言它的相对精度变差。

## Per-tensor Scaling 的优点与问题

整张 tensor 共用一个 FP32 scale：

```text
X_fp8[n, h] + one scale
```

优点是元数据少、GEMM/通信接口简单。问题是一个 outlier 可以决定全局 scale，使较小的值触及格式的下溢边界。这里需要区分 FP8 的浮点网格与 INT8 的等距格点。

对正常数，E4M3 在同一数量级区间 $$[2^j,2^{j+1})$$ 内的相邻有限格点间距为 $$2^{j-3}$$，这里 $$j$$ 是去掉 bias 后的指数，相对分辨率主要由三位 fraction 决定。用 $$2$$ 的整数次幂缩放，且缩放前后都不触及上溢、下溢边界时，只会移动指数，不会增加有效位数。例如按最近值舍入，$$1.1$$ 直接转 E4M3 得到 $$1.125$$；先乘 $$256$$，把 $$281.6$$ 舍入成 $$288$$，再除以 $$256$$，仍然得到 $$1.125$$。把正常数从“约 1”搬到“约几百”的编码区间，并不会自动获得更多有效精度；非 $$2$$ 的幂缩放可能改变舍入落点，但也没有这种精度保证。

真正的下溢反例是主体绝对值约 $$10^{-4}$$、单个 outlier 为 $$400$$。按当前最大值取 $$s=400/448$$，主体缩放后约为 $$1.12\times10^{-4}$$。原始 FP8 论文中 E4M3 的最小正常数为 $$2^{-6}$$，最小正 subnormal 为 $$2^{-9}$$；上述数值甚至小于最近值舍入到零的半步阈值 $$2^{-10}$$，因而会量化成零。若这些小值使用独立 scale，它们就有机会重新落回正常数范围。这才是更细粒度 scale 能改善这组数值表示的原因。[FP8 论文，表 1](https://arxiv.org/html/2209.05433v2#S3)

Per-tensor 适合分布相对稳定、outlier 可控的张量，也常是硬件/库最先支持的方案。这里的 tensor 指一个具体的权重或 activation 张量，并不意味着整个模型的所有层共享一份 scale；不同层的范围差异，应由各自的统计与参数反映。

## Per-channel 与 Block Scaling

更细粒度的 scale 让每组数值只需覆盖局部分布：

```text
per-tensor:  one scale for X
per-channel: one scale for each row/column
per-block:   one scale for each fixed-size block
```

若 block $$g$$ 的绝对最大值为 $$a_g$$：

$$
s_g=\frac{a_g}{F_{max}}
$$

较细粒度可以减轻不同组的动态范围互相牵制，尤其是前面的下溢反例；是否降低具体张量的误差，仍要看选择的 scale 与舍入落点。代价包括更多 metadata、scale 加载与广播，以及 block axis、shard 边界和 kernel 布局的配合。

“block scaling”也不是唯一格式。以 Transformer Engine v2.3 的三个 recipe 为例：

| Recipe | 共享尺度的范围 | Scale 表示 | 需要区分的性质 |
| --- | --- | --- | --- |
| `Float8CurrentScaling` | 每个 tensor | 高精度 scale | 使用本轮数据计算范围 |
| `Float8BlockScaling` | 配置的 1×128 或 128×128 block | FP32 容器，默认约束为 2 的幂，可放宽 | 不是 MX 的 32 元素格式 |
| `MXFP8BlockScaling` | 指定方向上连续 32 个值 | E8M0，表示 2 的幂 | 块方向与归约轴、转置表示有关 |

普通 block FP8 与 MXFP8 的 code 位宽都可能是 8，但 scale 的编码、块大小和布局不同，不能直接互换。按行和按列分组时，同一矩阵还会产生不同量化结果；该版本在需要矩阵及其转置时，从高精度输入分别量化两份表示，而不是转置已有低精度数据后再量化一遍。[TE v2.3 recipe 定义](https://github.com/NVIDIA/TransformerEngine/blob/5de3e1481cc1f5d6f7da6ac8938870b791b796db/transformer_engine/common/recipe/__init__.py)

## Current Scaling

Current scaling 在使用 tensor 的同一步计算当前 $$a_{max}$$ 和 scale：

```text
X
  → reduce amax(X)
  → compute scale
  → cast FP8
  → GEMM/communication
```

好处是 scale 跟随当前数据，不会仅因沿用了旧范围而错过本轮极值。代价是 amax reduction 位于 cast 之前，必须读取当前 tensor 并得到统计；具体是否与其他算子融合，决定它在关键路径上增加多少开销。

分布式场景不必一律全局同步。若多个 ranks 使用统一量化尺度，应在约定的通信组内协调统计；若每个 shard 带着自己的 scale 交给消费者，也可以保持局部尺度。数据由谁重建、scale 如何传递，必须先说清楚。

## Delayed Scaling

Delayed scaling 使用历史 amax 估计下一步 scale：

```text
step t:
  quantize with scale derived from history up to t-1
  record amax_t for a later scale update

step t+1:
  use updated scale
```

历史可以取最近值、窗口最大值或其他算法：

$$
s_{t+1}=f(a_t,a_{t-1},\ldots,a_{t-h+1})/F_{max}
$$

它让本轮 cast 不必等待本轮 amax 决定 scale，统计与更新仍然要做，并不由算法名称保证异步执行。风险是分布突然变大时，旧 scale 可能饱和；窗口最大值又可能因一个旧 outlier 长期过于保守。

具体算一遍：忽略 margin 等附加规则，历史 amax 为 1，因而旧反量化 scale 为 $$1/448$$。下一轮出现 10 时：

| 选择方式 | 本轮使用的 scale | 10 缩放后的值 | 饱和并重建后的值 |
| --- | --- | ---: | ---: |
| 沿用历史范围 | $$1/448$$ | 4480，截为 448 | 1 |
| 使用本轮范围 | $$10/448$$ | 448 | 10 |

这是旧范围失配的构造例子，不说明 delayed scaling 总比 current 差。范围稳定时，历史统计可以很好地工作；真正需要警惕的是分布突变，以及大值离开以后，历史窗口还让小值长期使用过大的反量化 scale。

TE v2.3 的 `DelayedScaling` 支持历史最大值、最近值或自定义算法；其 `reduce_amax` 默认在选定的 FP8 group 内同步，也允许关闭并保存各 rank 的局部状态。历史窗口、同步组与 checkpoint 中的 scale 状态，都属于 recipe 的一部分。[固定版本 DelayedScaling](https://github.com/NVIDIA/TransformerEngine/blob/5de3e1481cc1f5d6f7da6ac8938870b791b796db/transformer_engine/common/recipe/__init__.py)

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

先限定为每个 operand 使用一个标量 scale；使用 FP8 输入时：

$$
X\approx s_xX_8,
\qquad
W\approx s_wW_8
$$

于是：

$$
Y\approx s_xs_w(X_8W_8)
$$

若 $$X$$ 为 $$M\times K$$、$$W$$ 为 $$K\times N$$，per-row activation 与 per-output-channel weight scale 也可在点积外应用：

$$
Y_{mn}\approx s_{x,m}s_{w,n}\sum_k(X_8)_{mk}(W_8)_{kn}
$$

但 scale 若沿 reduction 维 $$K$$ 分块变化，就必须对相应分组的部分和分别缩放，不能从整个点积中提取一个 $$s_xs_w$$；MXFP8/block-scaled GEMM 的 kernel 契约正包含这层含义。

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

Transformer Engine 的 `fp8_autocast` 控制相关 TE 模块的低精度行为，而不是把 context 内任意 PyTorch 运算都自动转换为 FP8。生产 PTQ 同样需要 per-module policy。[TE v2.3 FP8 primer](https://github.com/NVIDIA/TransformerEngine/blob/5de3e1481cc1f5d6f7da6ac8938870b791b796db/docs/examples/fp8_primer.ipynb)

## Weight-only FP8 与 W8A8 FP8

### Weight-only FP8

权重以 FP8 存储，activation 仍 BF16。Kernel 可能读取 FP8 后转换，或者使用混合输入能力。主要收益是 weight memory/bandwidth，计算路径取决于硬件。

### FP8 W8A8

权重和 activation 都以 FP8 进入 GEMM，更可能使用完整 FP8 Tensor Core 吞吐，并同时降低两侧 operand traffic。代价是 activation scale/calibration 与更高数值风险。

### Dynamic FP8

权重静态 FP8，activation 每批在线量化。它在适应数据与 amax/cast 开销之间权衡。

同一个模型仓库的 “FP8” 标签可能对应其中任一种。Benchmark 前要从 runtime 日志/kernel trace 确认真实路径。

## FP8 KV Cache 是另一套误差传播

KV Cache 保存每层历史 keys/values，被未来每个 Decode step 反复读取。FP8 可以把 BF16 KV 的数值 payload 近似减半，真实缓存容量和读取流量还要加上 scale、对齐及具体实现的成本；它与 FP8 linear weights 不同：

- KV 是请求动态产生的；
- scale 可以在线生成，也可以由离线 calibration 固定；必须与 KV 量化、存储和读取时的粒度/版本一致；
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

如果网络带宽主导，减少 payload 有机会改善时间；小消息 Decode 中，quantization kernel 和 scale synchronization 则可能抵消节省。一套系统也可以为 dispatch 与 combine 选择不同精度，例如前者低位传输、后者保留 BF16；两条方向的误差传播和带宽目标不必相同，仍需对应实现与实测支持。

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

硬件支持不是一个能概括全部 recipe 的布尔值。以 TE v2.3 的支持检查为例：普通 FP8 路径允许满足条件的 Ada 及更高计算能力设备；Ada 分支还要求 CUDA 至少 12.1、cuBLASLt 至少 12.1.3。MXFP8 检查要求 compute capability 至少 10.0；该版本的 generic block-scaled GEMM 检查则针对 Hopper、CUDA 至少 12.9。这些是固定软件路径的条件，不是格式本身只能在哪种机器上被解码的数学限制。[TE v2.3 支持检查](https://github.com/NVIDIA/TransformerEngine/blob/5de3e1481cc1f5d6f7da6ac8938870b791b796db/transformer_engine/pytorch/fp8.py)

同版本默认 recipe 也按设备分支选择：compute capability 至少 10.0 时选择 MXFP8，否则选择 delayed scaling。可见“这个库默认 hybrid”或“设备支持 FP8，所以所有 FP8 recipe 都可用”都会丢掉必要条件。

更老的设备或不匹配的算子路径，即使框架能保存 FP8 tensor，也可能：

- 软件模拟；
- 先转换 BF16/FP16 再算；
- 缺少目标 shape kernel；
- 只获得容量收益；
- 运行得比 BF16 更慢。

要从 profiler/kernel 名称、instruction throughput 与硬件文档确认，不要从 checkpoint dtype 推断。

Shape alignment 也重要。TE v2.3 primer 对相关 FP8 Linear 的两个矩阵维度提出可被 16 整除的要求；其他库、kernel 和硬件限制可能不同。Padding 能满足 alignment，却增加无效计算，不能把这个数字当作所有 FP8 指令的统一要求。[固定版本 primer](https://github.com/NVIDIA/TransformerEngine/blob/5de3e1481cc1f5d6f7da6ac8938870b791b796db/docs/examples/fp8_primer.ipynb)

## PTQ、QAT 与原生 FP8 Training

### PTQ

从 BF16/FP16 checkpoint 收集 calibration statistics，生成 FP8 weights/scales。成本低，质量取决于模型与 granularity。

### QAT

训练/微调时模拟 FP8 quantization，让模型适应误差。成本高，但可能改善敏感模型。

### FP8 Training

从训练过程就使用低精度 recipe 管理 forward/backward formats、amax 和 scales。优化器状态、master weights、非 GEMM 输出并不因此全部变成 FP8。原始 FP8 格式论文的训练实验还采用了模拟 FP8：转换到低位可表示值后，算术在 FP16/BF16 等较高精度中完成，不能把这些结果解读为特定 FP8 硬件指令的跑分。[原论文 §4](https://arxiv.org/html/2209.05433v2#S4)

最终 checkpoint 的 master weights、保存格式与推理 artifact 仍需明确，不是训练使用 FP8 就自动产出任意 runtime 可加载的 FP8 inference checkpoint。

三者必须记录来源。相同 E4M3 codes，在不同 scale/granularity 下代表完全不同数值。

## 枚举字节来核对格式与缩放

8-bit 格式只有 256 种 bit pattern，可以直接枚举，而不需要下载模型。下面的标准库程序实现 E4M3FN、E5M2 解码，以及有限输入的最近偶数舍入和饱和。它将正负零分别保留，非有限输入的编码则主动拒绝；这是本程序的合同，不是所有 GPU cast 的统一行为。

代码逐个枚举非负有限格点，用距离选择最近值，距离相同时按最低 fraction bit 的奇偶选择。它不是高效 kernel，也不模拟任意设备的中间累加；其作用是核验前面的格式表、正常数缩放、下溢与历史范围失配。

<!-- example: fp8-format-and-scaling-reference -->
```python
from math import copysign, isfinite, isnan, isinf, isclose

FORMATS = {"E4M3FN": (4, 3, 7), "E5M2": (5, 2, 15)}


def decode_fp8(code, fmt="E4M3FN"):
    if fmt not in FORMATS or type(code) is not int or not 0 <= code <= 255:
        raise ValueError("supported format and one-byte integer required")
    exponent_bits, fraction_bits, bias = FORMATS[fmt]
    sign = -1.0 if code & 128 else 1.0
    exponent = (code >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = code & ((1 << fraction_bits) - 1)
    if fmt == "E4M3FN" and exponent == 15 and fraction == 7:
        return float("nan")
    if fmt == "E5M2" and exponent == 31:
        return copysign(float("inf"), sign) if fraction == 0 else float("nan")
    if exponent == 0:
        value = (fraction / (1 << fraction_bits)) * 2.0 ** (1 - bias)
    else:
        value = (1 + fraction / (1 << fraction_bits)) * 2.0 ** (exponent - bias)
    return copysign(value, sign)


POSITIVE = {fmt: [(code, decode_fp8(code, fmt)) for code in range(128)
                  if isfinite(decode_fp8(code, fmt))]
            for fmt in FORMATS}


def encode_finite(x, fmt="E4M3FN"):
    if fmt not in FORMATS or not isfinite(x):
        raise ValueError("supported format and finite value required")
    table = POSITIVE[fmt]
    magnitude = min(abs(x), table[-1][1])  # finite saturation
    code, _ = min(table, key=lambda pair: (abs(pair[1] - magnitude), pair[0] & 1))
    return code | (128 if copysign(1.0, x) < 0 else 0)


def quantize_scaled(values, fmt="E4M3FN", scale=None):
    values = list(values)
    if fmt not in FORMATS or not values or any(not isfinite(v) for v in values):
        raise ValueError("supported format and nonempty finite values required")
    fmax = POSITIVE[fmt][-1][1]
    if scale is None:
        amax = max(abs(v) for v in values)
        scale = amax / fmax if amax else 1.0
    if not isfinite(scale) or scale <= 0:
        raise ValueError("positive finite representable scale required")
    normalized = [min(fmax, max(-fmax, v / scale)) for v in values]
    codes = [encode_finite(v, fmt) for v in normalized]
    restored = [scale * decode_fp8(code, fmt) for code in codes]
    if any(not isfinite(v) for v in restored):
        raise ValueError("reconstruction overflow")
    return codes, restored, scale


assert decode_fp8(0x7E) == 448.0
assert decode_fp8(0x08) == 2 ** -6
assert decode_fp8(0x01) == 2 ** -9
assert decode_fp8(0x7B, "E5M2") == 57344.0
assert decode_fp8(0x01, "E5M2") == 2 ** -16
assert isnan(decode_fp8(0x7F))
assert isinf(decode_fp8(0x7C, "E5M2"))
assert copysign(1.0, decode_fp8(0x80)) == -1.0

for fmt, expected in [("E4M3FN", 254), ("E5M2", 248)]:
    finite_codes = [c for c in range(256) if isfinite(decode_fp8(c, fmt))]
    assert len(finite_codes) == expected
    assert all(encode_finite(decode_fp8(c, fmt), fmt) == c for c in finite_codes)

assert decode_fp8(encode_finite(1.1)) == 1.125
assert decode_fp8(encode_finite(1.1 * 256)) / 256 == 1.125
_, with_outlier, _ = quantize_scaled([1e-4, 400.0])
assert with_outlier[0] == 0.0
_, small_group, _ = quantize_scaled([1e-4])
assert isclose(small_group[0], 1e-4)
assert quantize_scaled([0.0, 0.0])[1:] == ([0.0, 0.0], 1.0)
assert isclose(quantize_scaled([10.0], scale=1 / 448)[1][0], 1.0)
assert isclose(quantize_scaled([10.0])[1][0], 10.0)
print({"E4M3FN_max": decode_fp8(0x7E),
       "E5M2_max": decode_fp8(0x7B, "E5M2"),
       "small_value_with_outlier": with_outlier[0]})
```

这段代码还揭示了“8-bit”不足以描述数据：同一个 `0x38`，按 E4M3FN 解码为 1，按 E5M2 解码却为 0.5；负零与 FNUZ 的处理更不能混用。真正的 artifact 必须保存明确 variant，以及 scale 的 dtype、axis、block 大小和布局。

## 用三类信号定位 FP8 问题

第一类是范围信号：逐层观察 amax、saturation、subnormal/zero rate 和 scale 分布。输出不准时，若大值大量饱和，先检查范围失配与历史更新；若主体大量变零，检查下溢和粒度；若始终在正常范围内却存在舍入误差，不能指望简单乘一个 2 的幂就增加 fraction 位数。NaN/Inf 还要沿着 cast、reduction、residual 追踪，因为不同 variant 的特殊值合同不同。

第二类是布局与生命周期：使用相同 codes 和 scales 做独立重建，再核对真实 kernel 的输入、block axis、scale 广播和 epilogue。保存与恢复 delayed history 时，要确认各 rank 的状态和同步组。仅模拟 cast 正确，并不能证明低精度矩阵与 scale 在设备上对齐。

第三类是完整执行成本：单 GEMM 的时间之外，还要纳入 amax、cast、bias/residual、输出转换、通信和 fallback。Prefill 与 Decode 分别测量，线上看 TTFT、TPOT 和满足质量/SLO 的 goodput。若 worker 加载 BF16 后在线转换，冷启动时间与转换峰值显存也属于成本。

这些信号最后仍要回到模型任务：层重建误差、logits 差异和 perplexity 可以定位变化，代码、数学、多语言、长上下文等任务则衡量是否满足具体用途。减少一个局部误差指标，不会自动保证所有生成质量改善。

## 小结

FP8 同时包含两层数轴：每个值内部的 exponent/fraction，以及 tensor 或 block 外部的 scale。前者决定有效位数和格式范围，后者决定当前数据怎样落到这些格点上。更换编码、缩放粒度或更新策略，改变的是不同部分的误差与成本，不能只看 dtype 名称。

E4M3FN 与 E5M2、普通 block FP8 与 MXFP8、current 与 delayed，都需要明确到各自的数值合同。只有再对齐乘法输入、累加、输出和实际 kernel，存储上的低位宽才有机会转化成运行时收益。

FP8 使用浮点编码处理低精度表示；[SmoothQuant]({% post_url 2026-08-02-smoothquant-w8a8-activation-outliers %}) 则处理整数 W8A8 的另一种困难：通过等价缩放，把难量化的 activation 分布迁移到更容易细粒度处理的 weights。两者的编码与执行契约不同，可以对照阅读。

## 参考资料

- [FP8 Formats for Deep Learning v2](https://arxiv.org/html/2209.05433v2)，格式、转换策略与模拟实验范围。
- [ONNX Float8 类型定义](https://onnx.ai/onnx/technical/float8.html)，FN/FNUZ、特殊值与 E8M0。
- [Transformer Engine v2.3 recipe 定义](https://github.com/NVIDIA/TransformerEngine/blob/5de3e1481cc1f5d6f7da6ac8938870b791b796db/transformer_engine/common/recipe/__init__.py)。
- [相同版本的支持判定与默认 recipe](https://github.com/NVIDIA/TransformerEngine/blob/5de3e1481cc1f5d6f7da6ac8938870b791b796db/transformer_engine/pytorch/fp8.py) 和 [FP8 primer](https://github.com/NVIDIA/TransformerEngine/blob/5de3e1481cc1f5d6f7da6ac8938870b791b796db/docs/examples/fp8_primer.ipynb)。
- [CUDA Math API 12.8 FP8 转换](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html)。
