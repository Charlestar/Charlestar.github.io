---
layout: post
title: "AWQ：为什么 4-bit 权重量化要先观察 Activation"
subtitle: "从显著通道、等价缩放到 W4A16 Kernel 与端到端 Serving 收益"
date: 2026-08-09 17:15:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 50
technology_year: 2023
mathjax: true
tags: [AI Infra, AWQ, 量化, W4A16, LLM Serving]
---

把 LLM 权重从 BF16 压到 4-bit，最直观的方法是按一组权重的最大绝对值确定 scale，再四舍五入到有限整数范围。模型体积确实缩小了，但少数对输出非常重要的权重可能被粗糙量化，误差沿几十层传播后明显损害生成质量。

AWQ（Activation-aware Weight Quantization）的关键观察是：判断权重是否重要，不能只看权重本身，还要看它乘到怎样的 activation 上。同样大小的两个权重，如果一个输入通道经常具有很大的激活幅度，它的量化误差就会对输出产生更大影响。

AWQ 使用少量 calibration data 收集 activation statistics，寻找值得保护的 input channels，再通过数学等价的缩放，让这些通道的权重更耐量化。它仍然生成规则的低比特权重，而不是让少数权重保留 FP16 的混合精度稀疏布局，因此更容易映射到高效 kernel。

## 先从普通 Weight-only Quantization 开始

考虑线性层：

$$
Y=XW
$$

权重 $W$ 以 group 为单位量化。对一组实数权重 $w$，对称 $b$-bit 量化可以写成：

$$
q_{max}=2^{b-1}-1
$$

$$
s=\frac{\max |w|}{q_{max}}
$$

$$
q=\operatorname{clip}
\left(\operatorname{round}\left(\frac{w}{s}\right),
-q_{max},q_{max}\right)
$$

推理时近似恢复：

$$
\hat w=sq
$$

误差为：

$$
\Delta w=w-\hat w
$$

输出误差则是：

$$
\Delta Y=X\Delta W
$$

最后一个式子说明，只最小化 $\lVert\Delta W\rVert$ 并不等于最小化实际输出误差。某 input channel 的 activation $X_j$ 越大，该通道权重误差对 $Y$ 的放大越明显。

## “Activation-aware”到底看什么

AWQ 用 calibration samples 运行模型，统计每层 input channels 的 activation magnitude。一个简化的重要性指标可以写为：

$$
a_j=\operatorname{mean}_{tokens}|X_{:,j}|
$$

原论文发现，保护少量与显著 activation channels 相关的权重，就能显著降低量化损失。这里的“约 1% salient weights”是论文实验观察，不是所有模型都必须使用一个固定 1% 阈值。

重要的是判断顺序：

```text
calibration tokens
  → observe input activation channels
  → identify channel importance
  → search equivalent scaling/clipping
  → quantize all weights into regular groups
```

AWQ 不是在在线推理时为每个 token 动态挑重要权重，也不是让 router 决定精度。Activation statistics 用于离线产生量化参数，部署后权重布局仍然固定。

## 为什么直接保留 1% FP16 不够理想

一种保护方式是：重要权重保持 FP16，其余 INT4。但任意散布的 mixed precision 会带来：

- 需要保存位置索引；
- kernel 同时走两条计算路径；
- 稀疏 gather/scatter 破坏连续访存；
- FP16 残差结果需要额外合并；
- 不同层的稀疏模式难以统一优化。

质量可能提高，硬件执行却不规则。AWQ 的目标是通过等价变换保护通道，同时让最终权重仍以规则 4-bit groups 存储，保留 weight-only kernel 的工程效率。

## 等价缩放怎样保护显著通道

对 input channel $j$ 选择正 scale $r_j$。令对角矩阵 $S=\operatorname{diag}(r)$，则：

$$
XW=(XS^{-1})(SW)
$$

也就是：

$$
X'_j=\frac{X_j}{r_j},
\qquad
W'_{j,:}=r_jW_{j,:}
$$

在没有量化时，这个变换严格等价。对 $W'=SW$ 做 group quantization 时，显著 input channels 对应的权重被放大，在有限整数级别中获得更高的相对分辨率；运行时再对 activation 做逆缩放，保持原线性层语义。

可以把量化后的计算写成：

$$
Y
\approx (XS^{-1})Q(SW)
$$

其中 $Q$ 表示量化再反量化近似。

scale 不能无限增大。放大一个 channel 可能抬高它所在 group 的最大值，让其他权重的量化 step 变粗。AWQ 因此根据 activation statistics 搜索缩放强度，并结合 weight clipping，在“保护显著通道”和“不要伤害同组其他权重”之间取舍。

## 一个两通道例子

假设两个 input channels：

```text
channel 0 average |activation| = 0.2
channel 1 average |activation| = 8.0
```

两通道权重量化绝对误差都约为 0.02。它们对输出的典型误差贡献约为：

```text
channel 0: 0.2 × 0.02 = 0.004
channel 1: 8.0 × 0.02 = 0.16
```

第二个通道的影响大约高 40 倍。AWQ 会倾向为 channel 1 选择更强的保护 scale，而不是因为它的原始 weight magnitude 未必最大就忽略它。

这个例子只说明机制。真实线性层包含多个 outputs、group coupling、非线性和层间传播，最终 scale 需要在 calibration forward 上搜索和验证。

## Group Size 决定精度、元数据与 Kernel Shape

4-bit 权重通常不是整个 matrix 共享一个 scale，而是每 $G$ 个权重使用一组 scale/zero point。Group 越小：

- scale 更贴合局部分布，量化误差通常更低；
- scale 元数据更多；
- dequantization/indexing 更复杂；
- kernel 的 vectorized load 和 packing 可能受影响。

Group 越大：

- metadata 少、layout 简单；
- 一个 outlier 更容易扩大整组 quantization step；
- AWQ scaling/clipping 的组内耦合更强。

官方 AWQ 示例常出现 W4、group size 128 等配置，但它不是跨模型/硬件的固定最优值。Checkpoint 中必须明确：

```text
weight bits
group size and group axis
symmetric/asymmetric scheme
scales and zero points dtype/layout
AWQ channel scales / clipping results
packing order
excluded modules
```

仅写一个 `quantization=awq` 不足以让任意 runtime 正确加载。

## Clipping 为什么有帮助

Min-max scale 被最大 outlier 决定。若极少数极端权重并不重要，却把量化范围拉得很宽，大多数权重会浪费整数 levels。

把 clipping threshold 从 $m=\max|w|$ 调到 $m'<m$：

$$
q=\operatorname{round}
\left(\frac{\operatorname{clip}(w,-m',m')}{s'}\right)
$$

会牺牲被裁剪 outliers，却给主体权重更细的 resolution。AWQ 搜索阶段结合 activation-aware objective 选择 scale/clipping，而不是只优化 weight MSE。

Clipping 结果是每层/每组量化 artifact 的一部分，不能在 serving load 时随意重算成另一套 min-max 范围。

## AWQ 是 PTQ，不需要反向传播

AWQ 属于 Post-Training Quantization。典型流程：

```text
FP16/BF16 checkpoint
  → prepare representative calibration text
  → collect layer activation statistics
  → search channel scales and clipping
  → fake/pseudo quantization evaluation
  → pack real INT4 weights + metadata
  → runtime kernel validation
  → task and serving benchmark
```

原方法不依赖 backpropagation，也不做基于 calibration set 的权重重建优化。这降低了量化成本和对特定 calibration samples 过拟合的风险，并有利于迁移到 instruction-tuned、多模态模型。

不需要训练不等于 calibration data 无关。数据应覆盖实际输入的语言、领域、长度和模态；过窄样本可能遗漏线上常见的显著通道分布。

## W4A16 的“A16”意味着什么

AWQ 常部署为 W4A16：权重以 4-bit 存储，activation 保持 FP16/BF16。它主要减少：

- 权重 HBM 容量；
- 从 HBM 读取权重的带宽；
- checkpoint/权重传输体积。

这不一定表示 Tensor Core 直接执行原生 INT4×FP16 矩阵乘。不同 kernel 可能：

1. 读取 packed INT4；
2. 按 group load scale/zero；
3. 在寄存器中 dequantize 到 FP16/BF16；
4. 与 activation 做矩阵乘并累加；
5. 融合 bias/residual 等后处理。

真正的数据类型与指令路径由 GPU 架构和 kernel 决定。模型文件小了，不代表执行一定使用最快的硬件指令。

## 权重显存能省多少

理想权重主体从 BF16 的 2 bytes 降到 INT4 的 0.5 bytes，即 4 倍压缩。但实际还包括：

$$
M_{AWQ}
=M_{packed\ INT4}
+M_{scales/zeros}
+M_{unquantized\ layers}
+M_{alignment}
+M_{runtime\ workspace}
$$

Embedding、LM head、norm、少量 sensitive modules 可能不量化；group scales 和 padding 也占空间。因此应读取加载后的真实 GPU memory，而不是直接用总参数量除以 4。

AWQ 不压缩 KV Cache。长上下文/高并发场景中，权重节省出的显存可以转给 KV，从而增加并发；但每 token KV bytes 仍由层数、KV heads、head dim 和 KV dtype 决定。

## 为什么 Decode 往往比 Prefill 更受益

Decode 每步 token 数小，GEMM 的算术强度较低，反复读取大权重，容易 memory-bandwidth-bound。4-bit weight-only 可以显著减少每步权重 bytes。

Prefill 的矩阵 $m$ 更大，同一块权重被更多 tokens 复用，GEMM 更可能 compute-bound。此时 INT4 unpack/dequant 开销、kernel 吞吐与 FP16 Tensor Core 的差异决定是否加速。

简化 roofline 判断：

$$
T\approx\max
\left(
\frac{FLOPs}{Compute\ Throughput},
\frac{Bytes}{Memory\ Bandwidth}
\right)
$$

AWQ 明显降低 `Bytes`，但若 workload 已由 `FLOPs/Throughput` 主导，或者 W4 kernel 的 compute throughput 较差，端到端不一定更快。

所以必须分开测 Prefill TTFT 和 Decode TPOT，不能只用模型加载显存推断速度。

## Packing Layout 是 Runtime 兼容性的核心

两个 checkpoint 都写着 W4A16 AWQ，packing 仍可能不同：

- 8 个 4-bit values 如何排列到 32-bit word；
- signed/unsigned code；
- zero point 是否存在；
- scales 按 output/input/group 哪个维度排列；
- GEMM tile 需要怎样 interleave；
- 权重是否预转置；
- 特定 GPU kernel 的 alignment。

Runtime 若按另一 layout 解包，最坏情况不是报错，而是输出数值完全错误。转换工具应输出 schema/version，加载时校验 tensor shapes、packing metadata 和小规模 dequant checksum。

不要把“能加载模型”当作正确性证明。

## AWQ 与 GPTQ 的边界

两者都常用于低比特 weight-only PTQ，但思路不同：

- GPTQ 以近似二阶信息和逐列/块重建降低量化误差；
- AWQ 根据 activation 识别重要通道，通过等价缩放和 clipping 保护它们，不依赖反向传播/重建。

最终都可能产出 4-bit grouped weights，但量化参数、packing 和 kernel backend 未必相同。选择不应只比较 perplexity，还要同时看：

```text
calibration/quantization time
task quality
supported model modules
runtime/kernel/hardware availability
Prefill/Decode performance
checkpoint portability
```

算法名字不能替代 backend 兼容矩阵。

## AWQ 与 SmoothQuant 的边界

SmoothQuant 的目标是 W8A8：同时量化权重和 activation。它利用等价缩放把难量化的 activation outliers 部分迁移到较易量化的 weights。

AWQ 的典型目标是 W4A16：activation 保持高精度，用 activation statistics 决定怎样保护低比特 weights。

```text
AWQ:
  activation informs weight protection
  common execution: W4A16

SmoothQuant:
  shift quantization difficulty from activation to weight
  target execution: W8A8
```

二者都使用 activation-aware 等价变换，但解决的精度组合和 kernel 路径不同，不能因为都含 scaling 就视为同一算法。

## 哪些层更敏感

实际量化配置可能对以下模块做不同处理：

- embedding / LM head；
- attention Q/K/V/O projections；
- MLP gate/up/down；
- MoE routed/shared experts；
- multimodal projector/vision encoder；
- very small matrices 或不满足 group alignment 的层。

是否跳过某层必须通过 ablation 决定。逐层 error proxy 可帮助定位，但最终生成质量是多层误差共同作用，不能只用单层 MSE 排序。

MoE 还要考虑每个 expert calibration 覆盖率：冷 expert 在小 calibration set 中可能几乎没有 token，activation statistics 不可靠。可以扩大数据、按路由覆盖率检查，或对低覆盖 experts 使用更保守策略。

## 质量验证不能只看 Perplexity

最低验证矩阵包括：

### 数值

- 逐层 output cosine/error；
- logits difference 与 top-k overlap；
- FP fake-quant 与 real packed-kernel 对齐。

### 语言模型

- perplexity；
- knowledge/reasoning/code/math；
- multilingual 与长上下文；
- instruction following 与结构化输出。

### Serving 语义

- greedy generation diff；
- sampling 分布与拒绝/安全策略；
- speculative decoding acceptance rate；
- LoRA/Adapter compatibility；
- multimodal inputs。

Fake quant 质量正常、real kernel 输出错误，通常说明 packing/dequant/kernel contract，而不是 AWQ 搜索本身。

## 性能评测要固定哪些条件

```text
model and AWQ artifact revision
bits / group size / symmetry / zero point
runtime and quantization backend
GPU architecture
kernel implementation/version
Prefill token buckets
Decode concurrency and context distribution
KV dtype/cache capacity
CUDA Graph/compile settings
```

报告：

- 实际权重与总 GPU memory；
- 可分配 KV blocks/最大并发；
- TTFT、TPOT、E2E 与 goodput；
- GEMM/kernel 分解；
- HBM bandwidth 与 SM utilization；
- 首次加载/权重 repack 时间；
- 质量指标。

如果只比较“AWQ 模型能在一张卡运行，BF16 不能”，这是容量收益，不是同硬件同 workload 的速度比较。两种价值都重要，但口径要分开。

## 为什么某些 Runtime 中 AWQ 反而更慢

常见原因：

- 当前 GPU 没有匹配形状的高效 W4A16 kernel；
- runtime 在每次 forward 做额外 repack/dequant；
- group size/layout 与 kernel 不匹配；
- Prefill compute-bound，FP16/BF16 GEMM 更快；
- 小矩阵的 scale/index overhead 比省下带宽更大；
- Graph/compile 无法覆盖量化 custom op；
- CPU scheduling 或 Attention 才是瓶颈；
- 节省显存后没有提高 batch/concurrency，容量收益没有转化为 goodput。

vLLM 文档中的硬件/量化支持会随版本更新，历史版本也曾明确提示其 AWQ 路径可能低于未量化吞吐。部署前应查询当前 backend 并在目标 GPU 上实测，不能把论文 TinyChat 数据套到另一套 runtime。

## 从 BF16 Checkpoint 到生产 AWQ Artifact

### 1. 固定源模型

记录 model/tokenizer/Adapter revision，验证 BF16 baseline。

### 2. 构造 Calibration Set

覆盖实际语言、领域、长度、结构化输出和多模态输入；记录每层/每 expert token 覆盖率。

### 3. 搜索与 Fake Quant

生成 channel scales/clipping；在未用于 calibration 的 validation set 做质量评估。

### 4. Real Packing

按目标 runtime/backend layout 写出 INT4、scales、zeros 与 manifest；不要先按一种 layout 打包再在线昂贵转换。

### 5. Kernel 对齐

小输入逐层比较 fake-dequant reference 与真实 packed kernel，覆盖非对齐 shape、Prefill/Decode、TP/EP shard。

### 6. Serving Benchmark

同流量 trace 比较 BF16/AWQ 的质量、显存、TTFT/TPOT/goodput，而不是只跑离线 tokens/s。

### 7. 发布与回滚

AWQ artifact 使用独立 revision/checksum；worker readiness 前验证 kernel/backend；保留 BF16 或已验证格式的回退路径。

## 小结

AWQ 的“Activation-aware”不是量化 activation，而是用 activation distribution 判断哪些 input channels 的权重误差更重要。它通过等价 channel scaling 和 clipping 保护显著权重，同时保持规则的低比特 grouped layout，典型执行为 W4A16。

可以抓住七点：

1. 输出误差是 $X\Delta W$，只看 weight magnitude/MSE 不足以判断重要性；
2. AWQ 离线观察 activation，部署时不做逐 token 动态精度选择；
3. 等价变换放大显著通道权重、反向缩放 activation，在量化前后平衡误差；
4. Group size、clipping、scale/zero 和 packing layout 共同定义 artifact；
5. W4A16 主要降低权重容量/带宽，不会压缩 KV Cache，也不保证原生 INT4 计算；
6. Decode 更可能受益，Prefill 是否加速取决于目标 kernel 与 roofline；
7. 质量、real packed-kernel 正确性、显存和 SLO goodput 必须一起验收。

AWQ 解释了 weight-only INT4 怎样尽量保住质量。下一步再看 FP8、W8A8 和 W4A8 时，问题会转向 activation 也进入低精度之后，outlier、scale granularity、Tensor Core format 与累加精度怎样共同决定真正的计算吞吐。

## 参考资料

- [AWQ Paper: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [MIT HAN Lab: llm-awq Official Repository](https://github.com/mit-han-lab/llm-awq)
- [AWQ Project Page](https://hanlab.mit.edu/projects/awq)
- [vLLM: Quantization](https://docs.vllm.ai/en/stable/features/quantization/)
- [NVIDIA TensorRT: Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [SmoothQuant Paper](https://arxiv.org/abs/2211.10438)
