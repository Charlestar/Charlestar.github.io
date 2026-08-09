---
layout: post
title: "SmoothQuant：把 Activation Outlier 迁移到 Weight"
subtitle: "从等价通道缩放到 W8A8 INT8 GEMM、Q/DQ 融合与 Serving 收益"
date: 2026-08-02 09:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 40
technology_year: 2022
mathjax: true
tags: [模型量化, GPU优化, LLM推理]
---

INT8 矩阵乘已经在 GPU 和 CPU 上存在多年，为什么大语言模型直到 SmoothQuant 之后才更容易稳定地使用 W8A8？障碍通常不在 weight，而在 activation。

Transformer 的 activation 里经常存在少数幅度很大的通道。若整个 tensor 共用一个 INT8 scale，这些 outliers 会把量化范围拉得很宽，大多数普通值只能挤在少量整数格点上；若缩小 scale 来照顾普通值，outliers 又会被截断。两种选择都会累积误差。

SmoothQuant 的做法不是删除 outlier，也不是在运行时给它们单独走高精度分支，而是在量化前对线性层做一次数学等价的通道缩放：压小难量化的 activation 通道，同时放大 weight 中对应的 input channel。原始浮点计算保持不变，量化难度却从 activation 迁移到了通常更容易按 channel 处理的 weight。

完整路径可以概括为：

```text
calibration data
  → collect activation statistics
  → choose per-input-channel smoothing factors
  → fold factors into weights / preceding operator
  → quantize transformed weights to INT8 offline
  → quantize transformed activations at runtime
  → execute a real INT8 × INT8 GEMM
  → rescale, fuse epilogue, and produce the next tensor
```

只完成前四步会得到 INT8 weights，却不一定得到 W8A8 加速；只有运行时和 kernel 真正消费 INT8 activation，W8A8 的计算路径才闭合。

## INT8 量化究竟损失在哪里

先看最常用的对称均匀量化。设浮点 tensor 为 $x$，量化 scale 为 $s_q>0$，一种写法是：

$$
q=\operatorname{clip}
\left(
\operatorname{round}\left(\frac{x}{s_q}\right),
-127,127
\right)
$$

反量化近似值为：

$$
\hat{x}=s_q q
$$

有的 runtime 会使用完整的 $[-128,127]$ 编码范围，或引入 zero point 做非对称量化；这里使用对称形式便于说明。无论具体约定如何，误差主要来自两处：

- **Rounding error**：连续实数只能落到离散整数格点；
- **Clipping error**：超出表示范围的值被截到边界。

scale 大，覆盖范围宽，但相邻格点间距也大；scale 小，普通值更精细，但极值更容易饱和。对近似均匀、范围稳定的 weight，这个折中比较容易处理。对带有稳定通道 outlier 的 activation，一个 tensor 共用一个 scale 时就会很困难。

例如，一个 activation tensor 的绝大多数值落在 $[-2,2]$，少量 outlier 接近 40。若 scale 要覆盖 40，则一个整数步长约为：

$$
s_q\approx\frac{40}{127}\approx0.315
$$

大量小值会被粗粒度舍入。若按 2 来定 scale，分辨率提高约 20 倍，但 outlier 会严重饱和。SmoothQuant 要改变的正是这个分布，而不是 INT8 编码规则本身。

## 等价变换怎样迁移量化难度

考虑一个不写 bias 的线性层：

$$
Y=XW
$$

$X$ 的最后一维和 $W$ 的输入维相同。为每个 input channel 选择正数 $s_j$，组成对角矩阵 $S$：

$$
Y=XW=(XS^{-1})(SW)
$$

定义：

$$
X'=XS^{-1},\qquad W'=SW
$$

则有：

$$
X'W'=XW
$$

对第 $j$ 个 input channel 来说：

$$
X'_j=\frac{X_j}{s_j},\qquad W'_j=s_jW_j
$$

如果某个 activation channel 的幅度异常大，就为它选择更大的 $s_j$。该 activation channel 被除小，而 weight 的对应 input channel 被乘大；两者相乘时缩放抵消，所以高精度线性层输出不变。

这条等式有三个重要边界：

1. $S$ 沿 **input-channel / GEMM K 维**定义，左右两边必须使用同一组因子；
2. 变换在量化前是数学等价的，量化后误差会重新分布，但不可能凭空消失；
3. smoothing factor 与后续 INT8 quantization scale 不是同一个概念。

第三点尤其容易混淆。$s_j$ 改造 tensor 的分布，通常离线确定；$s_q$ 把改造后的 tensor 映射到 INT8，可能静态保存，也可能在线计算。

## Alpha 在 Activation 与 Weight 之间做平衡

SmoothQuant 使用 calibration statistics 估计每个 activation input channel 的最大幅度，同时观察 weight 对应通道的幅度。一种常见表达是：

$$
s_j=
\frac
{\max(|X_j|)^\alpha}
{\max(|W_j|)^{1-\alpha}}
$$

其中 $\alpha\in[0,1]$ 控制把多少量化难度迁往 weight：

- $\alpha$ 较大时，activation outlier 被压得更多，weight 通道被放得更大；
- $\alpha$ 较小时，activation 改变更温和，weight 分布也更接近原状；
- 合适的 $\alpha$ 取决于模型、模块、量化粒度和目标 kernel，不应机械地固定为一个值。

假设某通道的 activation 最大幅度为 64，weight 最大幅度为 1，且 $\alpha=0.5$：

$$
s_j=\frac{64^{0.5}}{1^{0.5}}=8
$$

activation 最大幅度从 64 降到 8，weight 最大幅度从 1 增到 8。乘积没变，两侧的范围却更均衡。

实际实现还会处理极小统计值、epsilon、clamp、不同模块的 alpha 和 scale folding。公式说明的是核心机制，不等于所有 backend 的 artifact 格式完全相同。

## 为什么 Weight 更适合接住 Outlier

把数值范围从 activation 移到 weight 并不意味着 weight 不会受伤，而是 weight 更容易获得细粒度 scale。

Activation 由请求在运行时产生。若每个 input channel 都单独动态统计和量化，会引入 reduction、scale 计算、内存读写与复杂 kernel 接口。很多执行路径因此只给 activation 使用 per-tensor 或 per-token scale。

Weight 是静态的，可以离线处理：

- 为不同 output channel 或 group 保存独立 quantization scale；
- 提前完成 layout transform 和 packing；
- 把 metadata 与 checkpoint 一起缓存；
- 对少量敏感模块选择不同 recipe。

因此，即使 smoothing 后某些 weight input channels 变大，细粒度 weight quantization 往往仍能较好地表示它们。

还要分清两条轴：SmoothQuant 的 $s_j$ 沿 input channel 重塑 $X$ 与 $W$；weight 的 INT8 scale 可能沿 output channel 或某种 group 计算。前者保证等价变换，后者决定整数编码精度，它们不能互换。

## Calibration 阶段真正收集什么

SmoothQuant 是 post-training quantization，不需要重新训练模型，但需要有代表性的 calibration data。对每个目标线性层，至少要得到 activation input channels 的范围统计，例如：

```python
# 示意代码，不绑定某个框架
amax = zeros(hidden_size)

for batch in calibration_data:
    x = capture_linear_input(batch)
    amax = maximum(amax, abs(x).amax(over_all_dims_except_hidden))
```

只拿随机 token 或极短文本做 calibration，统计出的范围未必覆盖生产分布。数据集至少要考虑：

- 真实输入语言和领域；
- 常见与长尾 prompt 长度；
- system prompt、tool schema、代码块等结构；
- 长上下文位置上的 activation 分布；
- MoE 模型中访问频率不同的 experts；
- 多模态模型的不同输入模态与 projector。

最大值对异常样本非常敏感，有些实现会比较 percentile、clipping 或其他稳健统计。但改变统计方式会改变论文 recipe，需要以质量实验验证，不能只因为 histogram 看起来更平滑就认为更好。

Calibration set 只用于选择变换和量化参数；最终质量必须在未参与 calibration 的 validation set 与任务集上测量，否则容易把参数调到样本本身。

## 离线 Smoothing 怎样消除额外算子

直接根据公式执行 $X'=XS^{-1}$，会在每个请求中增加一次逐元素乘法。它的 FLOPs 很少，却可能产生额外 kernel launch 和显存读写，抵消低精度收益。

更好的方式是尽可能把 $S^{-1}$ fold 进产生 $X$ 的前序参数，把 $S$ fold 进当前 linear weight。例如在可等价合并的 normalization / linear 边界上，离线改写相邻参数，使运行时自然产生 $X'$。当前权重则保存为已经平滑并量化后的 $W'$。

理想 graph 变为：

```text
before:
previous op → X → multiply by S^-1 → quantize → INT8 GEMM with S·W

after folding:
modified previous op → X' → quantize → INT8 GEMM with transformed W'
```

folding 必须尊重模型拓扑。一个 activation 若分叉给多个 consumers，或同时经过 residual path，不能只修改一条支路而假设仍然等价。工程实现应沿 graph 确认所有 consumers，并逐层比较 smoothing 前后的高精度输出。

验证离线变换最直接的方法是先不做 INT8：

$$
\operatorname{maxabs}(XW-X'W')<\epsilon
$$

若此时已经不一致，问题属于轴、转置、bias、folding 或 graph rewrite，而不是量化精度。

## Smoothing 之后为何仍要在线量化 Activation

SmoothQuant 把 activation range 变得容易处理，但 GEMM 输入仍然是浮点 tensor，必须经过 Quantize 才成为 INT8。运行时大致有三种策略。

### Static Per-tensor Scale

Calibration 阶段为某个 activation tensor 固定一个 scale。线上无需再做 amax reduction，开销最低，但流量分布漂移时可能饱和，也可能因范围留得太宽而损失分辨率。

### Dynamic Per-tensor Scale

每次执行时根据当前 tensor 的 amax 计算 scale。它适应请求分布，但 reduction 和 quantization 是真实成本；若不能与上游算子融合，小 batch Decode 尤其容易被固定开销主导。

### Dynamic Per-token Scale

每个 token row 使用独立 scale，可减少不同 token 范围差异造成的误差。代价是更多 scale metadata、更复杂的 GEMM 接口和额外的统计工作。是否支持取决于硬件与 runtime，不是导出 checkpoint 时写一个配置就会自动生效。

SmoothQuant 解决 channel outlier，per-token scale 解决 token 之间的范围差异，两者处理的轴不同，可以组合，也有各自成本。

## 从 Q/DQ Graph 到真实 INT8 GEMM

在显式量化 graph 中，Quantize/Dequantize 节点表达低精度边界：

```text
BF16 activation
  → Q: activation to INT8
  → INT8 GEMM with INT8 weight
  → scale using s_x and s_w
  → bias / residual / activation epilogue
  → BF16 output or requantized INT8 output
```

若 activation 和 weight 分别有 scale $s_x$ 与 $s_w$，整数点积可以写成：

$$
Y\approx(s_x q_x)(s_w q_w)
=s_xs_w\sum_k q_{x,k}q_{w,k}
$$

整数乘加通常需要更宽的 accumulator，随后再应用 scale 并转换到输出 dtype；准确的 accumulator、rounding 和 saturation 规则由 kernel contract 决定。

Q/DQ 节点只是语义，不保证性能。优化器需要把它们与 GEMM、bias、activation 或 residual epilogue 融合。如果实际执行变成：

```text
INT8 weight → dequantize to BF16 → BF16 GEMM
```

那么只有权重存储可能变小，并没有使用 INT8 Tensor Core。若又单独 launch activation quantization 和 output dequantization，端到端速度甚至可能更差。

确认计算路径时，应查看 runtime engine、kernel 名称或 profiler trace，而不是只检查 checkpoint dtype。

## W8A8 的内存收益应该怎样算

单看 dense weights：

```text
BF16: 2 bytes / parameter
INT8: 1 byte / parameter + scales and metadata
```

因此权重主体接近减半，但不能直接宣称整个服务显存减半。总显存还包括：

```text
model weights
+ KV Cache
+ runtime activations
+ CUDA Graph pools
+ temporary workspaces
+ allocator fragmentation
+ communication buffers
```

SmoothQuant 不会自动压缩 KV Cache。若服务主要受长上下文 KV 容量限制，W8A8 权重节省对最大并发的提升会小于权重占主导的短上下文场景。

Activation 是否常驻 INT8 也取决于 graph。若每层 GEMM 后马上转回 BF16，主要收益来自 GEMM 输入和 weight；若算子链能够保留低精度并融合，才可能进一步减少 activation 带宽。

## Prefill 与 Decode 的收益为什么不同

Prefill 通常有较大的 $M$ 维，GEMM 更接近 compute-bound。目标 GPU 的 INT8 Tensor Core 吞吐、shape、并行切分和融合都合适时，W8A8 更容易产生计算吞吐收益。

Decode 的 batch 较小时，矩阵乘经常更受 weight memory bandwidth 限制。INT8 权重比 BF16 小约一半，理论上有带宽优势；但在线 activation reduction、quantize、scale 读取、kernel launch 和调度延迟可能占据更大比例。

因此应分别报告：

- Prefill latency 与 TTFT；
- Decode step latency 与 TPOT；
- 不同 batch / sequence bucket 的 tokens/s；
- 在同一延迟 SLO 下的 goodput；
- quantize、GEMM、epilogue 各自的时间。

只报告一个大 batch 离线吞吐，会掩盖真实在线请求中的固定开销。

## Tensor Parallel 与分片之后还等价吗

量化 artifact 要与并行分片的轴一致。若 weight 先 SmoothQuant、再 INT8 quantize、再 tensor-parallel shard，部署端必须以同一顺序解释 smoothing factors、quantization scales 和 packed layout。

常见风险包括：

- exporter 与 runtime 对 weight 转置约定不同；
- input-channel smoothing factors 在 column/row parallel 中切错轴；
- per-channel weight scales 没随 shard 一起切分；
- padding 后的 hidden dimension 未同步处理 metadata；
- AllReduce/ReduceScatter 前后错误地重复 dequant 或 requant；
- 某些 rank 走 INT8，另一些因 shape 不支持而 fallback。

最低限度要在单卡 reference、每个 shard 的局部输出、collective 后的完整输出三个层面比较误差。

## SmoothQuant、FP8 与 AWQ 的边界

三者都在解决低精度推理，但执行契约不同：

| 方法 | 典型路径 | Activation | 核心问题 | 主要收益 |
| --- | --- | --- | --- | --- |
| SmoothQuant | W8A8 INT8 | INT8 | 把 activation outlier 迁移到 weight | 权重容量、带宽与原生 INT8 GEMM |
| FP8 | FP8 weight + FP8 activation | FP8 | 用浮点编码、scale recipe 覆盖动态范围 | FP8 Tensor Core 计算与较低带宽 |
| AWQ | W4A16 | BF16/FP16 | 用 activation 判断哪些 weights 更需保护 | 更强的权重压缩，常偏向 Decode 带宽 |

SmoothQuant 的 activation-aware 体现在重塑 activation/weight 分布，并在运行时量化 activation；AWQ 的 activation-aware 主要用于离线判断 weight importance，典型 W4A16 路径并不把 activation 量化到 INT4。

FP8 有指数，较容易容纳跨数量级数值；INT8 是等距格点，依赖 smoothing 与 scale granularity 缓解 outlier。FP8 是否优于 INT8 仍取决于硬件代际、kernel、模型质量和部署目标。

## 质量验证不能只看 Perplexity

量化误差可能只集中在少数层、token 或任务上。完整验证至少分四层。

### 1. 等价变换

关闭 quantization，比较 smoothing 前后的 layer outputs 与最终 logits。这里应接近数值误差范围。

### 2. Fake Quantization

用浮点算子模拟 quantize/dequantize，比较不同 alpha、clipping 与 scale granularity，定位敏感层。

### 3. Real Kernel

使用真正 packed INT8 weights 和目标 W8A8 kernel，逐层对比 fake-quant reference。它能发现 layout、scale broadcasting、rounding、padding 和 accumulator 实现错误。

### 4. End-to-end Tasks

除 perplexity 外，还要覆盖：

- 长文本生成与长上下文检索；
- 数学、代码、结构化输出；
- 多语言与领域数据；
- tool calling / JSON schema；
- greedy decoding 下的 token 一致率；
- sampling 下的分布与任务成功率。

同时记录 activation saturation rate、zero rate、分层误差、logit distance 和异常请求。平均指标不变，不代表长尾样本没有退化。

## 性能不升反降时怎样定位

遇到“模型已经 W8A8，延迟却没有下降”，可以按以下顺序检查：

1. **Kernel 是否真实 INT8 × INT8**：排除 weight dequant 后走 BF16 GEMM；
2. **Shape 是否命中优化实现**：hidden size、M/N/K 对齐、TP shard 都会影响选择；
3. **Q/DQ 是否融合**：单独的 reduction、quantize 和 dequantize 可能吃掉收益；
4. **Fallback 覆盖率**：统计多少层、多少请求 bucket 真正走低精度；
5. **Scale 粒度与 metadata**：过细粒度可能增加 load 和广播成本；
6. **Prefill/Decode 分开看**：两者瓶颈和最佳 kernel 不相同；
7. **端到端瓶颈**：tokenizer、scheduler、collective、KV 管理可能已经占主导；
8. **功耗与频率**：比较相同功耗策略和稳定频率下的结果。

应同时保留 layer microbenchmark 和 serving benchmark。前者说明 kernel 能力，后者说明系统收益，两者不能互相替代。

## Artifact 需要记录哪些契约

一个可复现的 SmoothQuant checkpoint 不应只有 `.int8` 文件名。Manifest 至少记录：

```yaml
format: smoothquant-w8a8
base_model_revision: <immutable revision>
calibration_dataset_revision: <revision or checksum>
smoothing:
  alpha: <global or per-module mapping>
  statistic: <amax / percentile / other>
  excluded_modules: [...]
weight_quantization:
  dtype: int8
  granularity: <per-tensor / per-channel / group>
  symmetric: true
activation_quantization:
  dtype: int8
  mode: <static / dynamic>
  granularity: <per-tensor / per-token>
packing:
  backend: <runtime and version>
  layout: <layout id>
parallelism:
  tp_size: <n>
validation:
  quality_report: <artifact>
  kernel_report: <artifact>
```

同样叫 W8A8 的两个 artifact，可能在 alpha、scale 轴、rounding、clipping、packing 与 runtime contract 上完全不同。版本化这些信息，才能安全缓存、灰度和回滚。

## 一条可执行的落地路径

1. **固定 BF16 基线**：记录模型 revision、质量、显存、TTFT、TPOT 和 goodput；
2. **选择目标 backend**：先确认硬件与 runtime 支持的 INT8 GEMM、scale 粒度和 shape；
3. **建立 calibration set**：覆盖生产语言、长度、任务和长尾结构；
4. **收集 channel statistics**：保存 per-layer activation amax 与异常分布；
5. **搜索 smoothing recipe**：比较 alpha、clipping 和敏感模块排除策略；
6. **验证高精度等价性**：在量化前确认 graph rewrite 与 scale folding 正确；
7. **导出目标 layout**：完成 INT8 quantization、packing、sharding 与 manifest；
8. **对齐 fake quant 与 real kernel**：逐层覆盖边界 shape 和并行配置；
9. **运行端到端质量集**：重点观察长上下文、代码、数学和结构化输出；
10. **运行真实 traffic benchmark**：分别看 Prefill/Decode 与各 batch bucket；
11. **用 profiler 确认覆盖率**：定位 fallback、未融合 Q/DQ 与 collective 开销；
12. **灰度并监控漂移**：跟踪 saturation、质量代理指标和 SLO，保留 BF16 回退。

这条路径先锁定 backend，是因为量化粒度和 packing 必须服务于实际 kernel。先做一个抽象 checkpoint、部署时再临时转换，往往会把本可离线完成的工作带进启动或请求热路径。

## 小结

SmoothQuant 的核心不是“把异常值消掉”，而是利用线性层的通道缩放不变性，让 activation 和 weight 的量化难度重新平衡。

可以抓住八点：

1. INT8 的等距格点很怕 activation channel outliers；
2. $XW=(XS^{-1})(SW)$ 在量化前保持计算等价；
3. smoothing factor 重塑分布，quantization scale 映射整数，两者职责不同；
4. alpha 决定量化难度向 weight 迁移的程度，需要按模型和 backend 验证；
5. 离线 scale folding 应避免在热路径新增逐元素 kernel；
6. W8A8 必须在线量化 activation 并执行真实 INT8 GEMM，checkpoint 为 INT8 不足以证明加速；
7. Prefill 看计算吞吐，Decode 还要权衡权重带宽与 Q/DQ 固定开销；
8. 等价性、fake quant、real kernel、任务质量与 SLO goodput 必须逐层验收。

W8A8 把 weight 与 activation 都压到 8-bit，比较适合原生 INT8 矩阵乘。下一篇会继续降低 weight 精度，讨论 W4A8 如何在更高压缩率下组织两级 scale、INT4 weight 解包和 INT8 Tensor Core 计算，以及为什么“4-bit 权重”与“4-bit 乘法”不是同一件事。

## 参考资料

- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
- [MIT HAN Lab: SmoothQuant Official Repository](https://github.com/mit-han-lab/smoothquant)
- [NVIDIA TensorRT: Quantization Schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
- [NVIDIA TensorRT: Quantization Workflows](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-workflows.html)
- [NVIDIA TensorRT: Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
