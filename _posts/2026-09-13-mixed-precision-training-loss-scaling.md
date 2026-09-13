---
layout: post
title: "混合精度训练：从 FP16/BF16 的数值边界到一次正确的参数更新"
subtitle: "区分计算与持久状态的精度，推导 loss scaling，并沿梯度累积、裁剪和跳步追踪训练目标"
date: 2026-09-13
last_modified_at: 2026-09-13
author: iStar
catalog: true
series: distributed-training
series_order: 47
technology_year: 2017
mathjax: true
tags: [分布式训练, GPU优化]
---

把模型计算从 FP32 换成 16-bit，并不只是用更少的字节表示同一个数。一些反向梯度可能在传播途中舍入为零；另一些更新虽然已经算出来，却小到无法改变低精度权重。前向 loss 仍然有限，训练也在正常执行，但参数更新已经偏离了原来的意图。

混合精度训练的关键，是把不同精度放在不同位置：矩阵乘的输入可以更窄，点积累积可以更宽；前向计算可以使用低精度副本，跨许多 step 积累的参数与 optimizer state 则采用另一种保存策略。Loss scaling 还会主动移动反向梯度的数值范围。

本文沿着一次参数更新解释这些选择。技术年份 2017 指 [Mixed Precision Training 预印本](https://arxiv.org/html/1710.03740v3)首次公开，该工作发表于 ICLR 2018；不是说此前不存在低精度训练。论文部分主要核对其 §3 的权重更新、loss scaling 与累积精度，框架行为固定以 PyTorch v2.5.1 为基准，不将历史论文的显存或速度结论当作现代模型的普遍保证。

## 同样占两个字节，FP16 与 BF16 的风险为什么不同

浮点数把有限的位数分配给符号、指数和 fraction。指数决定大致能覆盖多大的数量级，fraction 决定同一数量级中，相邻可表示数离得多近。下面各格式都另有 1 个符号位；正常数还有隐含的首位 1。

| 格式 | 指数 / fraction 位数 | 最小正正常数 | 最大有限值 | 在 $$[1,2)$$ 内的间距 |
| --- | --- | --- | --- | --- |
| FP16 | 5 / 10 | $$2^{-14}$$ | 65504 | $$2^{-10}$$ |
| BF16 | 8 / 7 | $$2^{-126}$$ | $$(2-2^{-7})2^{127}$$ | $$2^{-7}$$ |
| FP32 | 8 / 23 | $$2^{-126}$$ | $$(2-2^{-23})2^{127}$$ | $$2^{-23}$$ |

BF16 与 FP32 的指数范围相近，但 BF16 的数值网格远比 FP32 粗，而且在同一数量级也比 FP16 粗。因此，“BF16 范围更大”和“FP16 局部分辨率更细”可以同时成立。Google 的 [BF16 格式说明](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)解释了这种位数分配；其中 TPU 对 subnormal 的处理是该硬件语境，不能直接推广到所有 GPU 算子。

FP16 的最小正常数不是最小正数。若支持 subnormal，最小正 subnormal 是 $$2^{-24}$$。在 round-to-nearest-even 模型下，小于 $$2^{-25}$$ 的正数才会舍入为零，恰好半步也按偶数规则舍入为零；例如 $$0.75\times2^{-24}$$ 反而会舍入到 $$2^{-24}$$。具体硬件路径若采用 flush-to-zero，还要单独核对。

另一种丢失发生在完全正常的数值范围内。设权重 $$w=1.5$$，每次更新希望减去 $$\delta=2^{-12}$$。FP16 在这里的间距是 $$2^{-10}$$，更新只有四分之一个间距，于是：

$$
\operatorname{round}_{16}(1.5-2^{-12})=1.5
$$

如果每次都直接更新并覆盖 FP16 权重，下一次又从 1.5 出发，小更新可以反复被吞掉。保存 FP32 参数，再从它生成低精度计算值，则允许这些小更新先在更细的网格上积累，积累到足够大时再改变计算副本。

前一个问题是梯度量级太小，后一个问题是更新相对权重太小。Loss scaling 主要针对前者；高精度参数更新主要针对后者，不能用一个开关代替两种机制。

还要把“真实梯度很小”和“梯度因为表示问题变小”区分开。收敛附近的小梯度可能正是目标函数要求的信号，不能为了让监控曲线好看就随意放大学习率；反过来，某一层长期出现大量精确零，也不能仅凭 loss 下降就断言没有下溢。应对照相同输入下的高精度路径，判断零来自模型结构、数值舍入，还是训练本来就没有监督信号。

## 高精度要放在会积累误差的位置

矩阵乘的一个输出是许多乘积之和：

$$
c_{mn}=\sum_{k=0}^{K-1}a_{mk}b_{kn}
$$

低精度输入、高精度累积，是把输入搬运与主要计算的收益，同长点积的累积风险分开处理。它不等于完整 FP32 GEMM：输入 cast 时已丢掉的位不会被 FP32 accumulator 找回，实际乘法与归约顺序也取决于后端。

参数更新还有更长的时间尺度。Adam 类 optimizer 的一阶、二阶状态会跨 step 保存，并涉及梯度平方、移动平均和除法；这些状态的精度不能仅从矩阵乘输入 dtype 推断。经典显式混合精度方案保留 FP32 master parameter，再产生低精度计算参数；原生 autocast 又可以让参数本来就保持 FP32，只在算子执行时转换。

这两条路径甚至可能得到相同的主体字节数，却有不同组成。设有 $$P$$ 个可训练参数，以 Adam 类状态示意：

| 持久状态 | 显式低精度计算副本方案 | 本文 FP32 参数 + autocast 方案 |
| --- | ---: | ---: |
| 参数 | 低精度计算参数 $$2P$$ + FP32 master $$4P$$ | FP32 参数 $$4P$$ |
| 参数梯度 | 此例采用低精度 $$2P$$ | FP32 $$4P$$ |
| 一阶、二阶状态 | FP32，共 $$8P$$ | FP32，共 $$8P$$ |
| 主体合计 | $$16P$$ bytes | $$16P$$ bytes |

左列若改用 FP32 梯度累积，主体会变成 $$18P$$；右列也不能再凭空加一份“AMP 自动生成的 FP32 master”。表格省略 scalar step 等小项，也不包含 activation、临时转换、通信、workspace 与 allocator。分片怎样改变这些状态的 ownership，见 [ZeRO/FSDP]({% post_url 2026-08-06-data-parallel-zero-fsdp-memory-sharding %})。

这里的右列是具体前提，不是 optimizer 的普遍承诺。PyTorch v2.5.1 的 [AdamW 初始化代码](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/adamw.py)使用 `zeros_like(p)` 创建 moments：参数保持 FP32，状态才自然得到 FP32；先把模型永久转成 BF16，并不会因此自动获得 FP32 moments 或 master copy。

混合精度降低某些 activation 和计算输入的体积，不代表训练总显存减半。若真正占峰值的是反向保存状态，还要结合 [Activation Checkpointing]({% post_url 2026-09-09-activation-checkpointing-rematerialization %})讨论生命周期，而不是继续在参数文件大小上估算。

## Loss scaling 移动的是反向传播的数值范围

对本次反向中固定、且不参与求导的正标量 $$S$$，定义：

$$
\widetilde{\mathcal L}=S\mathcal L
\quad\Longrightarrow\quad
\nabla_\theta\widetilde{\mathcal L}
=S\nabla_\theta\mathcal L
$$

因此，先对放大的 loss 反向，再在 optimizer 消费梯度前除以 $$S$$，理想数学中得到的仍然是原梯度。Scale 不是学习率；它只是让梯度在反向链中经过更合适的表示区间。

例如某个反向量为 $$g=2^{-26}$$，在前述 FP16 舍入模型里会变成零。令 $$S=2^{12}=4096$$，则：

$$
Sg=2^{-14}
$$

它已经进入 FP16 正常数范围。若这条反向路径传播的是放大后的梯度，最终再在 FP32 梯度或更新路径中解除缩放，就有机会保留原来的量级。这里的顺序极重要：不是先算出一个已经下溢成零的梯度，再乘 4096 把它救回来。

Scale 也不是越大越好。同一反向路径中更大的量可能被推向溢出，所以动态 loss scaling 会根据非有限梯度降低 scale，并在连续成功更新后尝试增大它。[论文 §3.2](https://arxiv.org/html/1710.03740v3#S3.SS2)讨论了缩放的数值动机；[PyTorch v2.5.1 GradScaler](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/amp/grad_scaler.py)给出了框架中的状态与调节行为。

这是一种根据可观察溢出来调整范围的控制策略，不是扫描每个中间张量并保证所有小梯度都被保留。没有发现非有限梯度，只能说明相应检查未发现这类异常，不能反证不存在下溢。某个异常大的梯度还可能迫使全局 scale 降低，让别处较小的梯度更接近下溢边界；因此持续的 scale 剧烈变化值得定位到具体层和算子，而不只是修改初始 scale。

这套机制有明确边界：

- 如果前向 activation 本来就超出 FP16 范围，缩小反向 scale 不会扩大前向格式的范围。
- 如果输入、mask 或 loss 公式本来产生 NaN，缩放不修复这些逻辑错误。
- BF16 通常不需要为 FP16 那样窄的指数范围使用同样的 scaling 策略，但它仍有舍入、溢出和下溢边界。
- Scale 不增加 fraction 位数，也不同于 [FP8 推理]({% post_url 2026-08-01-fp8-formats-scaling-llm-inference %})中把某个张量映射到编码范围的量化 scale。

尤其是从 BF16 预训练参数切到 FP16 时，数值可能超出 FP16 的最大有限值。GradScaler 的 scale 不保证始终大于等于 1；它下降也不意味着任何不兼容的模型最终都能靠缩放稳定训练。

## Autocast 选算子，GradScaler 管梯度，两者并不替 optimizer 决定一切

本文采用的普通 PyTorch 路线是：模型参数保留 FP32，前向和 loss 放在 `torch.autocast` 上下文中，反向放在上下文外。Autocast 按算子策略选择执行 dtype，不是把整个模型永久改成 FP16，也不是所有算子都必须用同一种精度。[v2.5.1 autocast 文档与实现](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/amp/autocast_mode.py)明确区分了这些行为。

矩阵乘与线性层、数值敏感的归约和 loss，可以采用不同策略。原地算子、带显式 `out` 的调用、显式指定 dtype 的操作，也不能机械套用普通 autocast 的转换规则，具体范围以该版本的 [AMP 算子规则](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/amp.rst)为准。

若某个子区域需要 FP32，可以关闭该局部区域的 autocast，并把来自低精度上游的输入显式转成 FP32。仅关闭 autocast，不会自动把已经是 FP16 的张量变回 FP32；显式转回也不能恢复上游已经舍入掉的信息。

GradScaler 则负责放大 loss、解除梯度缩放、依据非有限梯度控制更新，以及更新 scale。它不自动决定全局 loss 分母、不替 optimizer 指定 moments 精度，也不证明一次有限梯度对应的 optimizer 更新一定有限。这些仍属于训练代码的契约。

## 一次参数更新拆成多个 micro-batch 后，scale 必须跨窗口保持不变

假设一次有效更新由 $$G$$ 个 micro-batch 组成，每个 micro-batch 的 loss $$\mathcal L_j$$ 是对相同数量训练单元的均值，目标为：

$$
\mathcal L=\frac1G\sum_{j=1}^{G}\mathcal L_j,
\qquad
g=\frac1G\sum_{j=1}^{G}\nabla_\theta\mathcal L_j
$$

每个 micro-batch 对 $$S\mathcal L_j/G$$ 反向，梯度 buffer 最终保存：

$$
\widetilde g=\frac SG\sum_{j=1}^{G}\nabla_\theta\mathcal L_j
$$

所有 micro-batch 使用同一个 $$S$$，窗口结束后才可以统一除掉它。如果中途更换 scale，梯度 buffer 里混在一起的数字就不再属于同一种单位。

取两个未归一化梯度 $$g_1=2,g_2=4$$，分别用 $$S_1=8,S_2=16$$，累积得到 $$8\times2+16\times4=80$$。最后按 16 解除缩放，只得到 5，而目标梯度之和是 6。提前 unscale 第一部分、随后继续往同一个 buffer 加 scaled gradient，也有同样的单位混用问题。

所以 [v2.5.1 梯度累积示例](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/amp_examples.rst)要求：整个有效 batch 累积完之前，scale 保持不变，梯度保持 scaled；`unscale_`、`step` 和 `update` 放在更新边界。

这里还有一个独立于 AMP 的分母问题。最后一个窗口若只有 3 个 micro-batch，就不能仍机械除以配置值 4；各 micro-batch 的有效 token 数不同，也不能用“各自均值再平均”假装整体 token 均值。前面的公式限定了等分母前提；packing 与变长样本会把这道问题进一步放大。

## 裁剪一定要看到解除缩放后的梯度

忽略实现中用于数值稳定的 epsilon，非零梯度的 global-norm clipping 可以写成：

$$
\operatorname{clip}(g,\tau)
=g\min\left(1,\frac{\tau}{\lVert g\rVert_2}\right)
$$

零梯度单独定义为仍返回零。若 $$g=(3,4)$$、阈值 $$\tau=1$$，正确结果是 $$(0.6,0.8)$$。但如果先把 $$Sg$$ 按阈值 1 裁剪，再除以 $$S=128$$，会得到 $$(0.0046875,0.00625)$$，范数仅为 $$1/128$$。

这不是轻微舍入差异，而是把裁剪阈值实际缩小了 128 倍。正确边界是：

```text
窗口内：scaled backward → scaled backward → …
窗口末：unscale 一次 → clip 一次 → scaler.step → scaler.update
下一窗口开始前：清空梯度
```

裁剪也不能从窗口末搬到每个 micro-batch 后。它是非线性操作：以两个标量梯度 3 和 -2、阈值 1 为例，先求和得到 1，再裁剪仍为 1；分别裁剪再相加却得到 0。即使 scale 全部一致、每个局部梯度都有限，这两种算法仍然不同。若目标就是对完整有效 batch 的梯度做 norm clipping，就必须先完成该目标要求的累积与同步。

下面是单 optimizer 的窗口函数，而非自包含训练程序。调用前须已定义 `model` 和 `loss_fn`，将模型移到目标 CUDA 设备、保留 FP32 参数并设置训练模式；各 `(x, y)` 也须位于同一目标设备。`microbatches` 是已经收齐的非空窗口，每项的 `loss_fn` 都返回相同数量训练单元上的均值。代码按 PyTorch v2.5.1 API 核对，用来展示顺序，未在本文环境执行 GPU 训练。

```python
import torch

scaler = torch.amp.GradScaler("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=False)


def train_window(microbatches):
    count = len(microbatches)
    if count == 0:
        raise ValueError("empty accumulation window")

    optimizer.zero_grad(set_to_none=True)
    for x, y in microbatches:
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(x)
            loss = loss_fn(prediction, y) / count
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0, error_if_nonfinite=False
    )
    scaler.step(optimizer)
    scaler.update()
```

同一 optimizer 在一次更新间只能显式 `unscale_` 一次；已经手动 unscale 后，`scaler.step` 会按状态避免重复解除缩放。若没有梯度裁剪等读取原梯度的操作，也可以由 `scaler.step` 完成相应处理。

示例让常规 GradScaler 跳步路径处理已检测到的非有限梯度。若把 clipping 的 `error_if_nonfinite` 设为 `True`，非有限 norm 可能先抛出异常，行为就变成中止而非平滑跳过。另一方面，`unscale_` 记录的是当时的检查结果；之后任意修改梯度若新引入 NaN，不能假设 `step` 必然再做一次完整检查。

BF16 路径常可关闭 scaler，但关闭后也失去了这套 scaler 的自动非有限跳步保护，不能把它描述成“dtype 换一下，其他保护完全一样”。DDP/FSDP 还要保证同步组内 scale、归一化与更新决策一致；全局 norm、分片梯度和 collective 的具体处理不在这个单 GPU 骨架里。

## 梯度溢出跳过更新后，哪只时钟应该前进

在本文核对的普通 optimizer 路径中，GradScaler 检测到非有限梯度会跳过 `optimizer.step()`。这意味着本次尝试没有消费梯度去更新参数和 moments，optimizer 内的更新计数也不因此前进；但是数据已经读过，前后向计算已经发生，scaler 仍可调低 scale。

于是训练同时拥有不同的时钟：消耗的数据或 token 数、尝试更新次数、成功 optimizer 更新次数。Scheduler 若按成功更新数定义，就不应每次尝试都无条件前进；若本来按已消费 token 数定义，则可以有不同的合理语义，关键是提前写清楚。

不能用 `scaler.step(optimizer)` 返回 `None` 判断更新失败。它转交的是 optimizer 的返回值，普通 AdamW 成功执行时也可能返回 `None`。多 optimizer、fused optimizer 或自定义 scaler 又有各自边界，不能用一段返回值判断包装成通用成功信号。

恢复训练同样需要这套语义。只保存模型和 optimizer，没有 scaler 状态、scheduler 与数据进度，可能改变后续跳步和学习率轨迹。若在累积窗口中途保存，还涉及尚未消费的梯度及 micro-batch 位置；最简单的完整保存点通常是明确完成一个更新边界之后。持久化问题可继续参阅[分布式训练 Checkpoint]({% post_url 2026-08-17-distributed-checkpoint-resharding-async-save %})。

## 判断混合精度是否成功，要看更新链而不只是 loss 是否有限

验证先固定参数初始化、输入顺序、随机性和有效 loss 分母，建立 FP32 参考。对照前向 loss、关键层梯度、梯度范数以及一次更新后的参数和 moments；用适合 dtype 与后端的容差定位第一处分歧，而不是要求所有 reduction 路径逐位相同。

随后再看短训练轨迹中的 scale、非有限梯度和跳步频率，确认小梯度没有异常消失、optimizer state 的实际 dtype 符合预期。有限的 loss 不能排除梯度错误，短程数值对照也不证明长程收敛或验证集质量。

排查时按第一次出现异常的位置前进最有效。前向已经非有限，先看输入范围与敏感算子；前向有限而某层反向首次溢出，检查该层梯度路径和 scale；梯度看起来合理而参数更新异常，再看 unscale、裁剪、optimizer state 与学习率。直接同时切换 dtype、batch 和学习率，虽然可能让训练重新跑通，却会失去判断原因为何的依据。

性能必须在相同全局训练目标下比较：有效 token、batch、序列分布与更新次数不能偷偷改变。预热后覆盖完整前向、反向和更新，分别记录 step 时间、有效 tokens/s、活跃显存峰值与持久状态；首次 optimizer state 初始化也要纳入显存检查。Tensor Core 的理论吞吐不是完整训练加速比，如何解释计算与搬运上限可接着看 [Roofline]({% post_url 2026-09-09-roofline-gpu-performance-model %})。

仓库内可以复算格式边界、更新吸收、scale 混用、裁剪顺序及不等分母反例：

```bash
node scripts/reviews/sep13-mixed-precision-examples.mjs
```

这组标准库 CPU 检查采用明确的最近偶数舍入模型与理想梯度代数，没有调用上面的 PyTorch 骨架，不模拟 GPU 的 flush-to-zero，也不构成真实 GradScaler、分布式更新或收敛验证。

本文的数值例子不是 GPU benchmark，也没有声称验证了某个大模型的收敛。混合精度真正需要守住的，是从 loss 到参数更新的整条链：低精度没有提前抹掉关键信号，高精度保存了长期累积，缩放在同一窗口中可逆，而裁剪和 optimizer 最终看到的仍是原来定义的梯度。
