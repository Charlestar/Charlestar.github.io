---
layout: post
title: "Activation Checkpointing：反向传播需要的张量，何时保存、何时重算"
subtitle: "沿着梯度依赖推导分段重计算，理解显存收益、执行等价性与并行训练中的真实代价"
date: 2026-09-09
last_modified_at: 2026-09-09
author: iStar
catalog: true
series: distributed-training
series_order: 45
technology_year: 2016
mathjax: true
tags: [分布式训练, GPU优化]
---

训练时的一个张量可能已经完成前向使命，却仍然不能释放：几百个算子以后，反向传播还要用它计算梯度。网络越深，这些“未来还会借用”的中间结果越多，显存峰值也越高。

Activation Checkpointing 改变的不是梯度公式，而是满足这些依赖的时间：某些结果不一直保存，等反向真正需要时，再从保留的边界重新算出来。Rematerialization 强调的正是这次重新物化。

这个 checkpoint 是当前训练 step 内的重算锚点，不是写到磁盘、用于故障恢复的模型快照。后者涉及参数、optimizer、随机数与数据进度的持久化，见[分布式训练 Checkpoint]({% post_url 2026-08-17-distributed-checkpoint-resharding-async-save %})。两种技术可以同时使用，但解决的是不同问题。

## 反向传播究竟要借用哪些前向结果

先看一个没有 bias 的两层 MLP。输入各样本按行排列，设 $$X\in\mathbb R^{B\times H}$$、$$W_1\in\mathbb R^{H\times D}$$、$$W_2\in\mathbb R^{D\times O}$$：

$$
Z=XW_1,\qquad A=\phi(Z),\qquad Y=AW_2
$$

给定损失对输出的梯度 $$G=\partial\mathcal L/\partial Y$$，链式法则得到：

$$
\frac{\partial\mathcal L}{\partial W_2}=A^TG,
\qquad
\frac{\partial\mathcal L}{\partial A}=GW_2^T
$$

$$
\frac{\partial\mathcal L}{\partial Z}
=(GW_2^T)\odot\phi'(Z)
$$

$$
\frac{\partial\mathcal L}{\partial W_1}
=X^T\frac{\partial\mathcal L}{\partial Z},
\qquad
\frac{\partial\mathcal L}{\partial X}
=\frac{\partial\mathcal L}{\partial Z}W_1^T
$$

这些式子说明：计算第二层权重梯度需要 $$A$$，计算非线性的局部导数需要与 $$Z$$ 有关的信息，计算第一层权重梯度需要 $$X$$。具体算子可能保存输入、输出、mask 或辅助统计；并不要求每种激活函数都同时保存完整 $$Z$$ 与 $$A$$。

因此，训练语境中的 activation 不仅指非线性激活层的输出，也泛指前向产生、反向还需要的中间张量。它与参数、参数梯度、optimizer state、算子 workspace 应分开记账。

若只保留 $$X$$ 以及正确版本的权重，那么 $$Z$$ 和 $$A$$ 都可以在反向前重建。省掉的是它们跨越后续前向过程的长生命周期；重建时仍然要占空间，所以不能把“没有永久保存”理解成“反向从此不需要这些张量”。

## 保存边界，让重建只覆盖当前区域

把一个由十二层组成的链划成三个区域，每个区域四层。普通前向会为后向留下许多内部状态；分段重算则只长期保留区域边界：

```text
第一次前向：
x0 ── 层 1..4 ── x4 ── 层 5..8 ── x8 ── 层 9..12 ── x12
留作边界           留作边界          留作边界                 loss

反向推进：
从 x8 重建 9..12 的内部状态 → 反传 9..12 → 释放该段内部状态
从 x4 重建 5..8  的内部状态 → 反传 5..8  → 释放该段内部状态
从 x0 重建 1..4  的内部状态 → 反传 1..4  → 完成
```

重点在于一次只为当前反向区域恢复内部状态，而不是在反向之前把全网络所有 activation 一次性重建出来。后者重新产生原来的峰值，无法实现预期节约。[Training Deep Nets with Sublinear Memory Cost §4.1](https://arxiv.org/html/1604.06174v2#S4.SS1)给出了这种分段调度；本文的 2016 技术年份指这篇神经网络代表工作，而不是说自动微分的重计算思想起源于这一年。

框架中的“释放”通常意味着不再持有对应 storage 的活跃引用，并不需要调用清空显存缓存的接口。若 Python 列表还收集着每层 hidden states，或区域返回了大量中间张量让外部保存，那么它们仍然存活。Checkpoint 包装不会推翻外部引用所要求的生命周期。

## 分段太细和太粗，为什么都会多占空间

考虑一个理想化链：共有 $$n$$ 层，每层代表性的保存量为 $$A$$，每段包含 $$k$$ 层，先假设 $$k$$ 整除 $$n$$。这里 $$k$$ 是段长，不是段数。

有两笔主要支出。跨区域长期保存的边界约有 $$n/k$$ 份；反向时正在重建的一个区域，约需 $$k$$ 份局部状态。忽略输入输出端点常数与其它训练状态：

$$
M_{saved}(k)\approx A\left(\frac nk+k\right)
$$

减小 $$k$$ 会降低局部重建峰值，却留下更多边界；增大 $$k$$ 会减少边界，却让当前区域变大。把 $$k$$ 暂时视为连续量，求导：

$$
\frac{dM_{saved}}{dk}=A\left(1-\frac{n}{k^2}\right)=0
\quad\Longrightarrow\quad k=\sqrt n
$$

这就是该对称模型下平方根保存量的来源。它讨论的是一类 activation 调度，不是整个模型的显存复杂度承诺。现实中的参数、参数梯度、optimizer、通信 buffer 与 workspace 都没有因为这条公式消失。

取 $$n=64,A=32\ \mathrm{MiB}$$，可以得到以下估算：

| 策略 | 长期边界主项 | 当前段内部主项 | 保存量模型 |
| --- | ---: | ---: | ---: |
| 不重算 | 不单列 | 64 份跨层存活 | 2048 MiB |
| 每段 4 层 | 16 份 | 4 份 | 640 MiB |
| 每段 8 层 | 8 份 | 8 份 | 512 MiB |
| 每段 16 层 | 4 份 | 16 份 | 640 MiB |

这些数字是公式代入，不是 GPU 显存测量；端点、共享 storage、反向临时梯度以及不同算子的保存需求都会改变实际峰值。逐 Transformer block 启用 checkpoint 通常仍要保留各 block 的输入，也不能直接宣称它实现了上述最优分段。

若每个边界大小为 $$A_b$$，区域内每层保存量为 $$A_i$$，更合理的近似是：

$$
M(k)\approx\frac nk A_b+kA_i,
\qquad
k_{opt}\approx\sqrt{\frac{nA_b}{A_i}}
$$

边界很大时更倾向减少边界数，区域内部很大时则更倾向小段。进一步遇到跳连、共享输入和不等成本层，这个一维模型只能帮助解释方向，不能代替对真实计算图的选择。

## 再做一次前向，不等于训练慢一倍

令原前向工作量为 $$F$$，原反向为 $$B$$，额外重算为 $$R$$。只按算术量计，倍率为：

$$
\frac{F+B+R}{F+B}
$$

只有在 $$B\approx2F$$、$$R\approx F$$ 的假设下，才得到 $$4/3$$：step 工作量增加约 33%。如果执行时间也按该工作量同比例增长，那么相同 batch 的吞吐变为原来的 $$3/4$$，下降约 25%。这两个百分比不能混用；从算术量到真实时间还多了一层性能假设。

真实时间还取决于重算区域。保留某些输出、只重算便宜操作时，$$R$$ 可能远小于一个完整前向；较深的嵌套重算又可能反复执行部分前缀。PyTorch 非重入实现还可以在需要的张量全部重建后提前停止，不能默认每个区域函数都会完整执行第二遍。

反过来，额外 FLOPs 也不保证按比例增加时间。原本较小的 micro-batch 若因显存释放而能够扩大，GEMM 利用率可能改善；重算还可能改变 kernel fusion、通信重叠和访存量。因此应分别回答两个问题：固定 batch 时多花了多少时间，以及同一显存约束下最可用的训练配置快了还是慢了。

## Transformer 中，哪些状态最值得重算

设本地 micro-batch 为 $$B$$、序列长度为 $$S$$、hidden size 为 $$H$$、Query heads 为 $$H_q$$、每元素字节数为 $$b_a$$。一份 hidden-state 张量占：

$$
M_{hidden}=BSHb_a
$$

取 $$B=1,S=4096,H=4096,b_a=2$$，一份就是 32 MiB。前向可能涉及 norm 输入、Q/K/V 投影、MLP 扩展维度与 residual 等多份张量，不能只数 block 的最终输出。

若显式物化一份形状为 $$[B,H_q,S,S]$$ 的 attention 矩阵，则它占：

$$
M_{attention}=BH_qS^2b_a
$$

相同序列、$$H_q=32$$、两字节元素时，仅这一份矩阵就有 1 GiB。它解释了为什么早期 Transformer selective recomputation 特别关注 attention 内部的大矩阵。[Reducing Activation Recomputation in Large Transformer Models §4–5](https://arxiv.org/html/2205.05198v1)讨论的具体保存量模型，依赖其 attention、GeLU MLP、dtype 与 dropout 假设。

现代后端改变了这份账单。[FlashAttention]({% post_url 2026-03-17-FlashAttention原理与实现详解 %})本来就不把完整 $$S\times S$$ 概率矩阵写成长期保存的张量，而会利用保存的统计量在反向按 tile 重建概率。若外层再包装 activation checkpoint，不能把一个不存在的 1 GiB 矩阵重复扣减一次。

此时应该重新观察真实 saved tensors：Q/K/V 是否保留，norm 保存哪些统计，SwiGLU 分支是否 materialize，MLP 扩展维度在哪里形成峰值。与其沿用旧文章中的固定字节系数，不如先核对目标后端的 forward/backward 接口。NVIDIA 的 [Megatron Bridge 重计算说明](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html)也把 full、selective 与 FlashAttention 后端的关系分别处理；该动态文档于 2026-09-09 查阅，配置应以部署版本为准。

“大张量适合丢掉”还不充分。若重建它需要昂贵 GEMM，代价可能很高；某个元素级操作输出虽大，但上游输入必须因此继续保存，实际峰值也未必下降。真正要比较的是保存与重算对整段依赖和同时存活状态的改变。

## 把重算区域变成一个明确的函数边界

下面展示一个边界包装，不是完整训练程序。代码显式使用非重入模式，并保留 RNG 状态；具体行为以 [PyTorch v2.8.0 的 checkpoint 实现](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/checkpoint.py)为本文核对基准：

```python
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class CheckpointedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            return checkpoint(
                self.block,
                x,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return self.block(x)
```

区域输入形成重建起点，模块参数通过该模块引用访问；内部适合重建的反向保存状态不再全部长时间驻留。非重入实现仍记录 autograd graph，并支持按需求提前终止重算。不能用“checkpoint 就是在 `no_grad` 里跑前向”概括它；传统重入实现才采用另一套执行方式，并对梯度接口和嵌套输入等存在不同限制。

选择边界时先让它可解释：传入什么、返回什么、读取哪些参数、是否访问外部状态。随后再扩大区域或引入选择性策略。多个小包装也不自动等于手工分成几大段；包装边界本身会成为长期保存的一部分。

## 重算必须恢复同一次计算，不只是相同形状

链式法则要求使用原前向路径对应的局部导数。设倒置 dropout 的保留概率为 $$p$$、采样掩码为 $$D\in\{0,1\}$$，其输出是 $$Y=WX D/p$$。当 $$W=X=1,p=1/2,D=1$$ 时，$$Y=2$$；若损失为 $$\mathcal L=Y^2/2$$，原来的权重梯度是 4。

若反向重建时错误采到 $$D'=0$$，用原来的上游梯度乘新的局部导数，就会得到 0。前向 loss 没变，但梯度已经不是这次计算的梯度。这说明保持随机路径不是复现实验的装饰条件，而是重算正确性的组成部分。

框架的 RNG 保存并不是整个进程状态快照。对于本文核对版本，它处理 Torch CPU 与所选择设备类型的 RNG 状态；Python `random`、NumPy、自定义 generator，以及区域内部新引入设备上的随机源，需要另外核对和管理。不能为了省少量开销就默认关闭保存，然后只用“loss 看起来正常”判断正确。

编译模式也有版本边界：PyTorch v2.8.0 的接口说明明确指出，在 `torch.compile` 下 `preserve_rng_state=False` 不会关闭 RNG 保存。显式参数表达调用意图，但仍应核对 eager 与编译路径分别如何执行它。

还有几类状态不能被 RNG 恢复覆盖：

- BatchNorm running statistics、计数器、列表追加等副作用，可能在重算时再次发生。
- Python 全局变量、循环闭包引用、数据相关分支若改变，可能使重算进入不同路径。
- 输入或参数被原地修改，会破坏重建所需的原版本。
- MoE 的路由与容量裁剪若依赖随机或外部状态，需要保持相同决策及其语义。
- `optimizer.step()` 若在尚有区域等待重算时提前更新参数，重建会读到错误权重版本。

非重入 early-stop 还意味着副作用既可能重复，也可能只重复函数的前半段；不能把它理解成严格调用两遍的函数协议。更稳妥的边界是只做张量计算，把日志、统计提交和外部状态变更放在区域之外。

同样，`determinism_check="default"` 在本文核对版本中主要比较重建张量的 shape、dtype 与 device，不比较全部数值，更不是梯度等价性证明。通过这道检查之后，仍然要运行数值对照。

## Selective Checkpointing 选择的是依赖，不只是算子名单

把整个 Transformer block 都重算很容易理解，但它可能重复最昂贵的矩阵乘。Selective Activation Checkpointing 尝试保留某些操作的结果，同时允许其他操作重算。例如保存昂贵 matmul 的输出、重建较便宜的逐元素操作，可以探索更细的时间—空间折中。

[PyTorch 对 AC、SAC 与编译重算的说明](https://pytorch.org/blog/activation-checkpointing-techniques/)区分了这些层次：SAC 允许设置操作级保存偏好；编译器则可以在联合前向/反向图中选择分割位置，调整哪些值跨边界保存、哪些操作重新执行。后者并不等价于“所有 GEMM 保存、所有逐元素操作重算”的固定规则。

原因是依赖会传递。保存某个小结果，可能切断一大片昂贵上游的重建需求；保存一个很大的结果，也可能让多个分支共享它。Kernel fusion、不同后端暴露的算子粒度和原地别名，都会改变可选择的图。SAC 缓存的也是真实操作输出及关联状态，不是只记录一个算子名字。

可以观察某候选区域带来的峰值下降 $$\Delta M_i$$ 和额外时间 $$\Delta T_i$$，但 $$\Delta M_i/\Delta T_i$$ 不是可独立相加的物品价值。两个区域可能释放同一峰值时刻的不同张量，也可能一个只是把峰值移到另一个时刻；简单贪心没有全局最优保证。

内存预算驱动的自动选择很有吸引力，不过上述官方说明中的 Memory Budget 接口仍带实验性边界。应把它理解成按预算搜索保存策略的方向，而不是可以跨版本照抄的稳定开关。手工 region、SAC 与编译器重算叠加后，也应重新检查实际图与代价，不能把三种预估节约直接相加。

## 进入并行训练，重算可能重新发起通信

Activation Checkpointing 主要减少 activation，[ZeRO/FSDP]({% post_url 2026-08-06-data-parallel-zero-fsdp-memory-sharding %})主要切分模型状态，两者可以组合。但它们的执行时间线不是完全独立的：参数何时 unshard、何时释放，与哪个区域正在重建有关。

若重算边界把 FSDP wrapper 一起包进去，参数 all-gather 的触发行为可能与只包装内部纯计算区域不同；不能只把“多算一次前向”理解成多做本地 GEMM。类似地，TP 的 collective、CP 的跨 rank 数据交换，如果属于区域的执行路径，也可能被重放。应核对目标框架的 wrapping 顺序、预取与 reshard 策略，而不是凭两种技术各自的公式相乘。

[Pipeline Parallel]({% post_url 2026-08-11-pipeline-parallel-microbatch-1f1b %})又加入了同时存活的 micro-batches。每个 micro-batch 保存得更少，确实可能降低 stage 峰值；但重算会拉长部分 stage 的反向时间，改变原来的均衡与气泡。局部最省显存的边界，未必带来最佳全流水线吞吐。

Offload 则是另一个选择：把张量搬到 host 或其他存储，反向再传回来，避免本地重算但支付传输代价。比较二者需要看关键路径、链路带宽与能否重叠；显存释放量相同，不代表时间成本相同。混合采用保存、重算与迁移时，要给每类张量明确唯一的恢复来源。

## 怎样证明省下了显存，却没有改坏训练

首先建立同一初始状态的两条路径：相同参数、输入、RNG、dtype、mask 与 loss reduction，一条不重算，一条开启目标策略。比较 forward loss、输入梯度、各参数梯度以及一次 optimizer 更新后的参数；再用短训练轨迹检查累积差异。只比较 loss 会漏掉前面 dropout 反例中的梯度错误。

对浮点实现采用与 dtype、后端相适应的容差，不把 bitwise equality 当作所有 kernel 的通用要求。若出现差异，应先缩小 region，定位第一个重建或梯度分歧，再加入随机操作、并行通信和编译优化。本文提供的是推导和验证路径，没有声称完成真实 GPU 训练复现。

博客仓库的 `node scripts/reviews/new-activation-examples.mjs` 可以先复算内存表、两层 MLP 梯度、纯函数链的分段重算与有限差分，以及 dropout 的错误重建反例。它只使用标准库，不执行上面的 PyTorch 包装，因此通过这些检查也不代表目标框架或并行训练路径已获验证。

测显存时必须覆盖完整 forward 与 backward，并处理 optimizer 首次初始化产生的持久状态。Warmup 后分别报告活跃分配峰值与 allocator 保留峰值；两者含义不同，保留给以后复用的空间不等于当前仍被张量使用的空间。模型本体、通信和外部库分配也需要与框架内统计对齐。

性能至少做两组对照：固定 micro-batch、序列长度和全局 batch，衡量重算的直接成本；再在同一显存容量下寻找更可用的 micro-batch，并保持训练目标的有效 batch 与归一化语义，观察端到端吞吐。记录 attention 后端、region 划分、保存策略、编译状态和并行拓扑，才能解释改动究竟改变了哪条关键路径。

Activation Checkpointing 最有价值的视角不是“开启一个省显存选项”，而是重新安排反向传播所需证据的生命周期：哪些一直保存，哪些在需要前重建，哪些重建代价会经过网络。只要梯度依赖、参数版本和随机路径仍然一致，就可以在时间与空间之间寻找更合适的执行点。

## 参考资料

- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- [PyTorch v2.8.0: torch.utils.checkpoint](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/checkpoint.py)
- [Current and New Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/)
- [NVIDIA Megatron Bridge: Activation Recomputation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html)
