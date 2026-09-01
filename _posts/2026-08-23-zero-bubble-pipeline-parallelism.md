---
layout: post
title: "Zero-Bubble Pipeline Parallel：为什么 Weight Gradient 可以延后计算"
subtitle: "从 Backward 依赖拆分、显存生命周期到优化器版本边界，理解接近零气泡的同步流水线"
date: 2026-08-23 09:00:00 +0800
last_modified_at: 2026-09-01
author: iStar
catalog: true
series: distributed-training
series_order: 35
technology_year: 2024
mathjax: true
tags: [分布式训练, GPU优化]
---

在普通同步 Pipeline Parallel 中，一个 micro-batch 的 backward 往往被调度器视为不可分割的操作：当前 stage 算完全部 backward，才把 activation gradient 发送给上一个 stage。这个抽象便于使用自动微分，却把两类依赖完全不同的计算绑在了一起。

对于一个带参数的算子，backward 同时回答两个问题：

1. loss 对输入 activation 的梯度是多少，以便继续向前一个 stage 反传；
2. loss 对本 stage 参数的梯度是多少，以便本轮结束时更新权重。

第一个答案位于 pipeline 的跨 stage 关键路径上，第二个答案大多只在本地 optimizer step 前有用。如果必须等两者一起完成才向上游发送梯度，本地的 weight-gradient 计算就会人为延长整条反向依赖链。

Zero-Bubble Pipeline Parallelism 的突破口不是取消 backward，也不是用旧权重近似训练，而是把 backward 拆成计算 input gradient 的 **B pass** 和计算 parameter/weight gradient 的 **W pass**：优先执行、通信 B，把没有跨 stage 依赖的 W 放进其他设备本来会空闲的时间槽。这样仍能让一个 optimizer step 内的所有 micro-batches 使用同一版本参数，却显著扩大了合法调度空间。

前一篇《Pipeline Parallel：Micro-batch 怎样流过 Transformer Stages》已经解释了 GPipe、同步 1F1B、warmup/cooldown、virtual stages 与 activation checkpointing。本文不再重复这些基础，而是沿着一张更细的依赖图，讨论五件更难的事情：B/W 为什么能拆、怎样用显存换 bubble、自动调度器在搜索什么、optimizer 边界怎样保持同步语义，以及实现时如何证明结果仍然正确。

## 从一个 Linear 层看清 F、B、W

设线性层忽略 bias 后为：

$$
Y=XW
$$

其中 $X\in\mathbb{R}^{n\times d_{in}}$，$W\in\mathbb{R}^{d_{in}\times d_{out}}$。上游传来的输出梯度记为 $G_Y=\partial L/\partial Y$。反向传播中的两个主要矩阵乘法是：

$$
\underbrace{G_X=G_YW^{\mathsf T}}_{B:\ \text{input gradient}}
$$

以及：

$$
\underbrace{G_W=X^{\mathsf T}G_Y}_{W:\ \text{weight gradient}}
$$

这里的命名来自 Zero Bubble 原论文：

- **F**：forward，使用 $X$ 和 $W$ 产生 $Y$；
- **B**：backward for input，产生 $G_X$，让梯度继续流向更早的 layer/stage；
- **W**：backward for weights，产生或累加 $G_W$，供 optimizer 使用。

传统框架常把 $G_X$ 与 $G_W$ 放进一次 backward 调用。可从公式直接看到，$G_X$ 并不依赖 $G_W$。只要 $G_Y$ 已经到达并且当前版本的 $W$ 仍然可读，就能先算 $G_X$、把它发送到上游；$G_W$ 可以在之后再用保存的 $X$ 和 $G_Y$ 计算。

“可以延后”只表示 W 的 deadline 比 B 晚，并不表示 W 可以省略。所有 micro-batches 的 W 仍必须在对应 optimizer update 之前完成并按既定精度累加。

## B 是跨 Stage 关键路径，W 通常不是

假设模型按深度切成 stage $0,1,\ldots,p-1$，micro-batch 为 $j$。对同一个 micro-batch，主要依赖可以写成：

```text
Forward:
F(0,j) -> F(1,j) -> ... -> F(p-1,j) -> loss

Input-gradient backward:
B(p-1,j) -> B(p-2,j) -> ... -> B(0,j)

Local weight gradient:
B(i,j) -> W(i,j)       for every stage i
```

`F(i,j)` 的输出 activation 要发给 `F(i+1,j)`；`B(i,j)` 的 input gradient 要发给 `B(i-1,j)`。因此 F、B 分别形成两条方向相反的跨设备链。

相比之下，`W(i,j)` 的结果属于 stage $i$ 自己的参数。它不需要发送给相邻 pipeline stage，也不会成为 `B(i-1,j)` 的输入。按照原论文采用的执行约束，W 在同 stage、同 micro-batch 的 B 之后执行，但它可以与其他 micro-batches 的 F/B 重新排序。

这就是调度自由度的来源：

```text
传统 BW： GY -> [ B + W ] -> send GX
拆分 B/W：GY -> [   B   ] -> send GX
                      \
                       -> [ W ] before optimizer step
```

如果 B 完成后仍等待 W，等待会沿 `stage i -> stage i-1 -> ...` 逐段放大；如果先发送 $G_X$，上游可以更早开始工作，而 W 能留到该 stage 的空闲槽中执行。

## “Weight Gradient 延后”不等于异步训练

容易产生的一种误解是：既然 W 被推迟，模型是不是在使用不完整梯度或 stale weights？答案是否定的。Zero-Bubble 调整的是**同一个同步迭代内部的执行顺序**，而不是把某些 micro-batches 推到不同参数版本。

对 iteration $k$，至少要维持以下语义：

```text
所有 F(*, k) 读取参数 theta_k
所有 B(*, k) 读取与 forward 匹配的 theta_k
所有 W(*, k) 累加得到 gradient g_k
optimizer 使用完整且正确同步的 g_k 产生 theta_(k+1)
iteration k+1 的 F 只能读取已提交的 theta_(k+1)
```

W 可以晚于本 micro-batch 的 B，却不能晚于使用其梯度的 optimizer step。更不能让 stage 先覆盖 $\theta_k$，再用 $\theta_{k+1}$ 去计算属于旧 forward 的 B。后者会改变链式求导使用的 Jacobian，已经不是一次合法的同步训练重排。

因此需要同时区分两个“延后”：

- **延后 W compute**：在 iteration 内调整依赖无关的本地计算，核心算法允许；
- **延后或错位 weight version**：跨 iteration 混用参数，必须通过明确的版本协议避免。

## 一张更完整的依赖 DAG

仅画 F/B/W 三种方块还不够。生产调度器至少要表达下面这些边：

```text
F(i-1,j) --activation--> F(i,j)
F(i,j) ----------------> B(i,j)
B(i+1,j) --grad--------> B(i,j)
B(i,j) ----------------> W(i,j)

W(i,0..m-1) --> local grad finalization
local grad finalization --> DP reduce / grad norm / overflow state
all required global state --> optimizer(i,k)
optimizer(i,k) --> F(i,*,k+1)
```

其中 `F -> B` 并不是说 B 只需保留 F 的最终输出；它代表本 stage 内所有 autograd saved tensors、RNG/recompute 状态和参数版本必须可用。`B -> W` 则对应原论文的显存状态转换：B 完成后能释放 B 独有的中间量，但仍要保留 W 计算所需的 activation 与 output gradient。

调度正确性的实质就是：生成一个满足所有 DAG 边、单设备资源互斥、通信匹配和显存上限的拓扑序，同时尽量缩短最慢 stage 的完成时间。

## 为什么合并 Backward 会制造“假依赖”

传统 autograd 把一个算子的 backward 作为统一函数很合理，因为单 GPU 上 B 与 W 都必须做，连续执行还能复用 cache。Data Parallel 中也常在某层 W 完成后立刻启动该参数 bucket 的 AllReduce/ReduceScatter，并与更早层的 backward 重叠。

但 Pipeline Parallel 的目标不同：上一个 stage 等待的是 $G_X$，不是 $G_W$。如果统一 backward 的内部顺序为：

```text
compute GX
compute GW
return GX to scheduler
```

那么即使 $G_X$ 早已计算好，调度器也看不到它，跨 stage send 只能等 $G_W$。抽象边界把“函数调用结束”错误地当成“input gradient ready”。Zero-Bubble 所做的细粒度化，是让 runtime 在 $G_X$ ready 时就推进 pipeline，并把 $G_W$ 变成独立的可调度节点。

这里的困难不在数学，而在框架：框架必须捕获 W 所需的输入、输出梯度、weight shard 和具体 kernel，确保稍后执行时这些对象仍属于正确的 micro-batch、layer、virtual chunk 与参数版本。

## W 能移动多远：三个不可越过的边界

W 的调度窗口很宽，但不是无限宽。

### 不能早于它的数据就绪

只有当本层需要的 $X$ 与 $G_Y$ 都可用时，才能计算 $G_W$。原论文和官方 runtime 将 W 放在对应 B 之后，形成清晰、易验证的 `B -> W` 顺序。

### 不能晚于梯度消费者

gradient clipping、DP gradient reduction、optimizer update、checkpoint commit 或任何读取 `.grad/main_grad` 的逻辑，都必须看到完整的 W 结果。若 W 还在另一个 CUDA stream 上运行，还需要 event 依赖，不能只依赖 Python 调用顺序。

### 不能跨越参数版本销毁

B 需要 forward 对应版本的权重，W 需要 forward 输入和 output gradient。optimizer 覆盖参数、释放参数 shard 或 FSDP 重新分片前，必须确认所有相关 B/W 已完成。否则调度在时间线上看起来合法，数值图却已经断裂。

## 显存生命周期从一段变成两段

在统一 backward 中，一份 micro-batch activation 常被粗略建模为：F 后分配，BW 后全部释放。拆开后，它更像两个连续状态：

```text
F done                  B done                  W done
  |                        |                       |
  |---- B 所需状态 M_B ----|---- W 所需状态 M_W ---|
                                                   -> free
```

论文在自动调度的内存模型中，把三个 pass 的净显存变化写成：

$$
\Delta M_F=M_B
$$

$$
\Delta M_B=M_W-M_B
$$

$$
\Delta M_W=-M_W
$$

也就是说，F 让一份待反向状态进入显存；B 把较大的 B 状态转换成较小的 W 状态；W 才完成最终释放。对论文分析的 Transformer 配置，$M_W<M_B$，所以尽早 B 即使暂不执行 W，也能降低在途 activation 压力。

但“较小”不是“免费”。如果把大量 W 一直堆到尾部，`(X,G_Y)` 或等价缓存会积累起来。调度器必须同时数清 waiting-for-B 与 waiting-for-W 两类对象，而不能在 B 完成时就把整份 activation memory 记为零。

## Transformer 中 B 与 W 的时间并不相等

手绘 schedule 经常把 F、B、W 都画成相同宽度。原论文明确把这当作解释手工方案的理想假设，而不是现实规律。

按照论文只统计主要矩阵乘法的分析，一个典型 Transformer layer 满足：

$$
T_W<T_F<T_B,
\qquad
T_B+T_W\approx 2T_F
$$

B 除了线性层 input-gradient GEMM，还承载 attention backward 中与序列长度相关的计算；W 主要对应参数梯度 GEMM。真实时间还会受 attention backend、TP/CP collective、MoE routing、recomputation、kernel fusion 和 shape 影响。

因此不能把论文图直接硬编码成 production schedule。一个在 `F=B=W` 下严丝合缝的排列，遇到 $T_W$ 只有半个 slot、B 又包含暴露通信时，仍会留下碎片化空洞，甚至让某个 W 反过来阻塞即将就绪的 B。

## Bubble 应该怎样计量

令 $m$ 为 micro-batch 数。原论文把一次 schedule 的 `cost` 定义为所有 stages 中最长的执行跨度，并用下式衡量 bubble rate：

$$
r_{bubble}
=
\frac{
\operatorname{cost}-m(T_F+T_B+T_W)
}{
\operatorname{cost}
}
$$

分子是相对于理想有效计算时间的额外跨度。这个定义隐含“通信能被计算完全覆盖”的理想下界；现实 trace 还应单独区分：

- 没有 ready node 的真正 pipeline bubble；
- ready node 存在但被显存上限阻止的 memory stall；
- P2P/collective 没有覆盖的 communication stall；
- 最慢 stage 造成的 load-imbalance wait；
- host 调度、allocator 或 kernel launch gap。

如果把所有 GPU idle 都叫 bubble，就会误以为换 schedule 能修复 stage 切分失衡或网络拥塞。

## ZB-H1：在 1F1B 显存预算内先减少 Bubble

原论文首先给出手工方案 **ZB-H1**。它大体保留 1F1B 的结构，却让各 stage 更早开始 B，并根据 warmup 中在途 micro-batch 数调整 W 的起点，最后用推迟的 W 填充尾部空洞。

论文的理想模型给出：

$$
\text{Bubble}_{1F1B}
=(p-1)(T_F+T_B+T_W)
$$

$$
\text{Bubble}_{H1}
=(p-1)(T_F+T_B-T_W)
$$

在 $T_F=T_B=T_W$ 时，H1 的 bubble 是普通 1F1B 的三分之一；峰值 activation memory 仍为 $pM_B$。它体现了最容易落地的一种选择：先不追求字面上的零气泡，只利用 B/W 拆分回收一部分空闲时间，并维持接近基线的显存上界。

下面只是一条 stage 的局部直觉，不是论文完整 schedule 的逐格复刻：

```text
1F1B: ... F3 | [B0 W0] | F4 | [B1 W1] | ... | idle
ZB-H1:... F3 |  B0 -> send | F4 | B1 -> send | ... | W0 W1
```

重要的是 B 完成后立即推进跨 stage 依赖，W 被移动到不阻塞下一条高优先级 B/F 的位置。

## ZB-H2：用更多在途 Activation 换理想零气泡

第二个手工方案 **ZB-H2** 在 warmup 中注入更多 F，用它们填住第一批 B 到达前的空洞，并在尾部重新排列 W，使整体时间线由梯形更接近平行四边形。

论文的理想分析为：

$$
\text{Bubble}_{H2}
=(p-1)(T_F+T_B-2T_W)
$$

当 $T_F=T_B=T_W$ 时该式为零；对应峰值 activation memory 约为：

$$
(2p-1)M_B
$$

而 H1/1F1B 的理想上界为 $pM_B$。所以 H2 不是“免费的 H1 增强版”，而是主动把近两倍在途状态作为调度缓冲。显存越宽松，scheduler 越能提前执行 F、为后续阶段准备更多 ready work；显存越紧，F 会因 admission 受阻，bubble 又会出现。

而且 H2 的“zero”来自等时假设。实际 $T_F,T_B,T_W,T_{comm}$ 不同，手工排列仍可能留下小洞。论文实验中也把 H1/H2 作为基线，真正的现实配置由自动搜索生成。

## 名称容易混淆：H1/H2、1p/2p 与 ZB-V

Zero-Bubble 的论文版本和官方仓库经历了命名演化，阅读资料时应先对齐概念：

| 名称 | 含义 | 关键显存假设 |
| --- | --- | --- |
| ZB-H1 | 在等时假设下设计的手工 schedule | 与 1F1B 相近，峰值约 $pM_B$ |
| ZB-H2 | 更激进的手工 schedule | 峰值约 $(2p-1)M_B$ |
| ZB-1p / ZB1P | 自动搜索或官方实现中，以约 $pM_B$ 为上限的一类 schedule | 约 1× 基线 activation memory |
| ZB-2p / ZB2P | 以约 $2pM_B$ 为上限的一类 schedule | 约 2×，通常更接近零 bubble |
| ZB-V / ZBV | 每设备两个 chunks，逻辑 stage 走 V 形路径 | 理想条件下约 1× 基线 memory |

不同版本资料的命名重心并不相同：arXiv 扩展版同时详细描述了手工 H1/H2、自动搜索 1p/2p 与 ZB-V；当前官方仓库则让 ZB1P、ZB2P、ZBV 并存，Quick Settings 还把 `--zero-bubble-v-schedule` 作为推荐入口之一。它们共享 B/W 拆分，但不是五个可以互换的别名，复现实验时必须同时固定论文/代码版本和具体 schedule 参数。

## 自动调度器为什么比手绘时间线重要

一个真实配置至少由这些量决定：

$$
(p,m,M_{limit},T_F,T_B,T_W,T_{comm})
$$

stage 数和 micro-batch 数决定 DAG 的规模；显存上限决定最多允许多少 F/B 后状态存活；F/B/W 与通信实测时间决定哪种空洞能被哪个节点填满。

原论文的启发式过程可以概括为：

1. warmup 中在显存允许范围内尽量排 F，缩短第一个 B 前的空洞；
2. steady phase 交替选择 F 与 B，优先维持跨 stage 流水；
3. gap 足够容纳 W 时插入 W；触及显存上限时也用 W 释放状态；
4. 当 F、B 用尽后清空剩余 W；
5. 对少量二值决策做网格搜索，选择 makespan 更短的方案。

它不是一个运行时“哪个 ready 就随机跑哪个”的 opportunistic scheduler。训练要求所有 ranks 得到可匹配、可复现的 send/recv 顺序，通常应先根据 profile 和配置生成确定性 schedule，再由 runtime 执行。

## 把 Schedule Search 写成约束问题

每个节点可用三元组 $(i,j,c)$ 标识：stage $i$、micro-batch $j$、pass 类型 $c\in\{F,B,W\}$。搜索变量是同一 stage 上节点的先后次序和结束时间。

约束至少包括：

### Forward 跨 Stage 依赖

$$
E_{i,j,F}
\ge
E_{i-1,j,F}+T_{comm}+T_{i,j,F}
$$

### Backward 跨 Stage 依赖

$$
E_{i,j,B}
\ge
E_{i+1,j,B}+T_{comm}+T_{i,j,B}
$$

### 单设备互斥与本地 DAG

同一 GPU 上两个不能并发的 compute nodes 必须确定顺序；每个 `W(i,j)` 还要排在相应 `B(i,j)` 之后。若允许不同 CUDA streams 并发，则资源模型要进一步描述 SM、HBM bandwidth 和通信 stream，不能简单删掉互斥边。

### 任意时刻的显存上限

按 $\Delta M_F$、$\Delta M_B$、$\Delta M_W$ 累加，任一前缀的 live memory 都不能超过 $M_{limit}$。

目标是最小化最晚完成的 stage，也就是 schedule makespan。原论文还给出 Integer Linear Programming 形式；规模适中时可用 ILP 校验启发式结果，大规模时则用启发式给出近优解或作为 ILP 初值。

## Profile 不是只测一个 `backward()`

自动搜索只有在 cost model 可信时才有意义。至少应分别 profile：

- 每个 stage/chunk 的 $T_F$；
- input-gradient 部分的 $T_B$；
- weight-gradient 部分的 $T_W$；
- forward activation 与 backward gradient 的 $T_{comm}$；
- recomputation、TP/CP/EP collective 是否已经包含在某个 pass；
- 不同 sequence length、micro-batch size、MoE token distribution 下的方差。

原论文先运行若干 profiling iterations，再把实测值送入调度搜索。它还让首尾 stages 少放一层 Transformer，以补偿 embedding 和 loss 的额外成本。这提醒我们：再聪明的 schedule 也无法消除最慢 stage 的持续性失衡。

实际系统还应区分冷启动与稳态。首轮 kernel autotune、CUDA Graph capture、allocator 扩容和通信连接建立不应污染长期 schedule；另一方面，动态 shape 或 MoE imbalance 如果每步都变化，就不能只依赖一次静态中位数。

## ZB-V：V 形映射怎样降低显存门槛

ZB-H2/ZB-2p 可以用更多 in-flight F 接近零气泡，但 activation memory 代价很高。arXiv 扩展版提出 **ZB-V**：把模型均匀切为恰好 $2p$ 个 chunks，每个物理 worker 持有两个 chunks，逻辑模型先从 worker 0 走到 worker $p-1$，再反向走回 worker 0，形成 V 形放置。

例如 16 层、4 个 workers：

```text
worker 0: layers  1-2  + layers 15-16
worker 1: layers  3-4  + layers 13-14
worker 2: layers  5-6  + layers 11-12
worker 3: layers  7-8  + layers  9-10

logical forward path:
w0.first -> w1.first -> w2.first -> w3.first
         -> w3.second -> w2.second -> w1.second -> w0.second
```

于是一个 micro-batch 的 forward 终点与 backward 起点都在 worker 0 附近，而不是让 backward 再从远端一路返回后，首个 worker 才能开始回收状态。论文在 $T_F=T_B=T_W$ 的理想条件下给出：ZB-V 可在峰值约 $pM_B$、也就是接近 1F1B 的 activation 预算下达到零 bubble。

代价是每台设备拥有两个 chunks，pipeline 边界数增加。官方仓库的理想比较将 ZB-V 的 PP 通信量记为 1F1B 的 2 倍；此外当前实现还要求能把 layers 均匀拆成两个 virtual chunks，并需要补偿首尾 embedding/loss 的不均衡。

## ZB-V 与 Interleaved 1F1B 不是同一种映射

两者都会让一台物理设备持有多个 model chunks，但路径不同。

普通 interleaved 1F1B 常用 cyclic mapping：

```text
w0.c0 -> w1.c0 -> ... -> w(p-1).c0
      -> w0.c1 -> w1.c1 -> ... -> w(p-1).c1
```

它通过缩短 virtual stage 的 compute slot 减少 bubble，并保持统一 backward 也能运行。ZB-V 则是先正向分配再反向分配的 V shape，同时依赖 F/B/W 细粒度调度。

因此：

- interleaving 是 **stage/chunk 粒度与放置方式**；
- B/W split 是 **backward 计算图粒度**；
- ZB-V 是对两者的一种特定组合。

不能看到 `virtual_pipeline_model_parallel_size=2` 就断言系统在运行 ZB-V；还要检查 chunk mapping、B/W 节点以及实际 schedule 表。

## 与 DualPipe 的边界：不是所有“双向”都是 ZB-V

DualPipe 由 DeepSeek-V3 技术报告提出，重点是在 MoE 训练中用两个方向的 micro-batch 流和定制的 forward/backward overlap，更充分地隐藏跨节点通信。官方实现明确要求模型模块提供适配自身的 `overlapped_forward_backward`，说明它不只是把一张通用 F/B/W 表交给普通 runtime。

Zero-Bubble 与 DualPipe 都会讨论 B/W、V shape、首尾空洞与通信覆盖，时间线外观容易混淆，但设计中心不同：

- **Zero-Bubble**：先拆除 W 对跨 stage B 的假依赖，再在同步语义和显存预算内搜索执行顺序；
- **DualPipe**：组织双向流水，并让特定 forward/backward 计算与通信互相覆盖，尤其服务于 DeepSeek 的 MoE 通信结构；
- **DualPipeV**：DeepSeek 官方仓库描述为从 DualPipe 经 cut-in-half 得到的简化 V-shaped schedule，不等同于原始论文中的 ZB-V 名称替换。

还要先对齐符号：DeepSeek DualPipe 文档中的 `B` 表示 full backward chunk，`W` 才单指 backward-for-weights；本文 Zero-Bubble 的 `B` 则只表示 input-gradient backward。两份资料的 bubble 公式即使使用相同字母，也不能逐项直接代换。DualPipeV 与 ZB-V 都使用 V-shaped placement，并不意味着它们拥有相同的节点粒度、依赖约束和 overlap 假设。

选择时不能只比较理论 bubble 公式。需要同时看模型是否允许所需的 chunk 放置、每设备参数份数、PP 通信量、MoE All-to-All、可实现的 kernel overlap 和 runtime 支持矩阵。

## 通信重叠的两种不同含义

“W 填 bubble”首先是一种**跨设备空间—时间重叠**：stage 0 做 W 时，stage 1 可能做 B，stage 2 可能做 F。它并不要求同一 GPU 同时运行 W 和 P2P。

另一种是**同设备并发重叠**：计算 kernel 在 compute stream 上运行，send/recv 或 collective 在通信 stream 上运行。后者要满足：

- 通信 kernel 能及时获得 SM/doorbell 等资源；
- stream priority 和 launch order 不会让大 GEMM 饿死通信；
- buffer 在 send/recv 真正完成前不能复用；
- CUDA event 正确表达 producer/consumer；
- NCCL communicator 的操作顺序在所有 ranks 一致。

Zero-Bubble 官方 generic runtime 文档专门区分 schedule、通信后处理和 runtime，并描述用 pre-communication 减少计算对通信启动的挤压。Megatron Core 也提供 warmup/flush P2P overlap、输出 tensor 伪释放等配置。这些工程手段决定理论空洞是否真的能在 GPU trace 中消失。

## W 与 Data Parallel Gradient Reduce 怎样组合

W 产生参数梯度后，Data Parallel 还需要 AllReduce 或 ReduceScatter。普通反向传播通常按 layer 逆序逐步让 gradient buckets ready；任意移动 W 可能改变 bucket ready 时间。

原论文附录提出：一个 W pass 内包含多个参数的独立 gradient computations，可以把不同 micro-batches 中针对同一参数的计算聚在一起，让该参数的完整累积梯度更早 ready，随后启动 DP communication，与剩余 W 重叠。

实现时必须回答：

1. gradient bucket 的 ready 计数按 layer、parameter 还是 micro-batch 更新；
2. fused gradient accumulation 是否直接写入 `main_grad`；
3. W 重排后，同一 buffer 是否存在并发写；
4. ReduceScatter 启动后是否还有 W 会继续修改该 bucket；
5. optimizer 是否等待所有 DP work handles 完成。

只重排 pipeline nodes、却沿用假设“标准 backward 顺序”的 DDP hook，很容易得到静默错误或无法覆盖的尾部 collective。

## TP、Sequence Parallel 与 Fused Kernel 会改变拆分位置

在线性层启用 Tensor Parallel 后，B/W 周围可能还有 AllGather、AllReduce 或 ReduceScatter。Megatron Core 的 tensor-parallel linear API 已经明确区分 input-gradient 与 weight-gradient，并支持让 input-gradient 通信异步地与 W 计算重叠；这从另一个侧面说明 B/W 在 kernel 层确实是可区分的工作。

但“公式可拆”不代表任意 backend 都可直接插入 hook：

- Transformer Engine 可能把 dgrad、wgrad、bias grad 与 epilogue 融成专用路径；
- Sequence Parallel 要为 dgrad 安排 ReduceScatter，为 W 准备 AllGather 后的输入；
- gradient accumulation fusion 可能让 W 直接累加到预分配 buffer；
- FP8 训练还牵涉 scale/amax 状态和特定 kernel 的保存张量；
- tied embedding 的 W 可能横跨首尾 pipeline stages 做额外同步。

官方 ZB-H1 quick-start 的思路是在 linear backward 时捕获 `total_input`、`grad_output`、`weight` 和 wgrad 函数，放进 `WeightGradStore`，由 scheduler 稍后 `pop`；该轻量分支也明确列出不支持的组合。当前 generic runtime 文档同样列有 backend 限制。生产接入时必须以所用版本的实际支持矩阵为准，不能把研究代码的调度图当成任意 Megatron Core 版本的开关。

## Activation Checkpointing 之后还剩什么可保存

Activation Checkpointing 会丢弃一部分 forward 中间量，在 backward 时重算。B/W split 后，需要更精确地定义重算边界：

- 为 B 重算出的 activation，W 是否也要用；
- B 完成后保留 $X,G_Y$，还是 W 到来时再次重算；
- dropout、attention mask、MoE routing 和 FP8 state 能否确定性重建；
- virtual chunks 之间的 saved tensors 由谁持有和释放。

最朴素方案是在 B 期间完成必要重算，再把 W 所需的最小张量保存到 `M_W` 状态。若为了少存 $M_W$ 又在 W 前重算，就增加了新的 F-like 节点和依赖，原来的 cost model 已不再成立。

所以显存评估不能简单使用：

$$
M_{peak}=\text{number of live microbatches}\times\text{one activation size}
$$

而应按每个 micro-batch 当前位于 F→B、B→W 还是已释放状态，外加 checkpoint workspace、communication buffers、parameter/optimizer shards 和 allocator fragmentation 逐事件模拟。

## 变长序列和 MoE 会让静态 Schedule 失配

对固定 dense Transformer，micro-batches 的 F/B/W 时间可能相对稳定。引入 variable sequence packing 或 MoE 后，`T(i,j,c)` 会随 micro-batch 改变：

- token 数改变 attention 与 MLP FLOPs；
- causal attention 的有效工作量不一定只和 padded length 成正比；
- MoE 每个 expert 收到的 tokens 不均，All-to-All 与 grouped GEMM 都会抖动；
- dropped/rerouted tokens 影响 backward 路径；
- activation checkpoint 重算可能遇到不同 kernel plan。

一个离线 schedule 仍可作为确定性骨架，但需要让数据构造尽量均衡每个 micro-batch 的 token workload，并为通信 timeout 和局部抖动留余量。若 runtime 允许动态改序，还必须保证所有 peers 对 send/recv 次序达成一致，不能让每个 rank 根据自己的局部计时独立决策。

## Optimizer Step 为什么会重新制造尾部同步

即使所有 W 都恰好填满 pipeline，标准 optimizer 前仍常有跨 ranks 同步：

- global gradient norm，用于 gradient clipping；
- FP16/BF16 mixed precision 的全局 NaN/Inf/overflow 标志；
- DP gradient reduction 完成确认；
- tied/shared parameter gradient 同步；
- 所有 stages 是否共同 skip update 的一致决定。

如果每个 stage 都在 iteration 尾部等待一次跨整条 pipeline 的 AllReduce，平行四边形会被截成统一的垂直边；下一轮的首个 F 不能沿 stages 逐步启动，所谓“zero bubble”又会暴露 optimizer synchronization。

保留这道 barrier 最容易保证正确性，也可能已经能获得 B/W split 的大部分收益。只有追求连 iteration 边界都连续时，才需要更复杂的 optimizer post-validation。

## Optimizer Post-Validation 怎样工作

原论文的方案利用一个观察：稳定训练中 NaN/Inf 或 clipping 并非每步都触发。它把“先获得完整 global state，再更新”改成“先基于逐 stage 累积的部分状态推进，下一轮 warmup 再验证”。

简化时间线如下：

```text
iteration k tail:
stage 0 local state -> stage 1 -> ... -> stage p-1
each stage receives prefix state
  if prefix has no NaN/Inf and norm is below threshold:
      perform a provisional optimizer step; mark STEPPED
  else:
      skip the eager step; mark SKIPPED

iteration k+1 warmup:
stage p-1 sends fully reduced state back -> ... -> stage 0
each stage validates its previous action against full global state
  STEPPED and still valid: keep theta_(k+1)
  STEPPED but invalid:     rollback, then apply the globally correct action
  SKIPPED:                 apply the globally correct clipped step, or keep skip on overflow
```

对 overflow flag，“任一 stage 为真”是单调的 OR reduction；对 squared gradient norm，各 stage 的非负局部贡献逐步累加。某个 prefix 已出现 overflow 时，可以确定最终必须 skip；prefix norm 已超阈值时，可以确定最终需要 clipping，但精确 clip coefficient 仍要等完整 norm，因此该 stage 先不做 eager update。反过来，prefix 暂未触发条件并不能证明后续 stages 也安全，这些提前执行的 stages 才是日后可能需要 rollback 的对象。

这套协议仍宣称保持同步优化语义，因为验证会保留正确的 provisional update，撤销被完整状态否决的 update，并让此前跳过的 stage 按完整状态执行正确动作。但它显著扩大了 optimizer 的事务边界：系统不能在验证完成前把 provisional state 当成不可逆提交，也不能对从未执行过 step 的 stage 调用 rollback。

更容易漏掉的是下一轮计算。回滚参数并不会自动撤销已经由错误 provisional 版本产生并发送的 activation。实现必须二选一：把 `validation(i,k) -> F(i,*,k+1)` 设为硬依赖，让本 stage 在首个下一轮 F 前完成验证；或把提前执行的 F/P2P 及其所有下游节点标成 speculative，并在版本被否决时使整条依赖子图失效、丢弃结果后确定性重放。若 runtime 没有后一种版本化重放能力，就必须采用前一种保守边界。

## Rollback 不是简单把 Weight 减回去

以 AdamW 为例，一次 step 同时改变：

- parameter $\theta$；
- first moment $m$；
- second moment $v$；
- optimizer timestamp $t$；
- 可能还有 master weights、loss scale 和 scheduler state。

原论文给出 AdamW 的 in-place arithmetic rollback：根据当前 $m,v,\theta,t$ 与本轮 gradient 逆推出更新前状态，避免额外保存完整历史副本。它是实数代数上的逆公式；浮点舍入、fused update、mixed-precision master state 或不同 kernel 次序都可能让回滚无法逐位恢复。这个技巧依赖 optimizer update 的具体代数形式，并且只有在数值和状态覆盖都经过验证后才安全。论文中的吞吐与确定性实验也不能替代崩溃一致性、反复 rollback 和长时间训练验证。

工程上不应把它泛化为“所有 optimizer 都能回滚”。包含随机操作、非可逆量化、稀疏状态创建、外部 fused optimizer side effect 或参数异步 offload 的实现，可能无法仅靠逆公式恢复。更保守的实现可以保留 pre-step snapshot 或直接保留同步 barrier，以吞吐换简单、可审计的提交语义。

## 参数版本边界必须成为显式状态机

每个 pipeline stage 至少需要跟踪：

```text
iteration_id
parameter_version
microbatch_id
virtual_chunk_id
F/B/W completion bitmap
gradient_reduce state
optimizer state: not_started / provisional / validated / rolled_back
```

建议把可见性规则写成断言：

1. 若 runtime 不支持版本化 speculative replay，`validation(i,k)` 必须先于本 stage 的首个 `F(i,*,k+1)`；
2. 若允许 `F(k+1)` 读取 provisional 参数，每个节点和消息都必须携带版本，并在该版本被否决时连同全部 descendants 一起失效、重放；
3. `B(k)` 不得读取已覆盖且无法恢复的 $\theta_{k+1}$，任意 B 都必须与对应 F 的精确参数版本匹配；
4. `optimizer(i,k)` 只能在本 stage（包括所有 virtual chunks）的全部 `W(k)`，以及该 stage 必需的 DP、tied-gradient reductions 完成后运行；
5. checkpoint、global step、consumed tokens 只在 update 验证后提交；
6. 任一 stage rollback 时，参数、optimizer、activation、通信结果和后继计算必须共同收敛到同一已验证版本。

同步语义不要求所有 GPUs 在同一个纳秒更新，但要求每个 micro-batch 沿整条 pipeline 看到逻辑一致的 iteration version，并且外部系统只能观察到完整提交的 step。

## Failure、Checkpoint 与恢复边界

普通 1F1B-Flush 最自然的 checkpoint 点是 pipeline 已排空、optimizer 已完成之后。Zero-Bubble 若把 iteration $k$ 的尾部与 $k+1$ 的 warmup 重叠，就会出现 provisional optimizer state 和 in-flight micro-batches。

工程上默认可用的 checkpoint 策略是制造明确的 drain point：

- **周期性制造 drain point**：每隔若干 steps 停止注入新 F，完成 post-validation 后保存一致 checkpoint；

理论上也可以设计专用事务 runtime，记录所有 in-flight F/B/W 的不可变输入、参数版本、依赖和可重放描述，在故障时终止未完成 kernel/通信，再从一致状态重建执行图。但普通 checkpoint 无法序列化正在运行的 CUDA kernel、NCCL/P2P 进度或任意 allocator 状态；没有专门的 quiesce、日志和确定性重放协议时，这不是可选的常规生产方案。

因此实践应优先使用 drain point。checkpoint manifest 只能在所有 ranks 确认同一 validated step 后提交；如果进程在 rollback/redo 中间失败，恢复应回到上一个已验证 checkpoint，而不是猜测各 stage 哪些本地更新已经生效。

## Runtime 为什么要把 Schedule 与 Execution 分开

官方 Zero-Bubble 仓库后来引入 generic pipeline runtime，把系统拆为三层：

1. **schedule**：只描述 F、B、W、BW 等 compute nodes 的本地顺序；
2. **post-processing**：根据节点依赖补出 send/recv、必要的 offload 等操作；
3. **runtime**：执行节点、维护 buffer、streams 与通信状态。

这种结构比在一个巨大的 `forward_backward_pipelining` 循环里写大量 rank/micro-batch 分支更容易验证。调度搜索只负责产生合法拓扑序，通信层统一生成匹配操作，runtime 则能为每个节点输出 trace。

一个最小节点不应只有 `type=F`，还需要 stage、micro-batch、layer group/chunk、iteration、peer、tensor metadata 和资源信息。否则 V schedule 或 interleaving 中，同一 physical rank 上的两个 chunks 很容易把 activation 取错。

## 实现 B/W Split 时真正要缓存什么

官方轻量 ZB-H1 实现展示了一种直观办法：在线性层 backward 遇到 wgrad 时，不立刻执行，而是把以下对象放进队列：

```text
total_input
grad_output
weight / target gradient buffer
weight-gradient function
microbatch and layer ownership (production code must make this explicit)
```

调度器执行一个 W 节点时，从队列取出对应任务并调用 wgrad kernel。这个模式的关键不是 `queue`，而是对象所有权：

- input storage 在 W 完成前不能被下一 micro-batch 覆盖；
- grad_output 若来自通信 buffer，recv buffer 不能提前复用；
- weight 指针必须仍指向匹配版本和 shard；
- gradient buffer 的累加 dtype、缩放与顺序必须和基线一致；
- exception/cancel 时队列必须被所有 ranks 一致地清理或回滚。

只捕获 Python closure 却不记录版本和生命周期，通常能跑通短测试，却很难安全支持重计算、offload、CUDA Graph 或异步 allocator。

## 一份可执行的正确性验证矩阵

调度优化上线前，应该先把“数学等价”和“状态机合法”分开验证。

### 算子级：证明拆分后的梯度正确

对 Linear、embedding、attention projection、MoE expert 等支持拆分的模块，固定输入与 output gradient，分别比较：

```text
baseline combined backward: (GX_ref, GW_ref)
split execution: B -> GX_test, later W -> GW_test
```

覆盖 BF16/FP16/FP8、bias、TP/SP placement、gradient accumulation fusion 和非连续 tensor。先在单 rank 验证，排除 pipeline 通信干扰。

### 单 Pipeline：证明一个 Step 等价

固定 initialization、data order、dropout RNG 和 loss scaling，比较 1F1B 与新 schedule 的：

- 每个 micro-batch loss；
- 每个参数最终 gradient；
- gradient norm、clip coefficient、overflow decision；
- optimizer 后参数与 moments；
- global step、LR scheduler 与 consumed tokens。

原论文在其实验实现中报告固定 seed 后 loss 可以 bit-to-bit identical。自己的实现若改变了浮点累加顺序，不应强行承诺普遍 bitwise equality；应先在 deterministic 配置争取逐位一致，再为允许重排的 fused accumulation 制定有依据的绝对/相对误差阈值，并做多步收敛对照。

### 多维并行：证明所有 Shards 同步

依次加入 TP、DP/Distributed Optimizer、Sequence/Context Parallel、EP 与 tied embeddings。每加一个维度都检查 process-group membership、gradient bucket ready 顺序、通信次数和参数 shard checksum，不要一次把所有并行轴同时打开。

### 异常路径：证明事务能收敛

主动注入：

- 某 stage gradient 出现 Inf/NaN；
- global norm 在最后一个 stage 才超过 clip threshold；
- P2P 超时或某 rank 退出；
- provisional update 后、global validation 前进程崩溃；
- rollback 或 redo 期间保存 checkpoint 的请求。

验收目标不是“不报错”，而是所有 ranks 要么提交同一参数版本，要么共同回到可恢复边界，不能出现一部分 stage 已进入 $k+1$、另一部分仍停在 $k$。

## 用事件 Trace 验证 DAG，而不是只看彩色 Timeline

每个事件至少记录：

```text
iteration, microbatch, physical_rank, pipeline_stage, virtual_chunk
node_type(F/B/W), parameter_version
ready_ts, issue_ts, gpu_start_ts, gpu_end_ts
memory_before, memory_after
peer, send_recv_id, tensor_shape, dtype
grad_bucket_id, optimizer_validation_state
```

离线检查器可以验证：

- 所有跨 stage F/B 边都满足先后关系；
- 每个 F 恰有匹配的 B/W，没有重复或遗漏；
- W 完成前 optimizer 不启动；
- send/recv ID 在 peers 上一一对应；
- 任意时刻模拟 live memory 不超过预算；
- 每个 micro-batch 的 parameter version 沿 stages 一致。

Nsight Systems 的彩色条能显示 idle gap，却不会自动告诉你某条 W 是否写错 gradient buffer。schedule event log 与 GPU trace 必须一起看。

## 性能验证要分解收益来源

建议至少报告以下指标，而不是只给一个 samples/s：

1. step time 与 tokens/s；
2. 理论 bubble rate、trace 实测 idle rate；
3. P2P 与 DP/TP/EP communication exposed time；
4. 每 stage 的 F/B/W P50、P95；
5. waiting-for-B、waiting-for-W 的峰值数量；
6. activation、communication buffer、optimizer 的分类峰值显存；
7. micro-batch size 与 local GEMM efficiency；
8. post-validation 正常路径开销与 rollback 触发成本；
9. schedule profiling/search 时间以及配置变化后的重生成成本。

比较 1F1B、interleaved 1F1B、ZB1P/ZB2P/ZBV 时，要保持 global batch、有效 token 数、模型/optimizer 和精度策略一致。若 ZB2P 因显存不足而把 micro-batch size 减半，就应同时说明更多 micro-batches 带来的 bubble 收益与较小 GEMM 的效率损失。

原论文中的吞吐提升来自特定 GPT-like 模型、A100 和网络配置，只能证明方法在那些实验中有效，不能直接当成其他模型和集群的容量承诺。

## 哪些场景最可能从 Zero-Bubble 获益

Zero-Bubble 更适合：

- PP degree 较大，1F1B 的 warmup/cooldown 占比仍明显；
- 模型和 global batch 允许足够的 micro-batches；
- stage 切分已经较均衡，主要空闲确实来自依赖链；
- W 在时间线上有可利用的移动空间；
- backend 能可靠拆分 dgrad/wgrad；
- activation memory 能支持目标 schedule，或可使用 ZB-V；
- 跨节点 PP 通信量相对可控，不会因更多 chunks 抵消收益。

如果 profile 显示 GPU 大部分时间在最慢 MoE stage、输入不均、数据加载或 TP collective 上等待，先换 Zero-Bubble 通常治标不治本。

## 哪些情况下保留 1F1B 更合理

以下条件会让传统 1F1B 更有吸引力：

- PP bubble 已被大量 micro-batches 摊得很小；
- local micro-batch 太小，再增大 $m$ 会让 GEMM 利用率下降；
- activation 显存已经极紧，目标实现又不支持合适的 V/offload 方案；
- 使用的 fused/FP8/MoE backend 无法安全拆分 W；
- 变长序列或专家负载导致 profile 高度不稳定；
- gradient clipping/overflow 经常触发，post-validation rollback 失去“异常稀少”的前提；
- 运维更重视简单 checkpoint、故障恢复与确定性调试。

Zero-Bubble 是 schedule 设计空间的一部分，不是所有 PP 作业的默认正确答案。理论上的零空洞如果换来更小 micro-batch、更多 P2P、复杂回滚和更脆弱的 backend 组合，端到端 goodput 反而可能降低。

## 一条稳妥的落地顺序

1. **固定一个可复现的 1F1B 基线**：保存 loss、grad、optimizer state 和 per-stage trace；
2. **只拆一个 Linear 的 B/W**：验证公式、dtype、buffer ownership；
3. **扩展到完整 Transformer layer**：覆盖 attention、MLP、norm、embedding；
4. **先实现类似 ZB-H1/ZB1P 的显存预算**：保留 iteration 尾部同步；
5. **建立 schedule event validator**：自动检查 DAG、通信匹配和版本；
6. **加入实测 cost model**：按目标 shape/profile 生成确定性 schedule；
7. **再接 TP/DP/SP/EP 和 recomputation**：逐维做数值对照；
8. **比较 interleaved 与 V mapping**：测 PP 流量和 activation，不只看 bubble；
9. **最后评估 post-validation**：先故障注入和 rollback，再考虑跨 step 连续执行；
10. **把 validated step 定为 checkpoint 边界**：不要持久化 provisional 状态；
11. **在真实 token 分布下压测**：同时观察吞吐、显存、尾部延迟与失败恢复。

这条路径把最危险的 optimizer 事务优化放在最后。仅靠 B/W split 与保守 step barrier，往往已经能判断该模型是否值得继续投入。

## 常见误区

### “W 不在关键路径，所以可以跨很多 Step 再算”

W 只是不阻塞上游 B；它仍是本轮 gradient 的组成部分，必须在本轮 optimizer 消费梯度前完成。

### “B 完成就能释放这份 Micro-batch 的所有 Activation”

B 后通常仍要保留 W 所需的输入和 output gradient，即 $M_W$ 状态。只有 W 完成后才能完全释放。

### “ZB-H2 的名字里有 Zero，所以现实里必然没有 Idle”

H2 的零 bubble 结论依赖 $T_F=T_B=T_W$ 等理想条件。真实配置需要 profile-aware 搜索，通信和 stage imbalance 仍会产生空洞。

### “ZB-V 就是把 Virtual Pipeline Size 设为 2”

ZB-V 还要求 V-shaped chunk placement、对应的 F/B/W schedule 与通信序列。普通 cyclic interleaving 不是同一算法。

### “Optimizer Post-Validation 只是把 AllReduce 挪到后台”

它改变了 update 的提交协议：先 provisional step，之后验证，异常时 rollback/redo。没有完整状态机和故障恢复设计，不能只移动一个 collective 调用。

### “论文报告 Bitwise Identical，所以所有实现都应该逐位一致”

论文报告的是其固定配置与实现结果。不同 fused kernels、累加顺序和并行组合可能造成合法的浮点差异；仍须建立逐算子、逐梯度和多步收敛验证。

## 小结

Zero-Bubble Pipeline Parallelism 的核心可以压缩成一句话：**把 backward 中必须尽快跨 stage 传播的 input gradient 与只需在 optimizer 前完成的 weight gradient 分开调度。**

从这个拆分出发，整套系统形成了四层设计：

1. **计算图层**：F、B、W 的真实依赖决定哪些重排数学上合法；
2. **调度层**：ZB-H1/H2、ZB1P/2P 和 ZB-V 在 bubble、显存、chunks 与通信之间选择不同点；
3. **运行时层**：saved tensors、gradient buffers、P2P、CUDA streams 与 backend 支持决定计划能否执行；
4. **一致性层**：optimizer、global norm、overflow、rollback、checkpoint 与参数版本决定它是否仍是同步训练。

延后 W 并没有减少训练 FLOPs，而是把原本串在关键路径上的计算移动到空闲槽。它之所以可能接近零气泡，不是因为依赖消失了，而是因为调度器终于看见了此前被统一 backward API 隐藏的并行性。

下一步若继续研究更复杂的 pipeline schedule，应把 Zero-Bubble 与 interleaved 1F1B、DualPipe 放在同一套评价框架里：先验证参数版本和梯度语义，再比较 makespan、分类显存、通信暴露与真实 token 分布下的 goodput。只有这样，“时间线更满”才会转化为可靠的训练效率。

## 参考资料

- [Zero Bubble Pipeline Parallelism（arXiv 扩展版，包含 ZB-H1/H2、自动调度与 ZB-V）](https://arxiv.org/abs/2401.10241)
- [Zero Bubble (Almost) Pipeline Parallelism（ICLR 2024）](https://openreview.net/forum?id=tuzTN0eIO5)
- [Zero-Bubble Pipeline Parallelism 官方实现](https://github.com/sail-sg/zero-bubble-pipeline-parallelism)
- [官方 ZB-H1 Quick Implementation](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/tree/zb-h1-quick-start)
- [官方 Generic Pipeline Parallel Runtime 设计](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/blob/main/docs/zero-bubble/pipeline_parallel_runtime.md)
- [Megatron Core Pipeline Parallel Schedules](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.pipeline_parallel.schedules.html)
- [Megatron Core ModelParallelConfig](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.model_parallel_config.html)
- [Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek DualPipe 官方实现](https://github.com/deepseek-ai/DualPipe)
