---
layout: post
title: "Roofline：从计算量、数据流量到可信的 GPU 性能判断"
subtitle: "推导算术强度与两类资源上限，再用 GEMM、缓存层级和串行阶段解释性能测量的边界"
date: 2026-09-09
last_modified_at: 2026-09-09
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 5
technology_year: 2009
mathjax: true
tags: [GPU优化, 工程实践]
---

同一份大权重矩阵，乘一个 token 的向量和乘几百个 token 的矩阵，单个 token 的成本可能差很多。权重没变，主要运算也还是乘加，区别来自数据被多少计算复用。

Roofline 把这个问题压缩成两个账本：要执行多少运算，要搬运多少字节。它不预测每个 kernel 的精确时间，而是给出资源上限，帮助判断下一步该减少搬运、增加复用，还是改善计算执行。

模型的计时范围必须先明确：一次 GEMM、一个带依赖的阶段和一个完整请求不是同一个对象。本文从单个范围开始推导，最后再把它放回完整时间线。所有数值例子均为假设下的计算，不是某款 GPU 的 benchmark；2009 对应 Roofline 代表论文的发表年份，同名技术报告在 2008 年已经出现。

## 两本账怎样推导出一条屋顶

设一次明确范围内完成 $$F$$ 个浮点运算，跨某个存储边界搬运 $$Q$$ 字节，实际耗时为 $$t$$ 秒。定义：

$$
P=\frac Ft,
\qquad
I=\frac FQ
$$

$$P$$ 是达到的计算速率，单位为 FLOP/s；$$I$$ 是算术强度，单位为 FLOP/byte。FLOP 是操作计数，FLOP/s 才是速率。本文将一次 fused multiply-add 按乘法与加法合计 2 FLOP 计数。

若所选计算路径的上限为 $$C$$ FLOP/s，所选存储边界的带宽上限为 $$B$$ byte/s，那么无论怎样调度，都至少要支付这两类资源时间：

$$
t\ge\frac FC,
\qquad
t\ge\frac QB
$$

两者共同给出：

$$
t\ge\max\left(\frac FC,\frac QB\right)
$$

这里取最大值，是允许计算与搬运充分重叠时的理想资源下界，不是说实现中两段时间一定完全重叠。指令依赖、同步、发射与启动等额外限制，只会让这个简单下界不够紧。

把时间不等式转换成吞吐上界：

$$
P\le\min(C,BI)
$$

水平线 $$C$$ 与斜线 $$BI$$ 取较小值，就形成了“屋顶”。它们的交点称为 ridge point：

$$
I^*=\frac CB
$$

```text
P (FLOP/s, log scale)
^
|                 +-------------------  C
|               / |
|             /   |
|           /     |     P <= min(C, B I)
|         /       |
|       /         |
+-----------------+-------------------> I (FLOP/byte, log scale)
                  I* = C / B
```

在双对数坐标中，带宽线斜率为 1；左侧增加一倍算术强度，带宽给出的吞吐上限也增加一倍。越过交点以后，继续增加复用不能把吞吐推过所选计算上限。

最初的 [Roofline 文档 §3](https://digital.library.unt.edu/ark:/67531/metadc934195/m1/1/)明确把流量放在缓存与 DRAM 之间，并关注可持续带宽。今天使用“算术强度”这个名称时，也应保留这条关键约束：必须先说清楚字节究竟跨越哪一层存储边界。

## 给一次 GEMM 逐项记账

考虑：

$$
Y_{M\times N}=X_{M\times K}W_{K\times N}
$$

共有 $$MN$$ 个输出，每个输出需要沿 $$K$$ 维乘加。按前述 FMA 口径，主要计算量为：

$$
F=2MKN
$$

假设三个张量均为两字节元素，输出直接覆盖写入，也就是一般 GEMM 中的 $$\beta=0$$。再作一个明确的冷缓存模型假设：输入在计时前未驻留更近的缓存，各输入元素只从外存读一次，各输出元素只写一次。于是：

$$
Q_{model}=2(MK+KN+MN)
$$

三项分别是读输入、读权重、写输出。[NVIDIA 的矩阵乘法指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)用相同的运算与字节关系解释 GEMM 的数学和访存限制。

这个账本有严格边界。若 $$\beta\ne0$$，还要读旧输出，增加 $$2MN$$ 字节；不同输入输出 dtype、bias、epilogue、scale 和 workspace 也会改变公式。真实分块可能重复读输入或产生额外事务；暖缓存又可能让一部分输入不经过 HBM。因而 $$Q_{model}$$ 不是任意运行条件下的实测 HBM 流量。

现在固定 $$K=N=4096$$，并构造一台**假想设备**：匹配该数值路径的计算上限为 120 TFLOP/s，HBM 带宽上限为 2 TB/s。这里的 T 使用十进制 $$10^{12}$$，不是具体硬件规格或测量结果。

这台假想设备的交点为 $$I^*=60\ \mathrm{FLOP/byte}$$。矩阵账本化简为：

$$
F=33{,}554{,}432M
$$

$$
Q_{model}=33{,}554{,}432+16{,}384M
$$

$$
I_{model}=\frac{2048M}{2048+M}
$$

代入三种行数：

| $$M$$ | 强度，FLOP/byte | 计算下界 $$F/C$$ | 搬运下界 $$Q/B$$ | 两者最大值 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.9995 | 0.280 μs | 16.785 μs | 16.785 μs |
| 32 | 31.5077 | 8.948 μs | 17.039 μs | 17.039 μs |
| 128 | 120.4706 | 35.791 μs | 17.826 μs | 35.791 μs |

当 $$M$$ 较小时，固定的权重读取占据大部分流量。更多输入行复用同一份权重，$$F$$ 随 $$M$$ 增长，$$Q$$ 却只小幅增长，算术强度因此上升。在这套假设里，前两行是带宽给出较低吞吐上限，第三行则换成计算上限。

注意第三行的整个矩阵乘时间下界比第一行更长，只是平均每行成本更低。增加 batch 可以提升吞吐，不代表每个等待凑批的请求都具有更低延迟。这里的 $$M$$ 还是“这次 GEMM 的行数”，未必等于请求数：prefill 可把多个 token 展成行，decode、分块与 MoE 路由又会形成不同 shape。

这张表解释了权重复用，不能据此宣布整个 prefill 必然计算受限、整个 decode 必然带宽受限。它们还包含 attention、norm、采样、通信与主机调度，每一段可能面对另一种约束。

## 在屋顶左边，不等于带宽已经跑满

假设某个点的强度只有 1 FLOP/byte。在上述假想设备上，它的带宽上限是 2 TFLOP/s，比 120 TFLOP/s 的计算上限低得多。但这只说明两个候选上限中哪条更低，不证明真实 HBM 正在以 2 TB/s 工作。

如果工作很小、可并行访问不足，GPU 可能连足够多的 memory requests 都没有发出；如果地址依赖串行，下一次读取要等前一次结果；如果 load 不合并，同样有效数据会产生更多事务。低算术强度与低带宽利用率完全可以同时出现。

屋顶右边也一样。理论计算量很多，不代表 Tensor Core 持续繁忙：shape 或布局可能走另一条指令路径，tile 尾部有无效工作，最后一波 thread blocks 可能填不满设备。更大的 tile 往往提高复用，却减少可并行 tile 数，并增加寄存器或 shared memory 使用。这是复用与并行度之间的权衡，不是“tile 越大越好”。

[NVIDIA GPU 性能背景](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)区分了数学、内存与延迟相关限制。诊断时也应保留这个层次：位置告诉我们潜在限制，离屋顶的距离才引出“实现为何未达到它”的问题。

`GPU utilization` 的忙碌比例、occupancy 和计算峰值利用率也不是同一个数。GPU 持续有 kernel 执行，可以很忙却只有少量有用计算；[occupancy](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html#occupancy)是 SM 上活跃 warp 数相对硬件允许最大活跃 warp 数的比例。这里“活跃”不等于当下能发射指令，更不保证流水线没有依赖等待；理论可达值和运行时采样的平均值也需分开。它们可以作为解释证据，不能代替耗时和资源速率。

## 算术强度必须带着存储层级

一个值可能从 HBM 进入 L2 一次，却被多个 thread blocks 从 L2 多次读取。于是同一 kernel 的 $$Q_{HBM}$$ 与 $$Q_{L2}$$ 不同，定义的强度也不同：

$$
I_{HBM}=\frac F{Q_{HBM}},
\qquad
I_{L2}=\frac F{Q_{L2}}
$$

如果分别建立对应层级的带宽上限，可以得到分层约束：

$$
P\le\min
\left(C,B_{HBM}I_{HBM},B_{L2}I_{L2},\ldots\right)
$$

每条线都必须使用同一层级的流量与带宽；用 HBM 强度乘 L2 带宽，会制造一个没有对应物理边界的数字。[NERSC 的 Roofline 说明](https://docs.nersc.gov/tools/performance/roofline/)把这种 hierarchical Roofline 作为区分不同数据搬运限制的方法。

还要区分算法需要的逻辑字节与实际事务字节。一次只使用很少元素但跨越许多 cache lines 的访问，逻辑流量很小，实际搬运却可能很多。反过来，重复运行同一个小矩阵时，权重留在 L2，HBM 计数可能低于冷缓存张量账本。若忽略这一点，就容易把缓存命中解释成“超过物理带宽”。

这也解释了三类常见优化为什么要重新画点，而不是只看点向上移动。Tiling 改变局部复用；fusion 可能省去中间张量的写回和再读取；[FlashAttention]({% post_url 2026-03-17-FlashAttention原理与实现详解 %})改变 attention 中间状态的物化方式。它们都可能改变 $$Q$$，有时还改变 $$F$$，所以优化前后未必处在同一横坐标。

减少 HBM 流量以后，新的瓶颈还可能转移到 L2、shared memory、寄存器依赖或计算执行。原来的带宽优化成功，不意味着继续沿同一方向就仍然有效；需要重新确认当前限制来自哪一级。

## 计算量减少了，FLOP/s 反而可能下降

选择水平屋顶之前，必须回答这里的 $$C$$ 属于哪种指令路径。FP32 CUDA Core、BF16 Tensor Core、FP8、稠密与结构化稀疏运算有不同峰值；INT8 的操作吞吐也不能直接当作任意 FP32 工作的 FLOP/s 上限。累加精度与有效工作量口径同样需要匹配。

再区分两种 $$F$$：完成问题所需的有用算法运算，和硬件实际执行的运算。Padding、重复计算、转换和某些算法改写会让两者不同。按有用运算报告 throughput，适合比较完成同一个问题的效率；按实际执行运算分析指令利用率，适合解释硬件在做什么，但不能在一张图里悄悄切换分子。

设某实现执行 100 单位工作、耗时 10 单位，速率为 10；优化后只需 40 单位工作、耗时 5 单位，速率变为 8。它虽然 FLOP/s 更低，实际已经快了一倍。更高吞吐可能来自做了更多工作，而不一定来自更快完成目标，这也是 [Time-Based Roofline](https://arxiv.org/html/2009.04598v1)所强调的分析动机。

对量化尤其需要重算账本。[FP8]({% post_url 2026-08-01-fp8-formats-scaling-llm-inference %})可能同时改变字节与可用计算路径；[W4A8]({% post_url 2026-08-04-w4a8-progressive-quantization-kernel %})还要支付解包、scale 和格式转换成本。权重保存成 4 bit，并不能让所有工作都落在一个名为“4-bit 峰值”的水平线上。

因此性能比较先固定可接受的结果精度、shape、输出语义和统计范围，再比较耗时。若优化改变了算法近似或模型质量，应把这个变化单列出来，不能只在速度图上把它当成同一个问题的等价实现。

## 先取得可信的时间，再计算坐标

CUDA 提交通常是异步的。CPU 上围住一次 kernel launch 的计时，可能主要测到提交耗时；GPU 仍在执行时，CPU 的计时已经结束。若要测 CPU 观察到的完成时间，需要明确同步边界；若要测 GPU stream 上的区间，可以使用 CUDA events。[CUDA 12.8 Best Practices 的计时章节](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html#timing)说明了两类计时的差别。

单 stream 的概念顺序是：

```text
预热与准备输入
→ 在目标 stream 记录 start event
→ 提交需要测量的工作
→ 在同一 stream 记录 stop event
→ 等待 stop 完成
→ 读取两个 event 之间的时间
```

这个区间不自动包含其他 stream 上没有依赖汇合的工作，也不一定只是各 kernel 的纯执行时间之和：两事件之间的依赖等待、主机未及时提交形成的空隙，也可能在范围内。计时工具不会替代对执行范围的定义。

常规稳态测量应预热库初始化、编译和 autotune，并预先准备输入与必要 buffer；如果研究的是冷启动，就应有意保留这些开销，使用另一个清楚的指标名称。多次采样观察分布，避免只选最快一次掩盖抖动，也不要把编译首轮混入稳定阶段后再只报平均值。

数值正确性检查应先于性能计时，并覆盖与优化相关的维度、尾部 shape 和数据条件。容差需要符合 dtype 与算法误差预期；一组随机输入通过不等于整个定义域已经证明正确。本文没有执行 GPU benchmark，因此这些步骤是测量方法，不是已经取得的性能结果。

## 分析工具提供计数，也会改变运行条件

获得正常运行的时间线后，先选择占关键路径、且能代表实际 shape 的 kernel，再采集资源计数。可用 $$F,Q,t$$ 重建 Roofline 点，也可以检查实测带宽 $$Q/t$$ 与有效计算速率 $$F/t$$，看当前资源是否真的接近对应上限。

工具指标必须回到定义：统计的是哪类浮点指令，包含哪些 tensor operations；字节经过 HBM、L2 还是另一个接口；是否包含读写与无效事务。不同架构与工具版本的指标名称可能不同，应查询本机支持的 sections、sets 与 metrics，而不是把一条旧 counter 名当作通用公式。

[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)在 2026-09-09 查阅时显示为 13.3 文档，说明了 replay、缓存控制和序列化等行为。采集一组完整指标可能需要多次重放；分析时进程运行很久，不等于目标 kernel 在正常运行中也需要同样时间。

更重要的是实验条件会改变。Replay 可能在不同 pass 前清理缓存；序列化可能消除原有 kernel 并发；时钟控制与温度状态也会影响可用吞吐。分析得到的冷缓存计数，不能随手除以正常暖缓存运行的时间，拼成一个看似精确、实则不存在的性能点。

关闭清缓存也不是自动更正确。如果问题是稳态复用，应该匹配并报告暖缓存条件；如果问题是独立请求的冷数据成本，则需要另一种范围。正常运行、计数运行与经验上限测量，至少应在 shape、工作量、缓存状态与硬件条件上可比较。

理论峰值与经验屋顶也要区分。经验屋顶来自特定 microbenchmark 的可持续结果，通常比数据手册更有解释力，但不是不可越过的物理定律。一个 kernel 超过某条经验线，可能来自不同条件或更有效的数据路径；应先核对口径。所有带宽比值还要统一单位：GB 为 $$10^9$$ byte，GiB 为 $$2^{30}$$ byte，不能把两者不加换算地相除。

## 汇总全程序，为什么会丢掉真正的等待

继续使用前面的**假想 120 TFLOP/s、2 TB/s 设备**。构造两个必须严格串行的阶段：B 依赖 A 完成，不能让一个阶段的计算隐藏另一个阶段的访存。下表的 MB、GFLOP 均采用十进制：

| 阶段 | 运算量 | 流量 | 计算下界 | 搬运下界 | 阶段下界 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.12 GFLOP | 200 MB | 1 μs | 100 μs | 100 μs |
| B | 12 GFLOP | 2 MB | 100 μs | 1 μs | 100 μs |

按阶段保持依赖，端到端资源下界是：

$$
t_{serial}\ge
\max\left(\frac{F_A}{C},\frac{Q_A}{B}\right)
+\max\left(\frac{F_B}{C},\frac{Q_B}{B}\right)
=200\ \mathrm{\mu s}
$$

若先把两本账各自求和，再只画一个总量 Roofline，就得到：

$$
t_{aggregate}\ge
\max\left(
\frac{12.12\ \mathrm{GFLOP}}{120\ \mathrm{TFLOP/s}},
\frac{202\ \mathrm{MB}}{2\ \mathrm{TB/s}}
\right)
=101\ \mathrm{\mu s}
$$

101 μs 仍是合法下界，但比保留阶段顺序的 200 μs 松得多，不能解释成可达到的端到端时间。汇总把第一阶段闲置的算力与第二阶段闲置的带宽混在一起，却没有保留“它们发生在不同时间”的约束。

真实程序未必全部串行；独立 streams、异步通信和流水线可以重叠。正确做法不是一律相加，也不是一律取最大值，而是从时间线和依赖图确定关键路径，对能重叠的工作再施加资源限制。

这也说明 [CUDA Graph]({% post_url 2026-07-29-cuda-graph-llm-serving-static-replay %})为何可能让端到端变快，却没有让某个 GEMM 的点明显移动：它主要减少主机提交与执行间隙。Fusion 则可能同时减少间隙和中间流量，改变 kernel 边界本身。单 kernel 的屋顶只是完整性能图景中的一部分。

## 让模型提出问题，让测量作出判断

一条有用的 Roofline 结论应当能够还原成具体证据：在什么工作范围、什么精度和缓存条件下，多少运算对应多少实际流量，耗时多少，离匹配的计算或带宽参考还有多远。

如果屋顶较低是因为复用少，可以尝试 batching、tiling 或减少物化；如果点离屋顶很远，应继续检查并行度、依赖、指令路径与启动开销；如果局部已经很好而端到端仍慢，就返回时间线寻找通信、调度和串行等待。每次优化都重新计算其改变的工作量与流量，再验证同一目标下的时间收益。

Roofline 最有用的地方，不是把程序贴上“计算受限”或“带宽受限”的标签，而是把一个笼统的“GPU 为什么不快”拆成可核算的问题。计算量给出工作，字节流量给出搬运，执行依赖给出顺序；这三者放在同一口径下，性能判断才真正可解释。

博客仓库的 `node scripts/reviews/new-roofline-examples.mjs` 可以复算 GEMM 表、分层带宽约束、串行阶段的 200 μs 与汇总的 101 μs，以及计算速率下降但耗时缩短的例子。它只使用标准库验证算术与模型关系，不执行 CUDA、读取硬件计数器或测量实际 GPU 时间。

## 参考资料

- [Roofline 原作者文档：模型定义与 DRAM 流量口径](https://digital.library.unt.edu/ark:/67531/metadc934195/m1/1/)
- [NVIDIA: Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [NVIDIA: GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
- [NERSC: Roofline Performance Model](https://docs.nersc.gov/tools/performance/roofline/)
- [CUDA C++ Best Practices Guide 12.8.0](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Time-Based Roofline for Deep Learning Performance Analysis](https://arxiv.org/abs/2009.04598)
