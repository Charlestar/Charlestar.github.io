---
layout: post
title: "FP4 Attention：矩阵乘快了以后，概率、Scale 与数据交接为什么成了瓶颈"
subtitle: "沿 Direct-P 的概率编码和 Blackwell TMEM 生命周期，区分非因果前向、单次更新与长期训练的证据"
date: 2026-09-30
last_modified_at: 2026-09-30
author: iStar
catalog: true
series: attention-long-context
series_order: 80
technology_year: 2026
mathjax: true
tags: [注意力机制, 模型量化, GPU优化]
---

如果 Attention 里的两次矩阵乘变快四倍，整个 Attention 会变快四倍吗？问题不只在于还有多少普通指令，更在于第二次矩阵乘的输入尚未存在：第一次计算得到的 score，必须经过归约、概率转换、量化、打包和发布，才能成为下一次矩阵乘可消费的 operand。

2026 年 9 月 3 日公开的 [Hardware-Aware FP4 FlashAttention-4](https://arxiv.org/abs/2609.04105v1)把这段中间路径作为研究对象。它是 Robert Hu 基于 HAO 实现与 FA4 执行结构开展的独立预印本，不是原 [FlashAttention-4 论文](https://arxiv.org/abs/2603.05451v1)团队发布同一工作的新版。

本文聚焦 v1 的 Direct-P、TMEM 数据所有权，以及 causal training 的精度边界。论文最重要的结果不只是某个更大的吞吐数字：其非因果前向使用 MXFP4 概率与 V，而保留下来的训练路线反而使用 FP8 P/V。两者回答不同问题，不能统称为“FP4 Attention 已经无损训练”。

## P 既是一个数值结果，也是下一条指令等待的输入

先把 Attention 拆成三段：

$$
Z=\frac{QK^T}{\sqrt d},\qquad
P=\operatorname{softmax}(Z),\qquad
O=PV
$$

QK 和 PV 可以交给 Tensor Core，softmax 却包括求最大值、指数与归约等不同操作。[FlashAttention]({% post_url 2026-03-17-FlashAttention原理与实现详解 %})将它们分块执行，不把完整的平方规模 Z、P 写回 HBM，但依赖关系不会因此消失。

这里要分清两个常被混用的量：P 是归一化概率；kernel 内部常先使用相对于某个行参考 m 的未归一化权重 $$p_j=\exp(z_j-m)$$，累计分母和加权分子，最后相除：

$$
O=\frac{\sum_jp_jV_j}{\sum_jp_j}
$$

当 online softmax 更新行参考时，旧分子与分母需要一致地重缩放。不能把每个 tile 各自归一化成和为 1，再把 tile 输出相加；那会让每个 tile 获得相同总权重，而不再是整行 softmax。

低精度路径还要求把这个中间权重变成打包 payload 与 block scales。只把 QK/PV 换成 FP4 指令，其余路径仍可能串在两个矩阵乘之间。论文的 [§3.1](https://arxiv.org/html/2609.04105v1#S3.SS1)强调：优化发生在“第一个合法 P tile 发布之前”，才可能让 PV 更早开始。

用一个不含重叠的时间模型看上限：原耗时为 1，矩阵乘占 0.6，其他工作占 0.4；矩阵乘快四倍后，总时间仍为 $$0.4+0.6/4=0.55$$，只快约 1.82 倍。若额外量化又增加 0.1，总时间成为 0.65，只快约 1.54 倍。这是人为串行算例，不是 Blackwell benchmark；真实异步流水要分析关键路径，不能机械累加各段时间。

## 四个 bit 只描述 payload，Scale 仍要保存、搬运与消费

本研究使用的 E2M1 payload，其非负可表示值为：

$$
\mathcal F=\{0,0.5,1,1.5,2,3,4,6\}
$$

仅靠这些数无法覆盖模型中不同张量的范围，需要块级缩放。普通重构可以理解为 $$\widehat x=sq$$，q 来自有限集合，s 决定这一块的数值网格放在哪里。

论文中 NVFP4 Q/K 使用 16 元素块的 E4M3 scale；Direct-P 的 P/V 使用 32 元素块的 E8M0、即二次幂 scale。它们不仅决定精度，也决定 scale 页怎样对应矩阵指令的读取布局。Payload 占 4 bit，不等于含 scale、对齐与布局后的总存储恰好只有 4 bit/元素。[§3.2](https://arxiv.org/html/2609.04105v1#S3.SS2)给出了该实现中的具体数值约定。

概率对范围尤其敏感。一个值小，不代表整片小值的概率质量可以忽略；如果很多小概率共同承担一大部分分母，全部舍入为零会改变整行权重。若 block scale 自身先下溢到零，则不是几个 payload 消失，而是整块消失。

选择更宽范围的 scale 也不等于免费提高精度。二次幂缩放便于构造与硬件消费，但只保留少数 payload 档位；更细的 scale 可能减少误差，却增加稳定化、逆缩放和数据交接的成本。这里的选择是范围、精度与执行开销之间的共同设计。

## Direct-P：最终只剩八种结果，还需要先算精确指数吗

常规路径先计算较高精度的指数，再把它量化到 E2M1。Direct-P 反过来问：如果下一条指令最终只看到八种非负 payload，能不能直接判断 score 应落入哪一档？

该论文采用一个需要特别说明的幅度约定。设块幅度 $$\alpha=2^{k_B}$$，其实际重构权重为：

$$
\widetilde p=\frac{\alpha q}{6}
$$

因此有效步长是 $$\delta=\alpha/6$$。存储的 E8M0 scale 编码的是 α，不是 δ；硬件缩放乘法实际消费 αq。P、V 两边都采用这一约定时，乘积的输出修正还要除以 36。这里的 /6、/36 来自[论文 §4.2 的编码约定](https://arxiv.org/html/2609.04105v1#S4.SS2)，不是所有 MXFP4 实现自带的格式公理。

理想 payload 应是对下面的量进行 E2M1 舍入：

$$
x=(z-m)\log_2\mathrm e-k_B+\log_2 6,
\qquad q=Q_{E2M1}(2^x)
$$

这里 $$\mathrm e$$ 是自然常数，$$k_B$$ 是块 scale 的指数。这个变换只是把 $$\exp(z-m)/\delta$$ 写成二次幂形式。

正值相邻档位的中点是 0.25、0.75、1.25、1.75、2.5、3.5、5。理想转换取决于 x 落在哪些对数阈值之间，而不要求保留指数函数的全部尾数。

作者用低成本的 affine 近似配合硬件转换器来提高编码一致率，通用 fast 配置的中间值为：

$$
\widehat u(x)=\max(0,1.5x+1.2),
\qquad\widehat q=Q_{E2M1}(\widehat u(x))
$$

这不是精确 softmax 的代数变形。x=0 时，精确指数为 1，近似值为 1.2，两者都舍入到 1；x=1 时，精确值为 2，近似值为 2.7，后者会舍入到 3。算例避开了中点，不涉及 tie-breaking。更准确的实数近似也未必更快：新增依赖指令可能恰好拖延 P 的发布。

Direct-P 因而是一条可调的近似概率路径，而非把指数“无误差地删掉”。它还在部分位置保留硬件指数，并对特定极端 logit 模型层使用数值保护；不能把一个 affine 式子当作所有模型、范围和 mask 条件下可直接替换 softmax 的完整实现。

## 分子与分母必须描述同一份被表示出来的概率

即使 payload 都正确生成，还有一个更隐蔽的问题：PV 使用量化后的权重，归一化分母却可能来自原始指数。两者相除会得到一个内部不一致的算子。

设重构后的 V 为 $$\widehat V$$。Direct-P 让两边使用同一组 represented weights：

$$
\widetilde O
=\frac{\sum_j\widetilde p_j\widehat V_j}
{\sum_j\widetilde p_j}
$$

对于不同 block，需要连同各自 α 一起累加；不能只把 payload q 求和而忽略块 scale。[§4.3](https://arxiv.org/html/2609.04105v1#S4.SS3)的 normalizer 正是由 PV 消费的 payload 与 scale 产生。

看一个原创两元素反例。未归一化权重为 `[1, 0.4]`，设 α=1，乘 6 后为 `[6, 2.4]`，按最近档位取 payload `[6, 2]`，重构为 `[1, 1/3]`。若 V 两个值都为 2：

$$
\frac{2+2/3}{1+1/3}=2
$$

这是归一化加权平均应有的性质：所有 V 相同，结果应保持这个值。但若分子已经量化、分母仍使用 1.4，会得到：

$$
\frac{2+2/3}{1.4}=\frac{40}{21}\approx1.90476
$$

一致性修复并不恢复丢失精度。仍用相同权重，把 V 改成 `[0, 6]`：精确结果是 $$12/7\approx1.71429$$，一致量化结果是 1.5，错配分母则是 $$10/7\approx1.42857$$。前者说明近似仍然存在，后者说明还额外引入了不一致；这两类错误不应混为一谈。

上述例子直接量化已知权重，只解释重构与归一化约定，不复现 affine converter。真实 kernel 还要保证 masked key 的权重为零，正确处理零 scale、全 mask 行与非有限输入，不能在分母为零时继续套用这个除法。

数值上等价的表达式也可能有不同中间范围。论文在特殊保护路径中记录过：先做 scale/6，再乘 payload 和，会先产生可被 flush-to-zero 的 subnormal；改成先对 payload 和除 6，再乘 scale，避免了该失效。这是[§4.4 的特定边界](https://arxiv.org/html/2609.04105v1#S4.SS4)，不是浮点乘法可以随意重排的证明。

该保护路径还会把抽样的 K/V 行一起移到前部，尽早获得参考值。对无其他位置相关约束的非因果 Attention，K 的列次序和 V 的对应行同步置换，精确加权和不变；但因果 Attention 的可见性依赖原始位置，不能只重排 K/V、保留原来的三角 mask 就照搬。布局优化一旦越过算子边界，首先要重新核对语义，而不是只问指针能否按新顺序读取。

## 为什么概率已经压小，下一块 QK 仍然没有地方写

数值路径只解决“生产什么”。要让流水真正前进，还要知道“生产到哪里，以及谁仍在读取”。

Blackwell 的 TMEM 是矩阵操作使用的片上存储。[NVIDIA PTX 的 SM100 说明](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)描述了 128 行、512 列、每格 32 bit 的结构，并按列分配，而不是一个可任意按字节切割的普通数组。其容量计数为 $$128\times512\times4=256\ \mathrm{KiB}$$。

在论文沿用的 D128 双 query-stage 布局中，两块 FP32 score bank 加两块 FP32 output accumulator，各占 128 列，正好用完 512 列。读完一段 score 后，这些地址再被 P payload 与 scales 覆盖，随后交给 PV 消费：

```text
同一 score bank 的生命周期：
QK 写 score → 读取 score 片段 → 覆盖为 P/scale → PV 消费 → 可再次写 QK
```

如果 PV 还没读完就发出下一次 QK，新的 score 会覆盖仍被需要的 P。增加一个 barrier 可以报告“已经消费”，却不能凭空产生第五块合法的 score bank。FP4 payload 更小，也不自动缩小 FP32 score/output 的分配列数。[§2.4 与 §8.1](https://arxiv.org/html/2609.04105v1#S2.SS4)把它定位为存储所有权限制，而不只是算术不够快。

另一个边界来自生产与消费粒度。一个 N128 score tile 分成四个 N32 片段处理，但本文路线的 scaled-FP4 PV 指令一次需要 K64。因此第一片完成还不能发出有用的 PV，要等相邻两片：

```text
N32 片段 0 + 片段 1 → 第一半 K64 PV
N32 片段 2 + 片段 3 → 第二半 K64 PV
```

分成四片降低 softmax 的 live register 规模，并允许第一半提前发布，不表示 QK 被拆成了四次小矩阵乘，也没有减少 score bank 的 TMEM 大小。另一 query stage 仍可交错执行，所以这个局部生命周期图不意味着整个 CTA 串行运行。

这解释了一个重要诊断原则：Tensor Core 活跃度低，可能是 operand 尚未准备好或目标地址仍被占用，而不是矩阵工作量太少。只减少指令数、提高理论 FP4 峰值，未必改变关键路径。性能模型要同时画出数据依赖与资源 ownership，不能只写两条 GEMM 的 FLOP 数。

## 前向用 FP4，不等于训练也能沿用同一份精度约定

这篇论文有两条应分开记忆的路线：

| 边界 | Direct-P 非因果前向 | 保留的 causal training 路线 |
| --- | --- | --- |
| QK 输入 | NVFP4 Q/K | NVFP4 Q/K |
| P/V | MXFP4 | FP8 |
| 关键机制 | 直接生成概率 payload，归一化 represented P | 保存量化 Q/K、scales 与 LSE，反向重建概率 |
| 证据范围 | 算子计时、输出误差、固定输入模型替换 | 反向、含投影子层、完整更新、长期轨迹分别测量 |

所谓 full FP4 在该文中仅指 Attention 内 Q、K、P、V 的 payload，不表示 score 与输出 accumulator 也是 FP4，更不是整个 Transformer 或 optimizer 都采用 FP4。投影权重与 activation 又是另一组精度选择。

反向需要 Attention 概率，但保存完整 P 会重新引入平方规模状态。作者保存前向实际使用的量化 Q/K、block/global scales 与 log-sum-exp，在 backward 中重建 score 和概率。这比另建一条独立 BF16 score 路径更贴近前向的 represented operands；但“从相同量化状态重建”不等于恢复原始未量化 Q/K，也不意味着离散舍入函数突然拥有了精确的常规梯度。

梯度矩阵乘还需要不同物理方向的 Q/K/V 等视图。论文让投影或梯度的生产者在结果仍活跃时发布适合消费者的 FP8 布局，避免每层再启动独立转置、量化 kernel。其 dO 路径选择 E5M2 而非把所有梯度统一成 E4M3，是对范围需求的单独处理。[§7.1–§7.2](https://arxiv.org/html/2609.04105v1#S7.SS1)给出了这些精度与布局约定。

最值得保留的负面结果是：论文所有被测试的 MXFP4 P/V 训练轨迹都发生了发散，因此最终训练路线选择 FP8 P/V。这是对已测试方法、模型和训练配置的观察，不是数学上证明所有 FP4 Attention 永远无法训练；但也不能省略它，只留下“FP4 前向更快”。

## 把计时边界扩大，收益会留下多少

第一层是非因果 forward kernel。论文在 GB200 的九个 D128 shape 上报告 Direct-P fast 相对 HAO BF16 的几何平均加速 2.023 倍，但其平均 relative L2 约为 0.3366，不能称作无损。Q/K/V 在计时前已经量化，部分 scale 整理与 K/V 排列也在边界外，因此不等于完整 Attention layer 快两倍。[§5.3 与 §6.1](https://arxiv.org/html/2609.04105v1#S5.SS3)明确了这些口径。

第二层是反向及其直接生产者。一个很有说服力的例子来自 B1/S4096/D128：BF16 backward 为 0.501 ms，保存 Q/K 的重建核心为 0.356 ms；但加上 E5M2 dO 与统计量 publisher 后，变成 0.508 ms。独立核心节省的时间，几乎全被相邻准备工作消耗了。继续把投影、RoPE、布局发布和消费者纳入同一测量，才知道融合是否真正保留收益。

第三层是完整模型更新。单 GB200、8B 模型、S4096、local batch 4 的 FP8 P/V 路线，从相邻 BF16 基线的 854.516 ms 降到 751.722 ms，约 1.137 倍。这包含低精度 QKV/O 投影、loss、backward 与 optimizer，不是 Attention 单项贡献。该实验只有固定 synthetic tokens 的 10 次预热与 21 次测量更新，不能用于证明收敛；而且该性能实验的初始 logits 相对 BF16 已有明显差异，并非只是尚未来得及证明质量一致。MXFP4 路线另有自己的 BF16 计时基线，也不能跨列拼一个更好的倍数。[§7.3–§7.5](https://arxiv.org/html/2609.04105v1#S7.SS3)把这三种边界分别列出。

第四层才是长期训练。匹配实验使用 64 张 GPU，local batch 4，梯度累积 4，effective global batch 为 $$64\times4\times4=1024$$。BF16 与保留的 FP8 P/V 路线各完成约 100B tokens，但最后共同验证点 loss 分别为 2.3048、2.3948，相差 0.0900。稳定下降不是质量相同；路线同时改变了投影与 Attention，差距也不能单独归因于 P/V。每条路线只有一次训练轨迹，不能据此估计跨运行方差或宣称统计等价。[§7.7](https://arxiv.org/html/2609.04105v1#S7.SS7)

## 判断低精度优化，先问它保留了哪种一致性

数值检查至少要同时看绝对/相对误差与任务指标。对于非零向量 v，$$\cos(v,2v)=1$$，但将 2v 对照 v 的 relative L2 为 1；单看 cosine 会完全遗漏幅度翻倍。算子误差也不等于任务 loss，残差、归一化和后续层可能削弱或放大它，因此固定输入替换、短更新与长期训练不能互相替代。

执行检查则要对齐量化输入、scale layout、mask、累加精度和计时范围。不满足输入约定的 kernel 即使跑得很快，也不构成有效性能结果；用预先量化输入测得的收益，要进一步检查上游能否直接产生所需布局，以及新增临时张量和 launch 是否吃掉收益。

例如本文 fast 路线要求相邻 Q/K scale 已按 K64 消费方式折叠，普通 block-16 scale 页不能不经转换就传进去。两个张量都是“NVFP4”，并不足以证明它们满足相同的操作数约定。Scale 的分组方向、排列和任何 global factor，都属于计算定义；只检查 dtype 与 shape，会遗漏这类能够产生有限但错误结果的接口错配。

横向比较也要固定实验身份。同一个 shape 在 GB200、B300、GB300 上出现不同吞吐，可能同时改变硬件、实现、二进制与测量脚本。论文将部分外部结果标为跨运行背景，不能把两列数字直接相除就归因于某个优化。先获得同一输入、同一精度目标与同一计时边界下的基线，才有资格解释剩余差异。

本文没有运行这些 GPU 路线。作者截至 2026 年 9 月 3 日的[公开发布状态](https://github.com/MrHuff/fp4-fa4/blob/main/RELEASE_STATUS.md)也明确区分了源码与离线材料完整性、fresh-clone 的全部 GPU 验证：后者尚未全面完成，部分历史数据顺序、checkpoint 和原始记录不可重新获取。论文报告值得分析，但不应被包装为本文已独立复现、拿来即可生产部署的结论。

配套 CPU 算例可复算编码档位、分母错配、误差指标与简化资源计数：

```sh
node scripts/reviews/sep30-fp4-attention-examples.mjs
```

它不执行 Direct-P 硬件 converter，不模拟 TMEM 并发，也不验证训练质量。这个方向真正展示的趋势，是低精度优化开始把表示、布局和生产者/消费者一起设计：少几个 bit 只是起点，真正的收益取决于那些 bit 能否准时到达正确的指令，以及整条数值链是否仍在计算一个可接受的目标。
