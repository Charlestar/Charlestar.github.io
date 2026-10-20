---
layout: post
title: "Mamba-2 与 SSD：循环状态怎样变成分块矩阵乘"
subtitle: "从一次状态更新展开因果矩阵，再推导块内计算、边界状态与固定大小推理缓存"
date: 2026-10-21
last_modified_at: 2026-10-21
author: iStar
catalog: true
series: attention-long-context
series_order: 55
technology_year: 2024
mathjax: true
tags: [注意力机制, 机器学习, GPU优化]
---

自回归 Attention 把过去的 Key 和 Value 留在缓存里，新 token 到来时再查询这些历史。另一条路线是不保留逐 token 的历史表示，而是把过去持续汇总进一个固定大小的状态。这样做容易得到便宜的单步更新，却又带来一个训练问题：如果必须逐 token 更新，长序列怎样利用 GPU 擅长的大块矩阵乘？

Mamba-2 的核心 SSD（Structured State Space Duality，结构化状态空间对偶）把这两个视角连接起来：同一层既可以写成状态递推，也可以写成带特殊因果结构的矩阵乘；再沿序列分块，把主要工作组织为块内矩阵计算，只让压缩后的状态在块边界之间传递。

这一方法来自 2024 年 5 月 31 日提交的[原始论文](https://arxiv.org/abs/2405.21060)。下面围绕其核心算法推导，源码固定为官方 v2.2.2 对应的提交 [`8ffd905c`](https://github.com/state-spaces/mamba/tree/8ffd905c91d207f5c0cc84fc2a2fb748655094f0)。CPU 算例检验三种计算形式的等价性，不代表执行了官方 CUDA/Triton kernel、训练了模型或测量了速度与任务质量。

## 先定义一个 Head 的状态更新

暂时去掉 batch、多个 head 及外围投影，只考虑一个 head。令输入 $$x_t\in\mathbb{R}^{P}$$，写入向量 $$b_t\in\mathbb{R}^{N}$$，读取向量 $$c_t\in\mathbb{R}^{N}$$，状态 $$H_t\in\mathbb{R}^{N\times P}$$。核心更新定义为：

$$
H_t=a_tH_{t-1}+b_tx_t^\top,
\qquad
y_t=H_t^\top c_t\in\mathbb{R}^{P}.
$$

这里的 $$a_t$$ 是一个标量。它对这个 head 的所有状态维和通道共享，衰减旧状态；外积 $$b_tx_t^\top$$ 把当前输入写入状态；最后用 $$c_t$$ 从新状态读取输出。当前写入项不额外乘 $$a_t$$：衰减作用于上一步留下的状态。

例如 $$N=P=1$$、$$H_{-1}=0$$，两个时间步分别为 $$a=(0.5,0.25)$$、$$b=(2,1)$$、$$c=(1,3)$$、$$x=(4,2)$$。则 $$H_0=8$$、$$y_0=8$$；下一步 $$H_1=0.25\times8+1\times2=4$$、$$y_1=12$$。第二步输出中，第一步的输入贡献了 6，当前输入贡献了另一个 6。

这不是任意 SSM 都具有的形式。一般状态转移可以是矩阵，Mamba-1 的选择性 SSM 使用更一般的对角结构；SSD 进一步让同 head 的状态转移成为标量乘单位阵。这个结构限制，正是后面把衰减从 $$c_t^\top b_s$$ 内提取出来、组织矩阵计算的条件。

实际 Mamba-2 中，$$a_t,b_t,c_t$$ 由输入相关的投影与处理产生；“选择性”体现在这些更新参数可以随 token 改变。因此，即使在零初态、给定系数时能把 $$x$$ 到 $$y$$ 写成线性算子，也不能据此说整个神经网络是线性模型。固定的非零初态还会额外贡献与输入无关的一项，此时该核心对输入是仿射映射。

## 把递推展开成因果矩阵

先令 $$H_{-1}=0$$。连续代入递推式，每次写入都会受到它**之后**的衰减：

$$
H_t=\sum_{s=0}^{t}\left(\prod_{j=s+1}^{t}a_j\right)b_sx_s^\top.
$$

约定空乘积为 1，所以当前项 $$s=t$$ 的系数为 1。再代入读取式：

$$
y_t=\sum_{s=0}^{t}
\underbrace{\left(c_t^\top b_s\right)
\left(\prod_{j=s+1}^{t}a_j\right)}_{M_{t,s}}
x_s.
$$

把各时刻的 $$x_t^\top$$、$$y_t^\top$$ 堆叠为 $$X,Y\in\mathbb{R}^{T\times P}$$，就得到：

$$
Y=MX,
\qquad
M_{t,s}=\begin{cases}
(c_t^\top b_s)\displaystyle\prod_{j=s+1}^{t}a_j,&s\le t,\\
0,&s>t.
\end{cases}
$$

矩阵严格遵守因果可见性：第 $$t$$ 行只能读取 $$s\le t$$。衰减的下标从 $$s+1$$ 开始，而不是 $$s$$，否则会错误地让新写入在同一步被额外衰减。

若初始状态非零，还应加上：

$$
y_t^{\mathrm{initial}}=
\left(\prod_{j=0}^{t}a_j\right)H_{-1}^\top c_t.
$$

这项在分块算法中不能丢，因为每个块接收到的通常正是一个非零初态。把独立小块全都当成零初态计算后直接拼接，只保留了块内作用，遗漏了历史跨块传递。

## 与 Attention 的联系不经过 Softmax

令 $$B,C\in\mathbb{R}^{T\times N}$$ 分别堆叠 $$b_t^\top,c_t^\top$$，再定义含衰减的下三角 mask：

$$
L_{t,s}=\begin{cases}
\displaystyle\prod_{j=s+1}^{t}a_j,&s\le t,\\
0,&s>t.
\end{cases}
\qquad
M=L\odot(CB^\top).
$$

其中 $$\odot$$ 是逐元素乘积。把 $$C$$ 看作 Query、$$B$$ 看作 Key、$$X$$ 看作 Value，这就具有未归一化的 structured masked attention 形式。若所有 $$a_t=1$$，它退化为因果、未归一化的 linear attention 核心。

这里没有 softmax，没有要求权重非负，也没有要求一行权重和为 1。一般的 $$b,c$$ 可以令 $$c_t^\top b_s$$ 为负；读取输出也可能相互抵消。因此，SSD 的对偶不是“现成 softmax Attention 去掉 softmax 后功能不变”，更不是任意已训练 Transformer 可以不经训练就换成该递推。

[论文第 4–5 节](https://arxiv.org/html/2405.21060v1#S4)建立的是特定结构的 SSM 与结构化 masked attention 之间的联系。对偶首先给出同一个核心算子的不同计算方式，再帮助设计模型；模型是否具有需要的检索、记忆或任务能力，仍然取决于架构、状态维度与训练。

## 低秩的是跨时间分割的部分

如果 $$N$$ 很小，是否意味着整个 $$M$$ 都是低秩矩阵？不是。取 $$N=1$$、所有 $$b_t=c_t=1$$、$$a_t=0.5$$，则 $$M$$ 是对角线全为 1 的下三角矩阵，行列式为 1，任意序列长度下都可以满秩。

真正有用的结构出现在任意一个时间分割处。固定分割点 $$k$$，只看过去输入 $$s\le k$$ 对未来输出 $$t>k$$ 的作用：

$$
M_{t,s}=
\left[
\left(\prod_{j=k+1}^{t}a_j\right)c_t
\right]^\top
\left[
\left(\prod_{j=s+1}^{k}a_j\right)b_s
\right].
$$

左侧因子只依赖未来的 $$t$$，右侧因子只依赖过去的 $$s$$，中间维度为 $$N$$，所以这个跨分割子矩阵的秩至多为 $$N$$。对衰减 mask $$L$$ 本身，对应中间维度只有 1，这就是这里 1-semiseparable 结构的直观含义。

这也解释了固定状态为什么足够：过去对未来的影响可以通过 $$N\times P$$ 的状态传递，不必携带每个过去 token 的 $$x,b$$。它压缩了未来读取历史的通道，但不是完整保存所有历史明细的无损档案。

## 把一段序列拆成块内作用和初态作用

取一个半开区间 $$[l,r)$$ 作为 chunk。令 $$H_{l-1}$$ 是进入本块的状态，对块内每个 $$t$$ 定义：

$$
d_t=\prod_{j=l}^{t}a_j.
$$

输出分成两部分：

$$
y_t=
\underbrace{\sum_{s=l}^{t}
(c_t^\top b_s)\left(\prod_{j=s+1}^{t}a_j\right)x_s}_{y_t^{\mathrm{local}}}
\; + \;
\underbrace{d_tH_{l-1}^\top c_t}_{y_t^{\mathrm{carry}}}.
$$

第一部分假设本块从零状态开始，只涉及块内 token，可以用一个小的下三角矩阵乘。第二部分只需把进入本块的状态读出来，并施加从块首到当前位置的衰减。注意 $$d_l=a_l$$：旧状态在块内第一个时间步就已经受到一次衰减。

块的最终状态也可以分开：

$$
S_{[l,r)}=\sum_{s=l}^{r-1}
\left(\prod_{j=s+1}^{r-1}a_j\right)b_sx_s^\top,
\qquad
\alpha_{[l,r)}=\prod_{j=l}^{r-1}a_j,
$$

$$
H_{r-1}=\alpha_{[l,r)}H_{l-1}+S_{[l,r)}.
$$

于是每个块只需交出两样东西：一个标量衰减 $$\alpha$$，以及它在零初态下积累的状态 $$S$$。所有块的 $$y^{\mathrm{local}}$$ 和 $$S$$ 都可以先独立计算；等边界状态确定后，再把 $$y^{\mathrm{carry}}$$ 加回去。

## 块边界递推为什么适合 Scan

把块看成状态上的仿射变换 $$F(H)=\alpha H+S$$。相邻两块的复合为：

$$
(\alpha_2,S_2)\circ(\alpha_1,S_1)
=\left(\alpha_2\alpha_1,\ \alpha_2S_1+S_2\right).
$$

这种复合满足结合律，因此可以使用并行 prefix scan；但它一般不满足交换律，块顺序不能随意改变。顺序扫描也正确，只是并行深度不同。重要的是，边界扫描的序列长度从 $$T$$ 缩短为约 $$T/Q$$，其中 $$Q$$ 是 chunk 长度。

[作者的算法解释](https://tridao.me/blog/2024/mamba2-part3-algorithm/)把完整过程分成四步：块内输出、零初态块末状态、边界状态传递、初态对输出的贡献。块内和状态读写可以组织成批量矩阵乘，利用 GPU 的矩阵计算单元；只有边界联系需要处理跨块依赖。

对单 head 核心，块内矩阵构造及其乘输入的工作量为 $$O(TQ(N+P))$$，状态汇总和读出为 $$O(TNP)$$。对于 $$T\ge1$$，令块数 $$K=\lceil T/Q\rceil$$，边界顺序 scan 为 $$O(KNP)$$；在 $$1\le Q\le T$$ 时可简写为 $$O((T/Q)NP)$$，而 $$Q>T$$ 仍有一个块。当 $$Q,N,P$$ 固定时，对序列长度 $$T$$ 是线性的；取 $$Q=N=P$$，主要量级为 $$O(TN^2)$$。这不包含外围投影、卷积、归一化、反向保存和实际 kernel 调度成本。

一个实现细节非常容易被忽略：官方 [`ssd_minimal.py`](https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/mamba_ssm/modules/ssd_minimal.py)为了简洁，在第 3 步显式构造 chunk 数量上的二维衰减矩阵，再做 `einsum`。若有 $$K$$ 个 chunk，这一部分包含 $$K^2$$ 的规模，长序列时不再是真正线性。作者也明确说明，优化后的实现将其替换为实际 scan。不能因为教学代码行数很少，就把理想算法的复杂度直接赋给其中每一条执行路径。

## 用三种 CPU 计算核对同一个算子

下面依次实现逐 token 递推、完整因果矩阵、分块状态传递。采用上文 $$N\times P$$ 的状态方向，显式传入初态；官方缓存采用的最后两维方向不同，不能直接按内存布局混用。三种实现共享的只有输入检查和简单点积，不让矩阵实现调用递推充当“独立参考”。

先定义输入范围和直接递推。为同时检查一般代数与数值边界，教学接口允许 $$a_t\in[0,1]$$；实际有限连续参数生成的衰减通常在开区间内。Gate 为 0 是检验不能通过累计乘积相除的重要边界。

<!-- example: mamba2-ssd-recurrence -->
```python
import math
from numbers import Real


def finite_real(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("expected a finite real number")
    try:
        valid = math.isfinite(value)
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise ValueError("expected a finite real number")


def check_core(a, b, c, x, initial):
    if not initial or not initial[0]:
        raise ValueError("state dimensions must be positive")
    n, p, length = len(initial), len(initial[0]), len(a)
    if not (len(b) == len(c) == len(x) == length):
        raise ValueError("sequence lengths differ")
    for matrix, width in [(initial, p), (b, n), (c, n), (x, p)]:
        for row in matrix:
            if len(row) != width:
                raise ValueError("inconsistent shape")
            for value in row:
                finite_real(value)
    for gate in a:
        finite_real(gate)
        if not 0 <= gate <= 1:
            raise ValueError("gate must lie in [0, 1]")
    return length, n, p


def dot(left, right):
    return sum(u * v for u, v in zip(left, right))


def recurrent(a, b, c, x, initial):
    length, n, p = check_core(a, b, c, x, initial)
    state = [row[:] for row in initial]
    outputs = []
    for t in range(length):
        state = [[a[t] * state[i][j] + b[t][i] * x[t][j]
                  for j in range(p)] for i in range(n)]
        outputs.append([sum(c[t][i] * state[i][j] for i in range(n))
                        for j in range(p)])
    return outputs, state
```

第二种直接构造 $$M$$。对于每行 $$t$$，从当前 token 逆向更新衰减系数，避免针对每一对位置反复计算整个乘积。最终状态也从展开式独立求出。

<!-- example: mamba2-ssd-dense -->
```python
def dense(a, b, c, x, initial):
    length, n, p = check_core(a, b, c, x, initial)
    mixing = []
    for t in range(length):
        row, decay = [0] * length, 1
        for s in range(t, -1, -1):
            row[s] = dot(c[t], b[s]) * decay
            decay *= a[s]
        mixing.append(row)
    outputs, initial_decay = [], 1
    for t in range(length):
        initial_decay *= a[t]
        outputs.append([
            sum(mixing[t][s] * x[s][j] for s in range(length))
            + initial_decay * sum(c[t][i] * initial[i][j] for i in range(n))
            for j in range(p)
        ])
    state = [[0] * p for _ in range(n)]
    decay = 1
    for s in range(length - 1, -1, -1):
        for i in range(n):
            for j in range(p):
                state[i][j] += decay * b[s][i] * x[s][j]
        decay *= a[s]
    state = [[state[i][j] + decay * initial[i][j]
              for j in range(p)] for i in range(n)]
    return outputs, state
```

第三种只在每个 chunk 内处理位置对，再通过边界状态联系各块。为清楚呈现依赖，边界按序传递；这不是并行 GPU kernel。它也接受最后一块不足 $$Q$$ 的情况，不要求序列长度能被 chunk size 整除。

<!-- example: mamba2-ssd-chunked -->
```python
def chunked(a, b, c, x, initial, chunk_size):
    length, n, p = check_core(a, b, c, x, initial)
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    state = [row[:] for row in initial]
    outputs = []
    for left in range(0, length, chunk_size):
        right = min(left + chunk_size, length)
        before = state
        # 1. Outputs within this chunk, assuming zero incoming state.
        local = [[0] * p for _ in range(right - left)]
        for t in range(left, right):
            decay = 1
            for s in range(t, left - 1, -1):
                weight = decay * dot(c[t], b[s])
                for j in range(p):
                    local[t - left][j] += weight * x[s][j]
                decay *= a[s]
        # 2. Zero-initial-state contribution at the chunk's end.
        summary = [[0] * p for _ in range(n)]
        alpha = 1
        for s in range(right - 1, left - 1, -1):
            for i in range(n):
                for j in range(p):
                    summary[i][j] += alpha * b[s][i] * x[s][j]
            alpha *= a[s]
        # 3. Pass the actual boundary state to the next chunk.
        state = [[alpha * before[i][j] + summary[i][j]
                  for j in range(p)] for i in range(n)]
        # 4. Add this chunk's incoming-state contribution to its outputs.
        prefix_decay = 1
        for t in range(left, right):
            prefix_decay *= a[t]
            outputs.append([
                local[t - left][j]
                + prefix_decay * sum(c[t][i] * before[i][j] for i in range(n))
                for j in range(p)
            ])
    return outputs, state
```

把三个代码块保存在同一个 Python 文件中，追加下例即可运行。使用有理数时，几种运算都保持精确值，可以直接比较全部输出与最终状态；浮点对照还要求输入及运算中间量处于 Python 浮点的有限范围，仅检查输入有限不能排除乘加溢出。实际浮点实现应按 dtype 和误差范围比较，而不是要求改变运算顺序后逐 bit 一致。

<!-- example: mamba2-ssd-example -->
```python
from fractions import Fraction as F

a = [F(1, 2), F(1, 4), F(0), F(1), F(3, 4)]
b = [[F(2), F(-1)], [F(1), F(1)], [F(-1), F(2)],
     [F(2), F(1)], [F(1), F(-2)]]
c = [[F(1), F(0)], [F(2), F(-1)], [F(1), F(1)],
     [F(-1), F(2)], [F(1), F(3)]]
x = [[F(4), F(1)], [F(2), F(-1)], [F(1), F(3)],
     [F(-2), F(1)], [F(0), F(2)]]
initial = [[F(1), F(2)], [F(-1), F(0)]]
reference = recurrent(a, b, c, x, initial)
assert dense(a, b, c, x, initial) == reference
for size in [1, 2, 3, 5, 8]:
    assert chunked(a, b, c, x, initial, size) == reference
```

例子特意包含非零初态、负的 $$b,c,x$$、零衰减、单位衰减和尾块。若使用“前缀累计乘积之比”计算区间衰减，遇到零会出现除零，长序列的小衰减还可能让前缀先下溢，再出现 $$0/0$$。这里直接递推区间乘积，不做这种相除。

官方实现通常在 log 衰减空间计算，并对 `segsum` 做了避免大前缀和相减的处理。改成 log-space 并不自动消除误差：两个很大的累计和相减，也可能丢失小区间信息。[固定版本源码](https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/mamba_ssm/modules/ssd_minimal.py)同时保留了 `segsum_unstable` 与更稳定的实现，可以直接比较它们的运算结构。

## 把离散核心放回完整 Mamba-2 Block

上面从离散的 $$a,b,c,x$$ 开始，还没有解释它们如何由模型输入产生。固定版本 [`mamba2.py`](https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/mamba_ssm/modules/mamba2.py)的 `step` 路径给出了一个可直接核对的对应：连续参数 $$A_{\mathrm{cont},h}=-\exp(A_{\log,h})<0$$，输入相关步长经 softplus 得到 $$\Delta_{t,h}>0$$，然后计算：

$$
a_{t,h}=\exp(\Delta_{t,h}A_{\mathrm{cont},h}),
\qquad
H_{t,h}=a_{t,h}H_{t-1,h}
+\Delta_{t,h}B_{t,h}x_{t,h}^\top.
$$

所以本文的 $$b_t$$ 可以理解为已经吸收步长的 $$\Delta_tB_t$$。官方最小离散例子则把 $$\Delta_t$$ 吸收到 $$X$$ 中：传入 `X=x*dt`、`A=A_cont*dt`，再与原始 `B` 相乘。两种放置对该写入项等价，但**不能同时给 B 和 X 都乘一次 dt**。其中最小函数参数名 `A` 表示 log decay，而不是本文已经取过指数的 $$a_t$$。

完整 block 还包括输入投影、用于生成/处理相关通道的 causal convolution 与 SiLU、$$D\odot x$$ 的 skip 项，以及 gating、RMSNorm 和输出投影；具体归一化与 gate 的组合由实现配置决定。只实现 $$H_t$$ 的更新并不能复现完整层的输出，也不能直接替换已有权重对应的官方 forward。

在常见配置下，多个 head 各自维护状态，$$B,C$$ 还可以按 group 在若干 head 之间共享。共享投影减少了参数或投影数据量，不代表这些 head 的 $$H$$ 被合并成一份：它们仍然接收各自的输入通道并维护状态。

## 固定状态怎样改变推理和恢复

单 head 的核心递推每一步需要 $$O(NP)$$ 工作，持久状态有 $$NP$$ 个元素，大小不随已经生成的长度 $$T$$ 增长。但模型还需要短卷积缓存。固定源码中的 `allocate_inference_cache` 分别分配约为 `(batch, heads, P, N)` 的 SSM 状态，以及 `(batch, conv_channels, conv_width)` 的卷积状态；传入的 `max_seqlen` 不决定这两个 tensor 的长度维度。

例如只计算一个假设配置的这两类状态：batch 为 1，32 层，每层 32 个 head，$$P=64$$、$$N=128$$，两种状态都用 2 字节元素。SSM 部分为：

$$
1\times32\times32\times64\times128\times2
=16\ \mathrm{MiB}.
$$

若每层一个 $$B,C$$ group，短卷积通道数为 $$32\times64+2\times128=2304$$，卷积宽度为 4，则卷积缓存另需 0.5625 MiB，合计 16.5625 MiB。这个数字只对应声明的配置，不包含模型权重、临时 buffer、其他层、训练激活和反向状态，也没有与某个 Transformer 建立质量等价比较。

状态固定不等于可以无损记住无限历史，也不等于可以像逐 token KV 一样随意删去尾部再获得正确的旧状态。若服务需要前缀复用，应在可恢复边界保存相应的 SSM **和卷积**状态；若推测解码需要撤销未接受的 token，也应恢复这些状态，或者从保留的边界重算。只改一个“当前 token 长度”计数，不会撤销已经写入的状态。

混合 Attention/SSM 模型仍有 Attention 层的 KV Cache。容量规划必须逐类计算，不能把纯 SSD 单层的常量状态结论推广成“整个混合模型不需要随长度增长的缓存”。

## 性能收益来自计算组织，而不只是复杂度符号

同一个 SSD 核心可以按 token 递推、完整矩阵、分块矩阵加状态传递三种方式计算。递推节省持久状态，完整矩阵暴露并行性却引入 $$T^2$$ 规模，分块则利用跨分割结构，把大部分工作放回适合硬件的矩阵计算。

这解释了算法为什么有获得速度收益的可能，却没有决定任意设备上的实际快慢。Chunk size 过小可能增加边界和 launch 开销，过大则增加块内二次工作与中间存储；状态维、head dimension、dtype、batch、序列长度、融合程度和数值实现都改变取舍。

测量时应区分核心 SSD kernel、完整 block、模型 prefill 和单步 decode。CPU 上三种公式相等，只证明这个离散算例实现了同一个算子，不能证明 GPU 路径的速度、反向梯度或语言模型质量。真正值得验证的是：在固定模型和输入条件下，哪种计算组织降低了目标阶段的时间，哪些状态和数值约束仍然被完整保留。
