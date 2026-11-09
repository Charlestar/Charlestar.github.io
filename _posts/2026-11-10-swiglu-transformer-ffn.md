---
layout: post
title: "SwiGLU：三次投影的门控结构与反向传播"
subtitle: "从逐 token 的通道变换推导 SwiGLU 前后向，区分参数预算、逐元素融合与 Tensor Parallel 配对"
date: 2026-11-10
last_modified_at: 2026-11-10
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 30
technology_year: 2020
mathjax: true
tags: [GPU优化, 机器学习]
---

Transformer 的 attention 在不同位置之间交换信息，FFN 则对每个位置的通道表示做变换。各位置共享同一组 FFN 权重，但这次计算本身不把不同 token 混合起来；它收到的输入，已经可以包含前面 attention 汇总的上下文。

SwiGLU 改变了 FFN 的内部结构：不再只把一次上投影送入激活函数，而是生成两路中间表示，让其中一路经过 SiLU 后调制另一路，再投影回模型宽度。额外的一路投影带来乘法交互，也改变了参数量、反向依赖与 kernel 的中间存储。

理解这些关系，才能判断一个 `fused_swiglu` 究竟融合了什么，为什么常见中间宽度接近模型宽度的 $$8/3$$，以及 gate/up 两路在并行切分时为何必须保持配对。

## 从两层 FFN 到两路投影相乘

把输入 token 展成矩阵 $$X\in\mathbb R^{N\times d}$$，其中 $$N$$ 是本次计算的 token 行数，$$d$$ 是模型宽度。先考虑没有 bias 的普通 FFN：

$$
Y=\phi(XW_1)W_2,
\qquad
W_1\in\mathbb R^{d\times h_0},\quad
W_2\in\mathbb R^{h_0\times d}
$$

第一层把通道扩展到 $$h_0$$，中间施加非线性，第二层映射回 $$d$$。SwiGLU 则使用三份权重：

$$
A=XW_g,\qquad B=XW_u,
\qquad
W_g,W_u\in\mathbb R^{d\times h}
$$

$$
S=\operatorname{SiLU}(A),\qquad
U=S\odot B,\qquad
Y=UW_d,
\qquad W_d\in\mathbb R^{h\times d}
$$

这里 $$\odot$$ 是逐元素乘法。$$A,B,S,U$$ 都是 $$N\times h$$；同一行、同一中间通道的两个值相乘，再由 down projection 汇总回输出通道。Gate、up、down 描述的是三份投影的角色，并不规定每个通道必须学习某种人工命名的语义。

该结构来自 [GLU Variants §2](https://arxiv.org/html/2002.05202v1#S2)。本文使用其中 Swish 参数为 1 的形式，也就是 SiLU：

$$
\sigma(a)=\frac1{1+e^{-a}},
\qquad
s(a)=\operatorname{SiLU}(a)=a\sigma(a)
$$

PyTorch `Linear.weight` 的存储形状与上面的数学记号互为转置：gate/up 权重存为 $$[h,d]$$，down 存为 $$[d,h]$$。阅读 [Transformers v4.46.3 的 LlamaMLP](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py)时，要把 `linear(x)` 对应到 $$XW$$，不能拿存储形状直接代入另一套矩阵方向。

## 门控不是概率，也不是把 GELU 换成 SiLU

Sigmoid 输出在 0 与 1 之间，但真正乘到 $$B$$ 上的是 $$s(A)=A\sigma(A)$$。当 $$A<0$$ 时它可以为负；当 $$A$$ 足够大时，它可以大于 1。因此这个 gate 不是概率、不是二值开关，也没有 softmax 归一化。

例如：

$$
s(1)\approx0.731059,\qquad
s(-1)\approx-0.268941,\qquad
s(2)\approx1.761594
$$

它可以改变另一路表示的大小和符号。这里的门控也不同于 MoE router 选择哪些 experts：一个 dense SwiGLU FFN 仍然计算这些中间通道，不会因为名字带 gate 就自动变成稀疏执行。

更重要的是，SwiGLU 有两条独立的线性路径。只把 $$\phi(XW_1)W_2$$ 中的 GELU 改成 SiLU，仍然没有 $$XW_u$$ 这一路，不是同一个架构。因为非线性和逐元素乘积位于投影之间，也不能把三份权重预先相乘成一个固定的 $$d\times d$$ 矩阵；即便去掉 sigmoid，两路线性结果的乘积一般也已经不是关于 $$X$$ 的线性函数。

## 中间宽度为什么常接近模型宽度的三分之八

为什么常见 SwiGLU 中间宽度接近 $$8d/3$$，而不是普通 FFN 的 $$4d$$？先固定无 bias 的主要参数预算。

普通 FFN 若取 $$h_0=4d$$，参数数目为：

$$
P_{FFN}=2dh_0=8d^2
$$

SwiGLU 有三份投影，参数数目为 $$P_{SwiGLU}=3dh$$。令两者相等：

$$
3dh=2dh_0
\quad\Longrightarrow\quad
h=\frac23h_0=\frac83d
$$

这是比较两矩阵与三矩阵结构时得到的预算关系，不是 SwiGLU 公式要求的固定宽度。[原论文](https://arxiv.org/html/2002.05202v1#S2)据此缩小中间层，[LLaMA §2.2](https://arxiv.org/html/2302.13971v1#S2.SS2)也采用了这类宽度选择。其他模型可以按自己的容量和训练配方选择不同 $$h$$。

以 $$d=4096$$ 为例，理想匹配值为 10922.6667。假设为了布局把它向上对齐到 256 的倍数，得到 $$h=11008$$。这是一个明确的对齐示例，也与固定版本 [LlamaConfig 的默认宽度](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/configuration_llama.py)一致，并不表示该配置类会替任意模型自动计算这个公式。

| 无 bias 结构 | 中间宽度 | 参数元素数 | BF16 权重 |
| --- | ---: | ---: | ---: |
| 普通 FFN | 16384 | 134217728 | 256 MiB |
| SwiGLU | 11008 | 135266304 | 258 MiB |

对齐后多出 0.78125% 参数，所以“匹配预算”也要区分理想实数宽度与实际整数 shape。

若一次乘加按 2 FLOP 计，普通 FFN 前向的主导 GEMM 工作为 $$4Ndh_0$$，SwiGLU 为 $$6Ndh$$。代入 $$h=2h_0/3$$，两者主项相同；但 sigmoid、逐元素乘、读写、启动与 shape 效率仍然不同。相同主导 FLOP 不等于相同实际延时，判断方法见 [Roofline]({% post_url 2026-09-09-roofline-gpu-performance-model %})。

## 沿乘积法则推导完整反向

给定输出的上游梯度 $$G=\partial\mathcal L/\partial Y\in\mathbb R^{N\times d}$$，先经过 down projection：

$$
D=\frac{\partial\mathcal L}{\partial U}=GW_d^T,
\qquad
\frac{\partial\mathcal L}{\partial W_d}=U^TG
$$

再看门控乘法。SiLU 的导数为：

$$
s'(a)=\sigma(a)+a\sigma(a)(1-\sigma(a))
$$

于是两路得到不同梯度：

$$
\nabla_A\mathcal L=D\odot B\odot s'(A),
\qquad
\nabla_B\mathcal L=D\odot s(A)
$$

Gate 路径的梯度需要另一条路径的 $$B$$；up 路径的梯度需要 gate 的激活值。两路并不是算完乘积后就可以任意丢弃或交换的独立分支。

最后经过两次上投影：

$$
\nabla_{W_g}\mathcal L=X^T\nabla_A\mathcal L,
\qquad
\nabla_{W_u}\mathcal L=X^T\nabla_B\mathcal L
$$

$$
\nabla_X\mathcal L
=(\nabla_A\mathcal L)W_g^T
+(\nabla_B\mathcal L)W_u^T
$$

这里计算的是 FFN 分支。如果完整模块输出为 $$X+\operatorname{FFN}(X)$$，输入梯度还要加残差路径的 $$G$$；若 FFN 前有 norm，还要继续经过 norm 的导数。不能把分支公式直接当作整个 Transformer block 的反向。

一个特别有用的零点例子是 $$A=0,B=3,D=1$$。此时 $$U=0$$，但 $$s'(0)=1/2$$，所以 $$\nabla_A\mathcal L=1.5$$、$$\nabla_B\mathcal L=0$$。输出为零并不意味着两条输入路径都没有梯度。

若三份权重和输入都需要梯度，反向共有六次主要矩阵乘，工作量约为 $$12Ndh$$，前后向合计 $$18Ndh$$，省略逐元素操作。冻结部分权重会减少所需权重梯度的 GEMM，但若仍需输入梯度，对应路径不能整个跳过。

## 把公式放进一个小矩阵例子

取一行输入和三份二维权重：

$$
X=\begin{bmatrix}1&-1\end{bmatrix},\quad
W_g=W_d=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
W_u=\begin{bmatrix}2&1\\1&-1\end{bmatrix}
$$

由此 $$A=(1,-1)$$、$$B=(1,2)$$，输出为：

$$
Y\approx(0.731058579,-0.537882843)
$$

若标量目标取 $$\mathcal L=Y_0+2Y_1$$，则 $$G=(1,2)$$，代入反向公式得到：

$$
\nabla_X\mathcal L
\approx(1.851904826,1.558259374)
$$

下面的标准库 reference 同时给出前向与全部一阶梯度，矩阵形状沿用前面的行向量约定。它针对非空、有限、数值适中的教学输入，不处理低精度饱和；backward 要使用同一次 forward 的输入、权重与保存状态。

<!-- example: swiglu-reference -->
```python
import math


def shape(a):
    if not a or not a[0] or any(len(row) != len(a[0]) for row in a):
        raise ValueError("nonempty rectangular matrix required")
    return len(a), len(a[0])


def transpose(a):
    return [list(column) for column in zip(*a)]


def mm(a, b):
    n, inner = shape(a)
    rows, cols = shape(b)
    if inner != rows:
        raise ValueError("matmul shape mismatch")
    return [[math.fsum(a[i][k] * b[k][j] for k in range(inner))
             for j in range(cols)] for i in range(n)]


def sigmoid(a):
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    e = math.exp(a)
    return e / (1.0 + e)


def silu(a):
    return a * sigmoid(a)


def silu_prime(a):
    p = sigmoid(a)
    return silu(a) * (1.0 - p) + p


def swiglu_forward(x, wg, wu, wd):
    n, d = shape(x)
    gate_in, h = shape(wg)
    if gate_in != d or shape(wu) != (d, h) or shape(wd) != (h, d):
        raise ValueError("FFN shape mismatch")
    a, b = mm(x, wg), mm(x, wu)
    u = [[silu(a[i][j]) * b[i][j] for j in range(h)] for i in range(n)]
    return mm(u, wd), (a, b, u)


def swiglu_backward(x, wg, wu, wd, saved, grad_y):
    n, d = shape(x)
    if shape(grad_y) != (n, d):
        raise ValueError("upstream shape mismatch")
    a, b, u = saved
    h = len(a[0])
    grad_u = mm(grad_y, transpose(wd))
    grad_a = [[grad_u[i][j] * b[i][j] * silu_prime(a[i][j])
               for j in range(h)] for i in range(n)]
    grad_b = [[grad_u[i][j] * silu(a[i][j])
               for j in range(h)] for i in range(n)]
    dx_a = mm(grad_a, transpose(wg))
    dx_b = mm(grad_b, transpose(wu))
    dx = [[dx_a[i][j] + dx_b[i][j] for j in range(d)] for i in range(n)]
    return (dx, mm(transpose(x), grad_a), mm(transpose(x), grad_b),
            mm(transpose(u), grad_y))
```

<!-- example: swiglu-small-example -->
```python
x = [[1.0, -1.0]]
wg = [[1.0, 0.0], [0.0, 1.0]]
wu = [[2.0, 1.0], [1.0, -1.0]]
wd = [[1.0, 0.0], [0.0, 1.0]]
g = [[1.0, 2.0]]
y, saved = swiglu_forward(x, wg, wu, wd)
dx, dwg, dwu, dwd = swiglu_backward(x, wg, wu, wd, saved, g)
assert math.isclose(y[0][1], -0.5378828427399902)
assert math.isclose(dx[0][0], 1.8519048263915063)
assert math.isclose(silu_prime(0.0) * 3.0, 1.5)
print(y, dx, dwg, dwu, dwd)
```

检查手推梯度时，可以令 $$\mathcal L=\sum_{ij}G_{ij}Y_{ij}$$，逐个扰动 $$X,W_g,W_u,W_d$$ 的元素，比较中心差分与上述反向。非方形输入更能暴露错误转置；只用单位矩阵或正方形 shape，某些方向错误可能恰好被掩盖。

仓库中的 `node scripts/reviews/swiglu-examples.mjs` 会提取并运行这两段代码，用独立标量循环及全元素有限差分核对，还检查合并投影、通道分片、共同置换和错配反例。它验证有限的标准库 CPU 算例，不测量 GPU 性能，也不代表实际低精度内核或多进程通信已通过验证。

## Fused SwiGLU 究竟融合了哪一段

至少要区分三个不同边界。

第一种只融合 $$S=\operatorname{SiLU}(A)$$ 与 $$U=S\odot B$$。Kernel 读取 $$A,B$$ 后直接写 $$U$$，不必单独把 $$S$$ 写成一个全尺寸中间张量。[Liger v0.5.2 的 SwiGLU 算子](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/swiglu.py)属于这条路线；其前向以 FP32 处理 sigmoid 相关计算，再按输出 buffer 的 dtype 写回。

这种融合保留实数公式，但不承诺与低精度分步执行逐位相同：先把 $$S$$ 舍入写回，再乘 $$B$$，与保留较高精度乘完再写回，舍入位置不同。应在目标 dtype 下检查误差，而不是把浮点差异一概视为公式错误。

第二种把 gate/up 两份权重按输出通道拼接：

$$
[A\ B]=X[W_g\ W_u]
$$

一次较宽的 GEMM 生成两路结果，再按约定切开。矩阵乘调用数可以改变，但两份权重和对应乘加并没有消失。固定版本的 [Liger MLP 包装](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/transformers/swiglu.py)同时展示了三份 Linear 与合并 gate/up 的不同组织方式。

第三种才是进一步把投影、门控乃至 down projection 作为更大的执行区域处理。能融合到哪里，取决于 tile、寄存器、shared memory 和后续消费者。仅凭一个函数名带 `fused`，不能断言整个 FFN 已经变成一次 kernel。

接口也必须匹配。上述 Liger `LigerSwiGLUMLP` 固定使用 `bias=False`，而该版本 HF LlamaMLP 可由 `config.mlp_bias` 控制 bias。带 bias 模型不能不检查权重和计算合同就直接套用这个替换。Contiguous 转换、支持的中间宽度和 dtype 同样会影响具体路径，而不是数学公式自动保证的能力。

## 省去一个激活结果，不等于省去所有反向状态

完整反向已经列出了依赖：计算 gate/up 权重梯度需要 $$X$$，门控梯度需要 $$A,B$$，down 权重梯度需要 $$U$$。一种直接保存方案因此是保留 $$X,A,B,U$$，反向从 $$A$$ 重算 sigmoid 与 SiLU，而不另外长期保存 $$S$$。

取 $$N=8192,h=11008$$，BF16 下每份 $$N\times h$$ 张量为 172 MiB，$$A+B+U$$ 三份合计 516 MiB；$$d=4096$$ 时 $$X$$ 另有 64 MiB。避免单独保存 $$S$$ 对应一份 172 MiB 的张量规模，但实际峰值是否下降同样多，还取决于生命周期、buffer 复用、其他保存状态和 workspace。

Liger 该版本的 backward 从 $$A$$ 重建 sigmoid/SiLU，并把保存的 $$A,B$$ 存储覆写为两路梯度。这依赖原值已经没有其他必须保留的消费者，属于该自定义 backward 的生命周期安排；不能把类似原地覆写直接搬进任意 autograd 图，也不能由此推断支持任意重复或高阶 backward。

如果进一步连 $$A,B$$ 都不保存，就需要从 $$X,W_g,W_u$$ 重新生成它们。重算 sigmoid 是 $$Nh$$ 级逐元素工作，重做两次上投影则增加约 $$4Ndh$$ 的 GEMM 主项，代价完全不同。是否值得重算，应结合[激活 Checkpointing]({% post_url 2026-09-09-activation-checkpointing-rematerialization %})的区域与生命周期分析，而不是只数少保存了几张 tensor。

## Tensor Parallel 必须让两路门控共享通道身份

沿中间宽度 $$h$$ 分片时，第 $$r$$ 个 rank 应同时拥有 $$W_{g,r}$$ 与 $$W_{u,r}$$ 的同一组列，以及 $$W_{d,r}$$ 的对应行。这样：

$$
Y=\sum_r
\left[s(XW_{g,r})\odot(XW_{u,r})\right]W_{d,r}
$$

每个 rank 在本地完成属于自己的门控乘积，再生成完整输出 shape 的部分和。最终结果要相加或按下游布局归约，不能把这些 partial outputs 当成普通张量直接拼接。输入梯度也需要合并相应的部分贡献，详细布局见 [Tensor Parallel]({% post_url 2026-08-08-megatron-tensor-parallel-transformer %})。

Gate 与 up 的形状相同，反而容易掩盖错配。把某一路的通道单独重排，乘法仍然合法，却变成 $$s(A_j)B_{\pi(j)}$$，已经不是原网络。若同时一致重排 gate/up 的列与 down 的行，则只是内部通道重新编号，可以保持函数不变。

合并 gate/up 后尤其需要注意物理排列。如果 packed 输出按“全部 gate，随后全部 up”排列，直接把整个 $$2h$$ 维连续等分，某个 rank 可能拿到只有 gate 的片段，而不是自己的 gate/up 配对。分片规划必须理解投影角色；shape 能整除并不足以证明语义正确。若扩展到带 bias 的实现，down bias 也应按合同只加一次，不能在每个部分和上重复添加后再归约。

## 架构变化与等价执行优化要分开验证

把已有两矩阵 FFN 改成 SwiGLU，需要新增一条投影并改变中间表示，不是只换 kernel 就能保留模型函数。原始 GLU Variants 的 T5 受控比较支持其特定训练设置中的质量结果，但不能推出任意 checkpoint 替换后都会更好。

相比之下，已经使用 SwiGLU 的模型，改用等价的 gate/up 拼接、逐元素融合或通道分片，可以先在固定权重与输入下核对输出和全部梯度。测试应覆盖负输入、零点、非方形 shape、尾部通道，以及两路通道有意错配的反例；再进入目标 dtype、内核后端和实际并行环境。

性能验证则需要把三次投影与门控区域一起放回时间线。少一次中间写回可能有利，但更宽 GEMM、contiguous copy、重算和梯度累加都会影响实际成本。记录真实宽度、token 行数、精度、保存策略和执行边界，比仅报告“用了 fused SwiGLU”更有解释力。

SwiGLU 的核心是一种具体的乘法结构：两路投影在相同中间通道相遇，SiLU 调制其中一路，再由 down projection 汇总。宽度预算、反向保存与并行布局，都从这条结构推导出来；沿着它逐项计算，才能知道哪些变化保留了模型，哪些变化已经重新定义了模型。

## 参考资料

- [GLU Variants Improve Transformer](https://arxiv.org/html/2002.05202v1)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/html/2302.13971v1)
- [Transformers v4.46.3 LlamaMLP](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py)
- [Transformers v4.46.3 LlamaConfig](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/configuration_llama.py)
- [Liger v0.5.2 SwiGLU Kernel](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/swiglu.py)
- [Liger v0.5.2 SwiGLU MLP Wrappers](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/transformers/swiglu.py)
