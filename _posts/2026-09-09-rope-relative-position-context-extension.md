---
layout: post
title: "RoPE：相对位置怎样进入 Attention，长上下文又改变了什么"
subtitle: "从二维旋转与多频率相位，推导位置插值、基数调整以及 KV Cache 的一致性边界"
date: 2026-09-09
last_modified_at: 2026-09-09
author: iStar
catalog: true
series: attention-long-context
series_order: 5
technology_year: 2021
mathjax: true
tags: [注意力机制, KV Cache]
---

Attention 用 Query 与 Key 的内积衡量内容匹配，但“相同内容出现在什么位置”同样会影响匹配结果。比较“甲追乙”与“乙追甲”，只有 token 集合显然不够。

不带位置信息、也没有位置相关 mask 的 self-attention 是置换等变的：输入行怎样重排，输出就对应重排。Causal mask 已经施加先后可见性的约束，因此不能说 decoder 完全不知道顺序；位置编码进一步给内容匹配加入明确的位置结构。

Rotary Position Embedding（RoPE）不把一个位置向量加进 embedding，而是按位置旋转投影后的 Query 和 Key。它的关键性质是：Q、K 各自使用绝对位置旋转，内积中却自然出现相对位移。这是 [RoFormer 论文 §3.2](https://arxiv.org/html/2104.09864v5#S3.SS2)提出的构造。本文从这个等式出发，再讨论为什么“公式能算到更长位置”不等于“模型已经具备可靠的长上下文能力”。

## 先让两个坐标随位置旋转

本文使用列向量。设一个 attention head 的投影结果为 $$q_m=W_Qh_m$$、$$k_n=W_Kh_n$$，下标 $$m,n$$ 是逻辑 token position，尚未应用 RoPE。

先只看二维向量，定义逆时针旋转矩阵：

$$
R(\phi)=
\begin{bmatrix}
\cos\phi&-\sin\phi\\
\sin\phi&\cos\phi
\end{bmatrix}
$$

选择一个角频率 $$\omega$$，单位是弧度/token。位置 $$m$$ 对应角度 $$m\omega$$，于是：

$$
\widetilde q_m=R(m\omega)q_m,
\qquad
\widetilde k_n=R(n\omega)k_n
$$

旋转发生在 projection 之后。把 RoPE 移到 projection 之前通常会改变结果，因为任意权重矩阵与旋转矩阵并不交换。Value 通常保持未旋转；它提供被注意力权重加权汇总的内容，而这里的位置结构进入 Q/K 的匹配过程。

旋转矩阵具有两个可以直接相乘验证的性质：

$$
R(\alpha)^T=R(-\alpha),
\qquad
R(\alpha)R(\beta)=R(\alpha+\beta)
$$

所以旋转后的点积是：

$$
\begin{aligned}
\widetilde q_m^T\widetilde k_n
&=q_m^TR(m\omega)^TR(n\omega)k_n\\
&=q_m^TR((n-m)\omega)k_n
\end{aligned}
$$

位置依赖只剩 $$n-m$$。这里的正负号不是可随意替换的记号：我们定义了列向量和上述旋转方向，因此得到 $$R(n-m)$$；若实现使用另一种旋转约定，公式也必须整体一致。

进一步展开，令 $$q=(q_1,q_2)^T$$、$$k=(k_1,k_2)^T$$，并记 $$\Delta=(n-m)\omega$$：

$$
\widetilde q_m^T\widetilde k_n
=(q_1k_1+q_2k_2)\cos\Delta
+(q_2k_1-q_1k_2)\sin\Delta
$$

相对位置并不是一个独立加到 score 上的偏置；它与内容坐标的两个组合共同作用。不同 Q/K 内容，即使位置差相同，也会得到不同匹配。

例如 $$q=(1,2)^T$$、$$k=(3,4)^T$$，取 $$\omega=\pi/6$$、$$m=2,n=5$$，则 $$\Delta=\pi/2$$，点积为 $$11\cos\Delta+2\sin\Delta=2$$。把两个位置同时加 10，位移仍为 3，固定这两个内容向量时结果不变。

## 一个 Head 需要一组不同的转速

实际 head 有许多维度。设参与旋转的维度为正偶数 $$d_r$$，将它们拆成 $$d_r/2$$ 个二维平面，每对坐标使用不同频率：

$$
\omega_i=b^{-2i/d_r},
\qquad i=0,1,\ldots,d_r/2-1
$$

其中 $$b>1$$ 是 RoPE base，具体数值属于模型配置。位置 $$m$$ 的整体变换是分块对角矩阵：

$$
R_m=\operatorname{diag}
\left(R(m\omega_0),\ldots,R(m\omega_{d_r/2-1})\right)
$$

每一块都满足前面的二维恒等式，因此整体也有 $$R_m^TR_n=R_{n-m}$$。高频平面转得快，邻近 token 就有较大的相位差；低频平面转得慢，在更长距离上逐渐变化。这里的“频率”描述坐标随位置旋转的速度，不是 token 内容本身的频率。

用 $$d_r=8,b=10000$$ 做一个便于复算的例子。定义波长 $$\lambda_i=2\pi/\omega_i$$，训练窗口长度取 $$L=4096$$，跨度 $$L$$ 对应的旋转圈数约为 $$r_i=L/\lambda_i$$：

| 平面 $$i$$ | 角频率 $$\omega_i$$ | 波长，token | $$L/\lambda_i$$ |
| --- | ---: | ---: | ---: |
| 0 | 1 | 6.283 | 651.899 |
| 1 | 0.1 | 62.832 | 65.190 |
| 2 | 0.01 | 628.319 | 6.519 |
| 3 | 0.001 | 6283.185 | 0.652 |

最后一个平面在这个跨度内尚未转满一圈，前几个平面已经重复许多圈。这不是说高频平面失效了：单个平面的周期不等于整组多频率状态同时重复。模型消费的是所有平面与内容的共同结果。

有些模型只旋转 head 中的一部分维度。设完整 Q/K head 维度为 $$d_h$$，旋转部分为 $$d_r\le d_h$$，其余内容坐标保持不变，则标准缩放点积可以写成：

$$
s_{m,n}=
\frac{
(q_m^{R})^TR_{n-m}k_n^{R}
+(q_m^{U})^Tk_n^{U}
}{\sqrt{d_h}}
$$

上标 $$R,U$$ 分别表示旋转与未旋转部分。RoPE 维度不是任意改写 softmax 缩放分母的理由；应保持模型定义的完整 score 契约。[MLA 的 decoupled RoPE]({% post_url 2026-06-20-multi-head-latent-attention-kv-compression %})正是在另一套投影结构下，把位置分量与可吸收的内容分量拆开。

## 布局不同，不等于数学不同

上述推导默认相邻坐标成对：$$(0,1),(2,3),\ldots$$，常称 interleaved 布局。另一种 split-half 布局把前半与后半逐项配对，例如八维向量配成 $$(0,4),(1,5),(2,6),(3,7)$$。

如果 $$P$$ 是从一种布局转换到另一种布局的置换矩阵，那么同步变换向量与旋转算子即可保持等价：

$$
q'=Pq,\quad k'=Pk,\quad R'_m=PR_mP^T
$$

$$
(R'_mq')^T(R'_nk')=(R_mq)^T(R_nk)
$$

但只替换 `rotate_half`，不转换投影坐标与频率对应关系，就不是等价改写。最危险的地方在于形状完全相同，程序可以顺利运行，错误却藏在数值里。

作为具体实现例，[Transformers v4.57.1 的 Llama 源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py)使用 split-half 配对，并在 `LlamaRotaryEmbedding.forward` 中以 FP32 计算相位与三角函数，再转换到所需 dtype。较大的 position 若过早转为低精度，相邻整数可能不再可区分；低精度 activation 不意味着位置运算也可以无条件使用相同精度。

## 相对位置性质并不承诺无限外推

不带额外幅度缩放的旋转是正交变换，所以 $$\|R_mq\|_2=\|q\|_2$$。它保持单个向量的范数，却会改变不同位置 Q/K 的夹角。

这不保证任意一对 Q/K 的 score 随距离单调下降。取 $$q=k=(1,0)^T$$、$$\omega=1$$，距离 3 时点积是 $$\cos3\approx-0.989992$$，距离 6 时却变成 $$\cos6\approx0.960170$$。多频率叠加可以形成距离相关结构，但不能据此把每一对 attention 权重解释成单调衰减函数；softmax 还会受其他可见 token 的 scores 影响。

“共同平移不改变点积”也有前提：固定旋转前的 Q/K 内容，并保持频率规则相同。真实多层网络的 hidden states 受前文、mask 与序列边界影响，不能把这个局部恒等式直接升级为整个模型在任意输入变换下的平移不变性。

最后，$$\sin$$ 和 $$\cos$$ 能在任意位置计算，只证明算子有定义。训练时只见过有限位置范围、相对距离和相位组合，扩大窗口后，模型仍可能遇到未适应的分布。长上下文扩展要处理的正是“位置公式连续可算”与“已学会可靠使用这些位置”之间的距离。

## 把窗口扩大八倍，首先可以压缩位置

设原窗口长度为 $$L$$，目标长度 $$L'=sL$$，其中 $$s>1$$。Position Interpolation（PI）的直接做法是将位置替换为：

$$
m'=m/s=mL/L'
$$

因此 $$m'\omega_i=m(\omega_i/s)$$，它等价于把每个角频率都除以 $$s$$。这是 [PI 原论文 §2.3](https://arxiv.org/html/2306.15595v2#S2.SS3)的位置映射定义；它讨论的扩展方案还包含继续训练，不能把这一步代数变换单独当作质量保证。

例如从 4096 扩展到 32768，$$s=8$$。采用从 0 开始的半开位置区间，最后一个下标 32767 被映射为 4095.875，落在 $$[0,4096)$$ 中。这是把目标坐标范围压回原来的连续尺度，不是把所有位置舍入成旧整数下标。

代价同样明确：原本相邻 token 的角差是 $$\omega_i$$，现在变成 $$\omega_i/8$$。PI 不只修改窗口之外的位置，也压缩短距离上的相位分辨率。原来的本地模式需要在新的坐标尺度下重新适应。

## 改 Base 为什么不等于给位置除一个常数

如果希望保留高频平面的局部分辨率，同时让最低频平面像 PI 一样减速，可以改变 base。以下推导一种静态 NTK-aware base change，要求 $$d_r>2$$；这里说明的是频率构造，不是宣称具备所有模型上的外推保证。

最高频 $$\omega_0=1$$ 与 base 无关，改变 base 自然不会改变它。再要求最低频变为原来的 $$1/s$$：

$$
(b')^{-(d_r-2)/d_r}
=\frac{b^{-(d_r-2)/d_r}}{s}
$$

解得：

$$
b'=b\,s^{d_r/(d_r-2)},
\qquad
\omega'_i=\omega_i\,s^{-2i/(d_r-2)}
$$

不同平面使用不同缩放比例，因而不存在一个共同的 $$m'$$ 能把它简化成对所有维度一致的位置插值。这与 [YaRN 附录 A.2](https://arxiv.org/html/2309.00071v3#A1.SS2)讨论的 base-change 关系相接。

仍取 $$d_r=8,b=10000,s=8$$，此时 $$b'=160000$$：

| 平面 | 原频率 | PI 后频率 | 静态 base change 后频率 |
| --- | ---: | ---: | ---: |
| 0 | 1 | 0.125 | 1 |
| 1 | 0.1 | 0.0125 | 0.05 |
| 2 | 0.01 | 0.00125 | 0.0025 |
| 3 | 0.001 | 0.000125 | 0.000125 |

两者的最低频相同，高频差别却很大。只看 `rope_theta` 数值变大，无法推断它与某个 linear scaling factor 等价。

“静态 base change”还不等于所有名为 dynamic NTK 的实现。[Transformers v4.57.1 的 RoPE 工具源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_rope_utils.py)把 linear scaling 与依赖 `seq_len` 的 dynamic 参数计算分开。读取一个配置时，要追到该模型、该版本实际采用的公式，不能只按类型名称拼装来自不同版本的参数。

## YaRN 同时调整频段与 Attention 温度

低频平面在训练窗口内可能尚未转满一圈，扩展后进入此前未覆盖的相位区域；高频平面已重复许多圈，统一大幅压缩又会损伤局部分辨率。YaRN 因此按频段选择更接近“插值”还是“保持原频率”，而不是让所有平面使用一个比例。[YaRN §3.2–3.3](https://arxiv.org/html/2309.00071v3#S3)给出了这一设计及温度修正。

使用前面定义的训练窗口旋转圈数 $$r_i=L\omega_i/(2\pi)$$，一个概念性的线性过渡写法是：

$$
\gamma_i=\operatorname{clip}
\left(\frac{r_i-\alpha}{\beta-\alpha},0,1\right),
\qquad 0<\alpha<\beta
$$

$$
\omega'_i=(1-\gamma_i)\frac{\omega_i}{s}
+\gamma_i\omega_i
$$

低频的 $$\gamma_i$$ 接近 0，倾向 PI；高频接近 1，倾向原频率；中间逐渐混合。实际库可能在频率对应的维度区间上实现 ramp，不能认为任何带 YaRN 名称的代码都逐点等于这条概念公式。

此外，改变可见 token 数量与 score 分布，会影响 softmax 的集中程度。若在全旋转 Q/K 上分别乘幅度 $$a$$，则：

$$
\frac{(a\widetilde q)^T(a\widetilde k)}{\sqrt{d_h}}
=a^2\frac{\widetilde q^T\widetilde k}{\sqrt{d_h}}
$$

对应 softmax temperature $$t=1/a^2$$，不是 $$1/a$$。论文使用过 $$a=1+0.1\ln s$$ 的经验关系；$$s=8$$ 时 $$a\approx1.207944$$，logits 倍率约为 1.45913。这些系数来自特定模型实验，不是应当硬编码到所有架构的常量。

若模型只有部分维度旋转，在 cos/sin 上加幅度只会直接放大对应分量，不自动等于把完整 score 乘 $$a^2$$。这也是为什么长上下文配置必须与模型的 attention 缩放实现一起核对：频率、幅度、旋转维度与 softmax 分母共同定义计算。

这些方案并不是按名称排列的一条必然升级路线。统一插值给出清楚的坐标映射，改变 base 保留了不同的局部结构，分频段处理又增加了区间参数与温度选择。最终需要适配的是已经训练出来的 Q/K 内容分布；换一个模型、原始窗口或扩展倍率，相同几何规则的质量结果也可能不同。

扩展训练还面对两种同时存在的需求：模型要学会利用更远的证据，也不能因为坐标尺度变化而失去原窗口内的能力。因此只在极长样本上检查一个平均指标不足以判断迁移效果。短样本回归、长样本中的不同距离以及目标任务分布，需要成为同一次评测的不同切面。

## 一份 RoPE 配置至少包含四种不同含义

部署时看到“最大长度 32K”，首先应问它描述的是哪一层。它可能只是运行时允许接纳的 token 上限，也可能来自模型配置中的位置容量；两者都不能单独说明 checkpoint 曾以什么频率、什么长度训练。

随后要把四类信息分开核对：原始位置尺度及训练窗口，实际使用的频率函数，是否额外修正 attention 幅度，以及允许保存和执行的最大长度。`factor` 描述相对哪个原始尺度、base 是否已经在导出时修改，都必须明确。若导出的权重本来已经适配了一套扩展规则，再按外部说明重复扩大一次，就执行了另一套模型函数。

最后还要检查作用范围：全部层是否共用规则，是否只旋转一部分 head 维度，以及 prefix cache、prefill worker 与 decode worker 是否加载同一配置。参数名字相同只能说明接口相似，不能替代数值契约。实践中值得把解析后的频率向量、缩放因子、旋转布局和模型 revision 一起记录，而不只保存用户输入的配置字符串。

## K 写进缓存后，位置规则就成为状态的一部分

以刚才固定版本的 Llama 实现为例，执行顺序是 projection、旋转 Q/K，再把已旋转 K 与 V 写入 cache。下次 decode 可以直接读取历史 K，无需为整个前缀重复旋转。

但这也把位置规则固化进了历史状态。一个物理 KV block 中的 K，不仅属于某个 token，还属于生成它时的 position、旋转布局与频率配置。由此可以推出几条具体约束：

- Paged Cache 的 block offset 只是物理槽位，不能代替逻辑 position。
- Chunked prefill 应继续同一条序列的位置；不能在每个 chunk 从 0 重启。
- Prefix reuse 不只核对 token IDs，还要核对模型、adapter、位置与 RoPE 配置。
- Context Parallel 的 rank-local 索引必须映射回正确的全局位置；独立样本打包时，位置和 mask 则应遵守各样本的边界规则。

[KV Cache 基础]({% post_url 2026-06-26-kv-cache-autoregressive-inference-foundations %})讨论了逻辑位置与物理块的分离；[Sequence/Context Parallel]({% post_url 2026-08-14-sequence-context-parallel-long-context-training %})进一步展开跨设备的序列与 mask 语义。RoPE 不负责分配 cache，但决定了 cache 中哪些数值可以被安全复用。

更微妙的问题出现在请求增长时动态改变频率。设历史位置 $$n$$ 的 K 按旧频率 $$\omega_{old}$$ 旋转，新位置 $$m$$ 的 Q 使用 $$\omega_{new}$$。点积包含：

$$
q_m^TR(n\omega_{old}-m\omega_{new})k_n
$$

固定新频率重新计算整个序列时，对应项却是：

$$
q_m^TR((n-m)\omega_{new})k_n
$$

即使暂时固定未旋转的内容向量，两者也相差历史位置相关的相位 $$n(\omega_{old}-\omega_{new})$$，通常不相等。动态规则因此必须说明何时改变参数，以及历史状态怎样维持所要求的语义。

仅将缓存 K 从旧角度重新旋到新角度，也未必等于用新配置完整重跑网络。后续层的历史 hidden states 已经过旧频率下的 attention，其内容本身可能不同；修正当前层 K 的角度不能一般性地修复这种差异。能否复用、需要重建哪些层级状态，应由明确的实现契约与数值对照决定。

## 验证要越过三个不同边界

第一层是代数检查。用很小的向量即可验证旋转保范数、直接旋转与相对位置式相同、固定内容的共同平移性质，以及一致置换下两种布局等价。前文数字都是公式代入，不是模型或 GPU 实测。

本文配套检查还复算了频谱表、PI 与 base change 的区别、YaRN 幅度平方，以及两套频率混用时的缓存反例。在博客仓库中运行：

```bash
node scripts/reviews/new-rope-examples.mjs
```

这些检查只执行标准库中的小向量运算，不加载模型；它们不能替代下面两层验证。

第二层是执行一致性。在固定模型权重、mask、dtype 和位置配置下，对照完整 causal prefill、逐 token decode 与分块 prefill 的 logits，并按后端精度设置合理容差。测试应主动跨越频率更新阈值，同时覆盖 left padding、长短混合 batch、prefix reuse、打包样本边界和 cache block 边界。逐层找第一个分歧，通常比只看最终生成文本更有定位价值。

第三层才是上下文能力。应同时看原窗口内的质量回归与窗口外表现，并改变证据所在位置、证据到问题的距离、干扰文本和需要组合的证据数量。长文本 perplexity、检索测试和真实任务回答的侧重点不同：能找回一个标记不等于能整合长文，平均困惑度稳定也可能掩盖某些位置区域的失败。

长序列请求能够完成，只证明容量和执行路径允许；要声称扩展后的有效上下文长度，仍需要对应任务分布上的质量证据。修改最大长度、选择某种频率规则、继续训练与完成评测，是不同的工作，不能用一个配置开关替代。

## 从位置几何走向系统契约

RoPE 的起点很小：两个坐标随绝对位置旋转，内积因此携带相对位移。多组频率把这一机制扩展到不同尺度；PI、base change 与 YaRN 则重新安排这些尺度，并可能改变 attention 温度。

它们与 [GQA 的 head 共享]({% post_url 2026-03-17-GQA分组查询注意力详解 %})不是同一件事：GQA 改变有多少份 K/V，RoPE 决定每份 Q/K 如何表达位置。进入 MLA 时，旋转能否与线性投影重排成为代数问题；进入增量推理时，频率与 position 又成为历史状态能否复用的前提。

理解 RoPE 因而需要同时守住三条边界：旋转恒等式给出什么保证，长上下文训练改变了什么分布，以及运行时怎样忠实维护这套规则。三者连起来，才知道一份“支持更长窗口”的配置实际上意味着什么。

## 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Extending Context Window of Large Language Models via Position Interpolation](https://arxiv.org/abs/2306.15595)
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
- [Transformers v4.57.1: RoPE 参数计算](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_rope_utils.py)
- [Transformers v4.57.1: Llama RoPE 与 Cache 执行顺序](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py)
