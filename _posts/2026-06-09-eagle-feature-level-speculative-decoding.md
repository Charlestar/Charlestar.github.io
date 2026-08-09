---
layout: post
title: "EAGLE：为什么推测解码要预测 Feature"
subtitle: "从 Feature Uncertainty、Advanced Token 到树形草稿与无损验证"
date: 2026-06-09 09:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: speculative-decoding
series_order: 20
technology_year: 2024
mathjax: true
tags: [AI Infra, EAGLE, 推测解码, LLM推理]
---

推测解码把生成过程拆成两部分：一个便宜的 drafter 先提出候选，目标模型再并行验证。它能否带来收益，很大程度上取决于两件事：候选是否足够接近目标模型，以及产生这些候选是否足够便宜。

最直接的做法是训练一台更小的语言模型充当 drafter。但小模型只看到 token 前缀，需要重新从文本中推断语义、语法和目标模型的生成倾向；模型太小，候选容易被拒绝，模型太大，又会消耗过多计算和显存。

EAGLE 选择了另一条路径：目标模型刚完成一次 forward 时，已经产生了包含上下文信息的隐藏特征。与其让另一个模型从 token 开始重复理解上下文，不如利用这些特征，预测目标模型下一步会形成的 feature，再复用目标模型自己的 LM Head 得到候选 token。

这个想法看起来只是把预测对象从 token 换成向量，真正的难点却是：**未来 feature 不仅由过去 feature 决定，还取决于中间实际采样出的 token。** EAGLE 的核心并非一句“预测 feature”，而是先识别这种 Feature Uncertainty，再用提前一位的 token 序列把不确定性显式交给 drafter。

本文沿着这个问题展开，并把论文算法连接到真实 serving runtime：草稿头如何训练，候选树如何构造，目标模型怎样一次验证多条分支，以及哪些 KV 和采样状态可以正式提交。

## 先固定记号：token、feature 与分布

对已经生成的 token 序列：

$$
t_1,t_2,\ldots,t_j
$$

目标模型执行 causal Transformer，得到各位置的隐藏表示。EAGLE 论文把进入 LM Head 之前、也就是目标模型靠近输出端的 contextual hidden state 记为 feature：

$$
f_1,f_2,\ldots,f_j
$$

论文也把它描述为 second-to-top-layer feature。这里最重要的不是“倒数第几层”这句简称，而是 checkpoint 和实现约定：它是与目标模型 LM Head 输入相对应的那组上下文化表示。

目标模型用 $f_j$ 计算下一个 token 的 logits 与概率分布：

$$
p_{j+1}=\operatorname{Softmax}(\operatorname{LMHead}(f_j))
$$

再根据当前解码策略得到：

$$
t_{j+1}\sim p_{j+1}
$$

因此，一次标准解码的因果关系可以写成：

```text
token prefix t1 ... tj
          │
          ▼
target Transformer
          │
          ▼
feature fj
          │
          ▼
target LM Head → distribution p(j+1) → token t(j+1)
```

feature 比离散 token 更丰富。两个前缀即使最后一个 token 相同，其 contextual feature 仍可能因为更早的语义和句法不同而明显不同。EAGLE 希望利用这种信息密度，让一个很小的自回归头就能延续目标模型的生成状态。

## 为什么 token 级草稿需要另一台语言模型

经典 draft model 近似目标分布：

$$
q(t_{j+1}\mid t_{\le j})\approx
p(t_{j+1}\mid t_{\le j})
$$

它必须完成一台语言模型通常要做的工作：把 token 映射为表示，反复通过 Transformer，从上下文恢复语义，再预测下一个 token。虽然层数和 hidden size 可以缩小，但它依然需要自己的权重和 KV Cache。

如果 drafter 与目标模型能力差距很大，某个候选 $\tilde t$ 在 $q$ 下概率很高，在 $p$ 下却可能很低。随机采样验证时，它的接受概率为：

$$
A(\tilde t)=\min\left(1,
\frac{p(\tilde t)}{q(\tilde t)}
\right)
$$

候选一旦被拒，建立在它后面的草稿 token 也不能继续使用。于是，drafter 的便宜程度和候选质量形成直接冲突。

目标模型的 feature 提供了一个中间点：它已经包含目标模型对上下文的计算结果，drafter 不必从离散 token 重新构造全部状态；同时，预测一个连续且较规则的 feature 演化，可能比直接拟合庞大词表上的多峰分布更容易。

但“给定 $f_j$，预测 $f_{j+1}$”仍然不成立。

## Feature Uncertainty 从哪里来

考虑目标模型已经算出 $f_2$。LM Head 将它映射为下一个 token 的分布 $p_3$：

$$
p_3=\operatorname{LMHead}(f_2)
$$

假设这个分布对多个 token 都保留概率：

```text
"is"     0.42
"was"    0.31
"seems"  0.14
其他      0.13
```

从同一个 $f_2$ 出发，采样可能得到不同的 $t_3$。而下一位置的真实 feature $f_3$ 是目标模型处理具体 token $t_3$ 后形成的：

$$
f_3=F(t_{\le 3})
$$

如果实际采到了 `is`，目标 feature 沿一条路径演化；如果采到 `was`，它会沿另一条路径演化。也就是说：

$$
f_2\not\Rightarrow \text{唯一的 }f_3
$$

一个只看到历史 feature 的回归模型，会面对同一类输入对应多个合理输出的监督信号。使用均方误差一类目标时，它容易预测这些分支的平均位置；这个平均向量未必对应任何真实 token 路径，也未必能通过 LM Head 产生好的候选分布。

这就是论文所说的 Feature Uncertainty：它不是 Transformer 隐藏状态本身“随机”，而是从当前 feature 到下一 feature 之间，夹着一次尚未告知预测器的 token 采样。

可以把缺失变量画出来：

```text
                         ┌─ sample "is" ───► feature branch A
current feature f2 ─► p3 ┤
                         ├─ sample "was" ──► feature branch B
                         └─ sample "seems" ► feature branch C
```

如果 drafter 不知道分支选择结果，就无法确定应该预测哪个 $f_3$。

## Advanced Token：把分支选择交给 drafter

EAGLE 的处理方式很直接：预测 $f_j$ 时，不只使用过去的 feature，还把已经采样出的 $t_j$ 告诉模型。

对 feature 序列：

$$
f_1,f_2,\ldots,f_{j-1}
$$

配对的 token 序列不是同位置的：

$$
t_1,t_2,\ldots,t_{j-1}
$$

而是整体提前一位：

$$
t_2,t_3,\ldots,t_j
$$

论文称它为 advanced token sequence。于是 drafter 学习的关系近似为：

$$
\hat f_j = D(
f_1,\ldots,f_{j-1};
t_2,\ldots,t_j)
$$

为什么这个错位恰好有效？因为当预测 $f_j$ 时，导致 feature 分支发生变化的 $t_j$ 已经出现在输入里。原先缺失的采样结果不再是隐变量。

例如，要从已知的 $f_1,f_2$ 预测 $f_3$：

```text
feature input:  f1, f2
token input:    t2, t3
                         ▲
                         └─ t3 已告诉 drafter 这次采样走了哪条分支
output:         f̂3
```

得到 $\hat f_3$ 后，复用目标模型的 LM Head 计算候选分布：

$$
\hat p_4=\operatorname{Softmax}(
\operatorname{LMHead}(\hat f_3))
$$

从中产生候选 $\tilde t_4$。这个刚产生的 token 又作为下一次 feature 预测的 advanced token：

$$
\hat f_4=D(f_1,f_2,\hat f_3;
t_2,t_3,\tilde t_4)
$$

流程因此可以自回归展开：

```text
target feature f2
      │ + actual sampled token t3
      ▼
draft head predicts f̂3
      │
      ▼
target LM Head gives p̂4 → sample candidate t̃4
      │
      ▼
f̂3 + advanced token t̃4 → predict f̂4
      │
      ▼
target LM Head gives p̂5 → sample candidate t̃5
      │
     ...
```

这里的“advanced”只是两个输入序列的对齐方式，不表示提前知道未来。$t_3$ 是目标模型在当前正式位置已经生成的 token，$\tilde t_4$ 则是 drafter 自己刚提出的候选；算法没有使用任何尚未产生的真实答案。

## 草稿头到底包含什么

原始 EAGLE 不训练另一台完整语言模型。论文中的模块可以拆成三部分。

### 目标模型的 Embedding

advanced token 先经过目标模型的 embedding layer：

$$
e_j=\operatorname{Embedding}(t_j)
$$

这保证 token 表示的词表、维度和语义空间与目标模型一致。Embedding 来自目标模型，在训练 EAGLE drafter 时保持冻结。

### 一个特征融合层

对每个位置，将 contextual feature 与 advanced token embedding 拼接：

$$
z_j=[f_{j-1}; e_j]
$$

若二者维度都是 $d$，拼接结果是 $2d$。一个全连接层把它压回目标 hidden size：

$$
h_j=Wz_j+b,qquad h_j\in\mathbb{R}^d
$$

它把“目标模型已经理解出的上下文”和“这一步实际选择的离散分支”融合到同一个表示中。

### 一个 Transformer decoder layer

融合后的序列进入一个自回归 decoder layer，输出下一位置的预测 feature。这个 decoder 是主要的可训练模块，与完整目标模型相比很小。

最后，目标模型原有的 LM Head 把预测 feature 映射回词表分布。LM Head 同样冻结并共享：

```text
advanced token ─► frozen target embedding ─┐
                                          ├─► FC ─► one decoder layer ─► f̂
target contextual feature ────────────────┘                           │
                                                                      ▼
                                                       frozen target LM Head
                                                                      │
                                                                      ▼
                                                             candidate logits
```

这种结构让训练目标紧贴某个目标 checkpoint。tokenizer、hidden size、embedding、LM Head 或目标模型内部表示发生变化时，原 drafter 不再天然兼容。EAGLE checkpoint 不能仅凭模型家族名称随意配到另一个 revision 上。

## 训练不仅要让向量接近

如果只最小化 $\hat f_j$ 与真实 $f_j$ 的欧氏距离，数值相近并不一定意味着经过 LM Head 后的 token 排序足够相近。EAGLE 同时使用回归与分类目标。

### Feature 回归损失

论文使用 Smooth L1 约束预测 feature：

$$
\mathcal L_{reg}
=\operatorname{SmoothL1}(\hat f_j,f_j)
$$

它在误差较小时近似二次函数，误差较大时转为线性，对少量异常维度没有纯 L2 那样敏感。

### Token 分布分类损失

真实 feature 与预测 feature 分别通过同一个冻结 LM Head：

$$
p_{j+1}=\operatorname{Softmax}(
\operatorname{LMHead}(f_j))
$$

$$
\hat p_{j+1}=\operatorname{Softmax}(
\operatorname{LMHead}(\hat f_j))
$$

再用交叉熵约束候选分布：

$$
\mathcal L_{cls}=H(p_{j+1},\hat p_{j+1})
$$

总损失写成：

$$
\mathcal L=\mathcal L_{reg}
+\lambda\mathcal L_{cls}
$$

原论文实验取 $\lambda=0.1$。这个具体值是论文设置，不应脱离模型、数据和实现直接视为通用最优超参数。

分类损失的作用很实际：最终系统关心的不是隐藏向量每一维是否完美复刻，而是 LM Head 输出的候选 token 是否更接近目标分布、能否被目标模型接受。

## 为什么训练时还要给 feature 加噪声

训练样本中的输入 feature 通常来自目标模型，都是准确的 $f_j$。真实草稿阶段则不同：第一步之后，drafter 会把自己预测的 $\hat f_j$ 继续作为后续输入。

```text
training:  true f1 → true f2 → true f3 → target
runtime:   true f1 → pred f̂2 → pred f̂3 → pred f̂4
```

很小的单步误差会沿自回归路径积累。这是典型的 train–inference mismatch：训练只在干净轨迹上学习，部署却会访问带误差的状态。

原论文在训练 feature 上加入均匀噪声：

$$
\epsilon\sim U(-0.1,0.1)
$$

使 drafter 在略有扰动的输入上也学习恢复合理的下一 feature。它不能彻底消除长路径误差，但能提高对自身预测偏差的鲁棒性。

这个设计也说明，草稿深度不能无限增加。越靠后的节点经历越多轮自回归预测，feature 偏差和 token 分支错误都可能累积；候选树需要在覆盖率与远端质量之间取舍。

## 从一条草稿变成一棵候选树

如果每一步只采一个 token，某个高不确定位置猜错后，后续整条草稿都会作废。树形草稿在同一位置保留多个有希望的候选，让目标验证有机会选择其中另一条路径。

例如，drafter 可以得到：

```text
                         t4=A
                       /      \
                  t5=B          t5=C
                 /    \             \
             t6=D     t6=E           t6=F
```

每个节点对应一个 token，也对应从树根到该节点的完整前缀。EAGLE 可以在少数几轮 drafter forward 中扩展出节点数多于深度的树；论文用过“3 次 forward 构造 10-token tree”这样的配置示例。

树不是越宽越好。增加节点会带来：

- 更多 drafter 采样和 feature 状态；
- 更宽的目标模型 verification forward；
- 更多临时 KV slot；
- 更复杂的 position、mask、父子索引与路径恢复；
- 在高 batch 场景下更明显的算力竞争。

原始 EAGLE 使用预先设计的静态 tree structure。它能为常见的候选概率形态分配预算，但不能根据每个请求当前位置的置信度动态改变形状；这正是后续 EAGLE-2 要解决的问题之一。

## Tree Attention 如何一次验证多条路径

目标模型必须在一次 forward 中评分整棵树，但不同分支不能互相看到。

以上面的树为例，节点 `E` 的合法上下文是：

```text
正式前缀 → A → B → E
```

它不能看到兄弟节点 `D`，也不能看到另一分支的 `C`、`F`。否则目标 logits 会建立在一个现实中不可能出现的混合 token 序列上。

runtime 通常把树节点展平为一个 verification batch，并构造 tree attention mask：

$$
M_{uv}=
\begin{cases}
0, & v\text{ 是 }u\text{ 的祖先或 }v=u\\
-\infty, & \text{其他树节点}
\end{cases}
$$

再为每个节点设置与其路径深度一致的 position id。于是一次目标 forward 虽然在物理 tensor 中并行计算所有节点，逻辑上每个节点只沿自己的根到节点路径做 causal attention。

```text
flattened nodes: [A, B, C, D, E, F]

visible ancestors
A: [A]
B: [A, B]
C: [A, C]
D: [A, B, D]
E: [A, B, E]
F: [A, C, F]
```

目标模型由此得到每条可能路径上各位置的真实分布。接下来不是选一个“整棵树得分”，而是从根开始，按照当前 greedy 或 sampling 规则递归决定可以接受的路径。

## Greedy 验证：沿着目标 argmax 前进

temperature 为 0 时，目标模型每步选择 argmax。验证从树根层开始：

1. 计算正式前缀下目标模型的 argmax；
2. 若树中存在这个 token，接受对应节点并进入它的子节点；
3. 在新前缀下继续比较目标 argmax；
4. 某一层没有匹配节点时，提交目标 argmax 作为新 token并结束本轮；
5. 若一路走到树叶，再使用验证 forward 已经计算出的下一目标 token。

例如：

```text
draft tree:       A
                /   \
               B     C
              / \
             D   E

target argmax path: A → B → X
accepted draft:     A → B
correction token:           X
```

正式序列推进 `A, B, X`。节点 `C, D, E` 都只是临时假设，不能进入用户输出和长期 KV Cache。

在确定性 kernel、相同 logits 处理和无数值 tie 的理想条件下，结果与目标模型逐 token greedy decode 相同。工程上仍要检查浮点舍入、不同 batch shape 和 kernel 选择是否改变极接近的 logits 排名。

## 随机采样验证：不能只比较 token 是否相同

有温度、top-p 或其他随机采样时，如果只接受与目标模型单次采样“恰好相同”的 token，会改变输出分布。无损 speculative sampling 必须使用接受—拒绝修正。

在某个已确定的树路径位置，设 drafter 提议 token $x$ 的分布为 $q$，目标模型分布为 $p$。接受概率为：

$$
a(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
$$

若拒绝，则从正残差分布采样修正 token：

$$
r(y)=
\frac{\max(0,p(y)-q(y))}
{\sum_z\max(0,p(z)-q(z))}
$$

候选通过接受路径贡献的概率质量是：

$$
q(x)a(x)=\min(q(x),p(x))
$$

拒绝后的正残差恰好补齐：

$$
p(x)-\min(q(x),p(x))
=\max(0,p(x)-q(x))
$$

两条路径相加仍是目标分布 $p$。这就是论文所说 lossless 的含义：算法输出分布与目标模型原始采样分布一致，而不是保证给定相同随机种子后文本逐字相同。

在候选树中，系统还要记录每个节点由哪个 proposal distribution 产生、兄弟候选采用什么抽样方式，并沿被选择的路径递归执行验证。不能把线性 speculative sampling 的公式不加修改地套到任意 top-k 树布局上；tree construction 与 tree verification 必须使用相互匹配的算法。

## KV Cache 必须按事务提交

树验证会为所有候选节点计算目标 K/V，但只有最终接受路径属于正式序列。如果 verification 直接把整棵树追加到长期 cache，下一轮 attention 就会读到没有发生过的分支。

更安全的理解是把一次验证看成事务：

```text
reserve temporary KV slots for all tree nodes
                    │
                    ▼
target tree-attention forward writes tentative KV
                    │
                    ▼
verification selects one accepted path
          ┌─────────┴──────────┐
          ▼                    ▼
commit path KV          reclaim rejected-branch KV
```

实现可以通过复制选中节点、块表重映射或临时 arena 回收完成，具体取决于 cache manager。但提交后的逻辑状态必须等价于目标模型只生成了被接受前缀和修正 token。

drafter 侧也有自己的 feature/KV 状态。若它展开了 `A-B-D`，目标最终只接受 `A-B` 并改为 `X`，下一轮 drafter 必须从正式路径 `A-B-X` 继续，不能沿 `D` 的预测状态向前滚动。

同样需要事务化的还有：

- repetition penalty 与 token frequency；
- grammar、JSON schema 或有限状态机状态；
- EOS、stop token 和最大输出长度；
- 随机数状态与 proposal probability；
- 流式输出缓冲；
- prefix cache 的 block 引用计数。

尤其不能先把草稿 token 流给客户端，再在验证失败后尝试撤回。大多数生成协议只有追加语义，因此只有已验证、已提交的 token 才能进入输出流。

## EAGLE 为什么可能更快

EAGLE 没有减少目标模型的权重，也没有绕过目标验证。它试图通过更便宜、更准确的 proposal，让一次目标 forward 正式推进多个 token。

设一轮包含：

- drafter 展开候选树耗时 $T_d$；
- 目标模型验证整棵树耗时 $T_v$；
- mask、采样、KV 提交等开销 $T_o$；
- 最终正式推进 token 数为 $L$。

平均每个正式 token 的时间近似为：

$$
\operatorname{TPOT}_{EAGLE}
\approx
\frac{T_d+T_v+T_o}{E[L]}
$$

只有它小于基线单步 decode 的 TPOT，系统才真正加速。

feature-level drafter 有几个潜在优势：

- 直接继承目标模型当前 contextual feature，不必重新理解整个 token 前缀；
- 可训练参数远少于一台完整 target-compatible draft model；
- 共享目标 embedding 与 LM Head，候选分布更贴近目标输出空间；
- 树形 proposal 能在不确定位置覆盖多个分支。

但它仍有不可忽略的成本：

- 需要保存和调度额外 drafter 权重与状态；
- 目标验证的 token/node 数大于普通单步 decode；
- tree mask、position 和 KV 整理可能阻碍某些优化路径；
- 高并发时，基线 continuous batching 本来已能高效利用 GPU；
- 长上下文下，每个额外验证节点都要读取大量历史 KV。

原论文在其模型、任务、硬件和实现组合上报告了约 $2.7\times$ 到 $3.5\times$ 的 latency speedup，并观察到较高的草稿准确率。这些数字证明方法在相应实验条件下有效，不构成任意模型和生产流量上的固定承诺。

## 三个指标不要混为一个

讨论 EAGLE 性能时，经常出现“接受率很高，所以加速很多”的跳跃。至少要区分三个量。

### Candidate acceptance rate

$$
\text{acceptance rate}
=\frac{\text{accepted draft nodes}}
{\text{checked draft nodes}}
$$

它反映候选质量，但分母口径会受树布局影响。把没有走到的兄弟分支算不算“checked”，不同实现可能不一致。

### Accepted length

每轮验证最终连续接受多少草稿 token，或者本轮总共正式推进多少 token。它更直接影响目标模型串行轮数。

同样是 80% 节点命中，一棵把预算浪费在大量兄弟分支上的树，和一条连续接受很深的路径，对串行步数的贡献不同。

### End-to-end speedup

$$
\text{speedup}
=\frac{T_{baseline}}{T_{EAGLE}}
$$

它还取决于 draft latency、verification width、batch、KV 带宽、采样和 runtime 实现。接受长度上升不保证端到端耗时同比下降。

因此，实验报告至少要同时给出三者，并说明统计是在请求级、step 级还是 token 级聚合。

## 一次可解释的性能实验

评估时先固定目标模型的解码语义：相同 tokenizer、chat template、temperature、top-p、logits processor、最大长度与 stop 条件。然后比较普通 decode 和 EAGLE。

### 正确性与状态边界

至少覆盖：

- Greedy 下，普通 decode 与 EAGLE 的 token 序列比较；
- Sampling 下，在可控小模型或小词表上检查输出分布，而非逐样本字符串；
- 候选在第一层、中间层和树叶被拒绝；
- 所有候选都通过；
- EOS、stop string、最大长度和取消请求；
- grammar/structured output 与自定义 logits processor；
- 验证后目标 KV、drafter 状态和正式 token 长度一致。

### 分解每轮耗时

记录：

```text
draft expansion time
tree preparation / mask time
target verification time
sampling and rejection-correction time
KV commit / reclaim time
accepted draft length
committed tokens per verification
```

只看 GPU kernel 时间可能遗漏 CPU tree construction 与同步；只看端到端时间又无法解释回退原因，两套数据都需要。

### 扫描真正影响结论的维度

按以下维度分桶：

- batch size 与请求到达率；
- prompt length 与生成长度；
- 对话、代码、摘要、数学等任务域；
- greedy 与不同 temperature/top-p；
- tree node budget、深度与分支布局；
- 数据类型、量化方式和并行策略；
- 开启或关闭 CUDA Graph、prefix cache 等 runtime 优化。

最终要找的不是一个最大倍数，而是 EAGLE 相对普通 decode 的工作区间：在哪些 QPS、上下文长度和采样配置下改善 TPOT 或 SLO goodput，在哪些区间应该自动回退。

## EAGLE-2 与 EAGLE-3 分别接着改了什么

原始 EAGLE 建立了 feature-level drafter、advanced token 与树验证这条主线，后续版本没有简单重复它。

### EAGLE-2：让树形预算跟随当前置信度

静态树对每一步使用相同布局，但不同上下文的候选分布差异很大。有的位置第一名概率占绝对优势，适合向深处展开；有的位置多个分支接近，更适合保留宽度。

EAGLE-2 使用 drafter 的置信度近似候选接受可能性，动态维护和扩展更有希望的节点。它主要改变 proposal tree 的预算分配，而不是推翻原始 EAGLE 的 feature uncertainty 处理。

### EAGLE-3：改变 drafter 的训练约束与特征来源

EAGLE-3 不再要求 drafter 严格复刻未来 feature，而是更直接地优化 token prediction；同时融合目标模型低、中、高层特征，并通过 training-time test 缩小训练轨迹与自回归草稿轨迹的差异。

因此，这三个版本适合按问题顺序理解：

```text
EAGLE-1: 怎样利用目标 feature，且消除 feature uncertainty？
EAGLE-2: 有限候选节点应该怎样动态组成一棵树？
EAGLE-3: drafter 的训练目标与输入 feature 怎样继续扩展？
```

如果跳过原始 EAGLE，直接看到 EAGLE-3 的“多层特征融合”和“取消 feature prediction 约束”，就很难理解它究竟取消了什么，以及 advanced token、目标验证和无损采样为何仍然重要。

## 部署前要确认的兼容边界

一个 serving engine 声称支持 EAGLE，不代表任意组合都能直接工作。至少确认：

- drafter checkpoint 与目标模型精确 revision 是否匹配；
- tokenizer、vocabulary、embedding tying 和 LM Head 是否一致；
- runtime 实现的是 EAGLE 哪个版本与哪种 tree layout；
- sampling、top-p、top-k 和 rejection correction 是否完整；
- tensor/pipeline parallel 下 feature 怎样传给 drafter；
- tree attention 是否有相应高效 kernel；
- 量化后的 target feature 与 drafter 训练分布是否兼容；
- prefix caching、chunked prefill、structured output 是否经过组合测试；
- CUDA Graph 是否覆盖实际 node budget，还是频繁回到 eager path；
- 额外权重与临时 KV 是否降低可用并发。

如果只能在少数预定义 tree size 上捕获 Graph，动态控制器应在这些离散 shape 中选择，而不是每轮产生全新的 tensor shape。若 feature 需要跨 GPU 传输，也要把通信时间计入 $T_d$，不能只测 drafter kernel。

## 小结

EAGLE 的关键洞察不是简单地用隐藏向量代替 token，而是把目标模型已经计算出的 contextual feature 当作便宜而高质量的草稿起点，同时正视从当前 feature 到未来 feature 之间存在一次 token 采样分支。

Advanced token sequence 把这个实际采样结果提前一位交给 drafter，使下一 feature 的监督目标重新变得明确；冻结的目标 embedding 与 LM Head、轻量 FC 和单层 decoder 组成草稿头；回归与分类损失分别约束 feature 几何和最终 token 分布；树形 proposal 再用一次带 tree attention 的目标 forward 批量验证多条候选路径。

算法的无损性来自目标模型验证和正确的接受—拒绝修正，而不是 drafter 永不出错。工程上的正确性则依赖候选 KV 的临时写入、接受路径提交和拒绝分支回收。最后能否加速，要由 accepted length 是否覆盖草稿、宽验证和状态管理成本来回答。

理解这些边界后，EAGLE-2 的动态树和 EAGLE-3 的训练改造就不再是孤立技巧，而是分别在“如何分配候选预算”和“如何训练更可扩展的 drafter”上继续推进同一条技术路线。

## 参考资料

- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://proceedings.mlr.press/v235/li24bt.html)
- [EAGLE 论文预印本](https://arxiv.org/abs/2401.15077)
- [EAGLE 官方实现与模型说明](https://github.com/SafeAILab/EAGLE)
- [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)
- [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)
