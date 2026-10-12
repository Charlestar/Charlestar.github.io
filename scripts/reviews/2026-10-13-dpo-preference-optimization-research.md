# DPO：调研与写作交接

日期：2026-10-13。阅读一手论文、作者实现与 TRL 固定版本说明；未运行大模型偏好训练、生成样本、调用评审 API 或复现论文对比结果。

## 选题与系列

已有 RLHF/GRPO 系统数据流文章解释 Actor/Reference/Reward/Rollout，WAR 文章讨论轨迹调度；没有 DPO 独立专题。新文讲清偏好数据、KL 正则解、pair loss、序列 log probability 和训练数据合同，不扩写 PPO/GRPO 全部系统内容。

root 确认 `series: distributed-training`、`series_order: 105`、`technology_year: 2023`，初次预印本为 2023-05-29，本文实际阅读 v3 为 2024-07-29。

建议标题：`DPO：成对偏好怎样直接改变语言模型的概率`。

## 实际阅读范围

1. [DPO v3 论文正文](https://arxiv.org/html/2305.18290v3)：实际阅读 §3 Preliminaries、§4 Direct Preference Optimization、§5.1、Appendix A.1、A.2、A.4 以及 B Implementation Details；核对正文 Eq.3–7 与有限支集下的数学解。未完整阅读所有实验、人工评估材料与完整附录，不称为全篇审计或复现。
2. [论文版本元数据](https://arxiv.org/abs/2305.18290)：实际核对 v1/v2/v3 日期与作者。历史事实可引用；不用摘要中“stable”“matches/better”作为普遍优越性保证。
3. [作者 trainers.py](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py)：实际阅读 `preference_loss`、`_get_batch_logps` 与相应输入/forward 函数；核对原始 sigmoid branch 的 policy/ref log-ratio 方向、masked shifted labels、sum 默认。链接读取当日 main，未固定 SHA，正文固定数学式，不承诺旧脚本和现代软件依赖直接兼容。
4. [TRL v0.23.1 DPOTrainer 文档](https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer)：实际阅读数据合同、sigmoid 默认、logged rewards/margins/accuracy、precompute_ref_log_probs、配置字段。它是明确固定的历史接口快照，页面当日显示更新版本 v1.15.0；本文不称 v0.23.1 为最新，也不照搬主线新字段解释此版本。
5. [TRL v0.23.1 Dataset formats](https://huggingface.co/docs/trl/v0.23.1/en/dataset_formats)：实际阅读 explicit prompt 的成对 preference 与 conversational 格式、prompt extraction/templating。正文以自行构造的 `prompt/chosen/rejected` 样例解释，不搬用文档示例。
6. [TRL v0.23.1 trainer 源码](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/dpo_trainer.py)，并读取[完整 raw view](https://raw.githubusercontent.com/huggingface/trl/v0.23.1/trl/trainer/dpo_trainer.py)：实际阅读 `_prepare_dataset`、`tokenize_row`、`compute_ref_log_probs`、`dpo_loss` sigmoid branch、`concatenated_forward` 的 prompt mask、shift、sum 和 truncation、reference cache 列。浏览器 GitHub blob 在中途截断，已经通过 raw view 补读相关函数，不把未见 remainder 说成已审。
7. [TRL v0.23.1 DPOConfig 源码](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/dpo_config.py)：实际阅读 beta、reference_free、label_smoothing、precompute 与长度配置；本文限定原始 DPO：sigmoid、label_smoothing=0、固定 reference、不混 loss、不启用 IPO/length-desensitization/weighting 等变体。

单来源正文衍生摘要控制在 200 words 内；数学从明确前提自行推导并用自造有限分布/目标序列复算，正文不长译论文，不复制原实验表格或 README 训练配方。

## 阅读发现：正文和注释也必须交叉核对

- v3 HTML Appendix A.4 Eq.21 中 sigmoid 内的 rejected/chosen 顺序与正文 Eq.7 相反；后续推导与论文正文也应交叉看。写作以正文 Eq.7、作者执行代码和直接微分复算为准，不传播附录排版/符号问题；无需在教学正文大篇幅讨论论文勘误。
- 作者代码和 TRL 固定源码的注释把 beta→0 简述为“ignore reference”。对固定有限 log-ratios，beta=0 实际使 loss=ln2、参数梯度为零，不能写成“不用 reference 后退化到 SFT”。reference_free 是另一明确变体，与 beta=0 不同。
- TRL v0.23.1 `tokenize_row` 分别 encode prompt/chosen/rejected，再拼 token IDs，并为 completion 追加 EOS。不能写成它必然先整体 encode 原始 prompt+completion。该版本基于已渲染字段的输入合同；任意 BPE 文本切片是否可拼，需要单独验证训练/部署边界，不能从数据字段名推断。

## 原创理论推导的前提

先固定一个 prompt x、有限完整回答集合 Y；令 reference 对所讨论回答 q(y)>0，beta>0，reward r(y) 固定。有限集合使配分函数 Z 有限；推广到所有 token 序列需要相应 support 与可归一化条件。

目标 `J(p)=Σ p(y)r(y)−βΣp(y)log[p(y)/q(y)]`；设 `p*(y)=q(y)exp(r(y)/β)/Z`，展开得 `J(p)=βlogZ−β KL(p||p*)`。由 KL 非负得最优 p*，不把此非参数、已知 reward 的解说成任何有限数据/神经参数/优化器皆自动找到全球最优。

将 reward 写成 `r(y)=βlog[p*(y)/q(y)]+βlogZ`，同 prompt 的奖励差取消未知常数。采用 Bradley–Terry 模型 `P(w≻l|x)=σ(r_w−r_l)` 后，参数化 policy 的 margin 为

```text
m = β[(log πθ(yw|x) − log πref(yw|x))
      − (log πθ(yl|x) − log πref(yl|x))]
loss = −log σ(m) = softplus(−m)
```

这是有假设的偏好概率模型；sigma(m) 不是答案事实正确概率或模型生成 yw 的绝对概率。

直接微分：`d loss/dm = −σ(−m)`；reference 固定时

```text
∇θ loss = −βσ(−m)[∇θ log πθ(yw|x) − ∇θ log πθ(yl|x)]
```

β 同时出现在推导的正则目标与分类 margin；在固定 reward、精确最优分布解下，增大 β 使 reweighting 更温和。有限离线训练中的实际 KL/质量还取决于数据、参数化、优化器、学习率、训练步数；beta 不是直接设置一个可保证的 KL 半径。

## 原创复算例

### 绝对 chosen 概率可下降

参考与旧 policy 在三个完整回答 `[w,l,o]` 上都是 `[0.4,0.4,0.2]`。新 policy 为 `[0.3,0.1,0.6]`；全为合法归一化概率。beta=1 时：

- old ratio=1，margin=0，loss=ln2≈0.693147。
- new ratio=3，参考 ratio=1，margin=ln3，loss=ln(4/3)≈0.287682。
- chosen 绝对概率从 0.4 降为 0.3，但 pair loss 下降。

这是目标函数不规定绝对 chosen 概率单调上升的分布反例，不声称一个独立 three-logit SGD 单步必走这条轨迹。共享参数、多条 pair、质量偏差下尤其不能把 relative margin 改善推成所有输出质量改善。

### 序列 log probability 与 mask

toy 输入 `[BOS,p0,p1,y0,y1,EOS,PAD]`，completion 目标为 y0/y1/EOS；对齐 label 示意 `[-100,-100,-100,y0,y1,EOS,-100]`。模型内部 shift 后 logits 在 p1/y0/y1 位置分别预测 y0/y1/EOS。prompt 是条件并仍能被 attention 看见，只是不计入回答 log probability。

如果三个目标的条件概率为 `[0.5,0.25,0.8]`，完成回答概率为 0.1，log probability 为 ln0.1≈−2.302585。把三个 log probabilities 取 mean 得到 −0.767528，是几何平均 token 概率的 log，已经不是完整回答概率的 log。标准 DPO 用 completion sum；变体归一化应另命名/推导。

是否包括 EOS 由“完整回答事件”的数据合同定义；本文例子包含 EOS。PAD 按真实位置 mask，若 PAD 与 EOS 共用 ID，不能用 token-ID 相等删去所有 EOS。

### 相同回答和截断

最终 token 序列、条件与 mask 完全相同的 chosen/rejected，会有 m=0、loss=ln2 且 pair 参数梯度互相抵消。这种 pair 没有排序信息。截断删去二者差异时须重新判断样本是否还表达原始偏好。

原 JSON prompt 一样，不自动保证最终条件一样；例如全长 keep_end truncation 面对更长 chosen 和更短 rejected，可能分别删除不同长度的 prompt。审核应检查最终 token-prefix/condition 以及有效 completion，而非只看原字段。

## 建议逐段大纲

1. 一条 prompt 的两份技术回答，为什么“相对更合适”提供了 SFT 单答案之外的信息；标签表达标准，不直接等于事实真值。
2. 从 SFT policy 到固定 reference，解释用概率分布而非文字质量直觉写目标。
3. 固定 prompt 的 KL 正则目标与有限支集最优解，自行展开 KL，不跳过 Z 或 support。
4. Bradley–Terry pair model、奖励常数抵消、margin 与 loss；说明模型估计偏好与文本生成概率分开。
5. small finite distributions 和梯度权重，错误方向 pair 权重大、已正确大 margin 权重小；展示 absolute-chosen 反例。
6. sequence log probability、teacher forcing、prompt/response boundary、mask/shift、sum 与 mean；用一个七 token 行清楚讲完。
7. 真正训练 batch 的 policy/reference forward 数据流，不需要每次在线采样，但样本采集和评估仍会生成，完整工程仍支付两分支和 reference 成本。
8. 数据质量：相同 prompt、chosen/rejected 根据同标准、有代表性问题、截断/模板/EOS、重复和泄漏、噪声与事实错标；不扩张成泛泛治理清单。
9. beta 与 offline support 的边界；reference logprob cache 必须绑定 reference weights、tokenizer、template、token IDs、mask、truncation 与数据版本。
10. 评估看 held-out pair margin 和真实任务生成表现；训练 reward accuracy 只看隐式 margin，不代表事实正确率；连接已有 RLHF/GRPO 系统文章和本批 Tokenizer。

## 应避免的表述

- DPO 是训练模型判断两个回答的分类头，或者需要再单独训练一个 reward model。
- 任何 preference dataset、任意优化器均与 PPO 完全等价且更好，或低成本自动得到真实人类 reward。
- chosen logprob 必然上升、loss/隐式 reward accuracy 改善即证明输出正确。
- beta 越大在每个有限数据训练设置都保证更小 KL，beta=0 退化为 SFT。
- completion logprob 是 mean、prompt 不计 loss 所以看不见 prompt、词数能替代目标 token 数。
- precomputed reference 分数可在更改 template/tokenizer/truncation/weights 后继续复用。
- 原始 DPO 训练没有在线 rollout，就意味着采集数据和评估阶段从不生成。

## 独立审核建议

KL optimum identity、奖励常数抵消、reference ratio 与 margin 正负、stable softplus、analytic gradient 的 finite difference、beta=0、chosen 概率下降但 pair loss 改善、masked shifted completion sum、EOS/PAD 同 ID、相同 pair 梯度、截断后 prompt/有效差异。全为 CPU 数值和合同检查，不能代替真实训练收敛/生成质量/GPU 显存测量。
