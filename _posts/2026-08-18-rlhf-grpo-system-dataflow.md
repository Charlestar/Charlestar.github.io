---
layout: post
title: "RLHF/GRPO 系统数据流：一次训练迭代里四类模型怎样协作"
subtitle: "从 Policy Snapshot、Rollout 与 Reward，到 Advantage、参数更新和一致性边界"
date: 2026-08-18 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: distributed-training
series_order: 80
technology_year: 2022
mathjax: true
tags: [分布式训练, LLM推理, 工程实践]
---

监督微调的数据流很直接：读取一批 token，执行 Forward/Backward，更新一次参数。到了 RLHF 或基于可验证奖励的 GRPO，训练程序突然同时出现 Actor、Rollout、Reference、Reward、Critic 等名字，GPU 也不再只做训练：同一份策略参数可能先以 FSDP shards 存在于训练引擎，再以 Tensor Parallel 权重进入推理引擎，生成结果随后被多个只做 Forward 的模型重复读取，最后才回到 Actor 参与反向传播。

表面上，这是“多放了几份模型”；实质上，它是一张带版本语义的分布式数据流图：

```text
prompt
  │
  ├─► rollout(policy snapshot v) ─► response tokens + old log-probs
  │                                      │
  │                                      ├─► reward / verifier ─► score
  │                                      ├─► reference ─────────► ref log-probs
  │                                      └─► critic (PPO) ───────► values
  │
  └────────────────────────────────────────► advantage estimator
                                                   │
                                                   ▼
                                      actor/critic update to v+1
```

任何一条边都不只是“传一个 Tensor”。它还隐含了 tokenizer、sampling configuration、response boundary、policy version、padding layout、数值精度和样本身份。只要其中一个契约错位，训练可能不报错，却在优化一个与实际 rollout 不同的目标。

这篇文章不试图罗列所有 RL 算法，而是沿着一次同步训练迭代，回答五个工程问题：每个角色到底负责什么，哪些数据在它们之间流动，训练与推理表示为何需要转换，哪些位置必须建立同步边界，以及怎样证明一条分布式 RLHF/GRPO Pipeline 没有悄悄改变算法语义。

## 先把 RLHF 看成一张分布式数据流图

传统 RL 可以画成 Actor 与 Environment 交互、Critic 估值、Optimizer 更新的图。LLM 把图中每个“小网络节点”放大成了一个分布式程序：Actor 训练可能使用 DP、TP、PP、FSDP 或 ZeRO；Rollout 可能使用另一组 TP/PP 和 Continuous Batching；Reward、Reference 与 Critic 又各有自己的模型大小、吞吐模式和显存压力。

HybridFlow 对这个问题的概括很准确：RLHF 图中的节点是分布式 LLM computation，边则可能是不同并行布局之间的 many-to-many data resharding。于是系统有两个尺度：

- **图内节点**负责一次完整的分布式 Forward、Generation 或 Update；
- **图间控制器**负责依赖顺序、资源映射、数据重分片和失败处理。

这一区分很重要。让中央控制器逐个调度 GPU kernel 会产生过高控制开销；让每个 rank 自己拼接整张 RLHF 流程，又会把跨角色依赖散落在所有 worker 中。更合理的结构是：节点内部由训练/推理框架高效执行，节点之间由一个能看见完整 iteration 的控制平面协调。

## 四类模型之外，还有一个容易被忽略的 Policy Snapshot

常见图里会画 Actor、Reference、Reward、Critic 四类模型，再把 Rollout 画成 Actor 的一个动作。工程上应把 Rollout 当作独立执行角色，同时把“产生当前经验的旧策略”当作独立版本概念。

| 角色或版本 | 参数是否更新 | 主要计算 | 核心输出 |
| --- | --- | --- | --- |
| Actor / current policy $\pi_\theta$ | 是 | Forward、Backward、Optimizer Step | new log-probs、loss、更新后的参数 |
| Rollout policy $\pi_{\theta_{old}}$ | 在一批经验内冻结 | 自回归生成 | response tokens、采样元数据，可选 generation log-probs |
| Reference policy $\pi_{ref}$ | 通常冻结一段时期 | Teacher-forced Forward | reference log-probs |
| Reward model/function $r_\phi$ | 通常在一次 RL phase 内冻结 | 打分、规则验证或环境交互 | sequence/token/step rewards |
| Critic $V_\psi$ | PPO 中更新 | Forward、Backward、Optimizer Step | token values、value loss |

“四类模型”描述的是职责；“旧策略”描述的是数据来源版本。两者不是同一个分类维度。

### Actor 是被优化的 Policy，不等于生成服务进程

Actor 表示待训练策略 $\pi_\theta$。它必须能够在给定 prompt 与已生成前缀时计算每个 response token 的 log-prob，并对 policy loss 执行反向传播。它通常还拥有 optimizer moments、gradient scaler、学习率状态和训练并行组。

Rollout 引擎也会装载 Actor 的权重，但那只是某个 committed Actor version 的推理表示。不能因为两个进程加载了相同 checkpoint，就把“Actor 模型”与“Rollout worker”视为同一状态机：前者可能已经完成几个 minibatch 更新，后者仍在用此前的快照生成。

### Rollout 把静态 Prompt 变成 On-policy Experience

Rollout 的输入不只是字符串，而是一组已经模板化和 tokenized 的 prompts，以及确定的 sampling contract，例如 temperature、top-p、最大输出长度、stop token 与每个 prompt 的采样条数。它的输出至少包括：

- 真实送入模型的 `prompt_ids`；
- 实际采样到的 `response_ids`；
- prompt/response/valid-token masks；
- EOS、截断、工具终止等结束原因；
- 生成它们的 `policy_version`；
- 采样配置与随机种子或可追踪 request ID；
- 可选的 rollout-time log-probs。

Rollout 做的是推理，但目标不是服务用户：它生产的是训练数据。吞吐优化不能破坏数据可解释性，更不能在生成途中切换权重。

### Reference 是约束坐标系，不是 Old Policy

Reference policy 常由 SFT 初始模型复制而来，用于衡量当前策略偏离初始行为分布的程度。对 response token $a_t$，系统需要得到：

$$
\log \pi_{ref}(a_t\mid s_t)
$$

它通常只做 teacher-forced Forward，不需要 KV Cache 长期驻留，也不持有 optimizer。Reference 是否永久冻结取决于算法；例如原始 DeepSeekMath 的 iterative GRPO 会在外层迭代更新 reference。无论采用哪种方案，更新节奏都必须是显式配置，不能由 worker 重启或权重加载顺便决定。

### Reward 把行为结果变成训练信号

Reward 不一定是一张可训练神经网络。它可能是：

- 从偏好比较数据训练出的 scalar reward model；
- 数学答案、单元测试、编译器或形式化验证器；
- 多个 scorer 的加权或约束组合；
- 多轮 Agent 与环境交互后得到的 outcome；
- process reward model 在若干 reasoning step 上给出的分数。

因此更准确的抽象是 `RewardFunction(sample) -> RewardRecord`。返回值除了总分，还应包含各 component、错误状态、scorer version 与适用 mask。把所有失败都折叠成 0 分，会把“答案错误”“验证服务超时”“解析器崩溃”混成同一种学习信号。

### Critic 预测的是 Return Baseline，不是 Reward Model 的别名

PPO 的 Critic 或 value model 估计 $V_\psi(s_t)$，用于从回报中减去 baseline、降低 policy-gradient 方差。Reward model 评价“这条 response 好不好”；Critic 预测“从当前 token 状态继续，期望还能得到多少回报”。两者输出语义和训练目标不同。

InstructGPT 的实现曾用 Reward Model 初始化 value function，但初始化来源相同不等于角色相同：RL 阶段 Reward 通常冻结，Critic 则随经验更新。系统若复用同一模型容器，也必须把参数版本和 optimizer ownership 分开。

## Old Policy 与 Reference Policy 为什么必须分开

PPO 更新依赖重要性比率：

$$
\rho_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{old}}(a_t\mid s_t)}
=
\exp\left(
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\theta_{old}}(a_t\mid s_t)
\right)
$$

其中 $\pi_{\theta_{old}}$ 是产生这批 response 的 behavior policy。在同步 PPO 或原始 GRPO 中，每轮采样开始前通常执行一次精确快照 $\pi_{\theta_{old}}\leftarrow\pi_\theta$，随后在这批 experience 的生成和全部 update epochs 内保持不变。Actor 开始更新后，两者才产生差异；这里不是 EMA，也不表示允许一个含混的渐进同步过程。

Reference $\pi_{ref}$ 则服务于 KL regularization。先定义采样 action 上的有符号 log-ratio：

$$
\ell_t
=
\log\pi_{\theta}(a_t\mid s_t)
-\log\pi_{ref}(a_t\mid s_t)
$$

注意 $\ell_t$ 只是当前样本上的 log-ratio，不是对整个 action distribution 求和后的 KL divergence，也不保证非负。PPO 风格的 RLHF 常把 $-\beta\ell_t$ 作为逐 token shaped reward；若把 KL 直接放进 loss，则必须明确采用哪一种 sampled estimator。DeepSeekMath 的原始 GRPO 使用：

$$
\widehat{D}_{KL,t}
=
\frac{\pi_{ref}(a_t\mid s_t)}{\pi_\theta(a_t\mid s_t)}
-\log\frac{\pi_{ref}(a_t\mid s_t)}{\pi_\theta(a_t\mid s_t)}-1
$$

这个估计量与有符号的 $\ell_t$ 不能互换。两条路径都依赖 reference log-prob，但 shaping 的位置和优化目标不同。现在再看 old 与 reference 的职责：

- old log-prob 回答“这条 action 当时有多大概率被采到”；
- reference log-prob 回答“当前行为离约束策略有多远”。

如果误把 reference log-prob 填入 PPO denominator，ratio 就不再描述 on-policy update；如果 rollout 后没有保存或重算 old log-prob，而直接拿更新后的 Actor 结果代替，多个 PPO epoch 的 ratio 会退化为错误值。字段名应明确写成 `old_log_probs` 与 `ref_log_probs`，避免通用的 `log_probs` 在不同阶段被覆盖。

## 一次同步迭代的完整时序

把一次同步 PPO/GRPO iteration 展开，可以得到下面的状态机：

```text
committed actor version v
          │
          ├─ freeze/export ─► rollout version v
          │                         │
prompts ──┴─► expand/sample ────────┤
                                    ▼
                         responses + masks + ids
                                    │
                  ┌─────────────────┼──────────────────┐
                  ▼                 ▼                  ▼
              reward(v_r)       reference(v_ref)   critic(v_c, PPO)
                  │                 │                  │
                  └──────────┬──────┴──────────────────┘
                             ▼
              old log-probs + rewards + values
                             │
                    advantage / returns
                             │
             actor minibatch update(s), critic update(s)
                             │
                  commit actor version v+1
```

同步的含义不是所有 GPU 每一刻都在 barrier，而是：进入 Actor update 的一批 experience 必须能证明来自被允许的 policy version；在 version $v+1$ 发布前，Rollout 不能把部分 layer 更新成新权重后继续生成；失败重试也不能把两个版本的样本悄悄拼成同一组。

## 阶段一：先提交一个可识别的 Policy Version

一次 rollout 开始前，控制器应产生不可歧义的版本标识。仅用 `global_step=42` 往往不够，因为 step 相同仍可能发生故障恢复、不同分支配置或重复发布。更稳妥的版本元组可以包含：

```yaml
policy_version:
  run_id: rl-run-20260818-a
  optimizer_step: 42
  update_attempt: 1
  weight_manifest_hash: sha256:...
  tokenizer_hash: sha256:...
  model_config_hash: sha256:...
```

这里不是要求每个 token 都携带大段 YAML，而是要求 manifest 中有稳定身份，sample record 引用一个短 version ID。只有所有目标 rollout ranks 完成权重加载、校验并清理旧状态后，版本才从 `loading` 进入 `ready`。

发布顺序应是：写入参数，完成设备同步，校验 manifest，再原子更新 active version pointer。先更新 pointer、后慢慢传 layer，会产生最危险的“半新半旧”模型。

## 阶段二：Prompt Batch 怎样扩展成 Sample Group

设 prompt batch 有 $B$ 个问题，GRPO 对每个 prompt 采样 $G$ 个 response。逻辑 batch 大小变为 $B\times G$，但物理执行不一定要把这 $BG$ 条请求同时放进一个 GPU batch。

每条样本必须保留两级身份：

```text
prompt_id = p17
group_id  = runA/step42/p17
sample_id = runA/step42/p17/candidate03
```

`prompt_id` 用来追溯原始数据；`group_id` 保证 GRPO 的均值与标准差只在同一道题的候选中计算；`sample_id` 用于 reward、reference 与重试结果 join。不能依赖数组当前位置，因为 Continuous Batching 会以完成顺序返回，长短 response 还会被重新排序或分桶。

PPO 可以每个 prompt 只采一条，也可以采多条，但它不天然要求 group-relative normalization。不要因为数据结构都叫 `n samples`，就把 PPO 的多样本批次与 GRPO group 当成相同统计单元。

## 阶段三：Rollout 生成时哪些状态会增长

自回归生成期间，每条请求持有 token history、sampling state 和逐层 KV Cache。设层数为 $L$，每 token 的 KV 元素规模与层数、KV heads、head dimension 相关，序列越长，Rollout 显存占用越大。训练 Actor 的 activation 则主要在 Backward 前保留，两者生命周期完全不同。

Rollout 结束后应保留 response tokens 与必要的行为统计，通常不应把生成 KV 当作训练数据跨阶段搬运。Actor、Reference 与 Critic 通常会对完整 token 序列执行 teacher-forced Forward；神经 Reward Model 可能按自己的 tokenizer 重新编码后 Forward，而规则、verifier 或 environment 则读取各自的语义输入与执行记录。它们都不需要直接复用 rollout KV；搬运整份 KV 不但代价高，还会把推理引擎的 layer layout 和精度细节泄漏到训练协议中。

此外，权重更新后旧 KV 必须失效。即使 prompt token 完全相同，KV 也是具体 policy version 的函数：

$$
KV = f_{\theta_v}(x_{\le t})
$$

把 $\theta_v$ 产生的 KV 交给 $\theta_{v+1}$ 延续 decode，会构造一个不存在于任何完整模型中的混合执行。Prefix Cache、远端 KV Store 和本地 block cache 都要参与 version invalidation。

## 阶段四：Reward Evaluation 需要保留可解释的分量

最简单的 outcome reward 对整条 response 返回一个 scalar：

$$
R_i=r_\phi(q,o_i)
$$

但生产数据结构最好保留分量：

```yaml
reward:
  total: 0.72
  components:
    correctness: 1.0
    format: 0.0
    safety: -0.2
    length_penalty: -0.08
  scorer_version: reward-20260801
  status: ok
```

这样才能区分 reward 分布变化来自策略进步、scorer 升级，还是某个解析规则突然大量失败。若使用多个神经 Reward Models，它们可能有不同 tokenizer 和 chat template；应把同一语义 response 交给各 scorer 自己的已版本化 preprocessing，而不是强迫所有模型复用 Actor token IDs。

### Outcome、Process 与 Environment Reward 的边界

Outcome reward 在 response 结束后才产生，常被放在最后一个有效 token，再结合 KL shaping 向前计算 return。Process reward 会在推理步骤边界产生多个分数，需要保存 step-to-token mapping。Agent 环境的 reward 还可能依赖工具返回、超时、动作是否合法和 episode 是否终止。

三者不能只靠一个 `[batch, seq]` 浮点 Tensor 表达。系统至少要区分：

- 哪些 token 是 policy action；
- reward 发生在哪个 action 或 step 后；
- `terminated` 与 `truncated` 的语义；
- scorer error 是否允许进入训练；
- 最终 score 是原始值、归一化值还是已加入 KL 的 shaped reward。

算法可以选择怎样归因，数据层必须先把事实保存完整。

## 阶段五：Reference Forward 只读，却并不便宜

Reference 对 prompt+response 做一次 teacher-forced Forward，取出 response positions 上真实 token 的 log-probs。它不需要采样，也没有 Backward，但仍需读取大模型权重并计算所有有效 token。

为了减少无效计算，常见实现会移除 padding 或把不同长度序列 pack 到紧凑表示。不过 packed sequence 的位置映射必须可逆：输出的 `ref_log_probs[j]` 要准确对应 sample $j$ 的第 $t$ 个 response token，而不能对应 prompt token、padding 或下一条 packed sample。

Reference 与 Actor 如果结构相同，可以通过 LoRA base、权重共享或分时驻留减少显存；这些是物理优化，不得改变逻辑语义。Reference 必须是明确定义的版本，而且其 Forward 应处于 eval mode，关闭会改变结果的训练态行为。

## 阶段六：PPO Critic 怎样产生 Values 与 Returns

Critic 对序列状态 $s_t=(q,o_{<t})$ 预测 $V_\psi(s_t)$。令 $d_t=1$ 表示该 action 后到达真正的 terminal，配合每步 reward $r_t$，GAE 可写成：

$$
\delta_t
=r_t+\gamma(1-d_t)V_\psi(s_{t+1})-V_\psi(s_t)
$$

$$
A_t^{GAE}
=\delta_t+\gamma\lambda(1-d_t)A_{t+1}^{GAE}
$$

return target 则可写成：

$$
\hat{R}_t=A_t^{GAE}+V_\psi(s_t)
$$

在文本 episode 中，prompt tokens 通常不作为 policy actions，padding 更不能参与 value loss。自然 EOS 或环境终止通常令 $d_t=1$；仅因 `max_new_tokens`、超时或资源上限而截断时，往往仍需对可继续状态做价值续估。若统一把所有最后位置当 terminal，长输出的 return 就可能被系统截断策略悄悄改变。

Critic update 与 Actor update 可以使用不同 micro-batch size，因为二者模型大小和 activation pressure 不同。但它们必须消费同一个 experience manifest，不能在 shuffle 后丢失 sample/token 对齐。

## 阶段七：PPO Actor Update 实际读取哪些字段

对每个有效 response token，clipped surrogate objective 为：

$$
L_t^{clip}(\theta)
=
\min\left(
\rho_t(\theta)A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
$$

Actor update 至少读取 `input_ids`、position/attention 信息、`response_mask`、`old_log_probs` 与 `advantages`。若 KL 已在 reward shaping 阶段进入 returns，这里读取的是已处理后的 advantage；若 KL 直接进入 loss，还需要 `ref_log_probs`，并按选定的 estimator 单独计算 regularization。Actor 以当前 $\theta$ 计算 `new_log_probs`，再构造 ratio、clip fraction 与 policy loss。

一批 rollout 可以拆成多个 minibatches，并重复若干 PPO epochs。此时 `old_log_probs` 始终来自 rollout snapshot，不能在每个 epoch 后被新 Actor log-probs 覆盖。随着多个 epoch 更新，当前策略逐渐离开 behavior policy，ratio 与 approximate KL 会增长；clip fraction 是判断这批经验是否被过度复用的重要信号。

训练框架还要明确 loss reduction。按 token 平均、先按 sequence 平均再按 batch 平均，会给长 response 不同权重。只要改变 packing 或 micro-batch 划分就改变 loss，说明 normalization contract 没有固定。

## GRPO 去掉 Critic 后，系统并没有只剩一张模型

DeepSeekMath 提出的 GRPO 对同一 prompt 采样 $G$ 个输出，用组内 reward 估计 baseline，从数据流中移除可训练 Critic。对 outcome reward，原始论文使用：

$$
\hat{A}_i
=
\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)}
$$

这是原论文的数学写法，它没有在分母显式加入 $\epsilon$。若组内 reward 完全相同，标准差为 0，工程实现必须预先规定行为，例如使用 $\max(\operatorname{std},\epsilon)$ 使该组 advantage 为 0，或把该组标为无训练信号并跳过；不能任由 `NaN` 进入跨 rank reduction。随后再把 normalized reward 赋给 response 中各 token 的 advantage。这样省掉一张与 Policy 相近规模的 value model、它的 optimizer state 和一次训练路径，但代价转移到了 Rollout：每个 prompt 必须生成足够多候选，并等 group reward 聚齐后才能计算 advantage。

GRPO 仍然需要：

- Actor/old policy 产生并解释 on-policy samples；
- Reward Model、规则或 verifier 提供组内 scores；
- Reference 在启用 KL regularization 时提供约束概率；DeepSeekMath 的原始 GRPO 目标包含这一项；
- rollout-to-train 的权重转换与版本屏障；
- group-aware shuffle、失败处理和 normalization。

所以“GRPO 没有 Critic”是模型状态减少，不是系统复杂度消失。

### Group 是不可随意拆散的统计单元

假设同一 prompt 的 8 条候选被分到两个 worker。前 4 条先返回并独立归一化，后 4 条稍后再归一化，得到的 advantage 与 8 条共同计算不同。若只保留成功完成的候选，长序列、工具调用或难例更容易失败，剩余 group 还会带选择偏差。

因此控制器需要定义 group completion policy：

1. 所有 $G$ 条都成功后提交完整 group；
2. 对可重试错误使用相同 policy version 重新生成指定 candidate；
3. 达到重试上限后整组丢弃，或采用算法明确允许的 partial-group 规则；
4. 记录实际 group size，禁止静默补入其他 prompt 的样本。

当组内 reward 标准差接近 0 时，直接除法会产生不稳定数值。实现必须规定 epsilon、是否跳过零方差 group，以及相关指标；不能让不同 ranks 各自根据本地 group fragment 做决定。

## Experience Record 是整条 Pipeline 的事实来源

一条可审计的 experience 不应只是 `input_ids` 与 `reward`。可以把逻辑 schema 设计为：

| 字段 | 粒度 | 生产者 | 主要消费者 |
| --- | --- | --- | --- |
| prompt/sample/group IDs | sample | 数据层/控制器 | 所有角色、重试与审计 |
| prompt/response token IDs | token | Actor tokenizer/Rollout | Actor、Reference、Critic、Reward adapter |
| response/action mask | token | Rollout/postprocess | loss、KL、value、metrics |
| termination reason | sample | Rollout/environment | return、末状态价值续估与质量分析 |
| policy version | sample/batch | 版本控制器 | admission、staleness 检查 |
| old log-probs | action token | Rollout 或 frozen Actor pass | PPO/GRPO ratio |
| ref log-probs | action token | Reference | KL penalty/monitoring |
| raw reward components | sample/step | Reward subsystem | aggregation、审计 |
| values/returns | action token | Critic/advantage stage | Critic 与 Actor update |
| advantages | action token | advantage stage | Actor update |
| valid lengths/packing map | sample/token | batcher | 所有 model Forward |

最好把 immutable facts 与 derived tensors 分开：tokens、versions、raw scorer outputs 是事实；shaped reward、advantage、return 是在某套 hyperparameters 下可重算的派生物。这样调整 $\beta$、$\gamma$、$\lambda$ 或 normalization 时，不必重新生成昂贵 rollout，也能进行离线一致性检查。

## Tokenizer 与 Chat Template 是算法契约的一部分

很多隐蔽错误发生在模型计算之前。例如 Rollout 使用模板 A 生成，Actor update 重新用模板 B 把字符串编码；或 Reward Model 需要自己的 BOS/EOS 规则，却直接消费 Actor padding 后的 IDs。字符串看起来相同，不代表 action positions 相同。

稳妥做法是：

- Actor/Rollout 共享一份已哈希的 tokenizer 与 chat-template contract；
- experience 保存真实执行过的 token IDs，而不是只保存可重新编码的文本；
- Reward adapter 若重新 tokenize，保留从语义 sample 到 scorer input 的映射；
- response start、EOS、tool tokens、image placeholders 等边界显式记录；
- tokenizer 或 special-token 配置改变时，创建新 run/version，而不是热更新旧 experience。

一条简单但有效的验收是：对随机样本，Rollout IDs 送入训练 Actor teacher forcing 后，逐位置 token 必须完全相等，response mask 的第一个 1 必须正好落在第一个被采样 action 上。

## Generation Log-prob 与 Training Log-prob 为什么可能不同

理想情况下，rollout engine 在采样 token 时记录的 log-prob 等于同一权重、同一输入下训练引擎 teacher forcing 的 log-prob。实际系统可能因为以下原因出现漂移：

- Rollout 使用量化权重，Actor 用 BF16/FP32 master weights；
- 两边 TP layout、kernel、softmax accumulation precision 不同；
- sampling 前应用了 temperature、top-k/top-p 或 logits processor；
- 一边记录 raw model log-prob，另一边记录截断后 sampling distribution；
- padding、position IDs、RoPE scaling 或 attention mask 不一致；
- 权重同步没有覆盖 tied weights、MoE experts 或 adapter。

因此必须先定义 `old_log_probs` 的概率空间。PPO ratio 应比较同一 action distribution；用于复现采样的 filtered distribution 与用于 policy objective 的 raw policy distribution 不能随意混用。

工程上常选择由训练 Actor 对冻结 experience 再做一次 Forward 来计算 old log-probs，以统一数值路径。这次计算必须发生在任何 Actor update 之前，完成后结果随 experience 一起 sealed，在后续 PPO epochs 中保持不可变；不能每个 epoch 重算 denominator。也可以复用 Rollout 输出以节省计算，但必须做逐 token parity test，并把允许误差与不支持的量化模式写进配置。

## 训练表示与推理表示为何需要 Reshard

训练 Actor 可能采用 FSDP/ZeRO-3：参数、梯度和 optimizer state 沿 DP group 分片，计算前临时 AllGather。Rollout 为了提高生成吞吐，可能使用较小 TP、更多 replicas，且不保存 optimizer。即使参数数值相同，物理布局也可能完全不同：

```text
training layout:
  PP=2, TP=4, DP=2, optimizer/grad/param shards

rollout layout:
  PP=1, TP=2, replicas=8, inference-only weights + KV cache
```

Actor 到 Rollout 的更新因此不是简单 `state_dict` 复制，而是 logical parameter name、global shape、训练 shard、推理 shard 与 fused layout 之间的转换。QKV fusion、gated MLP 的 gate/up 拼接、vocab padding、MoE expert placement 和 tied embedding 都需要显式规则。

HybridFlow 的 3D-HybridEngine 展示了一种做法：训练和生成在同一组设备上使用不同并行组，通过参数重分片切换阶段。verl 也把 trainer-to-rollout update 视为显式的 engine/checkpoint transfer 阶段，而不是普通的 `state_dict` 复制；具体 worker 与 sharding-manager 名称在不同版本间已有变化，因此部署文档还应固定框架版本或源码 commit。无论框架名称如何，正确性条件相同：Rollout version 的每个 logical tensor 都必须来自同一个 committed Actor version。

## 参数、Optimizer、KV 与 Experience 有四种不同生命周期

把显存里所有东西都叫“模型状态”，容易导致错误的 offload 与恢复策略：

| 状态 | 典型生命周期 | 是否跨 iteration | 版本依赖 |
| --- | --- | --- | --- |
| Actor parameters | 多个 optimizer steps | 是 | current policy version |
| Optimizer/gradient states | 训练 run | 是 | Actor shard/layout 与 step |
| Rollout KV Cache | 单次请求或共享 prefix | 通常否 | 精确绑定 rollout policy version |
| Experience batch | rollout 到若干 update epochs | 短期 | old/ref/reward versions |

Reference、Reward、Critic 又各有独立参数生命周期。Checkpoint 若只保存 Actor weights，却没有保存 optimizer step、reference/reward version、prompt cursor 和尚未消费的 experience 状态，只能恢复“一个模型文件”，不能保证从同一 RL iteration 继续。

## 角色映射：Colocate、Standalone 与 Hybrid 没有固定赢家

角色逻辑不决定 GPU placement。同一套 PPO dataflow 可以映射成不同物理方案。

### 全部 Colocate 适合显存可容纳、阶段明显串行的场景

Actor、Rollout、Reference、Critic、Reward 分时复用同一组 GPU，可以避免为每个只偶尔工作的角色预留整组设备；其中 Rollout 是 Actor snapshot 的额外物理表示，而不是第五类独立逻辑模型。代价是频繁 load/offload、allocator fragmentation、角色切换和无法跨阶段 overlap。若这些物理权重表示与 optimizer 无法同时驻留，还需要 CPU/NVMe offload。

### Standalone Pools 适合各阶段吞吐可独立扩展的场景

Rollout、Reward、Actor Training 分别常驻不同 GPU pools，能把完成的 samples 流式送去打分，也能为慢 Reward Model 独立增加 replicas。代价是 Actor 权重必须跨 pool 同步，experience 需要跨网络传输，空闲阶段还可能造成资源碎片。

### Hybrid Placement 把逻辑角色合并成少数 Worker Groups

例如 Actor+Rollout 共享 GPU 与权重，Reference 与 Actor base 共享，Critic+Reward 分时共置。verl 的编程模型将 Actor/Rollout/Reference、Critic、Reward 映射到 resource pools；OpenRLHF 也支持 colocated 与分离 placement。这里没有通用最优答案，应根据各阶段 profile 和模型大小求解，而不是照抄 GPU 数量比例。

资源规划至少要满足：峰值显存不 OOM、最慢 stage 不长期积压、权重切换/同步成本可接受，以及故障域不会把同一份唯一状态一起丢失。

## 同步边界是一条 Versioned State Machine

一个安全的同步 rollout loop 可以抽象成：

```text
ACTOR_COMMITTED(v)
  -> EXPORTING(v)
  -> ROLLOUT_LOADING(v)
  -> ROLLOUT_READY(v)
  -> GENERATING(v)
  -> EXPERIENCE_SEALED(v)
  -> UPDATING_FROM(v)
  -> ACTOR_COMMITTED(v+1)
```

每次状态转换都有完成条件：所有 shards 已收到、checksum 一致、旧 KV 已清除、所有 samples 已写入 manifest、group 已完整，或梯度 step 已成功且新的内存态权重完整可见。超时不是“继续下一步”的理由，而应进入可恢复的失败状态。

这里的 `ACTOR_COMMITTED` 指本轮可供后续计算和 rollout 发布的原子内存版本，不要求每轮都写一份持久化 checkpoint。后者通常每若干轮执行一次；频率越低，故障后需要重放的 iteration 越多。权重 active pointer 与持久化 checkpoint 必须使用不同状态名，避免把“可以开始下一轮”误判成“进程退出后一定可恢复”。

同步训练通常在 `EXPERIENCE_SEALED(v)` 后才允许 Actor 改变；之后多次 minibatch update 都清楚地以 $v$ 为 behavior version。若希望 Rollout 与 Actor update overlap，就进入异步或 one-step-off-policy 设计，需要额外 staleness budget 和权重发布协议，不能只删除 barrier。

## 一次失败不能被伪装成低 Reward

分布式 RL Pipeline 的失败比普通训练复杂，因为一次 iteration 会调用多个模型服务和外部环境。

### Rollout Worker 失败

控制器应根据 sample IDs 找到未完成 candidates，并在同一 policy version 上重试。若该版本已经从所有 Rollout workers 卸载，要么重新加载它，要么丢弃受影响 experience；不能直接用最新策略补齐旧 group，却仍标记成旧版本。

### Reward 或 Verifier 失败

网络超时、解析异常、sandbox 崩溃应记录为 `scorer_error`。只有算法明确把某种环境错误定义为惩罚时，才能转成负 reward。否则应重试、隔离或丢弃样本，防止模型学会迎合基础设施故障。

### Actor Update 中途失败

若 optimizer step 不是原子提交，就需要从上一个 committed checkpoint 恢复参数、optimizer、scheduler 与 RNG，再决定是否重放相同 experience。不能仅加载部分最新 model shards；不同 ranks 的 step 不一致会让下一次 collective 正常返回，却训练一个数值拼接模型。

### 重复消息与迟到结果

跨进程队列通常是 at-least-once delivery。Join 层应以 `(run_id, iteration_id, sample_id, role, role_version)` 去重，迟到的旧版本 Reward 结果不能覆盖新 attempt。幂等不是只防止存储重复，也防止同一 experience 被 Actor 多更新一次。

## Checkpoint 必须保存“训练位置”，而不只是 Actor 权重

可恢复 RL checkpoint 至少包含：

- Actor parameters、optimizer、scheduler、scaler 与 RNG；
- Critic parameters/optimizer（若使用 PPO）；
- active reference 与 reward/scorer manifests；
- committed iteration、policy version 与 prompt sampler cursor；
- tokenizer、chat template、sampling 和 reward aggregation 配置哈希；
- 已 sealed 但尚未消费的 experience manifest，或明确声明这些数据会被丢弃；
- 并行布局与 logical tensor metadata，支持安全 reshard。

恢复时要先决定语义：从上一个完整 iteration 重跑 rollout，还是复用 sealed experience 继续 update。前者成本高但边界简单；后者需要证明所有 old/ref log-probs、reward 和 advantage 配置仍可解释。两种策略都可行，最危险的是系统没有选择，却随机复用了磁盘上能找到的中间文件。

## 性能分析先找 Critical Path，再谈 GPU 利用率

一次同步 iteration 的简单延迟模型是：

$$
T_{iter}
\approx
T_{sync}
+T_{rollout}
+T_{prepare}
+T_{actor\_update}
+T_{critic\_update}
+T_{checkpoint,amortized}
$$

$T_{checkpoint,amortized}$ 是持久化 checkpoint 摊到每轮的平均成本；未触发落盘的 iteration 中该项为 0。若 Reference、Reward 与 Critic Forward 并行，$T_{prepare}$ 更接近它们的最大值加上 join/transfer，而不是简单求和：

$$
T_{prepare}
\approx
\max(T_{ref},T_{reward},T_{value})
+T_{reshard/join}
$$

Colocation 会让多个阶段串行，从而降低峰值显存但延长 critical path；Standalone 能 overlap，却可能让 Actor 等待最后一条长 rollout 或慢 reward。单看平均 GPU utilization 会掩盖这种依赖：一组 GPU 100% 忙于生成，并不说明 Actor training 没在空等。

Profiler timeline 应同时标注 `iteration_id`、role、policy version、sample/group counters 和 transfer phase，才能回答“谁在等谁”。只有 kernel 时间线而没有数据流身份，无法区分计算慢、队列阻塞和版本 barrier。

## 指标要覆盖数据量、等待时间与算法漂移

建议按 stage 建立一组可关联指标，而不是只报告训练 tokens/s。

### Rollout 指标

- prompt、response 与有效 action tokens 数；
- generation tokens/s、首条/整批完成延迟；
- response length 分布、EOS/length truncation 比例；
- KV peak、cache hit、preemption/recompute；
- 每个 policy version 的成功、重试和丢弃 samples；
- GRPO group completion latency 与有效 group size。

### Preparation 指标

- Reward/Reference/Critic 各自 Forward tokens/s 与 queue time；
- scorer error、timeout 与各 reward component 分布；
- old/ref log-prob parity 误差；
- pack/unpack 与跨 pool transfer bytes/time；
- advantage mean/std、zero-variance group 比例。

### Update 指标

- policy/value loss、entropy、approximate KL 与 reference KL；
- ratio mean/tails、clip fraction、gradient norm；
- Actor/Critic microbatch time、collective time 与 MFU；
- optimizer overflow/skipped step；
- rollout-to-update policy lag 与 experience age。

### 端到端指标

- iteration wall time 及各 stage critical-path 占比；
- sampled tokens / updated tokens / accepted groups；
- 每个 committed update 的 GPU-hours；
- 版本切换、失败恢复和 checkpoint 时间；
- held-out task、reward hacking probe 与安全 guardrail 指标。

Reward 上升不能单独证明训练改善；它可能来自长度偏置、格式投机、scorer 漂移或数据泄漏。系统指标必须与独立质量评测一起观察。

## 正确性验证应从一个 Sample 逐层扩展

### 第一层：单样本 Token Contract

固定一个 prompt 和 response，不做 sampling。验证 Actor、Reference、Critic、Reward adapters 看见的语义文本一致；Actor/Rollout 的 token IDs、response start、EOS 与 masks 完全符合预期。对 padding side、packed sequence、超长截断分别建立 golden cases。

### 第二层：单版本 Log-prob Parity

用同一权重与未过滤 logits，对 Rollout 与训练 Actor 逐 token 比较 raw-policy log-probs。分别测试 eager/graph、单请求/连续批处理、不同长度、TP size 和允许的 dtype。若实际 behavior policy 还应用 temperature、top-k/top-p 或 logits processor，应再建立一组测试，验证变换后的采样分布与记录下来的 behavior log-prob 一致。若使用量化 Rollout，必须把预期数值差异写成明确 tolerance，并评估它对 PPO ratio 的影响。

### 第三层：单迭代算法对照

在小模型与小 batch 上，用单进程 reference implementation 保存 raw tensors；再运行分布式实现，比较 reward aggregation、advantages、returns、loss、gradient norm 和一步更新后的参数。这里关注语义等价，不追求吞吐。

### 第四层：并行布局等价

改变 FSDP/TP/PP/DP mapping，确认 logical parameters、generated tokens（在固定可复现设置下）、teacher-forced logits 和一步 update 仍在容差内。Actor-to-Rollout reshard 后抽样重建完整 tensors，比对 manifest hash 与 tied/fused parameters。

### 第五层：故障注入与长时间训练

在权重传输、group rollout、Reward Forward、Optimizer Step 和 Checkpoint publish 各阶段注入 worker crash/timeout。验证系统要么提交完整 iteration，要么从可识别边界恢复，不产生混合版本。长跑还要监控 reward 分布、KL、entropy、length、重复率与 held-out quality，寻找缓慢漂移。

## 常见实现错误往往不会立即 Crash

### 把 Prompt Token 计入 Policy Loss

response mask 偏移一位，就会让 Actor 对数据集 prompt 做监督学习式更新，同时稀释真正 action 的 advantage。loss 仍会下降，因此必须检查 mask 覆盖的 token 数和边界样本。

### 把 Old 与 Reference Log-prob 覆盖到同一字段

两者 shape 相同，单元测试只看 shape 很容易通过。应使用强类型字段、producer metadata，并在 loss 前断言 version contract。

### 在 Group 完成前做 GRPO Normalization

按 worker local batch 求 mean/std 会改变算法；按 completion order 拼 group 则会混入其他 prompt。所有 normalization 必须基于 `group_id` 的完整集合。

### 权重更新后继续复用旧 KV

Prefix Cache 命中率可能很好，但这些 KV 属于旧 policy。更新边界必须清理本地与远端 cache，或把 policy version 纳入严格隔离 namespace。

### Reward Error 默认填 0

这会让基础设施不稳定性成为训练标签。错误状态、业务负 reward 与正常 0 分必须是三种不同值域。

### 只按 Sequence Count 规划吞吐

RL rollout 长度具有长尾，$B\times G$ 条 samples 并不代表相同 token 工作量。调度、反压和成本都应使用 prompt/response tokens、KV bytes 与实际完成时间。

## 一条可落地的同步实现顺序

如果从零搭建，不宜先追求异步 overlap。可以按以下顺序收敛：

1. 用单进程小模型实现 prompt→rollout→reward→advantage→one-step update；
2. 固化 Experience schema、token/mask 和 version 字段；
3. 分离 old、reference 与 current log-probs，建立 golden-tensor tests；
4. 先把各角色放在独立、清晰的 worker groups，跑通同步 barrier；
5. 引入 FSDP/TP 等单角色并行，并验证布局等价；
6. 实现 Actor-to-Rollout 权重 manifest、reshard 和原子发布；
7. 加入 checkpoint、幂等 join、重试与故障注入；
8. 完成分阶段 profiler 和端到端质量指标；
9. 最后根据 critical path 选择 colocation、offload、streaming reward 或资源池重映射。

每次只改变一层。若同时更换 Rollout 引擎、量化权重、启用异步、改 advantage estimator 再调整 reward，出现训练漂移时几乎无法归因。

## 本文刻意留下的三个边界

系统数据流清楚后，还有三类问题值得单独展开。

第一，**Rollout Engine** 需要比较训练与生成共用 GPU、分时 Hybrid Engine、独立推理池的容量与调度，并讨论长尾生成、Continuous Batching 和 Agent environment。

第二，**Policy Weight Sync** 需要深入 logical tensor mapping、FSDP/Megatron 到 vLLM/SGLang 的 reshard、全量/分桶/增量传输、checksum、CUDA IPC/RDMA 与原子激活。

第三，**Async RL** 需要正式定义 policy staleness、experience age、partial rollout、off-policy correction、背压与 bounded-lag，而不是把同步 barrier 简单删除。

这三类优化都建立在本文的 versioned experience contract 上。没有它们，系统吞吐越高，只会越快地产生无法证明来源的训练数据。

## 小结

RLHF/GRPO 的困难不只是“显存里同时放几张大模型”，而是要让不同计算角色围绕同一批 experience 保持一致：

1. Actor 是可训练 policy，Rollout 是其某个 committed snapshot 的生成执行者；
2. old policy 为 importance ratio 提供行为概率，reference policy 为 KL 提供约束坐标，两者不能混用；
3. Reward 产生评价事实，Critic 为 PPO 估计 baseline；GRPO 去掉 Critic，却增加了完整 sample group 的同步要求；
4. token、mask、sample/group ID、termination、log-probs、reward components 和版本必须组成可审计的 Experience Record；
5. 训练 shards、推理权重、KV Cache 与 experience 有不同生命周期，权重更新必须让旧 KV 失效；
6. Colocate、Standalone 与 Hybrid 只是物理映射，不能改变逻辑角色和算法依赖；
7. 只有经过单样本、单迭代、并行等价、故障注入和长跑质量验证，才能把“Pipeline 跑通”升级为“训练语义可信”。

当这张数据流图被明确描述后，资源优化才有稳定落点：可以移动角色、改变并行布局、压缩权重传输或重叠阶段，而不必猜测一次性能改动是否同时改坏了训练目标。

## 参考资料

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)
- [DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales](https://arxiv.org/abs/2308.01320)
- [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143)
- [verl：PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
- [verl：HybridFlow Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
- [verl：Engine Workers](https://verl.readthedocs.io/en/latest/workers/engine_workers.html)
- [verl：Reward Loop](https://verl.readthedocs.io/en/latest/advance/reward_loop.html)
- [verl：Full Determinism for Reproducible RL Training](https://verl.readthedocs.io/en/latest/advance/determinism.html)
- [verl：Rollout KV Cache Offload via Mooncake-Store](https://verl.readthedocs.io/en/latest/perf/rollout_kv_offload.html)
