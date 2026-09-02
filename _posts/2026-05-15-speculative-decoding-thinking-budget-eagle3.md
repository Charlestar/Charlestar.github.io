---
layout: post
title: "推理模型的推测解码：Thinking Budget 与 EAGLE-3"
subtitle: "把生成工作量与执行效率分开优化"
date: 2026-05-15 12:00:00 +0800
last_modified_at: 2026-09-02
author: iStar
catalog: true
series: speculative-decoding
series_order: 30
technology_year: 2025
mathjax: true
tags: [推测解码, EAGLE, LLM推理]
---

推理模型的响应时间经常被两个问题同时影响：模型可能先生成很长的推理轨迹，而这些 token 又必须自回归地逐个产生。Thinking Budget 与 EAGLE-3 分别作用在这两个问题上，却很容易被混成同一种“加速”。

Thinking Budget 决定模型被允许或被引导生成多少推理工作，可能改变答案质量和输出长度；EAGLE-3 属于推测解码，目标是在保持目标解码语义的前提下，用更少的目标模型串行轮次生成同样的 token 分布。

简单说：一个决定 **要做多少工作**，一个优化 **这些工作怎样执行**。只有先分开，才能判断一次延迟下降究竟来自模型少想了，还是相同输出真的算得更快。

## 推理请求中有哪些 token

不同模型和 API 的呈现方式不同，但可以把一次响应粗略分成：

```text
input prompt
    │
    ▼
reasoning / thinking phase
    │
    ├─ 可能以文本 token 暴露
    ├─ 可能以摘要或签名形式传递
    └─ 可能完全不展示给用户
    │
    ▼
final answer / tool call
```

即使推理内容不可见，它仍可能消耗模型计算、上下文和配额。服务端看到的 `max_output_tokens`、可见答案 token 与 reasoning token 不一定是同一个计数口径，做预算前必须以具体模型 API 的说明为准。

推理模型也不一定严格先“想完”再答；它可能在推理、工具调用和答案之间多次切换。因而预算耗尽时的行为不能只靠截断字符串处理。

## Thinking Budget 控制的是什么

Thinking Budget 是服务或模型暴露的一类生成控制。不同系统可能把它表示为：

- 最大 thinking token 数；
- `low/medium/high` 等离散 reasoning level；
- 总输出 token 中预留给推理的部分；
- wall-clock deadline 或计算预算；
- 由模型内部根据任务动态决定。

这些形式并不等价。以 token 限额控制工作量较直接，level 则是模型特定策略；deadline 还受排队与硬件速度影响。不能把一个 API 的 `thinkingBudget=1024` 机械翻译成另一模型的同等质量配置。

### 预算改变的是目标模型行为

设预算为 $B$，目标模型实际生成的推理 token 数为 $R(B,x)$，任务输入为 $x$。端到端质量与耗时可写成：

$$
Q=Q(B,x),\qquad
T=T_{queue}+T_{prefill}+T_{decode}(R+O)
$$

$O$ 是最终答案 token。调小 $B$ 可能降低 $R$，从而减少 decode 工作；但 $Q$ 也可能变化。它属于质量—成本策略，不是语义无损的系统优化。

### 更多思考不保证更好

预算只是上限或引导，不是质量分数。模型可能：

- 很快找到正确路径，长预算只增加重复；
- 在错误路线中持续展开；
- 复杂任务因预算太短无法完成验证；
- 简单任务根本用不完预算。

因此，不能以“平均 thinking tokens 更多”证明模型更聪明，也不能以“更短”直接证明更高效。需要在任务正确率、答案完整性、超时和成本上共同评估。

## Speculative Budget 控制的是另一件事

推测解码每轮先提出若干候选，再由目标模型批量验证。这里也有一个预算，例如：

- 线性草稿的 token 数；
- 候选树的总节点数；
- 最大树深；
- drafter 能使用的时间或显存。

记为 $K$。增大 $K$ 不会要求目标模型“想得更久”，只是让 runtime 一轮准备更多候选。若候选被接受，目标模型串行轮次减少；若大量候选被拒绝，额外草稿与验证工作会浪费。

两个预算可以画成二维坐标：

```text
                         speculative budget K
                              small ─────► large
                        ┌────────────────┬────────────────┐
thinking budget B small │ 少想 + 短草稿  │ 少想 + 宽验证  │
                  large │ 多想 + 短草稿  │ 多想 + 宽验证  │
                        └────────────────┴────────────────┘
```

横向比较应保持目标模型输出策略相同，判断执行是否变快；纵向比较允许质量变化，用来选择任务预算。把左上和右下直接比较，无法知道收益来自哪一维。

## 为什么推理轨迹可能更难起草

普通对话中有大量常见搭配和模板句，较小 drafter 容易预测。长推理则可能出现：

- 中间结论改变后续方向；
- 计算结果只差一个数字，token 分布就分叉；
- 自我检查或回溯；
- 代码、公式与自然语言交替；
- 工具结果在中途注入上下文。

这些位置的局部分布更依赖目标模型内部状态。独立小模型如果只学到表面语言风格，可能在“因此”“下一步”之类衔接词上猜对，却在关键数字、运算符或工具参数处被拒绝。

这不是说推理模型一定有更低接受率。目标/drafter 配对、训练数据、temperature 和任务都可能改变结果；正确做法是按阶段与任务统计 accepted length，而不是从“reasoning”标签直接下结论。

## EAGLE 从 token 草稿转向特征草稿

传统 draft model 只看 token 上下文，独立预测后续 token。EAGLE 的出发点是：目标模型当前 forward 已产生丰富的隐藏特征，drafter 若能利用这些特征，候选可能更贴近目标模型。

EAGLE-1 在特征层做自回归外推，使用目标模型靠近顶部的 contextual feature，并结合 token 信息生成候选。EAGLE-2 根据 drafter 的置信度动态组织草稿树，让验证预算更偏向有希望的分支。

这类方法仍遵循推测解码的基本边界：EAGLE 只负责 proposal，目标模型负责 verification；接受/拒绝与修正机制保证最终解码语义。EAGLE 输出不是无需验证的答案。

## EAGLE-3 改变了什么

EAGLE-3 论文针对前代方法的两个限制做了调整。

### 从受约束的 feature prediction 转向直接 token prediction

前代 EAGLE 训练 drafter 去逼近目标模型的未来 feature，再从 feature 得到 token。论文观察到，仅增加训练数据时，这个 feature prediction 约束会限制扩展收益。

EAGLE-3 不再把精确预测目标 feature 作为中间约束，而是让 drafter 直接优化未来 token prediction。目标不是复刻教师每个内部向量，而是产生更容易被目标模型接受的候选。

### 融合低、中、高层特征

目标模型顶层表示高度服务于 next-token prediction，不一定单独包含最适合多步草稿的全部信息。EAGLE-3 将不同深度的隐藏特征融合，给 drafter 同时提供较局部和较抽象的语义信号。

概念数据流可以写成：

```text
target model forward
  ├─ low-layer feature  ─┐
  ├─ mid-layer feature  ─┼─► fusion ─► EAGLE-3 drafter ─► token tree
  └─ high-layer feature ─┘                              │
                                                       ▼
                                              target verification
```

不同论文/实现选择哪些层、怎样归一化和融合，必须与训练 checkpoint 对齐；“多层融合”不是把任意几层 tensor 拼在一起就能工作。

### Training-time Test

普通 teacher forcing 训练时，模型总是看到真实历史 token；推理时却会看到自己生成的候选，错误会沿路径累积，这就是 train-test mismatch。

EAGLE-3 的 training-time test 在训练中模拟 drafter 的测试时自回归过程，使训练目标更接近实际会访问的候选状态。可以把它理解为：不仅教模型在标准答案前缀上预测，还让它面对自己会走到的草稿路径，再学习哪些 token 更可能被目标接受。

这也是为什么训练数据必须包含目标模型的输出/特征，并严格对应 tokenizer、chat template 与 checkpoint revision。换了 base model 的权重，hidden feature 分布和目标 token 分布都会改变，旧 drafter 不再具有相同契合度。

## 一棵候选树怎样被验证

线性 speculative decoding 每轮只有一条候选路径。树形 drafter 可以在不确定位置保留多个分支：

```text
                 A
              /     \
             B       C
           /  \       \
          D    E       F
```

每个节点表示一个候选 token，根到节点是一段完整前缀。runtime 需要：

1. 给节点分配 position；
2. 构造 tree attention mask，节点只能看到自己的祖先；
3. 将多个节点组织进一次目标 forward；
4. 按目标概率与采样规则确定被接受路径；
5. 只提交路径上的 KV，释放其他分支。

树更宽可以覆盖更多候选，但验证 token 数、临时 KV、mask 与采样开销也增加。EAGLE-2/3 的高质量或动态树可以提高预算利用率，却不能消除这项折中。

## Thinking Budget 如何影响 EAGLE-3

从算法上看，EAGLE-3 不需要知道一个 token 属于“思考”还是“最终回答”：它只根据当前目标特征提出后续候选并接受验证。

从 workload 看，Thinking Budget 会改变：

- 总共需要 decode 多少 token；
- reasoning/answer 各阶段的 token 分布；
- 不同任务在 batch 中占用多久；
- EAGLE-3 实际遇到的轨迹与训练数据是否匹配。

如果预算很短，固定的 drafter 权重和临时 KV 可能还没摊薄就结束；预算较长时，可加速 token 更多，但轨迹中的高不确定区也可能拉低接受长度。因此两者不存在“预算翻倍，EAGLE 收益也翻倍”的线性关系。

## 服务端应怎样联合控制

最稳妥的设计是保留两个独立控制回路。

### 质量/成本控制器

根据任务类型、用户等级、deadline 与质量目标决定 Thinking Budget。它监控：

- 任务正确率/成功率；
- 答案完整性与格式；
- thinking、answer 和 tool token 数；
- 总延迟、超时率与成本。

### 执行控制器

在不改变目标解码策略的前提下调整 speculative budget。它监控：

- proposed、accepted 和 committed token 数；
- mean accepted length；
- drafter 与 target verify latency；
- 临时 KV/显存；
- TPOT、吞吐和 SLO goodput。

一个简化策略是：

```text
if request is close to deadline:
    quality policy may lower remaining thinking allowance

if recent accepted length is low or draft cost is high:
    runtime lowers draft tree budget

if acceptance is stable and verification has spare capacity:
    runtime cautiously increases tree budget
```

两者的归因必须分开。第一条可能改变答案，后两条原则上只改变执行效率。所有动态调整还应有上下限、冷却窗口和回退基线，避免因短期噪声反复振荡。

## Budget 耗尽时需要明确语义

简单在第 $B$ 个 token 后硬切流，可能留下未闭合 JSON、半个工具参数或不完整答案。服务 API 应定义：

- 预算是软引导还是硬上限；
- 是否要求模型从推理切换到最终回答；
- tool call 进行中能否继续完成；
- 总输出上限是否另算；
- 耗尽后返回 `incomplete`、继续回答还是报错；
- 多轮请求是否需要把 reasoning signature/state 带回。

这些是产品与模型契约，不能由 EAGLE-3 runtime 擅自决定。推测验证还必须让 EOS、stop、grammar 和预算计数只按已提交 token 推进，不能把被拒候选也算作用户输出。

## 怎样训练与配对 EAGLE-3

EAGLE-3 drafter 不是一个通用插件 checkpoint。它至少与以下对象绑定：

- base/target model 的精确 revision；
- tokenizer 和 vocabulary；
- 取用的 target layers 与 feature dimensions；
- chat template 和训练输出分布；
- reasoning mode、语言和任务域；
- serving runtime 的 tree layout 契约。

训练数据常需要由目标模型生成或捕获其 hidden features，使 drafter 学习目标实际分布。若训练全是简短聊天，线上却是长数学推理，平均 loss 尚可也不保证关键阶段的接受率。

SpecForge 提供 EAGLE-3 训练和接入 SGLang 的工程路径；官方 EAGLE 仓库也发布配套实现与部分 checkpoint。选用时应检查“official/compatible”标记和目标模型 revision，不能只看模型名称相似。

## 一套二维实验把两种收益分开

准备简单问答、数学、代码和工具调用四类任务，为每类固定输入集。实验分三步。

### 1. 只扫描 Thinking Budget

关闭推测解码，使用多个 $B$，记录：

```text
quality / pass rate
thinking tokens
answer tokens
tool success
TTFT / TPOT / E2E
timeout / incomplete rate
```

先得到质量—token—延迟曲线，选择符合质量底线的预算范围。

### 2. 固定 B，只扫描 EAGLE tree budget

在每个选定预算下，比较 vanilla decode 与多个 $K$：

```text
proposed nodes per step
accepted / committed length
draft latency
verify latency
TPOT and throughput
peak memory
```

Greedy 场景验证 token 一致性；sampling 场景验证目标分布与任务质量统计，而不是要求同 seed 文本逐字一致。

### 3. 回放真实 QPS

低并发的 latency speedup 不一定转化为高并发 goodput。按线上输入、预算和到达率分布回放，寻找 EAGLE 相对 vanilla 的交叉点，并检查 drafter 权重/临时 KV 是否挤压目标并发。

最终报告应把结论写成：

```text
预算策略减少了多少目标 token，质量变化多少；
在同一预算和目标语义下，EAGLE-3 又减少了多少执行时间。
```

论文报告的最高 speedup 来自特定模型、硬件、任务和 tree 设置，不应当作任意 reasoning workload 的固定承诺。

## 小结

Thinking Budget 与 EAGLE-3 可以同时用于推理模型，却解决不同问题。预算策略控制目标模型愿意投入多少推理工作，天然牵涉质量；EAGLE-3 融合目标模型多层特征、直接预测 token，并用 training-time test 缩小训练/推理差异，目的是更高效地产生可被目标验证的候选。

系统设计中应保持两个独立旋钮、两套指标和两阶段实验。先在无推测基线下找到合适的质量预算，再固定预算验证 EAGLE-3 的无损性与性能。这样，“少想”和“想得一样但算得更快”才不会被写成同一件事。

## 参考资料

- [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)
- [EAGLE 官方实现](https://github.com/SafeAILab/EAGLE)
- [SpecForge：EAGLE-3 训练与 Serving 接入](https://github.com/sgl-project/SpecForge)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [Gemini API Thinking 文档：一种具体预算语义示例](https://ai.google.dev/gemini-api/docs/generate-content/thinking)
