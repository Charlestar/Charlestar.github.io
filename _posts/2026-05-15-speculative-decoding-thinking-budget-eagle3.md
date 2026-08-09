---
layout: post
title: "推理模型的推测解码：Thinking Budget 与 EAGLE-3"
subtitle: "控制生成长度不等于保证加速"
date: 2026-05-15 12:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
tags: [AI Infra, 推测解码, EAGLE, LLM推理]
---

> **校订说明**：原文声称进行了 A100 基准测试，并给出金融风控“真实案例”和多组无来源结果；这些内容无法验证，已全部删除。

## 1. 两个不同的“预算”

- **Thinking budget**：限制或引导推理模型生成多少思考 token，属于生成策略/服务策略。
- **Speculative budget**：每轮生成多少草稿 token 或多大的草稿树，属于推测解码执行策略。

减少 thinking token 可能直接缩短输出，但可能影响任务质量；增加 speculative budget 可能提高每次验证接受的 token 数，也会增加草稿和验证成本。两者不能用一个固定公式替代线上评测。

## 2. 推理模型为什么更难 draft

长推理轨迹可能频繁改变局部 token 分布。草稿模型若不熟悉目标模型的推理风格，接受长度会下降；温度、top-p、工具调用和结构化输出也会改变接受行为。

EAGLE 系列利用目标模型特征预测候选，EAGLE-3 通过多层特征融合增强草稿能力。它仍需与特定目标模型匹配，并不意味着任意闭源 API 都能训练同类草稿模型。

## 3. 正确性先于性能

Greedy 模式应验证 token 级一致；sampling 模式应验证目标分布，而不是要求相同 seed 下逐 token 相等。服务端还必须正确处理拒绝后的 KV Cache 回滚、停止条件和工具调用边界。

## 4. 自适应策略

运行时可以根据近期接受长度、draft/verify 时间和队列负载调整草稿 token 数。可靠指标包括：

- mean accepted length；
- acceptance rate（需明确分母）；
- draft/verify latency；
- TPOT、goodput 与峰值显存；
- 按任务类型拆分的质量指标。

不要只用高接受率判断成功：草稿过慢时，高接受率仍可能没有端到端收益。

## 5. Thinking budget 的服务语义

预算可以由最大 thinking tokens、剩余总 token、时间 deadline 或模型支持的控制 token 表达。服务端必须明确预算耗尽后的行为：要求模型直接作答、截断、还是返回未完成状态。简单硬截断可能切断结构或工具调用。

预算策略还应区分任务。简单检索题可能短思考即可，数学证明和复杂代码修复可能需要更长轨迹。可以用任务分类、用户等级或动态不确定性设定初始预算，但质量回归不可省略。

## 6. EAGLE 特征预测

EAGLE drafter 接收目标模型的 token/hidden feature，预测后续特征或 token 候选。相较完全独立小模型，它能利用目标模型已经计算的表示；代价是 checkpoint 与目标模型层选择紧密绑定。

EAGLE-3 的多层特征融合意在提供不同抽象层次的信息。训练与 serving 必须对齐取特征的位置、归一化、tokenizer 和目标 checkpoint revision，否则接受率可能显著下降。

## 7. 联合控制器

一个保守控制器可以分别维护 thinking budget 与 draft length：

```text
if deadline risk rises:
    reduce remaining thinking budget
if recent accepted_length falls or draft cost rises:
    reduce speculative length
if accepted_length stays high and GPU has verify capacity:
    increase speculative length gradually
```

每次调整设置上下限与冷却窗口，避免因短期噪声频繁振荡。

## 8. 质量评测

除速度外，按预算扫描任务正确率、答案完整性、工具调用成功率和超时率。推测解码本身应保持目标解码语义；thinking budget 改变了允许生成的推理长度，因此质量变化应归因并单独报告。

## 参考资料

- [EAGLE-3 paper](https://arxiv.org/abs/2503.01840)
- [SpecForge](https://github.com/sgl-project/SpecForge)
- [vLLM speculative decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/)
