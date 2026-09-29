# Jev 调研与写作交接

调研日期：2026-09-30。研究角色记录；不是 API 实测、模型复现或独立审核报告。

## 身份与来源甄别

对象是 TypeSafe AI 于 2026-09-15 发布的 Jev 决策模型，不是 JEPA。官方入口是 `typesafe.ai`、`docs.typesafe.ai`、官网链接的 `evals.typesafe.ai` 与 `github.com/typesafe-ai`。检索出现许多名称相似的第三方介绍站；不把它们当官方文档，也不使用这些站的转述补全未公开架构。

## 实际阅读清单

以下页面于调研日实际打开，阅读的是正文或明确列出的段落；单页只摘取所需事实，不翻译重写整页。

| 一手来源 | 实际阅读范围 | 本篇可用信息与边界 |
| --- | --- | --- |
| [TypeSafe 发布文](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 正文至 FAQ；FAQ 只展示首个回答，不能声称读了折叠回答 | 发布时间、厂商对架构/并行 sampler/RLCD 的高层描述；工作流评测与 schema 保证的限定 |
| [Vercel 采用情况](https://vercel.com/blog/ai-gateway-jev-model-launch) | 2026-09-18 正文 | 上线 24 小时约 13% 付费团队使用，是该平台的早期采用情况，不是全球模型质量排名 |
| [Introduction](https://docs.typesafe.ai/introduction) | 正文全部 | state 与多问题接口；各题在同一 state 上分开评估 |
| [System One](https://docs.typesafe.ai/concepts/system-one) | 正文全部 | 自然语言输入、有限类型输出、不写自由文本；名称是思维方式类比 |
| [State](https://docs.typesafe.ai/concepts/state) | 正文全部 | string / JSON object / array；与问题分开；当前文本输入 |
| [Choice](https://docs.typesafe.ai/primitives/choice) | Request、Response、Good practice 和多问题例子开头 | 选择最大概率项；全分布和为 1；最多 255 选项；问题 ID 不参与推理 |
| [Score](https://docs.typesafe.ai/primitives/score) | Request、Levels、Response、Reading、Writing good levels | 2–10 个有序描述，索引从 0 开始；score 是等级索引的概率加权均值，不是精确测量值 |
| [Noul](https://docs.typesafe.ai/primitives/noul) | Request、Response、Reading、Writing、多题路由例子 | yes 概率；可选 true/false 描述；无单独 confidence；不是程度评分 |
| [Confidence](https://docs.typesafe.ai/confidence) | 正文全部 | Choice/Score confidence 为分布形状统计量；页面未披露确切公式；阈值需任务验证 |
| [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer) | 正文全部 | RLCD 名称、目标和群体校准含义；不是训练算法技术报告 |
| [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out) | 正文全部 | 同 state 的备选分支可预先询问，再由代码决定使用哪些回答 |
| [Models](https://docs.typesafe.ai/models) | Current models、Aliases、Customizing、Language support、Data handling、Listing models | 当前版本与别名；英语优先；64k 总请求、state 加最长题 32k；版本迁移须重测 |
| [API reference](https://docs.typesafe.ai/api) | Endpoint、Request、Question types、Response、Answer types | 字段形状与语义；未运行认证请求 |
| [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13) | 开头及 Literal / Math / Date / Indirection / Large state / Adversarial / Contradictory / Structural invariants | 适用版本、数值与对抗输入限制；不同题型或反义题的概率不保证恒等 |
| [官方 LLM adapter](https://github.com/typesafe-ai/system-one-adapter-python) | README 的介绍、Usage、Response | 同接口也可包装 LLM；支持原生 structured-output；未审计其源码或运行 SDK |
| [Workflow evals](https://evals.typesafe.ai/) | Overview 首页全文 | 四工作流等权；参考答案来自模型共识；没有逐个打开工作流明细或重算结果 |
| [Just Ask Jev，v1](https://arxiv.org/html/2609.29429v1) | 摘要、§3.2–3.4、§4.1–4.5、§5–6 与复现声明；[元数据](https://arxiv.org/abs/2609.29429) | 9/24 外部对齐失败检测评测预印本；不是 Jev 架构论文；未逐项读附录或运行作者评测 |

另打开官方首页、SDK 导览及文档索引作导航。检索见到生态调查 `2609.30216`，本轮未深入阅读，不用其统计支撑技术判断。没有付费请求、没有模型权重、没有 GPU 或服务延迟实测。

## 文章应解答的问题

推荐主线：**为什么软件里的一个语义判断，值得用与聊天不同的模型接口来实现？** 从一条支持工单或 RAG 片段出发，先界定可确定规则与语义判断，再引入有限候选、概率、代码动作。不要把文章写成产品发布稿或 API 字段大全。

建议结构：

1. 一个具体系统为什么只需要判断而不是生成回答；简短交代 Jev 身份与近期关注的来源。
2. state 是可审查的材料，questions 是判断规则；同 state 的独立问题可以并行，但若下一问依赖上一问新产生的内容，仍需分阶段。
3. 通过同一个原创例子区分 Choice / Score / Noul；表格只负责精确映射，正文解释误用。
4. LLM constrained decoding 能保证一定范围的 schema 合法性，不能把 Jev 差异简化成“只有它能输出 JSON”。差异应落在公开的输出契约、训练目标与并行求值方式；内部实现披露有限。
5. 区分概率、confidence、校准和业务决策。分布集中不保证事实正确；校准看群体频率，须检查目标域和版本。
6. 代码接管路由、拒绝自动化、权限与精确计算。保留未知/其他/人工路径；已知能力和权限筛选先执行。
7. 解释并行省往返为什么可能快，再讲公开性能证据的范围。额外问题仍有成本，吞吐与时延不是一回事。
8. 哪些任务值得尝试，哪些不应该交给它；中文、有攻击性的输入、长杂乱 state 都要单独验证。
9. 用小型离线算例说明原理，并附可运行测试；明确不是 Jev API 实测。

## 必须守住的事实边界

- `choice` 是一个有限选项，不是自由生成答案；候选集没有正确项时仍可能返回语法完全合法但语义错误的项。
- `score = sum(k * p_k)` 是索引期望，不是“故障影响了多少百分比用户”等物理量。相同均值可以来自不同分布。
- Noul 的 `0.2` 是肯定命题的概率，不是“模型只有 20% 自信”；它也可以表达较明确的否定。
- `confidence` 不等于 `p_max`、正确率或已公开的归一化熵公式。页面只说由分布计算，不应逆向猜算法。
- 各问题“分开评估”是执行/上下文隔离，不等于这些问题涉及的随机事件统计独立。不能无条件把两个 Noul 概率相乘作为联合概率。
- 允许代码组合不代表模型保证跨题逻辑一致；反义问题可分别出错。若需要补事件概率，应先清楚定义同一事件，而不是把另一个自然语言问题当严格补集。
- 类型合法与语义正确、事实有据、动作授权是不同保证。厂商的“零幻觉”宣传只能追溯到有限输出/schema 约束，不能继承成事实不会错或不会被注入。
- 公开材料描述 RLCD 和并行 sampler，但本轮读到的资料没有足以复现内部模型的层数、参数规模、网络图、损失函数及训练配方。不要断言是 BERT、分类头套壳、某个 LLM 改名或某种特定 RL 优化器。
- “不用自回归生成输出文本”不等于“没有输入 token、没有语言模型来源、没有服务排队成本”，更不能推出无限问题 O(1) 延迟。
- 支持自然语言与结构化文本输入不代表支持图片；演示 harness 把游戏状态提供给模型，不能说模型直接看屏幕。本篇无需扩展未经一手核实的游戏通关故事。

## 公开评测的最低上下文

官方的 193.6 倍时延 / 444.6 倍费用宣传来自其四个固定工作流比较，参考是外部模型输出的共识，不是人工事实真值；LLM adapter 也被要求提供兼容的概率结构。发布文自己认为这些可能在真实收益的较高端。若引用，紧邻写清作者报告与这些限制，不在标题使用夸张倍数。

外部论文只适合一个短段：`jev-1.13.0` 在 31 个适用基准上的 generic Noul 中位 AUROC 为 0.886；这不是准确率。论文同时发现聚合 ECE 为 0.047、逐基准 ECE 中位数 0.168，并以理想校准的有限样本模拟中位数 0.074 作对照。结果提醒“总体看着校准”不足以证明每个域校准。研究限英语、被检测模型 2–7B，标签多来自 scorer；不推演为中文生产安全验证。论文是外部实验报告，本文未复现。若正文加其他结果，必须压缩本段，避免长篇依赖单一论文。

## 可原创复算的教学例子

以下全为人工设定数组与成本，不是真实 Jev 响应。

1. 三等级分布 `[0, 1, 0]` 与 `[0.2, 0.6, 0.2]` 的期望都是 1，但第二个并不确定在中等级；验证范围、和为 1、均值及差异即可，不计算伪造的 Jev confidence。
2. 二元决策在正确决策损失为 0、误报损失 `C_FP`、漏报损失 `C_FN` 的简化假设下，正向动作风险 `(1-p)C_FP`，负向动作风险 `p C_FN`；前者更小时选择正向，等价 `p > C_FP/(C_FP+C_FN)`，相等时需定义平局策略。只适用于概率适合该任务且成本假设成立，不能将 confidence 字段直接代入。
3. 两个事件各有概率 0.5，若完全相同联合概率为 0.5，若互斥为 0；相乘得到 0.25 只属于额外独立假设。可用于解释“多问题并行”不蕴含概率独立。
4. 人工分组概率均为 0.8 的 10 个样本中 6 个为正，组均值误差为 0.2。仅是一个校准桶的算术，不代表十个样本足以估计真实业务可靠性，也不是 Jev 实测。

下游接入测试建议记录模型版本、state、question 版本、答案分布、实际标签、执行/人工/失败路径，以及端到端 p50/p95 与人工比例。拟合阈值和报告最终指标应使用不同样本；用按时间或来源隔离的样本检查漂移。昂贵/不可逆动作仍必须走确定性的权限与确认机制。
