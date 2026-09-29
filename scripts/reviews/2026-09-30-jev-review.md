# Jev 新文独立审核

- 日期：2026-09-30。
- 对象：`_posts/2026-09-30-jev-structured-decision-model.md`。
- 流程：独立核实一手资料 → writer 完稿停止 → auditor 全文审核、补修与离线执行 → 主审构建和发布。

## 实际阅读范围

通读最终文章的正文、公式、请求体和 JavaScript。审核者独立打开以下材料，没有只复述 research 笔记：

- [Vercel 采用记录](https://vercel.com/blog/ai-gateway-jev-model-launch)：完整正文；只用于平台采用速度，不用于证明能力。
- [官方发布文](https://typesafe.ai/blog/introducing-system-one-models-and-jev)：发布信息、正文比较表、Evidence / Technical Results、Workflow evals、Hallucination and Type-safety、Doom/Wikiracing 限定和可见 FAQ。没有声称读取未展开的 FAQ。
- [Introduction](https://docs.typesafe.ai/introduction)、[System One](https://docs.typesafe.ai/concepts/system-one)、[State](https://docs.typesafe.ai/concepts/state)、[Confidence](https://docs.typesafe.ai/confidence)、[AI Primer](https://docs.typesafe.ai/introduction/machine-learning-primer)：实际读取正文。
- [Choice](https://docs.typesafe.ai/primitives/choice)：定义、请求、响应、候选上限和多问题示例；[Score](https://docs.typesafe.ai/primitives/score)：定义、等级、响应、Reading / Writing levels；[Noul](https://docs.typesafe.ai/primitives/noul)：请求、响应、Reading / Writing、多题组合示例。
- [API](https://docs.typesafe.ai/api)：请求/响应字段、三原语、错误状态；[Models](https://docs.typesafe.ai/models)：当前版本、别名、输入预算、语言和版本 ID；[fan-out](https://docs.typesafe.ai/patterns/fan-out)及 [Jev 1.13 已知限制](https://docs.typesafe.ai/model-jaggedness/jev-1.13)：正文。
- [System One Adapter README](https://github.com/typesafe-ai/system-one-adapter-python)：介绍、Usage、Response、Options；仅核实原生结构化输出选项，没有安装或审计其实现源码。
- [Just Ask Jev v1](https://arxiv.org/html/2609.29429v1)：摘要、§1–4.4、§4.5 的人类一致性段落、§6 与复现声明；并定位附录 E 的表 13、H 的费用口径和 J 的缓存版本说明。未逐项精读全部附录，也未重算作者缓存结果。
- [Type-Safe Is Not Error-Free v2](https://arxiv.org/html/2609.26758v2)：摘要、§1–4.5、§6–7、结果表和附录中的示例；未复现 API 实验。

没有 Jev 付费调用、SDK 运行、模型权重、训练配方、GPU 或真实延迟测试。没有把 SEO 转述站当成官方来源。

## 事实与推理审核

1. Choice 是单选，不因有全概率就变成多标签分类；问题 ID 与实际选项名的语义不同。Score 是从 0 开始的等级序号期望，Noul 是肯定命题概率且没有额外 confidence。未为 confidence 杜撰熵或最大概率公式。
2. 将“相互独立的小问题”改为“可分别判断的小问题”。同 state 并行评估不蕴含统计独立，也不能消除尚未获得外部事实的数据依赖。
3. 三档 Score 例子、相同均值的两种分布、校准与准确率的区别正确。补充 `p>0.5` 的分类规则和有限样本不确定性，避免“全答对”缺少明确判决定义。
4. 成本阈值推导补充非负且总和大于零的前提。加入完美复核后的区间为 `[0.02,0.60]`，边界平局优先复核。强调人为成本、复核无误差/无延迟是假设；代码只给建议，不执行操作。
5. Just Ask Jev 的 0.886 是 31 个适用 generic-Noul 基准的 AUROC 中位数，不是准确率。文章使用 §4.4 的汇总 ECE 0.047 / 逐基准中位 0.168 / 有限样本零假设参照 0.074，未混入附录 E 更大文件集的另一组数值。版本、英语、目标模型规模和评分器标签限制都已保留。
6. 选项名研究只在正文保留定性结论。核对 §4.5：Jev 在 1,200 英文题的 yes/no 名称—定义互换中，定义级选择翻转 32.50%；0/1 对照 2.08%，300 题重复调用的最高翻转率 1.33%。摘要最显眼的 70.4 个百分点差异属于 Laya，不能归给 Jev。该研究是闭源服务单时点观察，不保证现版、中文或其他任务同样表现。
7. 未把合法 schema 写成语义不会错，未把未知的 RLCD/网络结构写成已公开算法。官方工作流比较的模型参照和有利条件已限定；正文没有宣传未亲测的倍数或吞吐。
8. 与 LLM 的比较承认 constrained decoding 和原生 structured outputs；JSON 传输字符串与模型开放文本生成不同。精确计算、权限、数据新鲜度、幂等和真实副作用仍留给代码。
9. 日期、版本和能力限定均标在截至 2026-09-30 的范围内；未引用未经一手核实的游戏通关故事。

## 已执行离线检查

`node scripts/reviews/sep30-jev-examples.mjs`：12 组通过。

脚本从文章直接提取唯一 JSON 块并解析，提取唯一 JavaScript 块在不提供网络、文件或进程接口的 VM 上下文执行；没有维护第二份 `proposePriority` 实现。概率数组也从正文提取。

- 请求的三问题结构、具体版本、候选 fallback、Score 等级数等少量结构断言；这不是完整 SDK/API schema 兼容测试。
- 正文代码实际打印 `normal / review / elevated`。
- Score 1.65；中间集中的分布和两端分裂分布均值同为 1、方差与最高等级概率不同。
- 人造 100 样本的准确率/单桶频率差，以及两个等大组的误差互相抵消。
- 二元成本边界 `1/21`；三动作边界 0.02、0.60、端点和两侧相邻值。
- 在 10,001 个概率网格点上，正文函数选出的动作达到其声明的最小风险。
- 缺失值、NaN、Infinity、越界数、字符串、布尔值和对象都进入复核。
- 用明确标注的本地 stub 模拟失败状态及错误类型，确认没有把 timeout 当成 `p=0`，也没有拿 Choice confidence / Score 代替 Noul。此项不模拟真实传输时序，不验证 HTTP 超时或 SDK 行为。
- 三个边缘概率都为 0.5 的人造联合分布，联合事件概率分别为 0.5、0、0.25，说明并行不蕴含独立。

`node scripts/audit-posts.mjs` 通过：87 篇、9 个系列、24 个标签。`git diff --check` 通过。

本记录不预先宣称最终 Jekyll 构建、浏览器检查或线上部署成功；这些由主审在交接后执行。离线检查也不证明 Jev 的模型准确率、校准或产品可靠性。

## 主审交接后验证

- 全量 JavaScript 示例共 17 个套件通过，其中本篇 12 组；标准库 Python 的 Agent 示例也通过。
- 历史覆盖检查仍只核对 2026-09-09 的 77 篇快照；本篇由此独立记录覆盖，没有修改快照来冒充旧审阅。
- 最终版 production Jekyll 构建通过；生成站点检查通过：101 个 HTML 页面、126 个生成文件。
- 原生公式保留检查通过：80 篇数学文章，共 472 个行内公式、874 个独立公式。
- 本篇浏览器实际渲染 12 个公式，未见 MathJax 错误或页面横向溢出；检查了标题、正文、成本公式和表格，重建后重新加载确认最终修订。
- 以上是内容示例与站点交付验证，不是 Jev API、模型质量或性能实测。线上部署须以推送后的 CI 与线上页面为准。
