# Vertumnus 新文独立全文审核

- 审核日期：2026-09-30。
- 对象：`_posts/2026-09-30-adaptive-context-parallelism-vertumnus.md`。
- 流程：研究与大纲 → writer 完稿并停止修改 → 独立 auditor 全文核验与标准库复算；主审负责构建、页面和发布验证。

## 实际阅读范围

通读正文全部段落、公式、表格与状态示意。独立打开 [arXiv 提交历史](https://arxiv.org/abs/2609.04774)和 [v1 正文](https://arxiv.org/html/2609.04774v1)，实际读取 §2–§9，包括 §2.2 剩余 pair 计数、§3 的模型与 SLO 定义、§5 候选/排队代理/反事实成本、§6 配置切换、§7 两个缓存算法、§8 实现和 §9 实验设置、端到端结果、重配置与缓存消融。读取范围不是只看摘要，也不声称追读本文参考文献的全部原始论文或 RTP-LLM 源码。

确认 v1 首次提交为 2026-09-04；它是预印本，文章没有把它描述为已接受的会议论文。

## 审核结论与边界

正文没有发现需要纠正的概念、公式或数值错误。writer 完稿前已收到并采纳独立预审意见：

1. `q_w` 是未完成预测工作量形成的轻量代理，不是 continuous batching 下精确的排队时间。
2. GPU-time 的最小 degree 反事实沿用当前候选自己的 `(R,P)`；不是改用另一台小 worker 的实际缓存状态。`(C-Cmin)A` 只由这个固定模型推出。
3. 区分单次 engine step 内固定 CP、drain、共同输入边界、全 rank epoch 确认及调度/缓存元数据发布。
4. 逻辑前缀、源端驻留与目标立即可用不能混为一谈；pending KV 尚不能计入 ready hit。
5. 28.1% 的 mean TTFT 降幅来自 Qwen/Production，13.3 个百分点来自 Qwen/Tool-Agent，均是最高被测试负载下相对最强基线的论文报告，不是同一场景或作者本人实测。

另逐项确认 dense causal 计数没有泛化到 sparse/linear/hybrid Attention；全命中不会被误说成零 TTFT；8 卡配置仅说明资源守恒，不推出吞吐翻倍；20 秒窗口、1.1–5.1 秒 drain 加 switch 与不足 1 秒 switch 没有混淆；实验 40 张 prefill 加 24 张独立 decode，decode 非瓶颈的范围明确。

## 配套 CPU 检查

已实际执行 `node scripts/reviews/sep30-vertumnus-examples.mjs`，10 组全部通过。

该脚本覆盖 10 组数学与简化状态参考：

- 枚举长度 0–64 的所有 prefix/suffix 划分，复核剩余 causal pair 公式及正文 15/36 算例。
- 200→140 ms 与 400→560 GPU-ms 的反向变化；零剩余 pair 不表示零服务时间。
- 同缓存反事实的 GPU-time 恒等式，并构造错误使用另一台冷缓存 worker 的反例。
- `A=2,B=24/4` 与 λ 改变后的三次 degree 选择。
- 热缓存/冷缓存的 β=1、β=3 队列比较。
- 先过滤不合格 worker、再及时预留负载的简化调度反例。
- split/merge 的固定 8 张 GPU 预算。
- drain、配置 epoch 和一致确认后的可见性不变量。
- 缺失块安装前不能发布 ready prefix；边际缓存收益的 telescoping 计数。
- 50% 请求达标率与 10% 输入 token 加权达标率。

## 未执行事项

没有运行 RTP-LLM、分布式 collective、真实 prefix 复制、配置切换、GPU profiling、排队压力或性能基准。状态检查只是说明正文要求的不变量，不是对 Vertumnus 实现的集成验证或正确性证明。数值检查使用 JavaScript 双精度 CPU 运算，不模拟 GPU 浮点执行。

全站构建、公式保留和真实浏览器检查由主审另行执行，不在此预先声称通过。

## 主审发布前验证

主审重新通读文章和 10 组 CPU 检查并执行通过；2026-09-30 的首篇构建包含 84 篇文章，元数据检查通过。历史 77 篇覆盖快照仅证明其原有范围，不包括本篇；本篇依据上面的独立记录。

使用校验 SHA-256 的官方便携 Ruby 3.3.11 和现有 Bundler 依赖完成 production Jekyll 构建；站点校验通过 98 个 HTML 页面、123 个生成文件；公式检查通过 436 个原生行内公式和 855 个展示公式（77 个含公式的构建页面）。此前 13 个 JavaScript 套件及 Python 审核例程也已通过，本篇新增脚本另外执行通过。

实际浏览器打开本篇本地页面，20 个 MathJax 容器、零数学错误标记，页面宽度与滚动宽度一致；目视检查标题、正文与公式段落，目录跳转可用。以上不是对 GPU 性能或作者系统实现的复现。
