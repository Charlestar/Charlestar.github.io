# 2026-09-30 AI Infra 研究选题扫描

范围：与博客已有 83 篇内容衔接的一手研究，重点为 2026 年 7–9 月。这里是定向选题，不是文献计量排名，也不声称覆盖整个领域。日期均以直接打开的 arXiv submission history 为准；候选只完成摘要/版本核验，入选后才另记核心章节阅读。

| 方向与一手材料 | 首次公开 / 本次版本 | 已有博客之外的缺口 | 本批决定 |
| --- | --- | --- | --- |
| [Vertumnus：Adaptive Context Parallelism](https://arxiv.org/abs/2609.04774v1) | 2026-09-04 v1 | 从训练 CP 转到 serving 的 CP worker composition、cache-aware 路由和状态切换 | 写：系统控制面 |
| [Hardware-Aware FP4 FlashAttention-4](https://arxiv.org/abs/2609.04105v1) | 2026-09-03 v1 | FA3/FP8 后，QK/PV 不同量化路径、P 构造关键路径、训练稳定性边界 | 写：算子/数值；Robert Hu 的预印本，不是 FA4 官方发布 |
| [WAR：Workload-Aware Rollouts](https://arxiv.org/abs/2607.17299v1) | 2026-07-19 v1 | GRPO 数据流之后，同步 Agent rollout 的低负载 speculation 与高负载 cache scheduling | 写：训练数据生成系统 |
| [KVSET：The KV Cache Working Set](https://arxiv.org/abs/2609.27746v1) | 2026-09-23 v1 | 既有容量规划尚缺 prefix-page 栈距离、miss-ratio curve 与在线缓存容量估计 | 后续候选 |
| [Exo-GPU](https://arxiv.org/abs/2609.16389v1) | 2026-09-14 v1 | Triton GEMM 之后，异步 Tensor Core 指令与顺序/并行等价验证 | 后续候选 |
| [KernelOPT](https://arxiv.org/abs/2609.30059v1) | 2026-09-24 v1 | AI kernel 评测之后，尊重 compiler dispatch 的 agent 优化和端到端检查 | 后续候选 |
| [ForgeMegakernel](https://arxiv.org/abs/2609.12379v1) | 2026-09-11 v1 | 单 kernel 优化进一步到 decode megakernel、依赖与中间态 oracle | 后续候选；测试不能直接叫形式化正确性保证 |
| [SQD：Rethinking Heterogeneous System Disaggregation for Subquadratic Attention](https://arxiv.org/abs/2609.13134v1) | 2026-09-11 v1 | Attention–FFN 解耦之外，按随上下文增长的计算/状态类别划分异构执行 | 后续候选；必须区分 proxy hardware 与 analytical model |
| [AsyncFlow](https://arxiv.org/abs/2507.01663v2) | 2025-07-02 v1；2026-09-11 v2 | 异步 RL 流水及 staleness；原有 GRPO 文章以同步为主 | 后续候选；不能把 v2 更新说成 2026 年 9 月首次提出 |

本批选择三种互补尺度：集群 parallelism 与缓存、GPU 内部数值/依赖、同步 RL rollout。这样可在已有知识上继续深入，而不是同时新增数篇相近的 serving 调度概览。

研究成熟度说明：这些新论文目前按其可见版本视为作者报告；arXiv 公开不等于同行评审通过，作者实验不等于本地复现。博客的 CPU 算例用于核对公式与反例，不能替代分布式性能、GPU kernel 或收敛实测。
