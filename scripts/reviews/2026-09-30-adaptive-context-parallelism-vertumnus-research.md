# Vertumnus 调研与写作交接

调研时间：2026-09-30；角色：研究与大纲，不是独立审核记录。

## 来源与实际阅读范围

- 主论文：[Adaptive Context Parallelism for Production LLM Serving, arXiv:2609.04774v1](https://arxiv.org/html/2609.04774v1)。[版本页](https://arxiv.org/abs/2609.04774v1)直接确认 v1 为 2026-09-04 06:10:15 UTC。它是预印本，不应写成已获会议认可的结论。
- 已逐段读取 §2–§9：CP 与 prefix 条件、式 (1)–(3)、请求成本、split/merge 协议、Algorithm 1/2、实现与实验设置/各 workload 结果。不是仅根据摘要写作。阅读方式为 arXiv HTML；未运行作者实现、未做 GPU 测量，未审阅其全部外部参考文献。
- 对照既有文章：`sequence-context-parallel-long-context-training` 已讲训练 CP、mask、position 与通信；`pd-elasticity-fast-worker-startup` 已讲新增 worker；本篇的新问题是**固定 prefill GPU 预算下，选择既有 CP worker，并重组这些持久 worker，同时保持 cache 可用性**。

## 核心事实与限制

系统仅优化 P/D 分离系统的 prefill 池；decode 独立配置，不能把结果写成 TPOT 提升。CP degree 在一次引擎 step 内固定，不是每 token 任意改变 communicator。论文使用 RTP-LLM，先驻留模型并预建/warm communication groups，再切换，不能推广为任何引擎都无需 reload。

三个决策分别是请求路由、较慢的 worker composition 重组、prefix 副本管理。缓存是每个 worker 的物理状态；global trie 只是元数据，不能让远端 KV 瞬间可读。reconciliation 未完成的块不算 ready hit。

`L=P+R`；剩余 causal QK pair 数为 `W=R*P+R*(R+1)/2`。注意这是 **dense causal pair-count 特征**，不是所有 sparse/linear/hybrid attention 的精确 FLOP 定律。profile 模型是拟合近似：`E_C=d+aR+bP+(cR+alpha*W)/C`，系数来自固定 model/hardware 的无排队采样；不能把这个式子当真正线性加速定理。

原论文成本为 `J=q+beta*s+lambda*(C*s-Cmin*E_Cmin(R,P))`；其中 baseline 使用**当前候选相同的 P/R**，不是另一台低 CP worker 自己的 prefix hit。`beta>=1,lambda>=0`；`q` 为未完成工作量代理，不是排队论意义上的准确等待时间。选定后立即 reserve，避免并发请求都看见旧负载。

重组：停止 admission → redirect queued → drain in-flight/共同输入边界 → activate 预建 group → 发布新 epoch → 所有 rank、scheduler、cache metadata 一致后恢复调度。KV 保留不等于已按新几何完整可用，缺块按需迁移，源 pin、目标安装完才 publish。split 一个 `2C` 为两个 `C`，总 GPU 数不变；这不是云资源扩容。

实验限定：64 张 H20-3e，40 prefill +24 decode，两模型/三 workloads；20 秒监控窗口并不等于 20 秒切换耗时；转换测量 1.1–5.1 秒含 drain，switch 本身 <1 秒。Qwen3-30B-A3B 在最高测量负载、Production 上 mean TTFT 降低 28.1%；13.3 **百分点** token-weighted SLO 提升出现在 Tool-Agent，不能拼成同一运行结果。本文可少列宣传数字，着重测量边界。作者的 vLLM/SGLang 能力描述仅限其评估版本，不能当 2026-09-30 所有版本现状。

## 建议正文结构

建议标题：`Vertumnus：长请求应该占几张 GPU，为什么要连缓存一起调度`。

1. 用“同样长的两个 prompt，其中一个缓存了大部分前缀”开篇，说明只按原始长度分卡会失真。
2. 先区分 request latency 与 GPU-seconds，以及 worker 数与 CP degree；不重复训练 CP 长篇内容，链接现有文章。
3. 从 causal 三角形推导 `RP+R(R+1)/2`，给可数的 8-token 例子；然后说明 prefix 命中仍有 suffix→prefix attention。
4. 从 prefill profile 到 `q+s`，再增加 GPU-time opportunity cost；明确各种时间、单位和 queue proxy。
5. 用同一自造候选表解释为什么 cache-first、load-first、length-only 各自会失败；把 lambda/beta 视为需校准的权衡，不是免费性能开关。
6. 从“选择哪台 worker”转到“集群应有哪些 worker”：固定预算的 split/merge、持续窗口、hysteresis、capacity 优先。
7. 用一条状态序列讲新 epoch、旧请求排空和 pending cache；区分逻辑命中与真正可复用。
8. 测什么才能验证：pair count/路由 CPU 算例、epoch/cache 不变量；GPU 端则验证 TTFT、GPU-seconds、reconfiguration drain、transfer、decode capacity、不同负载与 cache warmup。
9. 回到近期趋势：parallelism、placement、cache 三者开始一起控制；保持预印本/适用范围边界。

## 自行推导的 CPU 例子（不是论文测量）

- `L=8,P=0,R=8`：36 causal pairs；`L=8,P=6,R=2`：`2*6+3=15`。仅剩 2 个 query 不意味着只做 3 对 attention。`P=8,R=0` 得 0 对未缓存 query，但生产 fully-cached prefill 可能仍需 logits/边界 token 等引擎约定，勿称请求计算总成本零。
- 枚举 `q_index in [P,L)`，`k_index<=q_index` 验证所有整数 `0<=P<=L` 与公式相同。固定 `L` 时复用 P 前缀省 `P(P+1)/2` 对，而不是 `P*L`。
- 假定同样 prefix 条件、`E_2=0.20 s,E_4=0.14 s`，两台 `q_A=0.05 s,q_B=0`：beta=1,lambda=0 时 J 为 .25/.14，选 B；lambda=1 时 GPU-time baseline .4 GPU-s，B 的 excess=.16，J_B=.30，选 A。lambda 的单位负责将 GPU-s 映射到 score 的秒量纲；仅是自造 profile，非硬件数据。
- 缓存偏好独立例：hot `q=.20,s=.04`，cold `q=0,s=.14`，lambda=0；beta=1 选 cold(.14<.24)，beta=3 选 hot(.32<.42)。不能由此宣称 beta=3 最优。
- 8 GPU composition：`[4,4] -> [4,2,2] -> [2,2,2,2]`。不是 `[4,4]->[4,4,2,2]`，reserve 与 drain 期间容量需显式算。
- token-weighted SLO：输入长度 `[100,900]`、是否达标 `[true,false]`，按请求是 50%，按输入 tokens 是 10%。百分点不能当相对百分比。
- 状态反例：物理保留 prefix 的 shard 还在但 epoch 不同/缺块，不可把它放入 ready map；只有全 rank acknowledge 且所有必要 blocks installed 才接请求。

## 审核重点

不要把 `P/R` 所依赖的 worker 下标丢掉；不要用另一 worker 的 P 去算 GPU-time baseline；不要把 `sum profile services` 当精确批处理等待；不要声称 CP 消除了 attention 的二次工作；不要把模型/weights 常驻所需显存假设隐去；不要将 cache node 存在等同完整祖先路径存在；不要把论文中的几个 best-case 百分比混为全系统稳定收益。
