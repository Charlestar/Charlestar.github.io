# 2026-09-09 独立逐篇复审：第 21–40 篇

范围按 PowerShell `Get-ChildItem _posts -File | Sort-Object Name` 的第 21–40 篇划定。本轮逐篇完整读取了 20 篇正文、公式、代码块和文本图，不只检查上一轮评审条目或已有 diff。原始发布时间、文件名、URL、系列顺序不变；修改文章的 `last_modified_at` 为 2026-09-09。

结论：20/20 完整读完，18 篇有局部修订，2 篇未发现新的高置信问题。这里的“未发现”不等于证明全部无误。一手材料的核查范围是下文列出的具体章节、文档与源码，**不声称完整精读所有引用论文或复现论文实验**。`main`/`latest` 文档按 2026-09-09 所见状态核对，不代表旧发布日期时已经支持全部功能。

## 本轮验证

- `node scripts/reviews/full-attention-examples.mjs`：13 组 CPU 参考检查全部通过。包含非空/空行 online softmax、FA2 反向有限差分、MLA 吸收和原始缩放、Ring 计数和时间、单 pivot 分布及有限轮数、DSA ties、序列/token 双预算等。
- `node scripts/audit-posts.mjs`：77 篇、9 个系列、24 个标签通过元数据、标题、代码/数学定界符、控制字符、本地图片与残留标记检查。
- 未执行 CUDA kernel、真实模型生成、训练、NCCL/RDMA、Kubernetes、量化校准或性能基准。数值小例不构成生产路径认证。
- 页面渲染、全站构建与最终提交由主审负责；本分工未提交或推送。

## 21. EAGLE-3 与 Thinking Budget

- 文件：`_posts/2026-05-15-speculative-decoding-thinking-budget-eagle3.md`
- 读完：是，完整正文。
- 核查点：推理预算与每轮草稿长度是不同控制量；TTT 不是增加在线目标推理预算；特征输入、候选树、接受长度与端到端收益边界；树 mask 图文。
- 一手来源：[EAGLE-3 v2 §3.1–3.2](https://arxiv.org/html/2503.01840v2)、[Gemini Thinking 文档](https://ai.google.dev/gemini-api/docs/generate-content/thinking) 的模型相关预算配置。
- 修改：tree mask 补上共享正式前缀和当前节点自身。原“只能看到祖先”容易漏掉 causal diagonal。
- 保留判断：没有把不同产品的预算参数写成统一 API；接受率依任务、目标/drafter 配对改变的限制合理。
- 未验证边界：未使用真实 reasoning 模型测预算—正确率曲线，也未重新训练 drafter；对生成内容难度的描述是分析而非性能保证。

## 22. 异步连续批处理

- 文件：`_posts/2026-05-16-asynchronous-continuous-batching-deep-dive.md`
- 读完：是，完整正文。
- 核查点：CPU 准备与 GPU 前向依赖、异步 token 提交、stream/event、pinned-buffer 生命周期、取消与回滚、时间模型。
- 一手来源：[vLLM MRV2 官方说明 §Async-First Design](https://vllm.ai/blog/2026-03-24-mrv2)、[Scheduler 源码 `_update_after_schedule` / `update_from_output`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/sched/scheduler.py)。后者明确在发出调度后先推进 host `num_computed_tokens`，再校正拒绝结果。
- 修改：删除把输入计算与输出提交排成同一长度不等式的写法，改成输入 scheduled/materialized frontier 与独立输出计数；加入 1000-token prefill 后正式序列已有 1001 token、最后输出尚无 KV 的反例；解释 host 计数不是 completion event；修正 GPU idle 箭头原来落在执行区间内的图。
- 验证：参考脚本复算 frontier 反例。
- 未验证边界：未用 GPU trace 证明任何具体实现真正重叠；理论 `max(G,C+F)` 明确只是理想稳态模型。

## 23. FairBatching

- 文件：`_posts/2026-05-17-fairbatching-deep-analysis-solving-prefill-starvation-in-llm-inference.md`
- 读完：是，完整正文。
- 核查点：TTFT/TPOT 的 deadline 与 slack、进度余量、初始预算公式、三组候选顺序、PAB 与吞吐口径、空集合。
- 一手来源：[FairBatching 论文 §3.2–3.3 与算法](https://arxiv.org/html/2510.14392)，特别是 `max(min slack, min TPOT SLO)` 的下限及 urgent/prefill/non-urgent 顺序。
- 修改：预算公式不变，修正“保证 deadline”式解读；10 ms slack 与 50 ms 下限的反例表明预算并非硬上限。伪码增加空 decode 分支，统一一次 `now()` 快照，用相同 slack 定义筛选和排序；说明 prefill 的 next_deadline 指首 token deadline，prefill-only helper 为教学占位而非框架 API。
- 验证：预算反例通过 CPU 检查。
- 未验证边界：没有训练/拟合论文 latency model，未验证 PAB 在真实集群的可承诺误差；教学伪码不包含生产级 admission、抢占和 KV 原子预留。

## 24. DeepSeek Sparse Attention

- 文件：`_posts/2026-05-18-deepseek-v3-2-sparse-attention.md`
- 读完：是，完整正文。
- 核查点：Lightning Indexer、weighted ReLU 分数、因果 Top-k、dense warmup 与 sparse training、Indexer KL 与主模型梯度隔离、MLA 执行形态、稀疏访问不等于删除历史 KV。
- 一手来源：[DeepSeek-V3.2 §2 的 DSA/indexer 与两阶段训练](https://arxiv.org/html/2512.02556)；本轮打开报告并核对 2048 选择预算及 detach 说明。
- 修改：Top-k 集合改为返回位置索引，限定 `0 <= s <= t` 和 `min(k,t+1)`，解释边界同分时不能通过“分数属于 top-k 值”过滤全部同分项。
- 验证：`[3,2,2,1]` 选 Top-2 的旧式值过滤返回 3 个，新式索引选择返回 2 个；极短因果前缀也测试通过。
- 未验证边界：测试的同分规则采用索引顺序，只用于参考，不声称与任意 GPU topk 同分结果一致；未复现 128K 训练、FP8 indexer 或稀疏 kernel 性能。

## 25. QuantSpec

- 文件：`_posts/2026-05-21-QuantSpec-Self-Speculative-Decoding-with-Hierarchical-Quantized-KV-Cache-for-Long-Context-Inference-Acceleration.md`
- 读完：是，完整正文。
- 核查点：upper/lower INT4 组合、浮点加性偏置与整数 zero point、K/V 量化轴、双全精度 buffer、拒绝位置与修正 token KV、target 相对 INT8 cache 的精确性。
- 一手来源：[QuantSpec §4.2–4.3、Appendix D/Figure 7 与 Appendix E](https://arxiv.org/html/2502.10424)。论文明确 INT8 target 与 INT4 draft 视图、近期 `2G` buffer、K channel-wise/V token-wise。
- 修改：说明 per-channel 是固定通道跨 token 分组、per-token 是固定 token 跨通道分组；补短前缀不足 G 的边界；buffer 轮换前必须等待拒绝回滚，并检查实际确认 KV 足够 G，避免把暂态填满当作确认填满；注明这是带安全边界的教学实现，而非逐行复刻论文。回归 reference 必须复现相同量化分组及 buffer 历史，不能把一次全精度重新 prefill 的正常误差直接认定为回滚错误。
- 验证：`F2 tentative=128, accepted KV=12` 不应轮换出 F1；短前缀 31 不可能保留 128 个有效 KV。CPU 检查通过。
- 未验证边界：未执行论文定制 kernel。论文把 signed residual 与 INT8 表示结合的边界需要按实际打包代码核对，正文没有把它等同于随意截取高低 nibble；量化不提供相对 BF16 原模型严格分布无损的保证。

## 26. FlashInfer-Bench

- 文件：`_posts/2026-05-22-flashinfer-bench-ai-auto-generated-gpu-kernel-production-deployment-practice.md`
- 读完：是，完整正文。
- 核查点：Definition/Workload/Solution/Trace 分工、const/var、DPS 与返回值约定、数值误差与随机采样 TVD、部署 fallback、安全隔离、局部与端到端 benchmark。
- 一手来源：[Definition schema 的 Overview/Top-Level Object Structure](https://bench.flashinfer.ai/docs/flashinfer-trace/definition)、[Solution schema](https://bench.flashinfer.ai/docs/flashinfer-trace/solution)、[FlashInfer-Bench 论文的 Verification/采样 TVD 检查](https://arxiv.org/html/2601.00227)。
- 修改：相同 axes/const 值只是维度兼容条件，不是同一算子充分条件；补入 dtype、输入输出契约、constraints 和 reference 语义。相加与相乘不能因 shape 一致合并。
- 保留判断：随机采样使用统计分布而非逐样本文本比较、性能收益需计入上下游 layout 与部署回退，这些边界合理。
- 未验证边界：未构建/执行 AI 生成 kernel、sandbox、DPS 或 `apply` 的完整部署链；当前文档 schema 不是旧版永久接口。

## 27. vLLM V1 EngineCore

- 文件：`_posts/2026-05-23-vllm-v1-engine-architecture-reconstruction-from-single-process-to-multi-process-enginecore-evolution.md`
- 读完：是，完整正文。
- 核查点：API/EngineCore/worker 边界、增量 IPC、process topology 的限定、统一 token 进度、prefix match 单位与物理 block、取消和输出生命周期。
- 一手来源：[vLLM Architecture Overview 的进程/EngineCore 部分](https://docs.vllm.ai/en/latest/design/arch_overview/)、[Scheduler `_update_after_schedule`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/sched/scheduler.py)。
- 修改：将“ModelRunner 返回后才增加 computed”改为真实的调度提前记账、结果返回后校正；明确 host computed 可包含在途工作，KV 释放还要等待在途写入安全结束。
- 验证：核对源码注释及 `request.num_computed_tokens += num_scheduled_token` 的调用位置；文中预算减法和进程公式限定未发现新的计算错误。
- 未验证边界：未启动多进程模型服务器或逐个测试 TP/PP/DP executor；现有进程数是典型主进程估算，不是所有优化 executor 的枚举。

## 28. Sorting-Free Sampling

- 文件：`_posts/2026-05-24-flashinfer-sorting-free-sampling-llm-inference-sampling-performance-breakthrough.md`
- 读完：是，完整正文。
- 核查点：普通 categorical、Top-k/Top-p/min-p、pivot 条件、递推分布证明、终止、dual pivot 缩区间、ties、seed/offset 与 API filter order。
- 一手来源：[FlashInfer 官方文章的 Dual Pivot 与 Theoretical Proof](https://flashinfer.ai/2025/03/10/sampling.html)、[top_k_top_p_sampling_from_probs API](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_probs.html)。
- 修改：证明补有限词表、正质量、top-k 边界无 ties 的前提；单 pivot 并非没有有限上界，严格边界时至多 `V-k+1` 轮，但该线性上界不够小；RNG 检查改为不复用随机数区间，而非“token 输出必须不同”。
- 验证：5 项互异概率、k=1…5 的完整递归枚举，最终概率均为 top-k 归一化分布，最大路径轮数均为 `V-k+1`。
- 未验证边界：未执行 GPU 浮点归约、dual-pivot 相邻可表示数停滞、CUDA Graph RNG 或真实词表性能；严格证明不直接涵盖未定义的 ties 策略。

## 29. Model Runner V2

- 文件：`_posts/2026-05-25-vllm-model-runner-v2-architecture-refactoring-gpu-native-and-zero-sync-design-analysis.md`
- 读完：是，完整正文。
- 核查点：稳定 request row 与 per-step gather、GPU 原生输入、staged writes/UVA、Gumbel-Max、top-k logprobs 的全局 LSE、idx_mapping、ModelState、版本默认与历史不支持项。
- 一手来源：[MRV2 官方文章 §1–4、Performance/Limitations](https://vllm.ai/blog/2026-03-24-mrv2)、[v0.25.0 release 的 Engine 节](https://github.com/vllm-project/vllm/releases/tag/v0.25.0)、[GPU buffer_utils.py](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu/buffer_utils.py)。
- 修改：仅修正同步图的 idle 标签原来指向 GPU run 区域；稳定 row、异步依赖与 sampler 的正文未发现新高置信错误。
- 保留判断：v0.18 的限制作为历史状态，与 v0.25 dense default 区分正确；收益数字限定了模型和 GB200 环境。
- 未验证边界：未在 GB200 运行、未穷举 LoRA/多模态/推测组合，也没有将 2026-03 官方博客的性能直接外推到当前版本。

## 30. SpecForge

- 文件：`_posts/2026-05-27-specforge-eagle3-speculative-decoding.md`
- 读完：是，完整正文。
- 核查点：EAGLE-3 advanced-token 对齐、真实特征与递归 drafter 表示、TTT、树 mask、loss mask、离线 schema、online/disaggregated 关系、YAML/runtime 契约。
- 一手来源：[EAGLE-3 §3.1–3.2](https://arxiv.org/html/2503.01840v2)、[SpecForge Training 的布局/数据/支持矩阵](https://sgl-project.github.io/SpecForge/basic_usage/training.html)、[Data Preparation](https://sgl-project.github.io/SpecForge/basic_usage/data_preparation.html)。当前 catalog 的 online/colocated 标为 reserved unsupported，online/disaggregated 受支持。
- 修改：tree attention 补上正式前缀与自身；前次已修正的 g_t + anchor x_(t+1) -> draft x_(t+2) 对齐本轮完整复核后保留。
- 未验证边界：YAML 为概念示例，未启动在线 SGLang/Mooncake 特征生产或训练，也未验证 checkpoint 配套的全部层 ID/词表组合。

## 31. MoE 与推测解码

- 文件：`_posts/2026-05-29-moe-speculative-decoding.md`
- 读完：是，完整正文。
- 核查点：expert assignments 与 unique footprint、树示例 9 个专家计数、精确采样、utility、EP/offload、Cascade/EcoSpec/MoE-Spec/SpecMoE 的不同边界。
- 一手来源：[Cascade §4–5](https://arxiv.org/html/2506.20675)、[MoE-Spec 的 expert budgeting 与 §4.2 Quality-Speedup](https://arxiv.org/html/2602.16052)、[SpecMoE 的 memory-constrained/self-assisted 方案](https://arxiv.org/html/2604.10152)、[EcoSpec §3 的 marginal expert cost/tree selection](https://arxiv.org/html/2607.12696)。
- 结果：未发现新的高置信错误，本轮未改文。保留 exact/approximate 区分；EcoSpec 不改变目标验证规则，MoE-Spec 跳过目标专家属于近似，不能并列为相同无损结果。
- 未验证边界：没有复现任一 MoE 性能论文或 expert predictor；文中 Score 为解释性启发式，不声称是所有方法统一目标函数。MoE 非线性路由使“多 token 一定便宜”不成立，正文已有限定。

## 32. llm-d

- 文件：`_posts/2026-06-01-llm-d-kubernetes-native-distributed-llm-inference.md`
- 读完：是，完整正文。
- 核查点：EPP/InferencePool/Proxy 边界、近似与精确 prefix routing、KVEvents 时序、恢复成本、P/D 协议、KV 字节、WideEP 和扩容延迟。
- 一手来源：[llm-d Disaggregated Serving 的 EPP/sidecar 流程](https://llm-d.ai/docs/architecture/advanced/disaggregation)、[Prefix-Cache Aware Routing 两条索引路径](https://llm-d.ai/docs/architecture/advanced/kv-management/prefix-cache-aware-routing)。明确 vLLM nixlv2 顺序协议与 SGLang 并发协议。
- 修改：KV 传输公式中 `d_kv` 改为每层每 token 的单侧 K 或 V 元素数，限定 K/V 同尺寸 MHA/GQA，避免与外部系数 2 重复计数。
- 未验证边界：没有部署 Kubernetes、NIXL、WideEP 或逐个测 Gateway failure mode；引用的 current 文档按检查日期，不承诺全部硬件/backend 组合可用。

## 33. NVFP4 KV Cache

- 文件：`_posts/2026-06-02-nvfp4-kv-cacheblackwell.md`
- 读完：是，完整正文。
- 核查点：E2M1 值集、16-value/E4M3+global scale、4.5 bits/value、3.56x/1.78x、80 层示例、cache storage 与 attention compute、PTQ 与运行时支持、量化质量边界。
- 一手来源：[NVIDIA NVFP4 KV 技术文章的格式/数据流/质量与性能图](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)、[TRT-LLM Quantization 的 NVFP4 KV Cache/ModelOpt 节](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)、[Hardware Support Matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)。
- 结果：未发现新的高置信错误，本轮未改文。正文已解释裸 FP4 与含 scale 的不同口径、原生计算与存储不是同义词、硬件支持需按具体 SM/backend 查矩阵。
- 验证：320 KiB/token、128Ki token=40 GiB、4.5 bits/value 以及理想比值均通过 CPU 复算。
- 未验证边界：没有进行模型校准/量化、SM100/103/120 实机测试或 fused attention benchmark；“约/最高 50%”保留为 NVIDIA 博文报告，不以此代替本地真实 page size。

## 34. Continuous Batching

- 文件：`_posts/2026-06-04-continuous-batching-iteration-level-scheduling.md`
- 读完：是，完整正文。
- 核查点：Orca iteration boundary、selective batching、首 token 与 O−1 decode 计数、request state/KV 映射、双预算伪码、取消/背压、吞吐 B/T(B)。
- 一手来源：[Orca 官方 OSDI 页面](https://www.usenix.org/conference/osdi22/presentation/yu) 的 iteration-level scheduling / selective batching 摘要、[vLLM Scheduler 的资源预算及调度循环](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/sched/scheduler.py)。本轮未完整阅读 Orca PDF。
- 修改：伪码增加递减 `seq_budget`、预算耗尽退出、零工作排除、每 request 一序列与 reserve 成功假设；修正观测时间线为 prefill -> first token -> repeated decode，而不是把 first token 排在 decode 后。
- 验证：分别测试序列数先耗尽、token 先耗尽、零预算和 2046-token 剩余 prefill；全部通过。
- 未验证边界：教学 scheduler 不是生产引擎实现，未实现分支序列、原子失败重试、多模态或分布式 executor。

## 35. Chunked Prefill

- 文件：`_posts/2026-06-06-chunked-prefill-long-prompt-scheduling.md`
- 读完：是，完整正文。
- 核查点：causal 跨 chunk 等价、prefill/decode 饥饿示例、Sarathi decode-maximal、7000-token 时间线、动态预算定义域、KV block 对齐、prefix state 边界、多模态。
- 一手来源：[SARATHI 的 chunked prefill/decode-maximal batching 设计](https://arxiv.org/html/2308.16369)、[vLLM Scheduler 的统一进度与 `_update_after_schedule`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/sched/scheduler.py)。
- 修改：严格“不清空 decode 就不 prefill”且现有请求有限输出时，新请求仅到达不足以推出无限饥饿；修正示例中未 prefill 的 D 直接进入 decode 的逻辑错误，区分有限长等待与 decode 集合不断补充的真正饥饿条件。补 D_t 是已获准而非总需求、0<=D_t<=预算、ceil 公式 C>0；computed 字段改为可提前记账的逻辑进度。
- 验证：2016×3+952=7000、每轮加 32 decode 不超过 2048、63-block 对齐例及有尾部空位时的具体边界均通过。
- 未验证边界：未在真实 GPU 比较混合 batch kernel、pipeline 平衡或 P99 ITL；数学等价限于正文描述的 causal 模型与相同状态/精度契约。

## 36. EAGLE

- 文件：`_posts/2026-06-09-eagle-feature-level-speculative-decoding.md`
- 读完：是，完整正文。
- 核查点：feature/token shift、冻结 embedding/LM head、Smooth L1 与 soft-label CE、均匀噪声、树 mask/position、精确独立匹配与残差采样、KV 事务、TPOT 聚合。
- 一手来源：[EAGLE §3、训练目标与 multi-round speculative sampling 附录](https://arxiv.org/html/2401.15077)、[EAGLE-3 §3](https://arxiv.org/html/2503.01840v2) 用于后续版本区别；核对原论文分类权重 0.1、噪声 U(-0.1,0.1)。
- 修改：Feature Uncertainty 一处 `p3=LMHead(f2)` 漏 Softmax，改为与全文一致的概率定义；平均每 token 耗时改成 `E[总轮耗时]/E[提交数]`，解释不是每轮比值的等权平均并要求正分母。
- 验证：Softmax 质量和为 1；两种耗时聚合不相等的数值反例通过。前次修复的“修正 token 尚无 KV”本轮保留。
- 未验证边界：未训练 EAGLE、未逐 GPU 验证浮点 ties；lossless 指目标采样分布，不是相同 seed 文本一致。

## 37. FlashAttention-2

- 文件：`_posts/2026-06-14-flashattention-2-parallelism-work-partitioning.md`
- 读完：是，完整正文。
- 核查点：延迟归一化、LSE、grid/warp 划分、D 的行归约与梯度缩放、causal tile 图、GQA、varlen bottom-right、Amdahl 数值。
- 一手来源：[FA2 §3/Algorithm 1–2 与 §4](https://arxiv.org/html/2307.08691)、[FlashAttention README 的 v2.1 causal 对齐和空 mask 行行为](https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag)。
- 修改：online softmax 补初始/中间空 tile 的逐行跳过、首有效 tile 旧缩放 0、全空行的零输出与 backward mask；因果矩阵左上 tile 应为“边”而非无 mask 的“算”。
- 验证：包含最前/中间空块与极大 logits 的 online softmax 对 dense reference；D=rowsum(P*dP)=rowsum(O*dO)；softmax 反向中心有限差分，通过。
- 未验证边界：未执行 atomic dQ、dropout、GPU occupancy 或 A100 性能；理论矩阵公式未覆盖任意非有限输入，容差与目标 kernel 需实测。

## 38. Ring Attention

- 文件：`_posts/2026-06-17-ring-attention-sequence-parallelism.md`
- 读完：是，完整正文。
- 核查点：rank owner 与环方向、P 次块计算/P−1 传输、online merge、双缓冲、通信字节方向、causal 不平衡、backward、TP/CP/Ulysses 布局、weak/strong scaling。
- 一手来源：[Ring Attention §3 与算法/附录实现](https://arxiv.org/html/2310.01889v4)、[官方实现](https://github.com/haoliuhl/ringattention)、[DeepSpeed-Ulysses 的 layout 说明](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md)。
- 修改：教学循环最后一轮不再额外传输；mask 使用明确 owner=(rank−step) mod P；补空局部状态合并；区分每卡发送与接收字节，系数 2 表示 K/V 而非双向；有限流水时间为 (P−1)max(compute,comm)+compute，8卡/3ms/5ms 示例为38ms。
- 验证：P=1/2/4/8，每 rank 访问所有 owner 恰好一次、传输 P−1 次；24ms/38ms、通信字节量通过；空块合并共用 online reference 测试。
- 未验证边界：没有多卡/JAX/NCCL 执行或故障演练；论文伪码是高层概述，本博客循环是显式处理末轮边界的教学变体；overlap 需要 profiler 证明。

## 39. MLA

- 文件：`_posts/2026-06-20-multi-head-latent-attention-kv-compression.md`
- 读完：是，完整正文。
- 核查点：joint latent、Q/K/V projection 形状、两次吸收、RoPE 不可交换、共享 positional key、缓存元素与具体模型压缩率、MHA/MQA mode、TP/cache contract。
- 一手来源：[DeepSeek-V2 §2.1、Appendix C 公式(37)–(47)](https://arxiv.org/html/2405.04434v5)、[DeepSeek-V3 官方 `MLA` 的 naive/absorb 与 softmax_scale](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)、[FlashMLA README 支持矩阵](https://github.com/deepseek-ai/FlashMLA)。
- 修改：补实际 down-projection 后 RMSNorm，不能跨非线性做权重吸收；明确吸收后执行维度虽然从 d_h 变成 d_c，score 分母仍是 sqrt(d_h+d_R)，而不是 sqrt(d_c+d_R)；给出形状和共享 softmax_scale 的源码依据，并保留长上下文额外修正。
- 验证：显式 K/V 与吸收后的 score、Value 聚合数值等价；误换分母确实改变概率分布，CPU 例通过。
- 未验证边界：未执行 FlashMLA、低精度 cache、TP 重分片或真实模型长上下文回归；公式的数学吸收不保证运行时自动采用高效路径。

## 40. FlashAttention-3

- 文件：`_posts/2026-06-23-flashattention-3-hopper-asynchrony-fp8.md`
- 读完：是，完整正文。
- 核查点：TMA/WGMMA 层级、producer-consumer、ping-pong、intra-group 两级流水、FP8 layout/scale、正交变换、全 mask 行、性能与误差口径。
- 一手来源：[FA3 v2 §2.2、§3.1–3.3/Algorithm 2、§4](https://arxiv.org/html/2407.08608v2)、[FlashAttention 官方 README](https://github.com/Dao-AILab/flash-attention)。核对3.9/989吞吐示例、570→620–640消融及H100报告范围。
- 修改：将误导性的串行 ASCII 流水替换为先发下一 QK 和当前 PV、等待下一 score、Softmax 与当前 PV 重叠、等待 PV 后 rescale 的依赖流程；补全 mask 行统计跳过但同步/barrier 不能漏；随机正交混合降低 outlier 改为典型效果而非任意输入保证，注明 Hadamard 归一化和常规快速算法的维度要求。
- 验证：空块 online reference 通过；矩阵正交等价为代数检查。原有FP8 cast、零块scale与动态范围说明已复查，保留。
- 未验证边界：未检查生成 SASS、barrier race、FP8 native WGMMA、Hadamard融合或H100 benchmark；参考脚本不模拟异步时序，不能认证 GPU kernel 无 race。
