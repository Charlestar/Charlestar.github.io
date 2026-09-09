# 服务、MoE 与量化文章独立全文审阅

- 审阅日期：2026-09-09。
- 范围：按文件名排序的第 41–58 篇，共 18 篇。下列每篇均已从 front matter 到参考资料完整阅读，不仅检查上一轮评审涉及的段落。
- 处理原则：保留文章深度、原发布日期与 URL；修正明确的数学、计数、生命周期和接口边界；将实现设计示意与项目实际能力分开。所有修改文章的 `last_modified_at` 均为 2026-09-09。
- 独立复算：`node scripts/reviews/full-serving-examples.mjs`，12 项通过。包括 DSpark 附录 A 的概率反例、OPT-66B 的 SI/IEC 换算、缓存后缀注意力工作量、MoE 去重计数、EPLB 逆表、量化缩放与范围边界。
- 内容检查：`node scripts/audit-posts.mjs` 通过（77 篇文章、9 个系列、24 个标签）；`git -c core.safecrlf=false diff --check` 通过。
- 证据边界：下述核验来自论文正文、官方文档和源码；数值测试是无模型的算术检查，没有执行 GPU 内核、分布式恢复、真实服务压测或复现论文性能。最终整站构建由主审统一运行。

## 41. `2026-06-26-kv-cache-autoregressive-inference-foundations.md`

- 全文阅读：完成。核查 causal KV 不变性、缓存/非缓存前向位置计数、KV 大小、block 分配、prefix sharing 和取消状态机。
- 依据：[Transformers Cache explanation](https://huggingface.co/docs/transformers/main/cache_explanation) 的 Attention matrices、Cache class 及自定义生成循环；[GPUDirect RDMA §3.3](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html#unpin-callback) 对 outstanding DMA 的清理要求。
- 修改：KV 不变性的条件补充 deterministic evaluation、相同权重/位置/mask 配置；长度相关配置回溯变化不能直接套用。将 `L + T - 1` 的定义域明确为正整数 prompt 长度与至少一次输出的普通自回归生成。
- 修改：取消流程原先在传输清理前释放 block，改为标记丢弃结果、等待 GPU、取消或排空传输、确认无 GPU/NIC 引用，再释放/递减引用。丢弃结果不等于停止 DMA。
- 复算：逐步位置计数覆盖单输出边界。
- 未验证边界：具体模型位置编码、speculative 多 token 前向、共享块和后端取消实现需要各自实测；本文公式不自动涵盖这些情况。

## 42. `2026-06-29-distserve-goodput-prefill-decode-disaggregation.md`

- 全文阅读：完成。核查 goodput、TTFT/TPOT、P/D 干扰、KV 传输和 placement 示例。
- 依据：[OSDI 2024 最终论文](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) §3、§4 与 §6.1–6.2；[OPT-66B 官方配置](https://huggingface.co/facebook/opt-66b/blob/main/config.json)。最终论文的最高 7.4× 请求率和 12.6× 更紧 SLO 保留，未混用早期 arXiv 的旧数值。
- 修改：补全 KV 公式中长度、层数、KV heads、head dimension、每元素字节数的定义。512 token、64 层、72 heads、128 维、FP16 得到 1,207,959,552 bytes = 1.125 GiB ≈ 1.208 GB；10 请求/秒约 96.64 Gbit/s，不能把论文近似的 1.13 GB 当成严格十进制量直接换算成 90 Gbit/s。
- 复算：字节数、GiB、GB/s、Gbit/s 均加入独立断言。
- 未验证边界：上述是整份逻辑 KV 的裸负载，不是逐链路实测吞吐；TP 切分、协议开销、缓存复用与拓扑会改变物理传输量。

## 43. `2026-07-02-mooncake-kv-cache-centric-disaggregated-serving.md`

- 全文阅读：完成。核查架构流程、块索引、跨节点加载、Conductor 成本模型、热块副本与 overload。
- 依据：[FAST 2025 最终论文](https://www.usenix.org/system/files/fast25-qin.pdf) §3.1 工作流、§3.2 Store、§4.1 缓存感知调度和 §5 overload。§4.1 的 prefill 预测同时使用 prompt 长度与 prefix 长度，而不是只看未命中长度。
- 修改：将复用/重算比较改成相同起止点、两侧都考虑排队的成本式。命中前缀 H 后计算 U 个新 token 仍需让新 query 关注 H，不能写作仅对 U 的独立 prefill。路由分数明确使用 `T_prefill(P_i, S-P_i; config_i)`，标明示意模型而非 Conductor 原码。
- 修改：只将不能隐藏且位于首 token 前的传输计入 TTFT；P 已发送首 token 的流程中 handoff 可能体现于随后 ITL。
- 复算：以 causal query-key pairs 展示 H=100、U=10 时后缀计算为 1055 对而非 55 对，以及错误成本式导致的决策翻转。
- 未验证边界：没有重新拟合 Mooncake 的 profiling 多项式或测量实际排队、流水重叠；完整前缀命中后的 logits/尾 token 处理由引擎实现决定。

## 44. `2026-07-05-nixl-kv-connector-disaggregated-transfer.md`

- 全文阅读：完成。核查 backend/connector 边界、内存注册、描述符、TP 和 layout、租约与失败清理。
- 依据：[NIXL 官方设计](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md) 的注册、描述符、metadata 与 transfer；[vLLM NixlConnector compatibility](https://docs.vllm.ai/en/latest/features/nixl_connector_compatibility/) 的 TP/block/layout 限制；[GPUDirect RDMA §3.3](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html#unpin-callback)。
- 修改：KV 随 TP 缩小限定为 KV heads 确实分片的情况，MQA/MLA、TP 超过 KV heads 等配置可能复制。保留并复核了上一轮 P block 小于 D block 的 no-HMA 条件及明确的 LBHNC/LBNHC permutation 条件。
- 修改：租约到期/epoch 不匹配只能证明控制面的旧操作失效，不能证明 DMA 已停止；回收前必须撤销/排空并确认 NIC/GPU 无引用，否则隔离缓冲区。注册整个 arena 时更不能把局部逻辑释放等同于注销。
- 修改：心跳/租约状态机标为上层可采用的设计，不声称所有 NixlConnector 版本均逐请求实现它。
- 未验证边界：compatibility 页面是动态文档；未运行特定 GPU、backend、HMA 或异构 TP 的端到端互通测试。

## 45. `2026-07-06-dspark-confidence-scheduled-speculative-decoding.md`

- 全文阅读：完成。核查 accepted-prefix 期望、吞吐目标、early-stop、两步调度、graph bucket 和精确采样条件。
- 依据：[DSpark 论文 v1](https://arxiv.org/html/2607.05147v1) §3.2.2/Algorithm 1、§5.2 的 ZOS 和 Appendix A；[官方 SGLang 介绍](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/) 的 Verify modes、Ragged verify under full CUDA Graph 和 ceiling 估计边界。
- 高置信修正：删除“只要选连续 draft 前缀并保留标准 rejection，就不会改变目标分布”的过强断言。候选是否被纳入必须满足论文的非预知条件；事后看见依赖候选 token 的未来 confidence 再回头选最优截断点，可能改变 proposal admission 分布。
- 修改：加入 Appendix A 的可复算反例，目标 `(0.7,0.3)` 被事后 global argmax 改成 `(0.85,0.15)`。说明 early-stop 不看未来置信度；其全局最优性还依赖论文给出的单峰条件，阶梯状 CUDA Graph 代价本身不保证单峰。
- 修改：两步 confidence 用于决定 aggregate capacity，当前轮候选排序仍由当前 confidence 决定。区分 pure shadow 的完整提交与 `capaccept` 的完整验证/限制提交。随机采样下，相同 seed 不保证不同调度消耗相同 RNG 序列。
- 复算：完整反例、utility、early-stop 和首 token 概率均已断言。
- 未验证边界：未执行作者实现的 GPU 验证或复现性能上界；示例验证的是反例与条件，不是对整个实现的形式化证明。

## 46. `2026-07-08-nvidia-dynamo-distributed-inference-runtime.md`

- 全文阅读：完成。核查 router、Indexer、worker discovery、KVBM tiers、P/D 和弹性边界。
- 依据：[Router design v0.9.0](https://docs.dynamo.nvidia.com/dynamo/v-0-9-0/design-docs/router-design) 的 cache tracking/成本与事件；[KVBM design v1.0.1](https://docs.nvidia.com/dynamo/v1.0.1/design-docs/component-design/kvbm-design) 的 block 生命周期、G1–G4 与 TransferManager；[v0.9.1 offloading 指南](https://docs.nvidia.com/dynamo/v-0-9-1/user-guides/kv-cache-offloading)。
- 修改：事件示意改用官方 `removed`；多层缓存获益比较改成“载入 H + 计算依赖 H 的 U”与“完整重算 H+U”，避免遗漏后缀 attention 或重复计入 load/onboard。
- 保留：现有 v1.0.1 KVBM 引用可打开并支持对应设计，没有因为其他新版页面变化就错误删除 KVBM。没有把不同来源版本的能力合成一份“所有最新版均支持”的兼容性承诺。
- 复算：缓存路径的相同起止点成本由 Mooncake/Dynamo 共同测试覆盖。
- 未验证边界：没有部署整个 Dynamo stack；动态 dev 页面及不同 engine 集成的组合需要固定镜像后验证。

## 47. `2026-07-11-llm-inference-capacity-planning.md`

- 全文阅读：完成。核查 benchmark 边界、SLO、内存、KV、Little 定律、P/D 容量和扩容算例。
- 依据：[vLLM bench serve 参数](https://docs.vllm.ai/en/latest/cli/bench/serve/) 的 percentile/goodput 指标；[Transformers Cache explanation](https://huggingface.co/docs/transformers/main/cache_explanation) 的生成时间线；[DistServe §2–3、§6](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) 的阶段资源差异。Little 定律按其稳定流量、相同系统边界的定义逐式检查，没有将平均量当成尾延迟保证。
- 修改：TPOT 仅对输出数大于 1 定义，单输出请求不能记零来降低平均值。首 token 已由 P 发出时的 handoff 不自动算进 TTFT。
- 修改：分开逻辑 context 与已物化 KV：普通生成在第 g 个输出刚采样完时一般为 ISL+g−1 个位置；最终预算、共享块、预留块与 speculative 情况另计。输出 token/s 的 `λ E[OSL]` 不是 decode forward 工作量，后者基线为 `λ E[max(OSL−1,0)]`。
- 修改：Little 定律使用实际进入 decode 边界的 λ_D 及相同边界的 residence time；网络项使用 D 实际还需接收的 KV，而不是把 P 本地命中当成 D 命中。workspace 与 graph pool 不重复计费。
- 复算：输出/前向差异、单输出边界、KV 位置和原有 20 GPU 示例均通过。
- 未验证边界：容量算例是基于假设服务率的规划示例，未形成任何真实负载的吞吐或 P99 承诺。

## 48. `2026-07-14-llm-inference-failure-recovery.md`

- 全文阅读：完成。核查 request migration、token replay、exactly-once 边界、PDB、late response、取消与恢复容量。
- 依据：[Dynamo Request Migration 文档源码](https://github.com/ai-dynamo/dynamo/blob/main/docs/pages/fault-tolerance/request-migration.md) 的 token accumulation 与剩余生成预算；[当前 Request Migration 限制](https://docs.nvidia.com/dynamo/dynamo/dev/kubernetes/fault-tolerance/request-migration) 的 n>1/guided decoding 边界；[Kubernetes Disruptions](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/) 的 voluntary/involuntary 区分；[GPUDirect RDMA §3.3](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html#unpin-callback)。
- 修改：重放 prompt+已提交输出后的 prefill 可直接产生下一个 token，不应多算一次 decode；保留原始 prompt/输出边界、剩余 max_new_tokens 和只对生成历史生效的 penalty 状态。
- 修改：orphan KV/取消流程先排空 DMA 再释放；epoch fencing 不会物理停止旧传输。
- 修改：恢复资源使用 GPU·秒工作量除以目标恢复秒数，得到额外 GPU 数的下界，并按完整部署实例取整；线性 `c_p × tokens` 仅适用于已校准的窄区间近似，不直接保证 SLO。
- 复算：恢复工作量/时间的量纲断言通过。
- 未验证边界：未进行 kill worker、网络分区或客户端重试故障注入；随机序列、grammar FSM 和多候选恢复依赖产品支持。

## 49. `2026-07-17-expert-parallel-moe-token-dispatch.md`

- 全文阅读：完成。核查 router、top-k、dispatch/combine、TP×DP×EP、通信量和 shared expert。
- 依据：[Megatron MoE/Parallel Folding](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)；[vLLM EP deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)；[DeepEP 固定提交 elastic.py](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/buffers/elastic.py) 的 expand、num_recv_tokens 与专家索引接口。
- 修改：`N × k` 明确是有效逻辑 token-expert 分配数，包括本地专家；不是必然发出同数量的跨机 hidden rows/packets。同目标 rank 去重、局部聚合、padding 和无效 `-1` 路由均须分开计数。
- 修改：combine 必须含每个有效逻辑专家的贡献，但 rank 内预聚合后可以只传回少于 k 份物理向量。
- 复算：3 个 token、top-2 共 6 个逻辑分配，但同 rank 去重后仅 3 个物理目的行的例子通过。
- 未验证边界：具体 transport 是否去重、如何压缩和聚合由实际后端决定；本文没有给所有后端共同的物理字节公式。

## 50. `2026-07-20-moe-expert-load-balancing-eplb.md`

- 全文阅读：完成。核查训练侧 bias、推理侧复制、logical/physical expert 映射、greedy 和迁移时序。
- 依据：[DeepSeek EPLB 源码](https://github.com/deepseek-ai/EPLB/blob/main/eplb.py) 的 replicate_experts、balanced_packing、rebalance_experts 返回结构；[DeepSeek-V3 v2 §2.1.2](https://arxiv.org/html/2412.19437v2) 的 auxiliary-loss-free balancing 与 sequence-wise auxiliary loss。
- 修改：max/mean 指标限定为非空统计窗口；空批次没有可用的比率，不能记作零负载不均衡。
- 高置信修正：伪代码对 `logical_to_physical[e]` 先按 `logical_count[e]` 截取再选副本。官方逆表是矩形数组，未使用槽位填 `-1`，直接按整个数组长度取模可选中非法 physical expert。
- 复算：greedy 副本分配 `[2,2,1,1]` 和 `-1` padding 的有效槽位过滤通过。
- 未验证边界：未对在线迁移并发、热点突变或复杂拓扑重平衡做压测；balanced packing 的等容量前提仍须由部署满足。

## 51. `2026-07-23-deepep-moe-dispatch-combine.md`

- 全文阅读：完成。核查 V1/V2 边界、ElasticBuffer、EPHandle、低延迟容量、异步完成与 shared expert overlap。
- 依据：[DeepEP 仓库及 V2/legacy 说明](https://github.com/deepseek-ai/DeepEP)；[固定提交的 elastic.py](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/buffers/elastic.py)，尤其 dispatch、num_max_tokens_per_rank、do_expand、num_recv_tokens 和 `-1` 无效专家定义。
- 修改：示意图 routed 分支补完整 dispatch → expert GEMM → combine 后再与 shared expert 合并，不能把完成 dispatch 当作 routed output 已就绪。
- 修改：逻辑 assignment 守恒与实际接收行/扩展后行数分开；去重、alignment padding、无效路由都不改变“只统计有效逻辑贡献”的原则。
- 保留并复核：`num_max_tokens_per_rank` 是源 rank 输入 token 上限，不是目的 rank 聚合接收数上限；未将 legacy 的资源使用结论直接套给 V2。
- 复算：逻辑/物理行差异由 EP/DeepEP 共同测试覆盖。
- 未验证边界：没有编译 DeepEP 或测量其 SM/QP/带宽；返回形状和完成条件以固定提交及选定模式为准。

## 52. `2026-07-26-wide-expert-parallel-moe-serving.md`

- 全文阅读：完成。核查扩大 EP 的 batch/显存/网络权衡、shared expert、DeepSeek 部署示例和 parallel folding。
- 依据：[DeepSeek 官方推理系统概览](https://raw.githubusercontent.com/deepseek-ai/open-infra-index/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) 的 System Design/Large-scale Cross-node Expert Parallelism；[Megatron Parallel Folding](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)。官方 prefill EP32/DP32、decode EP144/DP144 及物理 expert 数量核对无误。
- 修改：每 rank 等量 expert slots 要求 `(E+R)` 能被 `P_e` 整除，否则需不等量布局。
- 修改：目标 expert batch 推导先指逻辑 expert 收到的 assignment；复制后每份物理 GEMM 还要除以副本数。原示例全局 512 行、top-8、256 逻辑专家得平均 16 行/逻辑专家，若两份副本则平均 8 行/物理副本。
- 复算：9 slots/32 ranks、2 slots/144 ranks 以及逻辑/物理 batch 差异通过。
- 未验证边界：平均值不能保证热点 expert 或尾延迟；示例部署不能直接推导任意模型的最优 EP 度。

## 53. `2026-07-29-cuda-graph-llm-serving-static-replay.md`

- 全文阅读：完成。核查 capture/replay、shape/address、静态输入、private pool、padding、bucket 与示例代码。
- 依据：[CUDA Graphs 指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) §4.2.2 capture、§4.2.3 update、§4.2.4 conditional nodes；[PyTorch 2.14 graph memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html#graph-memory-management)；[vLLM CUDA Graph design](https://docs.vllm.ai/en/stable/design/cuda_graphs/)。
- 修改：静态限制限定为通常的 PyTorch capture/replay 路径，不能断言 CUDA Graph API 本身不支持 update 或条件节点。同地址原地更新权重不必然重捕获，但地址/shape/kernel 配置变化需重新评估。
- 修改：padding 浪费率 `(C−B)/C` 与相对有效工作的额外比例 `(C−B)/B` 区分；19→24 是约 20.8% 与 26.3%。内存按唯一 private pool 的 reserved bytes 加池外 buffers 等计算，不将 pool 内 workspace 再次相加。
- 修改：示例使用 `model.eval()`、初始化的静态输入与 `inference_mode()`；说明 pool sharing 的执行顺序/不并发要求。
- 复算：两种分母、共享 pool 去重计账通过。
- 未验证边界：代码未在 GPU 上执行；模型是否含不安全操作、实际 graph pool 占用与所选 vLLM 模式仍需设备验证。

## 54. `2026-07-31-torch-compile-llm-serving-graph-breaks.md`

- 全文阅读：完成。核查 Dynamo/FX/AOTAutograd/Inductor 链路、guard、graph break、custom ops、AOT 与编译回本模型。
- 依据：[PyTorch 2.14 compiler programming model](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.html) 的 graph breaks/guards/non-strict tracing；[torch.compiler AOT compile 官方文档](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_aot_compile.html) 的 experimental API、fullgraph 限制与 AOTInductor 区别。
- 修改：custom op 的 schema/fake/meta 正确不等于它自动支持 CUDA Graph；副作用、mutation/alias 及捕获安全是不同契约。
- 修改：普通 Dynamo 路径会按其语义保留支持的副作用或断图，不应笼统声称编译会删除副作用；风险限定为 non-strict tracing 的纯函数前提和 CUDA Graph 的 host/replay 边界。
- 修改：AOT compiled Python callable 与 `torch.export` + AOTInductor 分开；回本单位改为对应编译变体调用次数，用向上取整并定义无正向运行节省时不能回本。
- 复算：正收益、整除/非整除边界、零收益及负收益通过。
- 未验证边界：未实际编译文章示例；不同 PyTorch 版本的实验 API、后端 graph break 及缓存覆盖需固定环境验证。

## 55. `2026-08-01-fp8-formats-scaling-llm-inference.md`

- 全文阅读：完成。核查 E4M3/E5M2、normal/subnormal、scale、GEMM、KV cache、MXFP8 和性能边界。
- 依据：[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) Table 1；[Transformer Engine FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) 的 formats、delayed/current scaling、HYBRID 和 MXFP8。E4M3 最大 448、最小 normal/subnormal 的既有区分核对无误。
- 修改：`amax / max_code` 定义域为有限且非零 amax；全零块可用正 scale=1 和全零 code，NaN/Inf 必须另定策略。明确 scale 与 inverse scale 的约定。
- 修改：GEMM 外提 scalar/per-row/per-output-channel scale 必须不随 reduction K 变化；K-block scale 要分别缩放 partial sums，不能整体提出。
- 修改：KV scale 可来自离线校准，不必都是运行时动态统计；“必须外部缩放”改为允许 scale=1 的隐式情形。
- 复算：全零组、K-invariant factorization 与 K-varying 反例通过。
- 未验证边界：没有验证某个 FP8 变体的完整 NaN 编码表或执行 Tensor Core 内核；格式、scale 粒度和 accumulator 必须作为具体 kernel 契约确认。

## 56. `2026-08-02-smoothquant-w8a8-activation-outliers.md`

- 全文阅读：完成。核查等价缩放推导、alpha 的影响、校准、INT8 GEMM 和实际算子范围。
- 依据：[SmoothQuant 原论文](https://arxiv.org/html/2211.10438v1) §3.1 的量化乘法、§3.2 的等价变换与 scaling、Figure 5 和 §4 的 O1/O2/O3 recipes。
- 修改：INT8 点积明确索引 `M×K` 乘 `K×N`，说明每行 activation 与每输出通道 weight scale 可外提，仅因为它们不随 K 变化；groupwise K scale 要对 partial sums 处理，非对称量化还需 zero-point correction。
- 保留并复核：alpha 不等于单调改善全部层，outlier 迁移保持的是量化前等价性；模型并非所有算子都变成 INT8。
- 复算：缩放因子外提的合法情形及不合法反例通过。
- 未验证边界：没有运行校准数据集或量化模型，不能承诺本文示例 alpha 对用户模型最优。

## 57. `2026-08-03-awq-activation-aware-weight-quantization.md`

- 全文阅读：完成。核查 salient weights、activation-based scaling、搜索目标、clipping、量化格式和 TinyChat。
- 依据：[AWQ 论文 v4](https://arxiv.org/html/2306.00978v4) §3.1、§3.2 及 TinyChat；[官方 quantizer.py](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/quantizer.py) 的 `pseudo_quantize_tensor`、zero_point 默认参数和全零/小范围 scale 防护。
- 修改：对称量化教学式限定有限权重、整数 bit 数至少 2、全零组使用正 scale。明确它不是 AWQ 唯一部署格式，官方实现支持非对称 zero point。
- 修改：clipping 阈值范围与重新计算的 scale 定义补齐，避免把只裁剪、保持旧量化步长当成完整 clipping 优化。
- 保留并复核：“约 1% salient weights”是论文特定实验观察；activation 重要性、scale 对 group range 的假设、无梯度搜索与 reconstruction 区分均保留。
- 复算：全零组除零防护通过。
- 未验证边界：未复现 saliency/scale 搜索或质量结果；对称示例不充当官方所有模型 quantizer 的逐位复刻。

## 58. `2026-08-04-w4a8-progressive-quantization-kernel.md`

- 全文阅读：完成。核查 QoQ 两级量化、形状、INT8 protected range、register-level parallelism、reordering、metadata 和性能来源。
- 依据：[QServe 论文 v3](https://arxiv.org/html/2405.04532v3) §4.1 Eq.4–5、§5.2.1 reordering、§5.2.2 zero-point、§5.2.3 register-level parallelism 与 §6 评估。
- 修改：scale 运算明确是广播而不是矩阵乘法；按 `W_kn` 给出输出通道 scale、K-group scale 和 zero point 的索引。第二级反量化近似恢复第一级 INT8，不是无损复原。
- 修改：溢出例子写明先 round 再加 zero point 的顺序；120/16 四舍五入为 8、加 7 为 15，恢复为 128。不同取整/zero-point 放置可能改变临界例子，不能隐去实现约定。保留 protected range [-119,119] 的目的，但不声称任意未使用 nibble 都对应合法模型数值。
- 复算：论文式溢出例子、[-119,119] 各整数 group 端点的 nearest-rounding 恢复范围、metadata 开销均通过。
- 未验证边界：测试不是 INT4 packed kernel 的逐位模拟，也未测 overflow assembly 或吞吐；metadata 示例标为通用格式，不能视为所有 QoQ 布局开销。

## 结论

18/18 篇已完整阅读并作定点修订。最重要的新增问题是：非因果候选筛选对精确采样的破坏、取消/失败时 DMA 生命周期先后顺序、缓存复用成本忽略历史前缀依赖、SI/IEC 与输出/前向计数混用、逻辑专家分配和物理通信混用、EPLB 逆表 padding、Graph 私有池重复计费，以及量化 scale 在 reduction 维度上的错误外提条件。

本记录说明当前审阅覆盖与可复算证据，不等于证明所有文章或所有版本实现绝无错误；实机验证边界已逐篇保留。
