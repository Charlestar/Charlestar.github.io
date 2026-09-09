# 2026-09-09 全文复审：分布式训练与相邻系统文章

范围：按 `Get-ChildItem _posts -File | Sort-Object Name` 的审阅时顺序，第 59–70 篇，共 12 篇。以下各篇均从 front matter、开头正文读到参考资料末尾，包含公式、表格、代码/伪代码和文本图；输出被截断的部分已重新读取。不是对上一份报告的关键词复查。

本文区分三类证据：正文与一手资料对照、独立算术/状态反例、未执行的真实系统验证。来源访问日期为 2026-09-09；`main/latest` 是当日可见文档，不保证未来行为。没有运行 GPU、NCCL、训练收敛、硬件故障或跨节点恢复实验。“未发现新增问题”不等于证明不存在错误。

## 59. `2026-08-06-data-parallel-zero-fsdp-memory-sharding.md`

- 全文：已读完，审阅时 654 行。
- 核查点：DP 平均梯度与 micro-batch 权重；混合精度 Adam 的 16 bytes/parameter 假设；ZeRO-1/2/3 的持久状态公式及 7B、8 卡数值；临时完整参数/梯度、root module、prefetch；FSDP1/2 生命周期、mixed precision、hybrid group；`no_sync`、tied weights、offload 的限定。
- 一手依据：[ZeRO §3、§5 Model State Partitioning、§7 Communication Volume](https://arxiv.org/html/1910.02054v3)；[FSDP2 `fully_shard` 与 `MixedPrecisionPolicy`](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)。官方说明原始 dtype 的 shard 供 optimizer 使用，`param_dtype` 决定 all-gather/计算，`reduce_dtype` 单独控制归约。
- 结论：未发现需要新增修复的硬错误。112、38.5、26.25、14 GB 的理想 model-state 数值与文中假设一致；保留既有混合精度和峰值显存限定。
- 未验证边界：不同 PyTorch 版本的 root/prefetch/no_sync 组合未逐项执行；未测实际峰值显存、通信量或吞吐。

## 60. `2026-08-08-megatron-tensor-parallel-transformer.md`

- 全文：已读完，审阅时 644 行。
- 核查点：column/row parallel 形状、非线性所在位置、partial sum 与 bias；Attention head 切分和 GQA 复制条件；vocab-parallel cross entropy 的全局 max/exp-sum/目标 logit；TP/DP group 映射、全局 gradient norm 的重复计数；TP+SP、通信重叠和 tied embedding。
- 一手依据：[Megatron-LM 2019 §3、Figure 3](https://arxiv.org/pdf/1909.08053)；[Megatron 2021 §2.3 Tensor Model Parallelism](https://arxiv.org/pdf/2104.04473)；[Reducing Activation Recomputation §4.2.2–4.2.3](https://arxiv.org/pdf/2205.05198)。
- 结论：未发现新增硬错误。两层 MLP 的分块求和、Attention 输出投影和分布式 CE 恒等式一致；没有把 TP kernel 或所有 GQA 头数条件泛化成无约束实现。
- 未验证边界：未执行 BF16/fused kernels、vocab padding、checkpoint 转换和不同 GQA 头数的分布式 parity 测试。

## 61. `2026-08-11-pipeline-parallel-microbatch-1f1b.md`

- 全文：已读完，审阅时 424 行。
- 核查点：micro-batch/global batch、同步 flush 权重版本；GPipe 与 1F1B 的 warmup/steady/cooldown；在途 activation；bubble 分母口径；virtual chunks 的 cyclic mapping、通信边界数和显存权衡。
- 一手依据：[Megatron 2021 §2.2.1、§2.2.2 与 Figure 3/4](https://arxiv.org/pdf/2104.04473)。论文的 bubble/ideal-work 比与本文 bubble/total-time 比分母不同，不能因数值形式不同认定本文错误。
- 结论：未发现新增硬错误。`(p-1)/(m+p-1)` 是等时、均衡、忽略通信时的总时间空闲占比；正文已限定同步 1F1B，而不是把 PipeDream 异步语义混入。
- 未验证边界：没有生成所有 stage/micro-batch 的可执行 schedule；interleaved 的整除约束和通信 overlap 仍依赖具体 runtime。

## 62. `2026-08-14-sequence-context-parallel-long-context-training.md`

- 全文：已读完，审阅时 454 行。
- 核查点：Megatron SP 的 AG/RS 互为反向操作；CP 本地 Q 与全上下文 KV；online softmax 合并的未归一化统计；causal 负载均衡；Ulysses 式 head/token 转置；TP×CP group 与位置/RNG 映射。
- 一手依据：[Reducing Activation Recomputation §4.2.2](https://arxiv.org/pdf/2205.05198)；[Ring Attention §3](https://arxiv.org/html/2310.01889v4)；[Megatron CP Overview、Figure 1](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)；[`TransformerConfig.cp_comm_type`](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html) 对 `p2p/all_gather/a2a/a2a+p2p` 的定义。
- 修改：把 `offset_r+j` 限定为连续分片；非连续首尾配对使用 `index_map_r[j]`，区分物理位置与 packed sample 内 RoPE 位置；补齐逐 Query row 的初始化、全 masked block 跳过、首个有效 block 及 `l=0` 边界。
- 数值复核：8 个 token 的 `[0,1,6,7]` 不能用单 offset 表示；两个 zigzag shards 的 causal 有效配对数各为 18。带空块/全 masked 块的 online merge 与直接 softmax 相等；朴素 `-Infinity-(-Infinity)` 为 NaN。
- 未验证边界：空 row 的返回值由 backend contract 决定；未验证 dropout 的并行规模无关 bitwise RNG、CP backward 通信或具体 kernel。

## 63. `2026-08-17-distributed-checkpoint-resharding-async-save.md`

- 全文：已读完，审阅时 516 行。
- 核查点：训练快照/export/recomputation 的区别；logical shard metadata 与 optimizer FQN；save/load 方向的目标布局；一致快照、manifest-last 和 storage durability；async staging、队列反压；data/RNG cursor、PP flush、Young 式频率近似。
- 一手依据：[PyTorch DCP Overview、`load`、`state_dict.get_state_dict`](https://docs.pytorch.org/docs/main/distributed.checkpoint.html)（in-place 目标存储与 optimizer ID→canonical FQN）；[DCP async 教程 “Fully Asynchronous Staging with DefaultStager”](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html#fully-asynchronous-staging-with-defaultstager)（2.9 引入、staging/upload 两个完成边界及 optimizer 前等待示例）。
- 修改：同步 staging 后异步写盘只是其中一种路径，不能宣称全部 D2H 都是最低前台 stall；说明异步 staging 可与只读计算重叠，必须早于首次原地修改待保存状态完成。将“序列化通信 request”改为专门 quiesce/日志/重放协议，明确普通 state dict 不能恢复活跃 CUDA/NCCL 句柄。
- 数值/状态复核：两参数分时拷贝期间更新会得到不属于任何 committed version 的混合快照；8 单位 staging 与 5 单位只读计算可重叠，暴露等待是 3 而非 8，纯示意非实测。
- 未验证边界：未做远端对象存储 durability、writer failure、TP/PP resize 或多步恢复 parity；manifest-last 是文章提出的协议要求，不声称所有 DCP storage backend 自动提供同样保证。

## 64. `2026-08-18-rlhf-grpo-system-dataflow.md`

- 全文：已读完，审阅时 699 行。
- 核查点：Actor/rollout/old/reference 四种身份；behavior log-prob 与过滤后采样空间；PPO ratio/clipping、token mask；GAE terminal/truncation；GRPO group normalization 与 failure join；训练/推理布局、KV 版本失效、角色生命周期和 checkpoint 摊销。
- 一手依据：[DeepSeekMath §4.1.1 Eq.3–4、Algorithm 1、Appendix A.1.6](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS1)；[GAE §3 Eq.10–16](https://arxiv.org/pdf/1506.02438)；[HybridFlow 系统设计与 3D-HybridEngine](https://arxiv.org/html/2409.19256v2)。
- 修改：写出 k3 的无偏条件为固定状态下从当前 policy 采样；旧 rollout 的多轮训练不自动满足此条件，importance weighting 只在所述支持集与固定状态层面恢复期望，不随意改写原始 GRPO surrogate。分开持久 optimizer moments 与 step 内梯度；截断可保留 value bootstrap，但不能把下一条 packed response 的 advantage 接上。
- 数值复核：二元 `p=(0.7,0.3), q=(0.4,0.6), old=(0.5,0.5)` 独立求和验证三种期望；GAE 单段截断反例验证 value bootstrap 与 trace 截断必须分开。
- 未验证边界：未执行 RL 训练、奖励可靠性、importance sampling 方差/梯度无偏性、off-policy 状态分布或版本发布故障注入。

## 65. `2026-08-20-gpu-collective-communication.md`

- 全文：已读完，审阅时 820 行。
- 核查点：placement、root/in-place/count 语义；Ring AR/AG/RS 的轮数与字节；Tree 模型假设；FSDP、TP、CP、EP 映射；ragged counts、group ordering、stream buffer 生命周期；API/Queue/Kernel/consumer 测量；algbw/busbw。
- 一手依据：[NCCL Group Calls 的 enqueue/order 约束](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)；[nccl-tests PERFORMANCE 的 collective 换算因子](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)；[FSDP2 MixedPrecisionPolicy](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.MixedPrecisionPolicy)；[DeepEP `ElasticHandle.psum_num_recv_tokens_per_scaleup_rank`](https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffers/elastic.py)（每 token 每 rank 去重与 expert alignment 分开）。
- 修改：`3(p-1)B/p` 增加参数/梯度等字节前提并给一般式；区分 Top-k assignments、按目标 rank 去重后的行数和远端 payload；通信 imbalance 的全零轮记为 N/A，并区分网络与 expert compute 均衡。
- 数值复核：4 ranks、参数 20 bytes/梯度 40 bytes，逐轮枚举的两 AG+RS 为 60 bytes 而非 45；6 assignments 可对应 4 个 rank-token 行与仅 2 个跨 rank 行。
- 未验证边界：未跑 NCCL benchmarks，未识别某台机器实际算法/协议；DeepEP main 的特定 layout 仅作为反例，未泛化到所有 dispatcher。

## 66. `2026-08-21-multi-lora-serving-adapter-batching.md`

- 全文：已读完，审阅时 526 行。
- 核查点：LoRA A/B 方向、rank 与 scaling；共享 base GEMM、segmented/grouped 的请求映射；KV/adapter 争用和生命周期；cache identity 与 CUDA Graph identity 的区别；TP placement、混合量化、slot eviction/generation。
- 一手依据：[Punica §4 的 SGMV 与 batch segment indices](https://arxiv.org/pdf/2310.18547)；[S-LoRA §5.1 Unified Paging、TP 设计](https://arxiv.org/html/2311.03285v2)；[vLLM v0.25.0 `BatchExecutionDescriptor` 源码字段](https://docs.vllm.ai/en/v0.25.0/api/vllm/v1/worker/gpu/cudagraph_utils/#vllm.v1.worker.gpu.cudagraph_utils.BatchExecutionDescriptor)。
- 结论：未发现新增硬错误。当前正文的 CUDA Graph 复用限定与所引版本源码一致；具体 adapter ID 不在该 descriptor 内，rank/shape/地址/映射一致仍是前提。
- 未验证边界：未运行 multi-adapter numerical parity、TP LoRA、quantized base、slot concurrent eviction 或 graph capture/replay；没有把 S-LoRA 的统一 paging 说成所有引擎默认实现。

## 67. `2026-08-22-pd-elasticity-fast-worker-startup.md`

- 全文：已读完，审阅时 625 行。
- 核查点：desired/provisioned/ready、P/D 独立负载与互相反压；启动关键路径和带宽下界；分层加载、强 artifact identity、compile/graph/KV cache 的不同契约；VMM 不是进程快照；parallel group readiness、drain、异构兼容和回退。
- 一手依据：[ServerlessLLM 的 multi-tier checkpoint/loading/scheduling 设计](https://www.usenix.org/system/files/osdi24-fu.pdf)；[Dynamo ModelExpress “How It Works”](https://docs.nvidia.com/dynamo/knowledge-base/kubernetes/model-loading/model-express)；[vLLM “Faster Startup”](https://docs.vllm.ai/en/latest/configuration/optimization/#faster-startup)。
- 修改：启动各阶段相加限定为串行基线；流式 loading/初始化并行时，实际 ready time 由带资源约束的启动 DAG 关键路径决定。
- 数值复核：两块读取与拷贝流水、并行 runtime 初始化的简单 DAG，阶段总和 46，真实最长路径 27（虚构时间单位，不是启动性能数据）。
- 未验证边界：没有部署 Kubernetes、ModelExpress、VMM 或启动 canary；控制器/Ready Gate 是文章提出的工程协议，不声称现有产品默认实现所有检查。

## 68. `2026-08-23-zero-bubble-pipeline-parallelism.md`

- 全文：已读完，审阅时 800 行。
- 核查点：Linear dgrad/wgrad 形状与 DAG；F/B/W 净显存；H1/H2/1p/2p/V 的假设、路径和通信；自动搜索约束、TP/SP/DP、重算；post-validation、rollback、speculative activation、checkpoint 提交边界；DualPipe 字母含义差异。
- 一手依据：[Zero Bubble §2、§3、§4、§6、Appendix C](https://arxiv.org/pdf/2401.10241)；[官方仓库各 schedule 比较与命名](https://github.com/sail-sg/zero-bubble-pipeline-parallelism)。附录 C 的逆运算明确除以 betas 与 `1-lr*weight_decay`；§6 给出 V placement 和等时/内存结论。
- 修改：开头“下一轮 F 只读 committed 参数”明确为不启用 speculative post-validation 的基线，与后文 provisional/replay 协议对齐；补齐 AdamW 算术逆的非零分母前提。
- 数值复核：一般 AdamW 参数的正向/逆向还原；`beta1=0` 或 `lr*decay=1` 时不同旧状态映射到同一新状态，不能仅靠该逆公式恢复。
- 未验证边界：没有穷举全 schedule 或执行真实 optimizer rollback；不承诺浮点逆逐位恢复、任意 optimizer 可逆或任意 fused backend 支持拆分。

## 69. `2026-08-24-long-request-kv-watermark-scheduling.md`

- 全文：已读完，审阅时 563 行。
- 核查点：chunk 进度与 cache 命中、有效 KV 游标/物理分配；grow reserve、互斥 watermark/exhausted 边界；fairness、deadline/urgency；预留 handoff credit；preempt/offload/migrate/cancel 生命周期和 finish_reason；公式与状态机/伪代码一致性。
- 一手依据：[Transformers “How caching works” 的逐轮 forward→sampling 循环](https://huggingface.co/docs/transformers/main/cache_explanation)；[vLLM “Preemption” 与 “Chunked Prefill”](https://docs.vllm.ai/en/latest/configuration/optimization/#preemption)。
- 修改：TTFT 不无条件加首个 Decode/handoff，按首 token 可见前的实际关键路径计数；TPOT 明确 `N_output>=2`，单 token 为 N/A；状态清单补充可选暂停/抢占与恢复边；Critical/Emergency 停止新增准入但允许已有请求在安全预留内推进，避免与前文保护策略矛盾。
- 数值复核：Prefill 100+sampling 2+发送 3 的首 token 时间为 105，不能再加下一轮 Decode；三个到达时点的平均 TPOT 不代表每个 ITL；单 token 不执行除零。
- 未验证边界：未证明任意过载条件下 aging 必然完成请求；未运行真实 engine 的 watermark、migration、cancel 或混合 KV groups 压测。水位阈值是示意策略而非 vLLM 默认配置。

## 70. `2026-08-25-gpu-interconnect-topology.md`

- 全文：已读完，审阅时 1509 行。
- 核查点：HBM/copy/SM/NIC initiator；NVLink/NVSwitch/PCIe/NUMA/host staging 路径；GDR 注册、ordering bridge、ACS/IOMMU；端点/截面/超售率与方向口径；UUID/MIG/BDF、topo/sysfs/RDMA 映射；capability/status/counters/diagnostics；counter delta、benchmark 分层与假设；最后十步数据流。
- 一手依据：[CUDA 多 GPU §3.4.1.3、§3.4.2、§3.4.2.5](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)；[GPUDirect RDMA §3.7 Synchronization and Memory Ordering](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html#synchronization-and-memory-ordering)；[Linux P2PDMA 的 hierarchy/ACS 与生命周期约束](https://docs.kernel.org/driver-api/pci/p2pdma.html)；[NVSMI Topology 与 `-gpu/-nic` 定义](https://docs.nvidia.com/deploy/nvidia-smi/index.html#topology)；[DCGM Topology and NVLink 的四类证据及代际字段](https://docs.nvidia.com/datacenter/dcgm/latest/learn/core-services/topology-and-links.html)；[nccl-tests PERFORMANCE](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)。
- 修改：链路公式明确 bit/s→byte/s 的 `1/8`，避免 GT/s/symbol/s 与编码开销重复扣除；末尾接收链条改为 NIC 发起 DMA，再经 peer path 写入 HBM，再形成消费依赖，避免“写入已完成”出现在实际数据路径之前。
- 数值复核：单 lane 原始 8 bit/s、4 lanes、80% payload 效率得到 3.2 byte/s；这是量纲示例，不是任何 SKU 的产品数据。
- 未验证边界：未访问/修改真实 Linux GPU/NIC、ACS/IOMMU/BIOS；未测 GDR、NVLink、counter 或错误注入。平台专用资格、最新工具命令仍需按安装版本和整机官方 qualification 验证。

## 可复算验证与最终范围

- 独立脚本：`node scripts/reviews/full-training-examples.mjs`。
- 已执行结果：11 个 arithmetic/state examples 全部通过，Node 内置 `assert`，零外部依赖。
- 本轮新增修改：第 62、63、64、65、67、68、69、70 篇，共 8 篇；第 59、60、61、66 篇保留正文。
- 已修改文章的 `last_modified_at` 为 2026-09-09；文件名、URL、原始发布日期、系列顺序保留。
- 未在此分工内提交或推送；站点统一构建、渲染/链接校验由主任务完成。
