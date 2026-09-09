# Activation Checkpointing：调研与写作交接

状态：调研完成；2026-09-09。仅调研记录，不包含训练/GPU 实验复现。

## 选题与系列定位

- 固定文件：`_posts/2026-09-09-activation-checkpointing-rematerialization.md`。
- 建议标题：`Activation Checkpointing：反向传播需要的张量，何时保存、何时重算`。
- `series: distributed-training`、`series_order: 45`、`technology_year: 2016`、`mathjax: true`、`tags: [分布式训练, GPU优化]`。45 在现有 NCCL=40 与 PP=50 之间，符合主审安排，不改旧文。
- 现有 FSDP 文仅有一节介绍该技术，PP/Zero-bubble/SP 多次依赖该概念，没有独立推导；与 Distributed Checkpoint 的持久化训练快照必须显式区分。
- 2016 指神经网络 sublinear-memory 代表论文，不得写“重计算思想于 2016 首次发明”：论文 §2 自己指出自动微分 checkpointing 的更早历史。

## 实际查阅的一手来源

1. [Training Deep Nets with Sublinear Memory Cost, v2](https://arxiv.org/html/1604.06174v2)：实际读取 §2–4.4、§5.1 的口径说明。重点 §4.1 算法1的分段重计算、§4.3 的 O(n/k+k) 存储模型、§4.4 的递归代价。§5.1 明确 feature-map 估算不含参数及 workspace，不能称总显存 O(sqrt(n))。
2. [Reducing Activation Recomputation in Large Transformer Models, v1](https://arxiv.org/html/2205.05198v1)：实际读取 §3、§4.1–4.3、§5/Table2。该文 `sbh(34+5as/h)` 字节模型使用 16-bit 保存张量、1-byte dropout mask、4h GeLU MLP、显式注意力中间矩阵。§5 的 selective recomputation 针对 QK/softmax/dropout/PV 区域；不能直接套到 FlashAttention、SwiGLU、GQA 的现代 block。
3. [PyTorch v2.8.0 checkpoint.py](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/checkpoint.py)：实际读取 checkpoint 文档/分支、`_infer_device_type`/RNG 保存恢复、`CheckpointPolicy`、`create_selective_checkpoint_contexts` 与 non-reentrant 重算入口。固定版本作为 API 语义来源，不称其为当前最新版本。
4. [PyTorch 官方：Current and New Activation Checkpointing Techniques](https://pytorch.org/blog/activation-checkpointing-techniques/)：实际读取 Activation Memory Basics、AC、torch.compile/min-cut、SAC、Memory Budget API。强调 runtime-oriented min-cut 与“保存所有便宜/昂贵算子”的启发式不是同一个问题；该文自己把 memory-budget 接口标为 experimental，勿写成通用稳定生产 API。
5. [NVIDIA Megatron Bridge Activation Recomputation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html)：实际读取 Quick Guidance、Configuration、Full/Selective/FlashAttention 相关段落（latest 页，2026-09-09 查阅）。用于提示 full/selective、uniform/block 策略与版本/后端依赖，不复述可变 flag 清单当作跨版本教程。

PyTorch stable/checkpoint 页面在本次浏览重定向为极少内容，因此关键语义直接核对固定 v2.8.0 官方源码，不宣称从重定向页读到了完整文档。正文优先引用以上固定源码与原论文。

## 必须先定义的对象

- 激活：前向产生、后向还需要的中间张量；不只是非线性层输出。参数、参数梯度、optimizer state、输入、workspace 分开记账。
- 示例链：z=xW1，a=φ(z)，y=aW2。给定上游梯度 G，dW2=a^T G，dA=G W2^T，dZ=dA⊙φ'(z)，dW1=x^T dZ。具体 backward 保存 z、a 或其它辅助统计取决于算子。
- activation checkpoint 是当前 step 内的“重算锚点”；与磁盘训练状态 checkpoint 没有等价关系。
- “丢弃”是缩短张量存活期或不持有反向 saved-tensor 引用，不要求显式清空 allocator 缓存；Python 外部引用和返回所有 hidden states 会抵消节约。

## 独立推导与可复算例

### 分段账单（理想链，不是显存承诺）

取 n 个等成本层，每层代表性保存量 A；每段含 k 层，暂取 k 整除 n。保留约 n/k 个边界，反向时只物化一个段的约 k 份中间量：

`M_saved(k) ≈ A(n/k+k)`，忽略端点常数、共享 storage、参数/梯度/workspace；连续最优 k≈sqrt(n)，取整并按实际图测量。用 k 表示“每段层数”，不得与原论文表示“段数”的 k 混用。原论文的结论在这个对称模型下相同。

- n=64、A=32 MiB：无重算主项 2048 MiB；k=4 时 (16+4)A=640 MiB；k=8 时 (8+8)A=512 MiB；k=16 时 (4+16)A=640 MiB。
- 这些是模型估算，不是 cuda.max_memory_allocated 结果。逐 block checkpoint 的现实通常保留每层输入；不能自动声称它达到了 sqrt(n) 调度。
- 若边界大小 A_b 与段内每层保存量 A_i 不同，`M(k)≈(n/k)A_b+kA_i`，极值 `k≈sqrt(n A_b/A_i)`。这个推广可作为独立解释：为什么不同网络不能照搬“每 sqrt(n) 层切一次”。

### 时间账单

设原 forward 工作量 F、backward B、重算 R：理论工作量倍率 `(F+B+R)/(F+B)`。只有假设 B≈2F 且 R≈F，才有 4/3；同硬件、忽略其它变化时 step 时间增加约 33%，吞吐变为 3/4（下降25%，不是下降33%）。这不是普适实测数字；重算可能触发通信、改 fusion、允许更大 micro-batch，也可能改善端到端吞吐。

### Transformer 张量账单

- B=1、S=4096、H=4096、2 bytes/element 时，一份 BSH 张量是 33,554,432 bytes=32 MiB。
- 单个完整 attention matrix 若 H_q=32、shape=[B,H_q,S,S]、2 bytes/element，则为 1,073,741,824 bytes=1 GiB。这里只算“一份矩阵”，不是现代 attention 实际存储总量。
- FlashAttention 已不物化完整 S² attention 矩阵并在后向重算分块概率，因此不能再在外层 AC 账单中重复扣除一个不存在的 S² 缓冲。外层 checkpoint 的收益应重新测量 Q/K/V、MLP 等真实 saved tensors。

### 策略选择模型

可以把候选 region 的“实际峰值下降 ΔM_i / 额外时间 ΔT_i”作为观察指标，但 ΔM/ΔT 不是独立可加的物品价值：依赖闭包、张量重叠存活、共享输入、fusion、collective 使简单贪心没有全局最优保证。不要把估算表包装成精确 knapsack 最优解。

## 针对性递进大纲

1. **一次 backward 为什么要借用很久以前的 forward**：用两层 MLP 导数逐个列出被借用的张量，随即说明生命周期；不用一开始罗列框架 flag。
2. **少保存一次，意味着以后从哪里重建**：前向留分段边界、反向按逆序重建当前段。画简短时间线或依赖链，有图比口号更清楚；区分磁盘 checkpoint。
3. **分段太细与太粗都要付账**：从边界 n/k 与局部 k 两部分推导 sqrt(n)，再代入64层表。限制是等成本链及保存量，不是整个 Transformer 显存。
4. **一次额外 forward 到底贵多少**：给 F+B+R 及4/3条件，指出非重入 early-stop 并非必定执行完整第二次函数；吞吐和step时间百分比不同。
5. **Transformer 不同张量值得保留的程度不同**：BSH 与 BH_qS² 算例；老 Megatron selective 的历史目标；FlashAttention 改变保存张量后，应重选边界。自然过渡到现代 selective，不混淆两种“selective”的粒度。
6. **PyTorch 的执行契约**：默认显式写 use_reentrant=False；简短代码展示 region 函数及 checkpoint 调用，解释保存输入、参数引用、early-stop。可选小表讲 non-reentrant/reentrant，而不是长API字典。
7. **重算必须重新得到同一条计算路径**：dropout RNG、mutable buffers、Python/global state、in-place、循环closure、参数版本、MoE路由。保RNG不等于保任意外部随机源或任意设备；不建议为省少许时间默认关 preserve_rng_state。
8. **selective 与 compile 怎样改变保存决策**：MUST_SAVE/PREFER_RECOMPUTE、输出缓存不是只记算子名；SAC依赖具体aten op/backend/fusion。memory-budget仅介绍思想，标明实验性，不让读者照抄未核验私有接口。
9. **并行训练里的省显存可能变成额外通信**：TP/CP/FSDP被重算region包含的collective、参数AllGather时机、PP在飞micro-batch数都会变化；重算与offload不同；内链已有TP/PP/FSDP/SP/CP文章。
10. **如何证明真的省了、没有改坏训练**：同初始参数/输入/RNG比loss、输入与参数grad、一步更新及短程轨迹；record peak allocated/reserved分别解释；warmup并包含backward与optimizer初始化；固定batch与最大可行batch两组实验；记录版本、shape、dtype、attention backend、policy、并行组。未实测只提供测量方法。

## API 与正确性审核要点

- v2.8.0 non-reentrant 默认 early-stop；reentrant 整个区域重算、forward no_grad，且输入/输出 requires_grad、嵌套输入/grad API 有额外限制。不要“checkpoint就是no_grad”概括两种。
- `determinism_check='default'` 在此源码仅比较 shape/dtype/device，不比较数值或梯度，不构成语义等价证明。
- PyTorch RNG保存覆盖Torch CPU及所选设备类型；不是 Python random、NumPy RNG、自定义Generator和所有新设备的自动快照。若要写这些源，需显式状态管理。
- 函数副作用可能被再次执行或因early-stop只执行一部分；统计计数/日志/BatchNorm running stats等不宜藏在可重算region里。不能假设函数严格调用两次。
- 参数在原前向和重算之间须保持正确版本；optimizer.step 不能提前改变尚待重算的参数。
- `checkpoint_sequential` 在此固定版本的最后一段不checkpoint；写该API示例须说明，否则本文“所有段重算”的理想算法与真实API不一致。更建议直接checkpoint包装region。
- 原文Memory Budget图的“saved activations”不等于整个进程峰值显存；reserved和allocated亦不同。
- 不将理论降低activation数量写成减少参数、optimizer或KV serving cache。
- 最终文应保持逐步解释，不出现“新手/入门/小白”等读者人群标签，不把这份交接大纲机械改成通用模板。
