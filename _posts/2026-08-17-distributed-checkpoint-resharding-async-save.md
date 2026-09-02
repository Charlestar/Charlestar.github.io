---
layout: post
title: "Distributed Checkpoint：怎样保存并重分片多维训练状态"
subtitle: "从 Sharded State Dict、两阶段提交到异步 I/O，构建真正可恢复的训练快照"
date: 2026-08-17 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: distributed-training
series_order: 100
technology_year: 2023
mathjax: true
tags: [分布式训练, 工程实践]
---

当模型只在一张 GPU 上训练时，Checkpoint 看起来像一次 `torch.save(model.state_dict())`。进入 FSDP/ZeRO、Tensor Parallel、Pipeline Parallel 和 Context Parallel 后，任何 rank 都不再拥有完整训练状态：某个 weight 沿 output features 分片，另一个沿 input features 分片；optimizer moments 还会沿 Data Parallel group 再切一次；Pipeline ranks 只保存自己的 layers；data loader、RNG 与 LR scheduler 又属于不同范围的状态。

这时 Checkpoint 不再是“把当前进程的字典写进文件”，而是一次分布式一致性协议：

```text
distributed in-memory state
  → describe logical global tensors
  → plan which rank writes which shards
  → persist data and common state
  → validate all writers
  → publish completion metadata last
  → later map global state into a possibly different device mesh
```

一个目录里有很多 `.distcp` 或 rank files，只能证明发生过写入，不能证明它能恢复。真正的验收标准是：在故障后重新启动，加载完整训练语义，并让下一步 loss、gradient 与 optimizer update 符合预期。

## 先区分三种容易混淆的 Checkpoint

### Activation Checkpointing

也称 gradient checkpointing 或 recomputation。它在一次 Forward/Backward 内少保存 activation，Backward 时重算，用计算换 GPU 显存。进程退出后这些状态没有恢复价值。

### Training Checkpoint

用于从某个 committed training step 继续训练。除了 weights，还必须包含 optimizer、scheduler、RNG、数据进度和并行布局等恢复态。

### Export Checkpoint

用于推理、评测或发布，通常只保存模型 weights、必要配置和 tokenizer assets。它可以转换 dtype、合并 shards、去掉 optimizer，但通常不能无损恢复训练。

三者目标不同。把 Hugging Face weights 当作 training checkpoint，会丢掉 optimizer moments、GradScaler、consumed tokens 和随机数状态；把完整训练快照直接发给推理系统，又会携带大量无用状态和内部格式依赖。

## 一个可恢复 Step 到底包含什么

在 optimizer step 已提交、pipeline 已 flush 的边界，典型恢复状态包括：

| 状态 | 为什么需要 | 常见分布方式 |
| --- | --- | --- |
| Model parameters/buffers | 恢复模型数值 | TP/PP/FSDP shards 或 replicas |
| Optimizer moments | 保持 Adam 等更新轨迹 | 通常按 parameter shard，再叠加 DP sharding |
| FP32 master weights | Mixed precision optimizer 可能使用 | 与参数 logical mapping 对齐 |
| LR scheduler/global step | 保持学习率和 step 语义 | common 或 rank-0 state |
| GradScaler | 保持动态 loss scale | replicated，但必须一致 |
| RNG states | 复现 dropout、采样和初始化 | global + per-parallel-domain streams |
| Data loader/sampler cursor | 不重读或跳过训练数据 | 每个 DP worker 或全局 consumed tokens |
| Parallel metadata | 解释每个 tensor shard | global shape、axis、offset、replica id |
| Training configuration | 检查兼容性 | common metadata |
| Commit/integrity metadata | 判断快照是否完整 | 最后原子发布 |

在标准 step boundary，parameter gradients 通常不必保存，因为它们已被 optimizer 消费并清零；若在 gradient accumulation 中途保存，就还需要累积梯度、当前 micro-batch index 和 pipeline in-flight state，恢复协议会复杂很多。

因此生产系统通常选择明确的 committed boundary，而不是试图在任意 kernel 之间冻结整个训练状态机。

## 为什么不能先 Gather 到 Rank 0

最朴素的分布式保存方法是让所有 ranks 把 shards 发给 rank 0，拼成完整模型，再由 rank 0 写文件。这会同时制造三个瓶颈：

1. **显存/内存瓶颈**：rank 0 必须 materialize 完整 weights 和 optimizer state；
2. **网络瓶颈**：所有数据集中流向一个进程；
3. **I/O 瓶颈**：只有一个 writer，无法利用并行文件系统或对象存储吞吐。

若 checkpoint 总大小为 $C$，有效聚合写带宽为 $BW$，理想下界是：

$$
T_{save}\ge\frac{C}{BW}
$$

Gather-to-rank-0 的 $BW$ 会受单节点 NIC、host memory copy 和单 writer 限制；rank 0 还可能因完整 optimizer state OOM。Distributed Checkpoint 的基本方向是让每个 rank 直接保存自己负责的 shards，并用 metadata 把它们描述成一组 logical global tensors。

## Physical Shard 不等于 Logical Tensor

假设 MLP 第一层 weight 的全局 shape 是：

$$
W\in\mathbb{R}^{F\times H}
$$

在 TP=4 时，它可能沿 output dimension 切分：

$$
W_r=W[rF/4:(r+1)F/4,:]
$$

本地文件只知道一块二维数组还不够。Checkpoint metadata 至少要表达：

```yaml
key: decoder.layers.0.mlp.fc1.weight
global_shape: [F, H]
dtype: bf16
shard:
  axis: 0
  offset: [r*F/4, 0]
  local_shape: [F/4, H]
replication:
  data_parallel: replicated-or-main-replica
layout:
  qkv_or_gate_order: <if applicable>
padding:
  logical_shape: [F, H]
  physical_shape: <possibly aligned>
```

不同 tensors 的切分轴可能不同：

- Column Parallel weight 沿 output features；
- Row Parallel weight 沿 input features；
- vocab embedding 沿 vocabulary rows；
- LayerNorm weight 可能在 TP group 内复制；
- PP rank 只拥有部分 layers；
- ZeRO/FSDP 又沿 DP ranks 切 parameter/optimizer storage；
- QKV、gated MLP 与 MLA 可能带 interleaved logical layout。

“每个 rank 写一个文件，加载时按 rank 编号拼接”会把这些差异全部抹平。可重分片依赖 logical metadata，而不是文件命名惯例。

## Sharded State Dict 是一份加载意图

普通 `state_dict` 常把 key 映射到当前进程的完整 tensor。Sharded state dict 除了本地 tensor，还携带 global shape、shard offsets、replica 信息和变换规则。

保存时，它回答：

```text
这个 rank 当前拥有 global tensor 的哪一段？
这一段是唯一主副本，还是多个 replicas 中的一份？
应该由哪个 writer 持久化？
```

加载时，目标模型先根据新的 parallel configuration 生成“我现在需要哪些 shards”的 state dict。Loader 再从 checkpoint metadata 中计算 source shards 与 target shards 的交集，只读取并组合目标 rank 需要的区间。

这也是 PyTorch Distributed Checkpoint（DCP）和 Megatron Core distributed checkpoint 的关键使用模式：不是先恢复成单机完整模型再切，而是由目标 sharded state dict 指导 load/reshard。

## 改变 Parallel Degree 时怎样 Reshard

假设 checkpoint 按 TP=4 保存，恢复时改为 TP=2。对沿 axis 0 均匀切分的 tensor，每个新 rank 需要两个旧 shards 的组合：

```text
old TP=4:
  shard 0: [0, F/4)
  shard 1: [F/4, F/2)
  shard 2: [F/2, 3F/4)
  shard 3: [3F/4, F)

new TP=2:
  target 0: [0, F/2)     ← old shards 0 + 1
  target 1: [F/2, F)     ← old shards 2 + 3
```

若从 TP=2 改为 TP=4，则每个旧 shard 被切给两个 target ranks。真正系统还要处理：

- PP degree 改变导致 layers 重新归属；
- DP/FSDP world size 改变导致 optimizer shards 重分配；
- padding/alignment 与 logical shape 不同；
- tied weights 在多个 module keys 之间共享；
- QKV packing、gated MLP ordering 或 framework naming 改变；
- MoE expert count/EP layout 是否允许转换。

所以“支持 model weight reshard”不自动等于“支持完整 training state reshard”。Optimizer moments 必须跟随参数的 logical elements；若 optimizer 使用 flattening、parameter ids 或 framework-specific buckets，还要保存稳定的参数身份映射。

## Optimizer State 为什么最难迁移

以 Adam 为例，每个 parameter element 通常对应 first moment $m$ 与 second moment $v$。若 weight 被重分片，$m$、$v$ 和可能存在的 FP32 master weight 必须按完全相同的 logical transformation 重分片。

危险情况包括：

- optimizer state 用运行时生成的 integer parameter id，而新进程 id 顺序变化；
- FSDP flattening 把多个 parameters 合进一个 buffer，恢复时 flatten order 改变；
- tied parameters 在保存时去重、加载时却被当成两份；
- frozen/unfrozen parameter set 改变，optimizer param groups 不一致；
- TP layout 从 grouped QKV 改为 interleaved QKV；
- BF16 optimizer 与 FP32 master-weight optimizer 格式不同。

Checkpoint schema 应以稳定 parameter name 和 logical layout 为主键，再明确 param-group hyperparameters。加载后不能只检查 key 数量；应验证每个 parameter 的 optimizer state shape、dtype、step counter 与 global slice。

Megatron Core 当前也区分更快但受限的 optimizer checkpoint 格式与 fully reshardable 格式，这反映了一个现实取舍：越允许任意改变 model parallelism，保存 metadata、转换和加载规划通常越复杂。

## 一次可靠 Save 应该怎样提交

可以把保存分成四个阶段。

### 1. Snapshot

在一致的 step boundary 固定要保存的 state。若后续训练会原地更新 tensor，就不能让后台 writer 继续读取同一块可变 GPU memory；需要同步、copy-on-write 或复制到独立 staging buffer。

### 2. Plan

各 ranks 提交本地 shard metadata，协调出 global save plan：哪些数据是唯一 shards、哪些是 replicas、每个 writer 写哪些 chunks、目标文件和 offsets 是什么。

### 3. Persist

并行写入 tensor shards、common state 与可选 checksums。此时目标目录仍应被视为 incomplete。

### 4. Finalize

所有 writers 成功后，最后写入 manifest/metadata/commit marker。只有这个最终标记存在且能通过完整性检查，checkpoint 才进入可恢复集合。

```text
step-1000.incomplete/
  shard files
  common state
  local metadata

all ranks succeed
  → publish global metadata / manifest last
  → rename or mark as step-1000 committed
```

Megatron Core 的异步保存实现也把 planning、actual saving 和 finalization 分开，并在异步数据写完后才生成完成 metadata。核心原则与具体文件名无关：不能在数据落盘前先发布“完成”。

## 为什么只用 Barrier 还不够

Barrier 能证明所有仍存活的 ranks 到达某个同步点，但不能证明：

- OS page cache 已经 durable flush；
- 对象存储中的所有 parts 都可见；
- 文件没有截断或 bit corruption；
- 某个 writer 写到了错误 path/offset；
- metadata 与 data files 属于同一次 attempt；
- coordinator 在发布 marker 后没有遗漏失败状态。

可靠协议需要 storage-level completion、全局错误聚合和 manifest。对于重要快照，可以记录每个对象的 size 与 checksum，并在恢复前验证。Checksumming 会增加额外读写成本，应按故障模型选择全量、分块或抽样策略。

## 异步保存怎样缩短 Training Stall

同步保存的 step 时间可近似写为：

$$
T_{step,ckpt}
=
T_{train}
+T_{snapshot}
+T_{storage}
+T_{finalize}
$$

异步保存希望把较慢的 $T_{storage}$ 与后续训练重叠：

```text
training rank:
  freeze state → GPU-to-host staging → continue next step

background worker:
  host staging → storage write → checksum → finalize
```

前台 stall 至少仍包含形成一致 snapshot 和把可变 GPU tensors 转移到安全 buffer 的时间。异步不等于零开销，还会消耗：

- pinned host memory；
- PCIe/NVLink copy bandwidth；
- CPU cycles 与内存带宽；
- NIC/parallel filesystem bandwidth；
- background process/thread 与文件 descriptors。

如果后台写入还没完成就触发下一个 checkpoint，staging buffers 会堆积，最终把训练拖垮或耗尽 host memory。系统需要限制 outstanding saves、合并或跳过过期请求，并在退出前等待最终提交。

## Snapshot Immutability 是异步正确性的核心

假设 training thread 在 step 1000 发起 async save 后立刻执行 step 1001，optimizer 原地更新 parameter $W$。如果 writer 稍后才从同一内存读取，一部分 bytes 可能属于 step 1000，另一部分已属于 step 1001，形成从未真实存在过的混合状态。

避免方法包括：

- 将 tensor copy 到独立 CPU staging buffer 后再继续训练；
- 使用稳定的 GPU snapshot buffer；
- 采用 copy-on-write 或版本化 storage；
- 明确等待所有读取依赖完成后才允许 optimizer 覆写。

“Python 字典已经创建”不代表底层 tensor storage 已冻结。异步 API 的正确性 contract 必须落实到 storage ownership 和 stream synchronization。

## Checkpoint 频率怎样选择

保存太频繁会浪费训练时间和存储带宽；太稀疏则在故障时重算大量 steps。若一次 checkpoint 的不可隐藏成本为 $C$，作业平均故障间隔为 $M$，经典近似会让合理周期落在：

$$
T_{interval}\approx\sqrt{2CM}
$$

这只是起点。大规模 GPU 作业还要考虑：

- 作业级 MTTF 会随节点数和依赖组件增加而下降；
- scheduler preemption 是否可提前通知；
- 数据/模型规模让 checkpoint bytes 随时间是否变化；
- 异步 I/O 有多少真正被隐藏；
- 恢复、重新排队和数据校验的固定成本；
- 是否保留本地高频快照与远端低频 durable 快照两层策略。

实践中应从故障日志计算“每次故障丢失的 committed tokens”，再调整 cadence，而不是固定每 N 小时保存一次。

## Data Loader State 决定是否真正续训

只恢复模型与 optimizer，却从数据集开头重新读取，训练轨迹已经改变。对于 sharded/streaming/packed dataset，需要保存或可推导：

- global consumed samples/tokens；
- epoch、shard id、sample offset；
- shuffle seed 与 sampler generator state；
- 每个 DP worker 的 cursor；
- sequence packing buffer 中尚未消费的 samples；
- dynamic batch/token-budget scheduler 状态。

改变 DP world size 后，不能简单让每个新 rank 读取旧 rank 的 cursor。更稳健的设计用 global data position 或确定性 sample assignment，在新 DP mesh 上重新计算每个 worker 下一段数据，同时保证没有意外重复或跳过。

## RNG State 不是一个 Seed

训练中的随机性可能来自：

- CPU/Python/NumPy RNG；
- 每个 CUDA device 的 RNG；
- dropout、attention dropout 和 stochastic layers；
- TP/SP/CP 下按 shard 管理的 RNG tracker；
- data augmentation、sampling 与 packing；
- optimizer 内部的随机算法。

只保存初始 seed，无法恢复“已经消费了多少随机数”。Checkpoint 要保存各 RNG streams 的当前 state，或使用可由 `(global step, layer, micro-batch, global token)` 确定性映射的 counter-based RNG。

若恢复时改变 TP/CP degree，local tensor shapes 和随机数消费顺序也会改变。要追求不同 parallel layout 下的数值可复现，RNG mapping 必须绑定 logical elements，而不是 local rank 的调用顺序。

## Pipeline Parallel 应在什么时刻保存

最简单可靠的时刻是 1F1B pipeline 已 flush：

- 所有 micro-batches 完成 backward；
- gradients 已聚合并由 optimizer 消费；
- 所有 stages 使用同一个 committed global step；
- 没有 in-flight activation/P2P request；
- data cursor 已推进到对应 step 边界。

若在流水中途保存，就要序列化每个 stage 的 outstanding micro-batches、activation、activation gradient、Forward 参数版本、通信 request 和 schedule position。恢复这些瞬时状态通常比重做当前 step 更贵也更容易出错。

因此可以允许故障时丢掉当前未提交 step，但不能把“目录已经写了一半”误认为新 checkpoint。

## Load 不是 Save 的反向 `memcpy`

可靠恢复至少包含以下阶段：

1. 发现最新 **committed** checkpoint，而不是按目录名取最大 step；
2. 验证 manifest、format version、必要 files 和 checksums；
3. 构建目标 model/optimizer 和新的 parallel mesh；
4. 由目标 sharded state dict 生成 load plan；
5. 并行读取目标 ranks 需要的 byte ranges；
6. 应用 rename、reshape、reshard 或 format transform；
7. 恢复 common state、RNG 与 data cursor；
8. 全局确认所有 ranks 成功，再进入下一 training step。

加载失败不能让一部分 ranks 带着新状态继续、一部分保留初始化状态。错误必须全局传播并让本次恢复 attempt 失败。

## 格式版本与模型版本要分开

Checkpoint 至少存在两类版本：

- **Storage format version**：文件、metadata、backend 与 checksum schema；
- **Model/training schema version**：parameter names、QKV layout、optimizer 类型、tokenizer 和配置语义。

Storage reader 能打开文件，不代表当前代码能正确解释模型。模型升级需要显式 migration：key rename、tensor split/merge、transpose、dtype conversion、vocab resize 或 optimizer reset，并记录转换后的新版本。

不要在 loader 里默默 `strict=False` 后忽略 missing/unexpected keys。允许缺失的 key 应有白名单和初始化规则，转换后还要做数值或结构验证。

## 安全加载也属于 Checkpoint Contract

训练快照可能来自共享存储、外部模型仓库或旧任务。能够反序列化任意 Python object 的格式也可能执行非预期代码。PyTorch 新版本将 `torch.load` 的 `weights_only` 安全路径放到更重要的位置；分布式格式也应尽量使用 tensors、受控 metadata 和 allow-listed types。

同时要验证：

- checkpoint 来源与访问控制；
- manifest/checksum 或签名；
- path traversal 与符号链接；
- metadata 中的巨大 shape 是否造成内存耗尽；
- dtype/shape 与目标参数 contract 是否匹配。

“内部存储”不代表永远可信，尤其当多个训练任务可以写同一个 bucket/prefix 时。

## Retention 不能只保留最新一个目录

最新 checkpoint 可能在训练数小时后才暴露数值异常，也可能包含尚未发现的格式问题。常见保留策略包括：

- 最近若干个高频 checkpoints；
- 每隔更大时间窗口保留一个长期 checkpoints；
- milestone/evaluation-best checkpoints；
- 最后一个已验证可恢复的 checkpoint；
- 独立的最终 export weights。

删除旧快照前，应确认新快照已完成 finalize、通过 restore smoke test，并且没有 reader/转换任务仍在使用。对象存储的生命周期规则也不能早于 manifest 与 data 的引用周期。

## Restore Test 应怎样设计

Checkpoint 测试不能停在“`load()` 没报错”。可以分四层。

### 结构完整性

- 所有 required keys 存在；
- global shape、dtype、shard coverage 正确；
- 每个 logical element 恰好有一个主副本；
- manifest、size、checksum 与 format version 合法。

### 数值恢复

- 将 shards 聚合成小模型 reference，比对 parameters/buffers；
- 比对 optimizer moments、step counters 与 scaler；
- 检查 tied weights 和 padding 区域。

### 续训一致性

在固定 next batch 与 RNG 下比较：

```text
continuous run: step N → step N+1
restart run:    save at N → reload → step N+1
```

对比 loss、selected activations、gradients 和更新后的 weights。允许的浮点误差要按 dtype 与 collective order 明确设定。

### Reshard Matrix

覆盖目标支持的组合，例如：

```text
save TP2 PP2 DP4 → load TP4 PP1 DP4
save TP4 PP2 DP2 → load TP2 PP4 DP2
save FSDP world 8 → load FSDP world 4
```

若只测试同 world size、同 rank mapping 的恢复，就没有验证“分布式 checkpoint 可重分片”这一核心能力。

## 性能 Benchmark 应记录什么

同步与异步 checkpoint 都要报告：

- 总 logical state bytes、实际 storage bytes 与文件数量；
- snapshot/staging 前台停顿；
- background write 和 finalize 完成时间；
- GPU→CPU、host→storage 的有效带宽；
- 各 rank bytes/latency 与最慢 writer；
- 训练 step slowdown 和通信争用；
- peak host/pinned memory；
- load plan、read、reshard 和首个 step 时间；
- checksum/integrity 的额外成本；
- 不同 world size 与 storage backend 的扩展性。

只报告“异步写入耗时 30 秒”没有意义；如果前台 staging 仍停顿 8 秒，或者后台 I/O 让后续每个训练 step 慢 10%，这些都属于 checkpoint 成本。

## 一条可执行的落地路径

1. **定义 committed boundary**：只在 optimizer step 完成、pipeline flush 后保存；
2. **列出完整恢复态**：weights、optimizer、scheduler、RNG、data cursor 与配置；
3. **给每个 tensor 定义 logical placement**：global shape、axis、offset、replica；
4. **先做同步 sharded save/load**：验证同 parallel layout 下 next-step parity；
5. **加入 manifest-last 提交**：失败目录永不进入可恢复集合；
6. **实现 optimizer 与 data state**：不能只验证 model weights；
7. **覆盖 TP/PP/DP resize**：让目标 state dict 指导 reshard；
8. **再加入 async staging/I/O**：保证 snapshot storage 不会被训练覆写；
9. **限制 outstanding saves**：定义排队、跳过和退出等待策略；
10. **做故障注入**：writer 崩溃、文件截断、rank 丢失、metadata 缺失；
11. **自动执行 restore smoke test**：新 checkpoint 未验证前不删除上一个；
12. **分离 training 与 export 格式**：各自保持清晰 contract。

## 常见误区

### “每个 Rank 都写成功，就得到完整 Checkpoint”

还需要全局 shard coverage、replica 去重、common state 和最终 commit metadata。任何 writer 失败都不能发布新快照。

### “模型能 Load，训练就能续上”

没有 optimizer、scheduler、GradScaler、RNG 和 data cursor，只是从旧 weights 启动了一次新训练阶段。

### “异步 Save 不会影响训练”

Snapshot staging、host memory、PCIe、CPU、NIC 和 storage 都会与训练争用。异步只是提供重叠机会。

### “支持 FSDP Reshard 就能任意改变 TP/PP”

不同 parallel dimensions 对应不同 tensor axes、layer ownership 与 optimizer mapping，需要格式明确支持。

### “目录名最大的 Step 就是最新 Checkpoint”

最大 step 目录可能是失败 attempt。只能选择已 finalize 且通过完整性检查的快照。

### “恢复后 Loss 接近就足够”

错误 optimizer moment、data cursor 或 RNG 可能暂时不让 loss 明显跳变，却会改变后续轨迹。应做固定 next-step 的分层 parity。

## 小结

Distributed Checkpoint 的核心不是并行写很多文件，而是把分散的 physical shards 还原为一组有稳定身份的 logical tensors，并以一致、可验证的方式提交：

1. Training checkpoint 要覆盖模型、optimizer、scheduler、RNG、data cursor 和并行 metadata；
2. Sharded state dict 描述每个本地 tensor 对应的 global shape、offset、axis 与 replica；
3. 目标模型先声明需要的 shards，loader 才能在 TP/PP/DP 改变后直接 reshard；
4. Optimizer state 必须跟随 parameter logical elements，不能依赖易变的本地 id 或文件编号；
5. Save 应经过 snapshot、plan、persist、finalize，完成 metadata 最后发布；
6. Async save 必须冻结底层 storage，并控制 staging memory 与 outstanding requests；
7. Pipeline 最好在 committed flush boundary 保存，避免序列化 in-flight 状态；
8. 最新快照只有在 restore test 通过后，才有资格替代上一个恢复点；
9. Training 与 export checkpoints 应分离，分别优化可恢复性与可消费性。

至此，分布式训练专题已经从 Data Parallel/FSDP、Tensor Parallel、Pipeline Parallel，延伸到 Sequence/Context Parallel 与 Checkpoint。下一阶段可以继续深入多维并行规划：如何根据模型 shape、网络拓扑、显存和 global batch，在 TP×PP×CP×DP×EP 的候选空间里选择一组可执行配置。

## 参考资料

- [PyTorch Tutorial: Getting Started with Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
- [PyTorch Distributed Checkpoint API](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [NVIDIA Megatron Core: Distributed Checkpointing](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/dist_checkpointing.html)
- [NVIDIA Megatron-LM: Megatron Core Quick Start](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/QuickStart.md)
- [TorchTitan Checkpointing Guide](https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md)
- [Universal Checkpointing: A Flexible and Efficient Distributed Checkpointing System for Large-Scale DNN Training with Reconfigurable Parallelism](https://www.usenix.org/conference/atc25/presentation/lian)
