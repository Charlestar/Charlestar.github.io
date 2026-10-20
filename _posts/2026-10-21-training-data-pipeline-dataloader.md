---
layout: post
title: "训练数据管线：一个 Batch 怎样在 CPU 与 GPU 之间接力"
subtitle: "沿 DataLoader 的取样、分片、预取和传输，理解吞吐瓶颈与训练语义为何需要一起检查"
date: 2026-10-21
last_modified_at: 2026-10-21
author: iStar
catalog: true
series: distributed-training
series_order: 48
technology_year: 2017
mathjax: true
tags: [分布式训练, GPU优化]
---

假设 GPU 计算一个 batch 只需要 6 ms，但训练循环隔 12 ms 才拿到下一份数据。此时再优化矩阵乘，可能只是让 GPU 更早回到等待状态。另一种更隐蔽的问题是：吞吐提高了，实际送入模型的样本却重复、遗漏，或者在恢复训练后跳过了一段尚未消费的数据。

数据管线同时承担两项工作：及时准备输入，以及保持输入所代表的训练过程。理解它，需要把一个 batch 从数据源一直跟到模型，并分清“已经派发”“已经准备好”“已经传输”和“已经训练过”这几个位置。

下面以 PyTorch 2.8 的 DataLoader 接口和 v2.8.0 实现为范围。系列中的 2017 年锚定 [PyTorch 公开发布的年份](https://pytorch.org/blog/a-year-in/)，不是 DataLoader 或异步数据加载的起源年份。文中的 Python 算例只用标准库验证调度与分片规则；实际吞吐、H2D 重叠和 GPU 空闲时间，需要在目标硬件上测量。

## 一个 Batch 经过哪些环节

先考虑常见配置：map-style Dataset、自动组 batch、多个 worker，并启用 pinned memory。主要路径可以写成：

```text
训练主进程：Sampler → BatchSampler → 派发一组索引
                                      │
worker：              读数据 → 解码/变换 → collate
                                      │
主进程后台线程：             接收结果 → pin memory
                                      │
训练主线程：                    取得 CPU batch
                                      │
设备执行：                        H2D → 模型计算
```

Sampler 决定取哪些样本及其顺序；BatchSampler 把索引组合成一批。Worker 根据索引读取、变换数据，再由 `collate_fn` 整理成训练循环需要的结构。例如，图像可能被堆叠成一个四维 tensor；变长文本可能在这里 padding，并生成 attention mask。

一个容易影响瓶颈判断的细节是：**多进程加载时，常规 `collate_fn` 在 worker 的 fetch 路径中执行，不是所有 worker 完成后再由主线程统一拼 batch。** [v2.8.0 fetch 实现](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/data/_utils/fetch.py)还会优先使用 Dataset 可选的 `__getitems__` 批量取样接口；否则逐个调用 `__getitem__`。批量读取因此可以减少逐样本调用和存储访问的开销，但仍须返回与这组索引对应的样本列表，供后续 collation 使用。

当 `num_workers=0` 时，取样和 collation 都发生在主进程，没有后台数据 worker；若启用 pinning，也是在这条单进程取数路径中处理。上图独立的 pin-memory 线程属于多进程路径。把参数从 0 改为正整数，不只是增加了几条读文件线程，而是引入进程间传递、队列、worker 生命周期以及额外的内存开销。

Worker 通常返回 CPU tensor，再由训练进程负责 H2D。不要为了让示意图“更并行”就直接在每个数据 worker 中初始化 CUDA；这会把设备上下文和多进程 CUDA 生命周期问题引入数据层。[DataLoader 文档](https://docs.pytorch.org/docs/2.8/data.html#multi-process-data-loading)也建议多进程加载配合自动 pinning，而不是让 worker 返回 CUDA tensor。

## 先看生产速率，再讨论预取深度

设一条简化管线只有两个阶段：生产者准备一批数据耗时 $$t_p>0$$，GPU 消费一批耗时 $$t_g>0$$。如果完全串行，处理 $$K\ge1$$ 批需要：

$$
T_{\mathrm{serial}}=K(t_p+t_g).
$$

若生产和计算使用独立资源，可以理想重叠，第一批的准备仍然不能省略；之后的稳态节奏由较慢阶段决定：

$$
T_{\mathrm{pipeline}}=t_p+t_g+(K-1)\max(t_p,t_g).
$$

例如 $$t_p=12\,\mathrm{ms}$$、$$t_g=6\,\mathrm{ms}$$、$$K=6$$，串行为 108 ms，理想两阶段流水为 78 ms。GPU 依次在 12、24、36、48、60、72 ms 开始处理，每次用 6 ms；两批之间仍有 6 ms 等待。

下面直接计算准备完成和消费完成的时间。这里没有操作系统、文件系统、CUDA 或带容量限制的队列，是一个确定性的两阶段模型。

<!-- example: dataloader-two-stage -->
```python
def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def two_stage(batches, prepare_ms, compute_ms):
    if type(batches) is not int or batches < 0:
        raise ValueError("batches must be a nonnegative integer")
    positive_int(prepare_ms, "prepare_ms")
    positive_int(compute_ms, "compute_ms")
    prepared = computed = 0
    timeline = []
    for batch in range(batches):
        prepared += prepare_ms
        start = max(prepared, computed)
        computed = start + compute_ms
        timeline.append((batch, prepared, start, computed))
    return timeline


timeline = two_stage(6, 12, 6)
assert [row[2] for row in timeline] == [12, 24, 36, 48, 60, 72]
assert timeline[-1][3] == 78
assert 6 * (12 + 6) == 108
```

这时把预取队列调得再深，也不能让生产者长期超过每秒约 83.3 批。队列可以提前存好一些数据，暂时隐藏存储或解码抖动，但不创造生产能力。若先积累了 4 个 ready batch，消费速度是每 6 ms 一批，生产速度是每 12 ms 一批，按连续流量近似，净消耗为每 12 ms 一批，约 48 ms 会耗尽这份余量。具体离散时刻取决于生产和消费的相位，不能把这个近似当作真实 trace。

真实管线还有存储、解码、pinning 和传输。理想情况下，每个独立阶段的能力给出一个吞吐上界；但如果 worker 共享存储带宽、CPU 核心和内存带宽，能力不能简单相加。增加 worker 后吞吐下降，可能正是竞争和上下文切换已经超过并行收益，而不是“预取还不够”。

## prefetch_factor 限制的是提前派发的任务

在自动 batching 的常见路径中，一组 batch 索引就是一个任务。若有 $$W$$ 个 worker，`prefetch_factor=F`，DataLoader 会以约 $$WF$$ 个提前派发任务形成窗口。尚在读文件、正在解码、已经进入结果队列以及等待按序返回的任务，可能同时存在。

所以 `num_workers=8, prefetch_factor=2` 不意味着“总有 16 个准备好的 batch”，也不意味着“16 个 batch 已经在 GPU 上”。如果所有 worker 都被慢存储卡住，窗口可以很满，ready queue 却为空。[v2.8.0 DataLoader 实现](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/data/dataloader.py)在初始化/重置时预先派发任务，并在向调用者处理返回结果时补充新任务；GPU 传输不属于这个预取窗口。

内存也不能只按 `batch_bytes × W × F` 精确计算。这个乘积可以作为对张量数据量的粗略预算，但实际还包含 worker 内的临时解码结果、Python 对象、进程间共享存储、pinning 前后的副本以及训练主线程手中的 batch。变长输入还会让 batch 大小波动。应同时记录主进程与 worker 的内存，以及 pinned host memory，而不是只看 CUDA 显存。

默认 `in_order=True` 时，已经完成的后续任务可能要等待较慢的前序任务。这是有序返回带来的队头阻塞。PyTorch 2.8 的 `in_order=False` 可以不强制这种返回次序，但会改变训练循环看到的样本顺序。如果样本读取速度与类别、长度或来源相关，有限训练窗口内的分布也可能随之改变。它是一个有语义代价的调度选项，不宜作为通用加速开关。

## 两个 drop_last 操作的是不同集合

分布式训练中，`DistributedSampler(drop_last=...)` 和 `DataLoader(drop_last=...)` 虽然同名，却处在不同层。

Sampler 先解决“各个 rank 应获得多少个索引”。若数据量不能被 rank 数整除，`drop_last=False` 会补索引，使每个 rank 长度相同；`True` 则截去全局索引序列尾部。之后，每个 rank 的 DataLoader 再按本地 batch size 组批，其 `drop_last=True` 丢弃的是这个本地序列剩余的不足一批的索引。[DistributedSampler v2.8.0](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/data/distributed.py)的顺序是先构造全局序列、补齐或截断，再按 rank 步进抽取。

用 10 个样本、3 个 rank、本地 batch size 为 2 举例，不打乱时：

| 配置 | Rank 0 | Rank 1 | Rank 2 | 实际消费 |
| --- | --- | --- | --- | --- |
| Sampler 补齐，Loader 保留尾批 | 0, 3, 6, 9 | 1, 4, 7, 0 | 2, 5, 8, 1 | 12 次，10 个不同样本 |
| Sampler 补齐，Loader 丢尾批 | 0, 3, 6, 9 | 1, 4, 7, 0 | 2, 5, 8, 1 | 仍是 12 次，重复没有消失 |
| Sampler 截断，Loader 保留尾批 | 0, 3, 6 | 1, 4, 7 | 2, 5, 8 | 9 次，存在本地小 batch |
| Sampler 截断，Loader 丢尾批 | 0, 3 | 1, 4 | 2, 5 | 6 次，6 个不同样本 |

第三行的每个 rank 都有两个 batch，但最后一批只有一个样本；第四行则每个 rank 只有一批。因而“没有小 batch”“各 rank 步数一致”“每个样本恰好出现一次”不是同一个目标。验证集尤其要避免把补齐的重复样本直接计入指标后，仍宣称做了精确的一次遍历。

以下程序重现**不 shuffle 时**的索引规则，并把两个阶段分开。它不是 PyTorch 的替代实现，也不使用 Python 随机排列冒充 `torch.randperm`。

<!-- example: dataloader-two-drop-last -->
```python
def rank_indices(size, replicas, rank, sampler_drop=False):
    if type(size) is not int or size < 0:
        raise ValueError("size must be a nonnegative integer")
    positive_int(replicas, "replicas")
    if type(rank) is not int or not 0 <= rank < replicas:
        raise ValueError("rank is out of range")
    if type(sampler_drop) is not bool:
        raise ValueError("sampler_drop must be bool")
    per_rank = size // replicas if sampler_drop else (size + replicas - 1) // replicas
    total = per_rank * replicas
    order = list(range(size))
    if sampler_drop:
        order = order[:total]
    elif total > size:
        order += [order[i % size] for i in range(total - size)]
    return order[rank:total:replicas]


def local_batches(indices, batch_size, loader_drop=False):
    positive_int(batch_size, "batch_size")
    if type(loader_drop) is not bool:
        raise ValueError("loader_drop must be bool")
    groups = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    return [group for group in groups if not loader_drop or len(group) == batch_size]


padded = [rank_indices(10, 3, rank) for rank in range(3)]
assert padded == [[0, 3, 6, 9], [1, 4, 7, 0], [2, 5, 8, 1]]
assert sum(len(batch) for row in padded for batch in local_batches(row, 2, True)) == 12
truncated = [rank_indices(10, 3, rank, True) for rank in range(3)]
assert [local_batches(row, 2, True) for row in truncated] == [
    [[0, 3]], [[1, 4]], [[2, 5]]
]
```

实际 shuffle 时，所有 rank 必须使用相同的 sampler seed 和 epoch，先得到同一份全局排列，再分别取自己的切片。在创建当轮 DataLoader iterator **之前**调用 `sampler.set_epoch(epoch)`，才能让不同 epoch 使用不同排列。每个 rank 各自随意生成一个排列，再各取其中一段，不能保证全局无重复、无遗漏。

## IterableDataset 要同时区分 Rank 和 Worker

Map-style 路径由主进程的 sampler 给 worker 分派索引；普通 Dataset 不需要再依据 worker id 过滤一次，否则会把已经分派好的数据错误丢掉。

IterableDataset 不同。每个 worker 持有一份 Dataset 副本，分别执行自己的 `__iter__`。如果这些副本都从同一条流的开头读取，数据就会重复。跨进程训练时，还要同时区分 rank，不能只对本地 worker 分片。

设每个 rank 使用相同的 $$W$$ 个 worker，共有 $$R$$ 个 rank，可以定义全局消费者编号：

$$
c=rW+w,\qquad C=RW.
$$

若每个消费者能够观察到完全一致、稳定编号的数据流，让编号 $$i$$ 满足 $$i\bmod C=c$$，就能得到一个不相交且覆盖全部样本的划分。这个办法适合解释正确性；实际读取远端对象时，应优先在文件、对象或数据 shard 层划分所有权，避免每个 worker 都下载全量数据，再丢弃绝大部分。

即使分片没有重复，batch 数仍未必相同。23 个样本分给 2 个 rank、每 rank 2 个 worker，四个消费者各有 6、6、6、5 个样本。使用本地 batch size 3 且 `drop_last=True` 时，最后那个 worker 丢掉 2 个样本，总计消费 21 个；rank 0 有 4 批，rank 1 只有 3 批。

<!-- example: dataloader-iterable-shards -->
```python
def iterable_shards(size, replicas, workers):
    if type(size) is not int or size < 0:
        raise ValueError("size must be a nonnegative integer")
    positive_int(replicas, "replicas")
    positive_int(workers, "workers")
    consumers = replicas * workers
    return [[i for i in range(size) if i % consumers == consumer]
            for consumer in range(consumers)]


shards = iterable_shards(23, 2, 2)
assert list(map(len, shards)) == [6, 6, 6, 5]
worker_batches = [local_batches(shard, 3, True) for shard in shards]
rank_steps = [sum(len(worker_batches[rank * 2 + worker]) for worker in range(2))
              for rank in range(2)]
assert rank_steps == [4, 3]
assert sum(len(batch) for groups in worker_batches for batch in groups) == 21
```

原因是 IterableDataset 的不满批丢弃发生在**每个 worker 副本**的 fetch 路径，不是先合并同一 rank 的所有尾部再决定一次。此时 `len(dataloader)` 也只是基于 Dataset 长度和 batch size 的估计，无法替用户的分片代码推导真实批数。

普通同步 DDP 训练若假定各 rank 执行相同数量的 collective，而一部分 rank 提前耗尽数据，就可能出现等待或 collective 不匹配。应在分片/组批时保证一致的训练步边界，或明确采用能够处理不均匀输入的训练协议；不能仅凭“样本已经均匀分片”判断训练循环安全。

## 持久 Worker 保留的也包括数据状态

`persistent_workers=True` 避免每轮重建 worker，适合 worker 启动、导入或数据集初始化开销明显的情形。但 worker 中的 Dataset 实例也继续存活，主进程执行 `dataset.epoch = epoch` 不会自动改写其他进程中的副本。

[v2.8.0 worker 实现](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/data/_utils/worker.py)会在启动时初始化 Python `random`、PyTorch，以及可用时的 NumPy 全局随机状态，再调用 `worker_init_fn`。后续复用 worker、恢复一次迭代时，不会重新调用这个初始化函数。自建的随机生成器、第三方解码/增强库和 Dataset 自己的状态仍须明确管理；也不能期待 `worker_init_fn` 自动承担每轮 epoch 更新。

可复现性至少有三层：sampler 给出的样本顺序、每个样本采用的随机变换，以及多 worker 完成后向训练循环返回的顺序。固定一个 seed 不能替代这些层之间的设计。尤其当 worker 数量变化时，按 worker 连续推进的随机流会把不同随机数分配给样本。若业务要求增强与 worker 调度无关，可以由稳定的样本 ID、epoch 和增强轮次派生随机种子，并使用局部生成器；这样改变的是随机数的归属规则，而不是承诺所有算子跨硬件逐 bit 一致。

在使用 `spawn` 的环境中，Dataset、collate 函数和 worker 初始化函数还需要能够被序列化。把入口放进 `if __name__ == "__main__":`，把需要传给 worker 的函数定义在模块顶层，是避免递归创建进程及局部函数无法序列化问题的基本结构。先用 `num_workers=0` 定位数据正确性和异常，再打开多进程，通常更容易获得直接的错误栈。

## Pinned Memory 不等于已经完成 GPU 预取

Pinned memory 是 page-locked 的 **CPU 内存**。`pin_memory=True` 不把 batch 放入显存；它让支持的 tensor 数据位于适合 H2D 传输的 host buffer 中。自定义 batch 类型还需要提供合适的 `pin_memory()` 方法，否则不能仅根据外层参数判断内部所有 tensor 都已 pin。Pinning 本身需要资源，过量使用会挤压系统可分页内存。

`tensor.to(device, non_blocking=True)` 也不等于“传输与本批计算同时发生”。要把异步分成两个维度：主机调用是否需要等待，以及设备上两项工作能否重叠。若 H2D 和依赖它的计算都提交到同一个 stream，它们仍依次执行；计算不能在自己的输入复制完成前开始。

为了让下一批 H2D 与当前批计算重叠，可以使用独立 copy stream 和双 buffer，但必须同时处理三个不同问题：

| 问题 | 必须保持的关系 | 不能代替它的操作 |
| --- | --- | --- |
| 数据何时就绪 | Compute stream 等待对应 copy event，之后才能读这一批 | 仅设置 `non_blocking=True` |
| CUDA 张量释放后，底层存储何时可被分配器复用 | 张量跨 stream 使用时，通过 `record_stream` 或正确的手动同步告知分配器其使用寿命 | 仅等待最初 H2D 完成 |
| CPU 源数据何时可改写 | H2D 完成前不修改或复用对应 pinned host buffer | 对目标 CUDA tensor 调用 `record_stream` |

这三个关系分别保护“数据依赖”“设备存储寿命”和“源数据内容”。[PyTorch CUDA stream 语义](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-streams)与 [`record_stream` 文档](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.record_stream.html)明确区分前两者。对已发出的异步复制立刻重填 CPU buffer，即使 buffer 对象仍然活着，也可能复制到被修改的内容；对象未释放不等于内容不可变。

固定复用的 CUDA 双 buffer 还有一条显式覆写依赖：copy stream 再次写某个 buffer 前，必须等待此前读取它的计算结束。`record_stream` 管的是释放之后的分配器复用，不会阻止程序自己对仍存活的 buffer 执行 `copy_` 等原地写入；这类复用仍需 compute-complete event 或等价的 stream 同步。

正确的理想时间线应是：计算第 $$k$$ 批时复制第 $$k+1$$ 批；计算第 $$k+1$$ 批前只等待它自己的 copy 完成；只有相应读取结束后才复用各自 buffer。不同 stream 只是提供并行执行的可能，实际重叠还受 copy engine、链路带宽、kernel 占用和调度影响。Pageable memory 还可能涉及内部 staging，不能把它的 `non_blocking` 行为概括成与 pinned buffer 完全相同的后台 DMA 保证。

## 恢复训练要回到已消费位置

预取把数据状态分成了几个游标。假设训练刚完成 batch 100，sampler 已向 worker 派发到 batch 116，队列中还存着 101–108，其余任务仍在处理。保存“sampler 当前已经迭代到 116”，恢复时再从 117 开始，就跳过了 16 批尚未训练的数据。

反过来，若恢复时从 epoch 开头重放数据，却没有相应跳过已完成的消费区间，同一批样本可能在同一模型状态之后再次训练。因此恢复语义应围绕**与模型和优化器快照一致的消费边界**，而不是某条后台线程或远端流当前读到了哪里。

对于固定 map-style 数据、确定的 sampler 和变换规则，可以记录数据版本、epoch、seed、已消费的本地 batch 位置，以及恢复随机变换所需的状态，再确定性重建顺序。梯度累积时还要说明消费边界是在 optimizer step 之后，还是保存了未提交的部分梯度与 micro-batch 状态；不能把二者的计数混用。

随机变换也必须对应这个边界。Worker 可能已经为未消费的 batch 生成了增强，因此直接保存它“现在”的 RNG 状态，再从 batch 101 重读，不一定重现原来的增强。可以采用按样本身份派生的随机数，或从较早的一致数据状态确定性重放；若保存队列与 worker 状态，则还要覆盖尚未消费的结果及其对应进度，不能只保存一个超前游标。

流式数据则可能需要保存 shard manifest、各消费者进度、shuffle buffer 和随机状态。若 world size 或 worker 数发生变化，原先的局部 offset 不一定能直接映射到新的分片。数据读取端能够重放，不等于训练步骤具备端到端 exactly-once；恢复测试应实际比较断点前后的样本 ID 和更新边界。这与 [Distributed Checkpoint 的一致快照]({% post_url 2026-08-17-distributed-checkpoint-resharding-async-save %})是同一个问题的两侧：模型状态和输入进度必须描述同一段训练历史。

## 调优时保留一份可核对的消费记录

开始调性能之前，先用可识别的样本 ID 跑完一个小 epoch，核对各 rank 和 worker 的覆盖、重复、尾部处理、batch 数及顺序。再进行一次中断恢复，检查恢复后的消费序列。否则，一个“更快”的管线可能只是少训练了样本，或者对 padding、重复样本与有效 token 使用了不同分母。

然后分层测量：

1. 用形状和 dtype 相同的 GPU 常驻输入估计计算侧基线，但不要把这个数当作真实训练吞吐。
2. 保持变换和 collation，分别用本地缓存数据与真实数据源运行，观察存储是否主导等待。
3. 逐步改变 worker 数，记录完整 step 吞吐、`next(iterator)` 等待分布、CPU/存储负载与 host memory；不要只记录最快的一次。
4. 在生产能力足够、但存在短时抖动时调整 prefetch，再评估 pinned memory 和跨 stream 传输；每次只改变能够解释的机制。
5. 在真实多 rank 训练中重新检查最慢 rank 的到数与 step 完成时间，避免单卡加载器的改进被共享存储竞争抵消。

CPU 侧计时 `next(iterator)` 可以衡量训练主线程等数据的时间，但它不是 worker 总工作量；`to()` 返回耗时也不是 H2D 的设备持续时间。对长序列训练，还应同时报告样本数和有效 token 数：每秒 batch 数增加，可能只是 batch 变短或 padding 策略发生变化。CUDA events 适合设备区间，完整 step 的主机计时需要正确的完成边界；分析重叠关系时可沿 [GPU 性能剖析的时间线路径]({% post_url 2026-10-20-gpu-profiling-nsight-systems-compute %})定位，最终仍应回到无采集工具介入的正常运行。

数据管线的目标不是让队列永远显得很满，而是以可解释的资源成本，让正确的下一批输入及时就绪。生产速率决定长期能力，预取缓冲短时波动，分片与恢复规则决定实际训练了什么；三者都成立，吞吐改进才值得保留。
