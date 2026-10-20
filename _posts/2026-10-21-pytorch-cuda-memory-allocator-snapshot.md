---
layout: post
title: "PyTorch 显存分配：从 Tensor 生命周期到 OOM 快照"
subtitle: "区分已分配、等待复用和缓存中的显存，再沿分配事件定位真正需要改变的生命周期与布局"
date: 2026-10-21
last_modified_at: 2026-10-21
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 50
technology_year: 2023
mathjax: true
tags: [GPU优化, 工程实践]
---

删除一个 CUDA tensor 后，`nvidia-smi` 中的占用没有下降，是不是显存泄漏？OOM 报错时还有不少 reserved memory，是不是调用 `empty_cache()` 就能解决？这两个问题都无法只用一个“显存占用”数字回答。

一块存储可能仍被某个对象引用，可能已经收到释放请求却还不能跨 stream 复用，也可能已经成为 PyTorch 的空闲缓存，只是还没有还给 CUDA。要定位问题，需要把对象、分配器和设备这三个观察层分开，并追踪分配状态随时间怎样变化。

本文固定讨论 PyTorch 2.8 / v2.8.0 的 **native CUDA caching allocator**，随后单独说明其他 backend 的边界。系列中的 2023 年锚定 [PyTorch 2.1 中实验性 Memory Snapshot 工具的公开介绍](https://pytorch.org/blog/understanding-gpu-memory-1/)，不是缓存分配器的起源年份。下面的 CPU 账本是人为构造的状态模型；采集代码供 CUDA 环境使用，不代表已经执行真实 PyTorch/GPU 分配或复现 OOM。

## 删除对象和归还显存之间隔着什么

Python 的 `del x` 删除的是一个名字绑定。如果 `x` 的存储还被其他 tensor、view、容器或 autograd 所需的对象持有，存储本身就不满足释放条件。例如，把每轮的 GPU 输出保存在列表中，会让这些输出继续存活；把 loss tensor 不断累加而不切断不需要的计算图，也可能延长图及相关存储的生命周期。

即使最后一个持有者消失，缓存分配器通常也不会立即调用 `cudaFree`。缓存的目的恰恰是让后续分配复用已有显存，减少频繁向 CUDA 申请和归还的成本。更完整的状态关系是：

```text
仍有存储所有者
    │ 最后一个所有者释放
    ▼
free_requested
    ├─ stream 顺序已足以保证复用安全 ───────────┐
    └─ 有记录的其他 stream 使用 → 等待相应事件 ─┤
                                             ▼
                                      free_completed
                                             │
                                       空闲缓存 block
                                             │ 可整体归还时
                                       segment_free
```

`free_completed` 表示这块内存已经可以由分配器安全复用，不应理解为“GPU 上所有使用它的 kernel 都已在物理时间上完成”。在同一个 stream 上，后续工作会按序执行，CPU 端可以提前复用分配记录，而不会让新写入越过旧使用；跨 stream 使用才需要额外的事件或正确的手动同步。[`record_stream` 的官方说明](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.record_stream.html)专门解释了这个看似反直觉的现象。

同样，告诉分配器“这块存储还在另一个 stream 使用”，解决的是释放后的复用安全，不会给仍活着的 tensor 自动加读写锁。若程序主动对同一个 buffer 发起新的 `copy_()`，仍须建立与旧计算之间的依赖。这与 [数据管线中三种 buffer 生命周期]({% post_url 2026-10-21-training-data-pipeline-dataloader %})是一致的。

## 先建立同一时刻的显存账本

在 native snapshot 中，一个 segment 是分配器向 CUDA 取得的较大区间，内部可以拆成多个 block。对同一设备、同一快照中的 block，可以按状态定义：

- $$A$$：`active_allocated` 的 block 字节数，已经分配给调用方。
- $$P$$：`active_awaiting_free` 的 block 字节数，调用方已请求释放，但分配器仍在等待其他 stream 使用结束。
- $$I$$：`inactive` 的 block 字节数，分配器可以按其规则复用的缓存。
- $$R$$：这些 segment 的总字节数。

于是有一个简单但有用的分解：

$$
R=A+P+I,\qquad X=A+P,
$$

其中 $$X$$ 对应这份账本中的 active block 总量。由此可以看出，`reserved - allocated` 并非一个单独的“碎片量”：它既可能包含等待 stream 的部分，也可能包含整段空闲缓存和被活跃邻居夹住的空闲 block。

$$A$$ 也不是把所有 tensor 的 `numel() * element_size()` 相加。View 可能共享同一存储；一个小 view 也可能保持一份很大的底层存储。分配器的 block size 还可能因对齐、取整或复用策略，大于原始请求的字节数。

[v2.8.0 `memory.py`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/memory.py)列出了 `memory_stats()` 的统计口径：`allocated_bytes.all.current`、`active_bytes.all.current`、`reserved_bytes.all.current` 和 `inactive_split_bytes.all.current` 不应互换。字段后缀 `allocated` 和 `freed` 表示历史累计增减，`current` 才是当前值，`peak` 则是峰值。

还有一个值得单独区分的细节：native 的 `requested_bytes.all.current` 不是仅对仍为 `active_allocated` 的请求求和。在 [v2.8.0 的释放路径](https://github.com/pytorch/pytorch/blob/v2.8.0/c10/cuda/CUDACachingAllocator.cpp)中，`allocated_bytes` 在 `free()` 收到释放时减少，`requested_bytes` 则到 `free_block()`、能够安全复用时才减少，因此等待中的请求也可能计入后者。下面账本中的 `requested_live` 特意只统计 `active_allocated`，不冒充这个原生统计字段。

## 24 MiB 中为什么只有 8 MiB 整段空闲

构造两个普通固定大小的 segment，忽略 graph 私有池、expandable segments、分配取整策略和 stream/pool 匹配细节：

```text
Segment 0，16 MiB
┌───────────────┬──────────┬───────────────┬──────────┐
│ allocated 6   │ inactive4│ awaiting_free2│ inactive4│
└───────────────┴──────────┴───────────────┴──────────┘
  原始请求5.5 MiB

Segment 1，8 MiB
┌───────────────────────────────────────────────────┐
│                    inactive 8                     │
└───────────────────────────────────────────────────┘
```

账面上 $$R=24$$、$$A=6$$、$$P=2$$、$$I=16$$，单位都是 MiB。`reserved - allocated` 为 18 MiB，但当前已经完全 inactive 的 segment 只有第二段的 8 MiB。第一段的空闲部分仍与活跃分配共处，不能把整段一并归还给 CUDA。它们并非没有价值：符合分配器条件的新请求仍可复用其中的空闲 block。

下面的代码检查这个**自定义账本格式**。块按顺序排列，大小相加必须等于 segment 大小；`requested` 仅为 allocated/pending block 保留原始请求量。它不读取 pickle，不是 PyTorch snapshot parser，也不模拟真实 `empty_cache()`。

<!-- example: allocator-snapshot-ledger -->
```python
MIB = 1024 * 1024
STATES = {"active_allocated", "active_awaiting_free", "inactive"}


def byte_count(value, name, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"invalid {name}")
    return value


def summarize_segments(segments):
    totals = {state: 0 for state in STATES}
    reserved = requested_live = whole_inactive = largest_inactive = 0
    intervals = []
    for segment in segments:
        address = byte_count(segment["address"], "address")
        total = byte_count(segment["total_size"], "total_size", positive=True)
        blocks = segment["blocks"]
        if not blocks:
            raise ValueError("segment must contain blocks")
        used = 0
        all_inactive = True
        for size, state, requested in blocks:
            byte_count(size, "block size", positive=True)
            byte_count(requested, "requested")
            if state not in STATES or requested > size:
                raise ValueError("invalid block state or requested size")
            if state == "inactive" and requested != 0:
                raise ValueError("this toy format clears inactive requested size")
            totals[state] += size
            used += size
            all_inactive = all_inactive and state == "inactive"
            if state == "active_allocated":
                requested_live += requested
            if state == "inactive":
                largest_inactive = max(largest_inactive, size)
        if used != total:
            raise ValueError("block sizes must sum to segment size")
        intervals.append((address, address + total))
        reserved += total
        if all_inactive:
            whole_inactive += total
    intervals.sort()
    if any(left_end > right_start
           for (_, left_end), (right_start, _) in zip(intervals, intervals[1:])):
        raise ValueError("segments overlap")
    return {
        "reserved": reserved,
        "allocated": totals["active_allocated"],
        "pending": totals["active_awaiting_free"],
        "inactive": totals["inactive"],
        "requested_live": requested_live,
        "whole_inactive": whole_inactive,
        "largest_inactive": largest_inactive,
    }


example = [
    {"address": 0, "total_size": 16 * MIB, "blocks": [
        (6 * MIB, "active_allocated", 11 * MIB // 2),
        (4 * MIB, "inactive", 0),
        (2 * MIB, "active_awaiting_free", 2 * MIB),
        (4 * MIB, "inactive", 0),
    ]},
    {"address": 32 * MIB, "total_size": 8 * MIB,
     "blocks": [(8 * MIB, "inactive", 0)]},
]
ledger = summarize_segments(example)
assert [ledger[key] // MIB for key in ["reserved", "allocated", "pending", "inactive"]] == [24, 6, 2, 16]
assert ledger["whole_inactive"] == 8 * MIB
assert ledger["allocated"] - ledger["requested_live"] == MIB // 2
```

这个静态状态还说明不了调用 `empty_cache()` 后的全部行为。Native 普通全局池的回收路径可能先等待 outstanding events，再把相应 block 放回缓存、合并并尝试归还整段。因此，不能反过来断言“调用它永远处理不了 pending block”；准确的限制是它不能越过必要依赖强行释放，也不能让仍被调用方持有的 allocated 存储凭空消失。

## 空闲总量充足也不代表已有合适的 Block

再构造另一种布局：

```text
Segment 0： allocated6 | inactive4 | allocated2 | inactive4
Segment 1： allocated4 | inactive4
```

总计 12 MiB inactive，但最大的现成空闲 block 只有 4 MiB。一个 6 MiB 请求无法直接装进任意一块。把这些不相邻的空洞相加，并不能让一个 tensor 的存储跨越它们。

但这仍然**不能单独证明会 OOM**：分配器可能向 CUDA 申请新的 segment，或者在失败后释放符合条件的缓存再重试。真实复用还受 stream、small/large pool、私有池和分配策略限制；即使有一个够大的空闲块，也不意味着任意请求都能使用它。

Native 的基本策略包括寻找适合的缓存 block，必要时 split，在释放时尝试与相邻空闲 block 合并；无合适缓存时再向 CUDA 申请。申请失败时，还会尝试回收符合条件的非 split 缓存并重试。这是分配器管理的**虚拟地址区间布局**，不能解释为在整张 HBM 中寻找物理连续空洞；CUDA 设备指针的连续性不等于底层物理页面必须按同样方式连续。

因此判断碎片问题时，需要同时回答：请求多大、当时有哪些可复用 block、它们属于哪个 pool/stream、还有多少 driver 可用容量，以及回收重试发生了什么。只看一项 `reserved / allocated` 比值不够。

## 在 OOM 之前开启有界的事件历史

只在 OOM 后打印一个总量，通常已经丢失了“谁先申请、谁延迟释放、哪个短峰值触发失败”的过程。Memory Snapshot 可以记录当前 segment/block 状态，开启 history 后还会保存分配事件和调用栈。

以下包装器使用 v2.8.0 私有的下划线接口。把真实训练或推理窗口组织成无参数 `workload` 后调用 `capture_window(workload, "case-a")`；输出目录须可写，文件名应为本次诊断独有，避免覆盖旧现场。它自行开启并关闭进程中的 history，不应嵌套调用或接管已有采集会话。升级 PyTorch 时需要重新核对接口。

<!-- example: allocator-snapshot-capture -->
```python
import sys
from pathlib import Path
import torch


def capture_window(workload, prefix):
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA-enabled PyTorch environment is required")
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    print({"torch": torch.__version__, "cuda": torch.version.cuda,
           "allocator": torch.cuda.memory.get_allocator_backend()})
    torch.cuda.memory._record_memory_history(
        enabled="all", context="all", stacks="python", max_entries=100000
    )
    try:
        result = workload()
    except torch.cuda.OutOfMemoryError:
        try:
            torch.cuda.memory._dump_snapshot(str(prefix) + "-oom.pickle")
        except Exception as dump_error:
            print(f"snapshot failed: {dump_error}", file=sys.stderr)
        raise
    else:
        torch.cuda.memory._dump_snapshot(str(prefix) + "-ok.pickle")
        return result
    finally:
        had_error = sys.exc_info()[0] is not None
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception as stop_error:
            if not had_error:
                raise
            print(f"history stop failed: {stop_error}", file=sys.stderr)
```

历史是有界环形记录，新事件可能覆盖旧事件。`max_entries=100000` 限制的是事件条数，不是固定的文件字节数；栈深和记录内容会影响 host 内存与快照大小。应选择能够覆盖故障前窗口的预算，而不是长期以默认超大上限开启。

OOM 处理中的第一步应尽可能保存现场，不要先 `empty_cache()`、`gc.collect()` 或全设备同步再导出，令原本的布局与等待状态被改变。普通成功路径导出时，也不自动意味着设备完成了整个 workload：snapshot 反映的是导出时分配器看见的状态。若要比较若干稳定完成边界，应另行定义测量窗口和必要同步，不能一边改变程序时序、一边把结果当作未经干预的失败现场。

## 两种视图分别回答什么问题

在 [Memory Visualizer](https://pytorch.org/memory_viz) 中加载自己生成的快照，先确认设备和事件窗口，再分别查看两种视图。[官方快照文档](https://docs.pytorch.org/docs/2.8/torch_cuda_memory.html)解释了它们的用途。

Active Memory Timeline 帮助观察分配的存活跨度与调用栈。跨越许多 iteration 的长条，可能是合理持久状态，也可能是被容器或计算图意外保留的输出。第一次 optimizer step 后才出现并长期存在的状态，不应仅因“第二轮比第一轮占用高”就被定性为泄漏；需要确认后续稳定边界是否继续增长。

Allocator State History 则沿 `alloc`、`free_requested`、`free_completed`、`segment_alloc`、`segment_free` 和 `oom` 等事件检查布局。选择 OOM 事件，可以把失败请求大小与当时的 segment/block 结构放在一起看。事件中的 `device_free` 是当时 CUDA 报告的剩余容量，不是“PyTorch inactive cache 大小”。

指针地址会复用。同一个地址后来出现另一条分配，不意味着原 tensor 一直活着；应按地址加分配代次区分对象。相反，定位同一次分配时，可以结合地址/代次和调用栈在两个视图间追踪，判断它何时创建、何时请求释放，以及何时才可复用。

先找时间关系，再看一个总量。基线持续抬升，通常应先调查长寿命引用；基线稳定而某个操作短时抬高峰值，更像是临时 workspace 或激活重叠；大量 awaiting-free 则需要检查跨 stream 生命周期。三种现象可能同时存在，不能都用“碎片”概括。

## 修复应对应快照中的证据

若证据是仍然活跃的分配，先改变所有权或计算需求。检查保存输出的容器、closure、autograd 引用与循环依赖；日志中只需要标量时，不应长期持有整张 GPU tensor。`detach()` 可以切断图，但不会自动把数据搬到 CPU，也不会让被保留的 tensor 本身不占显存。对确实需要的训练激活，可评估 [activation checkpointing]({% post_url 2026-09-09-activation-checkpointing-rematerialization %})或调整 micro-batch，收益要连同重计算与训练目标检查。

若证据是生命周期重叠过长，检查跨 stream 依赖和 buffer 归还边界。补齐正确的 event/`record_stream` 是安全要求；但为了让曲线好看而在每个算子后全设备同步，会改变并发执行和吞吐。应在真实依赖处结束寿命，再测量，而不是移除仍然需要的同步或保护。

若证据是频繁变化的请求尺寸和大量难复用的 split 余量，可以考虑形状分桶、稳定临时 buffer 尺寸，或减少同一窗口内的尺寸组合。对 native allocator，`max_split_size_mb` 是限制 split 的调节项，适合在确有大量 inactive split 且其他办法不足时评估，不是常规默认答案。`expandable_segments` 是面向变化尺寸等场景的实验性选项，也不承诺解决所有 OOM；其释放与统计细节不能完全套用前面的固定 segment 模型。

[CUDA 内存管理文档](https://docs.pytorch.org/docs/2.8/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)还区分 `backend:native` 与 `backend:cudaMallocAsync`。后者要求 CUDA 11.4 或以上，某些 native 统计和 split 参数不适用。切换 backend 后应重新核对统计含义、运行正确性和性能，不宜把 native 快照中的解释直接迁移过去。

## 为什么设备总量仍可能对不上

Snapshot 只能看到通过 PyTorch allocator 管理的分配。NCCL、直接调用 CUDA 的扩展以及其他分配路径，可能持有它没有记录的显存。CUDA context、driver 开销与其他进程，也会影响设备工具看到的总量。

`torch.cuda.device_memory_used(device)` 基于设备层面的统计，不能无条件当作“当前 Python 进程的使用量”。把它减去本进程的 `memory_reserved()`，不等于得到了 NCCL 的精确占用：两者的进程范围、采样时刻和涵盖对象都可能不同。应在尽可能隔离的 GPU 上比较，结合按进程统计和相关库的行为缩小差异来源，再决定是否调查非 PyTorch 分配。

CUDA Graph 还会为固定地址重放保留私有池，不能只根据某轮 tensor 不再被普通代码引用，就推断相关地址应立即归还全局池。需要把 graph、输出 tensor 及共享池的生命周期一并分析，具体机制可结合 [CUDA Graph 的执行与内存约束]({% post_url 2026-07-29-cuda-graph-llm-serving-static-replay %})理解。

## 验收一项显存修复

保留相同模型、输入 shape/长度分布、dtype、batch、并发与 backend，在相同生命周期边界比较多轮运行。先核对结果和训练更新语义，再比较 allocated/active/reserved 的稳定基线与峰值、重试和 OOM 事件，以及正常运行的 step 时间。降低 reserved 但让每轮重新分配并增加同步，不一定是有价值的改进。

`empty_cache()` 可以把符合条件的未使用缓存交还给 CUDA，供其他应用使用，也可能在特定布局下帮助后续分配；它不删除仍然活着的存储，不是修复引用保留的办法。Native 失败路径本来就会执行缓存回收与重试，因此“每轮手动清一次缓存”不应代替生命周期诊断。[接口说明](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.memory.empty_cache.html)与真实调用路径需要一起理解。

最终应能回答三个问题：哪些数据必须继续存在，哪些 block 已经安全可复用，哪些 segment 可以归还。把对象寿命、stream 顺序和分配布局对应起来以后，OOM 才从一个笼统的容量报错，变成能够被验证和修复的具体事件。
