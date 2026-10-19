---
layout: post
title: "GPU 性能剖析：从 Nsight Systems 时间线到 Nsight Compute 计数器"
subtitle: "沿一次工作负载的提交、等待和执行找到关键路径，再把内核指标转化为可验证的优化判断"
date: 2026-10-20
last_modified_at: 2026-10-20
author: iStar
catalog: true
series: gpu-runtime-precision
series_order: 17
technology_year: 2019
mathjax: true
tags: [GPU优化, 工程实践]
---

一个矩阵乘在性能报告里占了最多时间，是否就应该先优化它？不一定。它可能与通信重叠，提前完成以后仍然要等待另一条 stream；也可能真正拖慢请求的是两个矩阵乘之间没有及时提交工作的空隙。

性能剖析需要同时回答两个问题：程序为什么在这个时刻才完成，一个具体 kernel 又为什么需要这么久。Nsight Systems 适合把 CPU、CUDA API 和 GPU 工作放在同一条时间线上；Nsight Compute 适合进入选定的 kernel，检查资源使用与指令执行。前者决定应该研究哪里，后者帮助解释局部机制，最后还要返回没有分析工具介入的运行，确认用户关心的指标真的改善。

本文沿这条路径展开，不重复 [Roofline 的资源上限推导]({% post_url 2026-09-09-roofline-gpu-performance-model %})。系列中的 2019 年锚定 [Nsight Compute 2019.1 新增 NVTX 上下文过滤](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html#updates-in-2019-1)，不是两款工具的起源年份。下面的命令按 2026-10-20 查阅的官方文档组织，使用前仍需查询本机版本支持的选项。

## 先说清楚哪一个完成时间值得优化

一次完整请求、一个推理 step、单个 kernel 和稳定负载下的一秒吞吐，不是同一个观察对象。先固定对象，才能判断优化是否有效。

例如，单请求延迟可以从应用接收输入计到输出完成；一次设备算子测试可以从已准备好的 GPU 输入计到目标结果完成。前者可能包含排队、CPU 数据准备和传输，后者则有意排除了它们。吞吐测试还需要固定到达方式与并发条件，不能用一个串行 microbenchmark 的加速比例直接推算线上容量。

建立基线时，至少保留模型或程序版本、输入 shape、dtype、设备、驱动、CUDA 与库版本，以及计时边界。编译、autotune、内存分配和初始化是否包含在范围内，也应明确：冷启动有自己的价值，但不能与预热后的稳态样本混合后只给一个平均数。

CUDA 提交通常是异步的。CPU 调用一个算子返回，只能说明相关主机调用已经返回，不能说明结果已经在 GPU 上就绪。若目标是主机观察到的完成延迟，需要在范围末端等待相关工作完成；若用 CUDA events 测量设备区间，则必须确保要测的工作位于两个 events 之间，跨 stream 工作还要有正确的依赖汇合。[CUDA 的计时说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html#timing)区分了这两类方法。

在每个小算子后加全设备同步虽然让计时容易理解，却会破坏本来存在的重叠。因此，调试正确性时的同步、单请求完成时的同步和生产流水线中的依赖，应分别设计。基线首先要代表目标工作负载，不能为了便于测量而把目标改成另一个程序。

## NVTX 给主机行为命名，不替 GPU 计时

时间线上只有一串相似的 CUDA 调用，很难判断它们属于 prefill、decode，还是某个测试的预热阶段。NVTX range 可以把这些主机侧的语义写进去：例如 `decode_step`、`projection`、`residual_norm`。

一个 push/pop range 表示 CPU 线程何时进入和离开这段代码。假设 CPU 在 range 内提交一个 kernel，随即离开 range，GPU 可能在 range 结束后才开始执行它。Range 的长度因此不是该 kernel 的 GPU 执行时间，也不隐含一次同步。

[Nsight Systems 的标注说明](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#marking-and-labeling-regions)解释了 host range 与 GPU 投影的关系：工具可以将 range 内发起的 GPU 工作关联出来，帮助理解其来源。关联解决“是谁提交的”，不会把主机时间和设备时间变成同一种时间。

Range 名称应围绕稳定的阶段语义，而不是每次都生成一个带长字符串的唯一名字。过密的细粒度标注会增加采集和阅读成本；真正有价值的层级，是能从请求或 step 逐步缩小到需要研究的算子。Push 与 pop 也必须配对，异常路径可以用 `try/finally` 或上下文管理器保证退出。

## 一个可以保存执行的采集工作负载

下面把一个矩阵乘与一个逐元素激活组合成一次 step，预先分配输出 buffer，并把预热、正常计时与采集模式分开。保存为 `workload.py` 后，在安装了 CUDA 版 PyTorch 的环境中执行。

它不是模型 benchmark：输入已经驻留 GPU，没有真实请求队列、网络、数据加载与多 stream 通信。示例的目的，是让标注和采集边界具体可见。接口参照 PyTorch 2.8 的 [NVTX range](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.nvtx.range.html) 与 [profiler start/stop 实现](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/profiler.py)。本文环境没有安装 PyTorch 与两款 Nsight 工具，因此没有执行这段 GPU 工作负载或生成实测报告。

<!-- example: gpu-profiling-workload -->
```python
import argparse
from contextlib import contextmanager
import json
import statistics
import time

import torch


@contextmanager
def region(name, enabled):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch environment is required")

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    x = torch.randn((256, 1024), device="cuda", dtype=torch.float32)
    weight = torch.randn((1024, 1024), device="cuda", dtype=torch.float32)
    product = torch.empty((256, 1024), device="cuda", dtype=torch.float32)
    output = torch.empty_like(product)

    def step(mark=False):
        with region("target", mark):
            torch.mm(x, weight, out=product)
        with region("activation", mark):
            torch.tanh(product, out=output)

    # Check the operator composition and output-buffer reuse before timing.
    reference = torch.tanh(x @ weight)
    step()
    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-5)
    for _ in range(20):
        step()  # Deliberately no range named "target" during warmup.
    torch.cuda.synchronize()

    metadata = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "x_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "dtype": str(x.dtype),
    }
    if args.capture:
        torch.cuda.profiler.start()
        try:
            with region("measured_step", True):
                step(mark=True)
                torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()
        print(json.dumps({**metadata, "mode": "capture"}))
    else:
        samples_ms = []
        for _ in range(30):
            started = time.perf_counter()
            step()
            torch.cuda.synchronize()
            samples_ms.append((time.perf_counter() - started) * 1000)
        print(json.dumps({
            **metadata,
            "mode": "host_observed_step_completion",
            "samples": len(samples_ms),
            "median_ms": statistics.median(samples_ms),
            "min_ms": min(samples_ms),
            "max_ms": max(samples_ms),
        }))


if __name__ == "__main__":
    main()
```

先运行 `python workload.py`，得到不带 profiler 的完成时间分布。这里每个 step 结束都同步，故有意测量串行 step 的主机完成延迟，不是让多个 step 排队后的最大吞吐。正确性比较检查的是这个示例的算子组合与 buffer 复用；替换成另一种算法或融合实现后，还要为新实现增加独立的数值参考与边界测试。

再记录工具版本和环境支持情况：

```bash
nsys --version
ncu --version
nsys status --environment
```

采集需要匹配设备、驱动、工具版本和目标平台权限。计数器权限不足时，应按管理员批准的配置处理，不能把失败后的空报告当作“没有瓶颈”。这里不要求为了运行示例而修改系统范围的权限策略。

## 先用 Systems 看工作为什么没有接上

用以下命令只采集 `cudaProfilerStart/Stop` 划定的区域，避免把初始化和预热淹没进时间线：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop -o trace python workload.py --capture
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum trace.nsys-rep
```

`--capture-range=cudaProfilerApi` 让 start/stop 调用控制采集；脚本在 start 前等待预热完成，在 stop 前等待目标工作完成。关闭 CPU sampling 与 context-switch trace 是为了从较小的采集范围开始，不表示 CPU 调度不重要。如果后续证据指向主机侧，再按平台支持情况增加相应数据源。[Systems 的 focused profiling 与 CLI 选项](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#focused-profiling)给出了这些控制方式。

汇总报告适合找高频调用和高累计成本，真正判断等待还要打开 `.nsys-rep` 时间线。一次 kernel 的主机提交、设备开始与设备结束，可以画成：

```text
CPU thread     [ CUDA launch API ] ---- continues running ---->
GPU stream            ... waiting ... [ kernel execution ]
                      ^ submitted     ^ begins            ^ ends
```

API 调用长度、提交后到执行前的间隔、设备执行长度，是三个不同观察量。很短的 API 调用后可以排着很长的 GPU 队列；一个较长的 API 调用也可能正在等待之前的设备工作，不能直接把它归为 CPU 计算慢。还应利用工具的相关性信息检查究竟是哪次调用发起了哪次工作，不要只凭视觉上相邻就认定因果关系。

沿一次目标范围读取时间线时，可以依次观察：输入是否已准备好、CPU 何时提交、工作落在哪个 stream、前驱何时完成、结果是否等待拷贝或其他 stream。Stream 内的顺序、events、显式同步和数据依赖，决定哪些工作必须等哪些工作；两个不同 stream 本身并不保证同时执行，更不保证没有共享资源竞争。

GPU 上一段空隙可能来自 CPU 尚未提交，也可能来自传输、依赖、其他进程、运行时行为或没有采集到的活动。先沿依赖寻找证据，再决定要不要增加 CPU sampling、线程调度或其他平台支持的 trace。仅凭一个空白区写出“Python 是瓶颈”，缺少了中间最重要的推理。

## 累计时长、忙碌区间和关键路径各自在算什么

假设一个任务先花 2 ms 准备输入，然后启动两条相互独立、能够完全重叠的分支：计算耗时 6 ms，通信耗时 9 ms。最后还要等两者完成，花 1 ms 整理结果。这里的数字和重叠条件是用于推导的假设，不是真实 GPU 的预测。

```text
prepare 2 ms --+--> compute 6 ms -------+--> finalize 1 ms
              |                       |
              +--> communication 9 ms +
```

完成时间由较慢分支决定：

$$
T=2+\max(6,9)+1=12\ \mathrm{ms}
$$

如果只把计算与通信的持续时间相加，会得到 15 ms；它们在设备时间轴上的区间并集则是 9 ms。一个数统计累计工作时间，一个数统计这两段活动至少有一段在执行的时间，而 12 ms 统计整个任务完成时间。多条重叠工作累计时长超过 wall time 是正常现象，不能据此得出“GPU 利用率超过 100%”。

现在把计算加速两倍，分支变成 3 ms，完成时间仍是 12 ms。把通信加速两倍，分支变成 4.5 ms，则：

$$
T'=2+\max(6,4.5)+1=9\ \mathrm{ms},
\qquad
S=\frac{12}{9}=\frac43
$$

这不是说计算值得忽略，而是当前等待链经过通信；通信缩短后，计算又成为较慢分支。优化会改变关键路径，不能永久沿用第一次分析得到的排序。真实 GPU 上两分支若争抢 SM、带宽或其他资源，单独测得的 6 ms 与 9 ms 也可能在并发时改变，此时必须返回真实时间线。

下面用标准库复算依赖图的最早完成时间和区间并集。图模型只表达给定持续时间与依赖，不模拟 GPU 调度或资源竞争。

<!-- example: gpu-profiling-critical-path -->
```python
import math


def nonnegative_time(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("time must be a finite nonnegative number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("time exceeds floating-point range") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("time must be a finite nonnegative number")
    return value


def earliest_finish(tasks):
    # name -> (duration, predecessor_names)
    state, finishes = {}, {}

    def visit(name):
        if name not in tasks:
            raise ValueError("unknown predecessor")
        if state.get(name) == 1:
            raise ValueError("dependency cycle")
        if state.get(name) == 2:
            return finishes[name]
        state[name] = 1
        duration, predecessors = tasks[name]
        duration = nonnegative_time(duration)
        start = max((visit(p) for p in predecessors), default=0.0)
        finishes[name] = nonnegative_time(start + duration)
        state[name] = 2
        return finishes[name]

    for name in tasks:
        visit(name)
    return finishes


def interval_union(intervals):
    checked = []
    for start, end in intervals:
        start, end = nonnegative_time(start), nonnegative_time(end)
        if end < start:
            raise ValueError("interval ends before it starts")
        checked.append((start, end))
    total, right = 0.0, 0.0
    for start, end in sorted(checked):
        total += max(0.0, end - max(start, right))
        right = max(right, end)
    return total


tasks = {
    "prepare": (2, ()),
    "compute": (6, ("prepare",)),
    "communication": (9, ("prepare",)),
    "finalize": (1, ("compute", "communication")),
}
assert earliest_finish(tasks)["finalize"] == 12
assert sum(end - start for start, end in [(2, 8), (2, 11)]) == 15
assert interval_union([(2, 8), (2, 11)]) == 9

faster_compute = {**tasks, "compute": (3, ("prepare",))}
faster_communication = {**tasks, "communication": (4.5, ("prepare",))}
assert earliest_finish(faster_compute)["finalize"] == 12
assert earliest_finish(faster_communication)["finalize"] == 9
print("wall=12 ms, summed branches=15 ms, union=9 ms")
```

在真实报告中，“关键路径”意味着沿着使目标最终完成的等待链追踪，而不是把 kernel 汇总表从大到小当作这条链。热榜能提出候选，依赖和时间线才能解释它们是否阻挡了完成。

## 选中具体实例以后，再打开 Compute

同一个 kernel 名字可以出现在不同 batch、序列长度和执行阶段，也可能因为不同 grid、缓存状态与并发环境表现完全不同。把所有同名 kernel 求平均，会把要研究的实例埋掉。

从 Systems 选出候选时，应保留所在阶段、输入 shape、dtype、stream、调用顺序及可获得的 grid/block 等信息，再在 Compute 中核对。对于带动态 shape 或长时间运行的服务，可以先把一次代表性输入缩成较小的复现程序，但必须检查复现是否仍保留了原来的瓶颈条件。

先查询本机支持的集合、sections 与指标：

```bash
ncu --list-sets
ncu --list-sections
ncu --query-metrics
ncu --nvtx --nvtx-include "target/" --set basic --launch-count 1 -o kernel python workload.py --capture
```

这里的 `target/` 对应脚本中的 push/pop range；末尾 `/` 用来说明这是 push/pop range 过滤，而不是 start/end range 的写法。`--launch-count 1` 只采集匹配范围内的第一个 kernel。虽然 range 只包住一个 `torch.mm`，一个库调用仍可能发起多个 kernel，所以它不承诺捕获整个矩阵乘，更不承诺第一个就是时间线中最值得研究的 kernel。应检查输出中的名字、launch 配置与实例，再按需要缩小范围或增加实际名称过滤。[Compute CLI 的 NVTX filtering](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering)规定了这套语法。

Systems 与 Compute 分开运行，不嵌套采集。脚本的预热没有同名 `target`，也就不会把预热阶段误当成目标实例；真实程序里若使用 launch skip 等过滤，还要核对它到底在匹配前还是匹配后的序列上计数。

## 指标应该回答一个已经提出的问题

`--set basic` 是缩小采集成本的起点，不是一次诊断的终点。看到某个异常后，再收集本机支持的定向 section，比一开始对整个程序执行 full collection 更容易解释。

比如，一个 kernel 耗时很长，但究竟是搬运量多、线程并行度不足，还是依赖限制了指令发射？这些假设需要不同证据：

| 要解释的现象 | 可以继续查看的 section | 需要保留的区分 |
| --- | --- | --- |
| 哪类执行资源接近其参考吞吐 | SpeedOfLight | SM 汇总吞吐不是某类 Tensor Core 指令的专属利用率 |
| launch 是否提供足够工作，资源限制了多少驻留 | LaunchStats、Occupancy | 理论 occupancy、实测 occupancy 与可发射 warp 不是同一个量 |
| 数据从哪一级存储搬运，事务是否高效 | MemoryWorkloadAnalysis | L1、L2 与设备内存流量不能混成一个 HBM 带宽数字 |
| scheduler 为何没有持续发射指令 | SchedulerStats、WarpStateStats | 先确认发射不足，再结合源码和依赖理解 stall 分类 |

具体 section 名和可用指标以 `--list-sections`、`--query-metrics` 的结果为准。表中是诊断方向，不是所有架构都具有完全相同计数器的承诺。[Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)对 section、吞吐与调度状态给出了定义。

以 memory 为例，较低的算术强度说明访存可能给出较低的资源上限，却不证明实际带宽已经跑满。如果一个 kernel 只有很少的 thread blocks，可并行访问不足，带宽利用率也会低；继续减少每个元素的几个算术操作未必解决它。如果实测事务明显多于有用数据，还要检查访问合并、重复读取与布局，而不只是报告一个带宽百分比。

Occupancy 也不是越高越好。增加 tile 或让更多中间值留在寄存器，可能降低可驻留 warps，却减少了全局访存；反过来，试图提高 occupancy 造成寄存器 spill，可能让时间更长。要比较的是完成同一问题的成本，再用这些指标解释变化。

Warp stall 同样需要上下文。某种等待占比高，不自动意味着应该消除它：只要其他 eligible warps 能持续填满发射机会，这种等待就可能已经被隐藏。先确认 scheduler 确实无法充分发射，再追查等待对应的指令、数据依赖或同步，才能把状态分类变成优化方向。

## 计数器采集会改变被测程序

硬件计数器不能总在一次执行中收集完。Compute 可能需要 replay，多 pass 之间的状态恢复、缓存处理和启动序列化都可能改变运行条件。因此，分析进程运行了几十秒，不代表目标 kernel 在正常运行里也耗时几十秒。

默认 kernel replay 与 range replay 的观察对象也不同。前者围绕单个 kernel 收集数据；range replay 面向一段 CUDA 调用与工作，能在相应支持条件下保留范围内并发，指标也关联到整个范围。不能笼统地说“Compute 下所有工作永远被串行化”，也不能把 range 汇总指标当成某个单 kernel 的私有指标。

缓存是另一个会改变结论的条件。默认 replay 的缓存控制可能让目标更接近隔离、冷缓存执行；真实工作负载中，前面的 kernel 却可能已经把数据带入 L2。关闭工具的清缓存选项，也不自动恢复所有原始条件：还要考虑 replay 模式、前驱执行和跨 pass 一致性。时钟控制、温度与并发进程同样需要记录。[官方 replay 与 cache control 说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)详细讨论了这些差别。

最危险的做法，是从一份冷缓存计数报告拿到字节数，再除以另一份暖缓存正常运行的耗时，得出一个并不存在的“实测带宽”。计数、耗时和资源上限必须描述可比较的范围与条件；如果条件有意不同，就应分别回答不同问题。

通信 kernel 还可能依赖其他进程或 GPU 同时取得进展。把这类 kernel 从并发环境中隔离，会改变性能，甚至使程序无法前进。不要直接套用本节的单进程矩阵乘命令去采集一个真实 NCCL 作业，应先阅读当前工具对 mandatory concurrent kernels 与多进程采集的限制。

## 一次优化怎样形成证据闭环

假设 Systems 显示一串很短的逐元素 kernel 之间存在提交空隙，且这串工作位于目标完成链上。一个可检验假设是：融合或执行图减少提交与中间物化，可能缩短该阶段。假设成立时，预期变化不只是“某个指标更高”，而是 kernel 数量或间隙减少，必要流量下降，最终完成时间也下降。

另一种情况是，时间线连续而目标 kernel 的访存量很大。若 Compute 显示大量中间读写，才有理由尝试消除物化；若显示可并行工作不足，则应研究 shape、分块和并行度。两个程序都可以被口头描述成“GPU 没跑满”，改法却不相同。

每轮只改变一个可解释的机制，先检查数值和内存语义，再比较匹配条件下的局部证据，最后关闭 profiler 重跑原始指标。除了典型 shape，还要覆盖尾部 shape、低并发与实际重要的输入分布；一个只对大 batch 有效的优化，可能拖慢更常见的小 batch。

优化后还应重新采一段 Systems 时间线：等待链可能已经转移，重叠关系也可能变化。若局部 kernel 更快而完整请求没改善，先检查它是否仍在关键路径，以及节省的时间是否被新的等待抵消，而不是立即宣布工具测错。

这样，性能结论就不再只是“某个 kernel 慢”或“某个利用率低”，而是一条可以复核的因果链：目标是什么，等待发生在哪里，哪种机制限制局部执行，改变后哪些证据跟着变化，以及正常运行最终改善了多少。

仓库中的 [CPU 复算脚本](https://github.com/Charlestar/Charlestar.github.io/blob/master/scripts/reviews/gpu-profiling-examples.mjs)会提取本文依赖图代码，检查并集、分支加速、依赖循环和非法时间等情况，并对 GPU 示例做 Python 语法解析。它不替代 CUDA 编译、硬件 trace、计数器采集或真实工作负载的性能验收。

## 参考资料

- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA Nsight Compute CLI NVTX Filtering](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering)
- [CUDA C++ Best Practices Guide 12.8 Timing](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html#timing)
- [PyTorch 2.8 NVTX API](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.nvtx.range.html)
- [PyTorch v2.8.0 CUDA profiler API 源码](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/profiler.py)
- [Nsight Compute 2019.1 Release Notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html#updates-in-2019-1)
