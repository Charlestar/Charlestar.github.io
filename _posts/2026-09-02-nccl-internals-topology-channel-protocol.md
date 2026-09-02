---
layout: post
title: "NCCL 内部机制：一次 AllReduce 如何变成拓扑图、Channel 与 GPU Kernel"
subtitle: "沿着 NCCL 2.31.2 的真实执行路径，理解调优、传输与故障诊断"
date: 2026-09-02 02:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: distributed-training
series_order: 40
technology_year: 2015
mathjax: true
tags: [分布式训练, GPU优化]
---

调用一次 <code>ncclAllReduce</code> 时，应用只交给 NCCL 两个指针、元素数量、数据类型、归约操作、communicator 和 CUDA stream。真正执行前，NCCL 却必须回答一长串更具体的问题：

- 这组 GPU、CPU、PCIe Switch、NVLink/NVSwitch 与 NIC 到底怎样连接？
- Ring、Tree、PAT、CollNet、NVLS 中，哪些候选在当前硬件、操作和数据类型上可用？
- Simple、LL、LL128 哪一种协议更合适？
- 数据应切成多少 chunk，每个 chunk 又怎样按 slice 和 FIFO step 流动？
- 使用多少 channel、多少 CTA 与多少线程，才能兼顾启动延迟、链路利用率和计算重叠？
- GPU 能否直接访问对端显存，是否要经过共享内存，网络数据面是否需要 CPU proxy 推进？
- API 已经返回，究竟只代表“提交成功”，还是通信已经完成？

前文《[Collective Communication：AllReduce、AllGather、ReduceScatter 与 All-to-All](/2026/08/20/gpu-collective-communication/)》讨论的是数学语义、tensor ownership 与算法通信量。本文不再重复“什么是 AllReduce”，而是从一次已经确定语义的 AllReduce 出发，沿 NCCL 的 host enqueue、拓扑建模、调优、plan、channel、GPU kernel、transport 与 proxy 一路向下，直到字节真正经过 NVLink、PCIe 或网络。

这条路径不是一条固定流水线。它更像多级决策树：communicator 初始化时先准备拓扑和连接能力；每次调用时再结合消息大小、操作类型、协议限制和注册状态选择执行方案；kernel 与 proxy 最后协作推进数据。理解这种分层，比记住某个版本的一组阈值更重要。

## 先锁定版本：公开契约与内部实现不是同一件事

本文的公开行为以 NVIDIA 当前的 [NCCL 2.31.2 User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html) 为准；源码分析固定在官方 release [NCCL v2.31.2-1](https://github.com/NVIDIA/nccl/releases/tag/v2.31.2-1)，对应 commit <code>7b83616</code>。所有 NCCL 源码文件链接都指向这个 tag，而不是持续变化的 master；独立的 nccl-tests 也在后文另行固定 commit。

这个边界非常重要：

1. <code>ncclAllReduce</code>、stream 语义、group 约束和 <code>ncclCommGetAsyncError</code> 属于公开契约，升级版本时应维持文档所承诺的行为；
2. <code>ncclInfo</code>、<code>ncclTaskColl</code>、<code>ncclKernelPlan</code>、work batch、channel mask 和 cost model 常数是内部实现，文件位置、字段与组织方式都可能变化；
3. 2.31 的 enqueue 代码已经拆到 <code>src/enqueue/</code> 与 <code>src/tuning/</code>，不能拿旧版本的单文件流程直接套用；
4. 2.31 还加入了 per-collective configuration/tuning 等新路径，部分操作可以走 symmetric kernel、copy engine、RMA 或其他专用实现。“NCCL collective 永远等于一个传统 monolithic communication kernel”已经不是准确的全局描述。

因此，本文讲的是“2.31.2 中经典 host-initiated collective 的主路径，以及它旁边已经存在的分支”，不是 NCCL 永久不变的 ABI。

## 一张图看完整调用链

先把各层放到同一张图里：

~~~text
Application thread
  ncclAllReduce(..., comm, stream)
        |
        v
Host enqueue
  ncclInfo -> argument / communicator checks -> task append
        |
        +-- default 2.31.2 path: ncclTaskColl
        |
        +-- optional enqueue-rearch path:
            raw task -> pre-tuning -> classify -> post-tuning -> task
        |
        v
Tuning and planning
  topology graphs + op/bytes/dtype/reg state
        -> algorithm + protocol + channels/CTAs + warps
        -> ncclKernelPlan + per-channel work batches + proxy ops
        |
        +-----------------------------+
        |                             |
        v                             v
GPU launch                        CPU proxy
  channel mask -> grid              NET / some staged paths
  block/CTA ~= active channel        post/test/flush/progress
  work batch -> device function      update FIFO head/tail
        |                             |
        +-------------+---------------+
                      v
Transport connection and buffers
  P2P / SHM / NET / CollNet / NVLS
                      |
                      v
        NVLink / NVSwitch / PCIe / host memory / NIC / fabric
~~~

这里至少有四种“图”，不要混为一谈：

- 应用的计算图，例如 CUDA Graph 或训练框架 graph；
- NCCL 探测到的硬件拓扑图；
- graph search 产出的 Ring、Tree、CollNet、NVLS 通信图；
- 一次 group 内任务与 launch 的执行计划。

标题里的“变成拓扑图”并不是说每次 AllReduce 都重新扫描 PCIe。拓扑探测和主要 graph search 通常发生在 communicator 初始化阶段；单次 collective 复用这些结果，并依据消息与运行状态做二次选择。

## 从 API 调用到可调度任务

### API 返回只意味着提交边界，不意味着数据已经可读

[CUDA Stream Semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html) 明确规定：带 stream 的 NCCL 调用在操作有效入队后返回，collective 随后在 GPU 上异步执行。正常情况下，可以用 CUDA event、<code>cudaStreamQuery</code> 或 <code>cudaStreamSynchronize</code> 判断 stream 上工作是否完成。

因此要区分三件事：

~~~text
host function returned
    != NCCL work completed on GPU
    != remote ranks have all left the collective
~~~

同一 stream 上后续 kernel 会自然依赖前面的 NCCL 工作；另一个 stream 若要读取结果，必须显式建立事件依赖。把 host 返回当成 tensor ready，会制造典型的跨 stream 数据竞争。

非阻塞 communicator 又增加一层含义：<code>ncclGroupEnd</code> 可能返回 <code>ncclInProgress</code>，此时甚至还不能假设通信 kernel 已经提交到用户 stream。应用必须先轮询 <code>ncclCommGetAsyncError</code>，直到 communicator 状态变为 <code>ncclSuccess</code>，再执行相关 CUDA 同步或后续操作。

### 从 ncclAllReduce 到 ncclInfo

在固定版本的 [src/collectives.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/collectives.cc) 中，<code>ncclAllReduce</code> 进入 <code>ncclAllReduceConfigImpl</code>。这里没有执行 Ring，也没有传输数据，而是构造一个 <code>ncclInfo</code>：

~~~text
func          = ncclFuncAllReduce
opName        = "AllReduce"
sendbuff      = caller send buffer
recvbuff      = caller receive buffer
count/dtype   = logical payload description
op            = Sum / Max / ...
comm/stream   = execution context
chunkSteps    = ALLREDUCE_CHUNKSTEPS
sliceSteps    = ALLREDUCE_SLICESTEPS
collConfig    = parsed per-call configuration
~~~

随后它调用 <code>ncclEnqueueCheck</code>。这里的意义不是“把参数复制到另一个结构体”这么简单，而是把公开 API 语义翻译为内部可检查、可调优、可聚合的描述。2.31 的 <code>ncclAllReduceConfig</code> 还允许每次 collective 带配置，这进一步说明算法与 CTA 等选择不再只能由 communicator 级环境变量决定。

### 2.31 的 per-collective 配置改变了什么

2.31 新增的不是另一个全局开关，而是一组带 <code>Config</code> 后缀的公开 collective API。以 AllReduce 为例，<code>ncclAllReduceConfig</code> 比普通调用多接收一个 <code>ncclCollConfig_t</code>；其他 collective 也有对应变体。固定 tag 的 [公开头文件](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/nccl.h.in) 把它定义为版本化、只在尾部追加字段的结构，并要求用 <code>NCCL_COLLCONFIG_INITIALIZER</code> 初始化。

2.31.2 中可以按单次调用表达的核心意图包括：

- <code>minCTAs</code>、<code>maxCTAs</code> 与 <code>nvlsCTAs</code>：给 channel/CTA 资源设置上下界，而不是直接描述物理链路数；
- <code>cgaClusterSize</code>：在支持的平台上控制 CUDA thread-block cluster 大小；同一个 group 内不一致属于未定义行为；
- <code>algSelection</code> 与 <code>forceAlgSelection</code>：过滤本次调用允许的算法，并决定“没有可行候选”时报错还是回退；
- <code>CTAPolicy</code>：覆盖或继承 communicator 级 CTA policy；
- <code>userProfilerTag</code>：给 profiler plugin 传递不影响执行的关联标识；
- <code>ext</code>：供 vendor library 扩展的链表；官方 NCCL 本身忽略这些 vendor-specific 选项。

这套 API 比长期设置 <code>NCCL_ALGO</code>、<code>NCCL_MAX_CTAS</code> 更适合做“只调整某个 bucket、某个专家通信或某次控制面 collective”的实验，但它没有解除 collective 的分布式契约。所有 rank 必须为同一个 collective 设置相同配置；头文件明确指出 NCCL 只做本地校验，因此 rank 间不一致不会神奇地变成一致的 schedule，反而可能表现为报错或 hang。配置对象及扩展链表还必须在调用期间保持有效。

也不要把 per-collective configuration 与 tuner plugin 混为一谈。前者是应用提供的约束与标签，后者向 NCCL 的 cost model 提供候选代价；它们最终都可能影响 task 的算法和资源字段，但进入系统的位置、生命周期和责任不同。本文后文再讨论 cost model 与 plugin。

### EnqueueCheck：先建立合法的提交事务

固定版本的 [src/enqueue/enqueue.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/enqueue/enqueue.cc) 中，<code>ncclEnqueueCheck</code> 大致完成这些动作：

1. 检查 communicator 指针、状态与 revoked 标志；
2. 确认异步初始化已经进入可通信状态；
3. 按检查模式验证参数、buffer 与 collective 条件；
4. 记录 <code>COLL</code> 日志和 profiler 事件；
5. 调用 task append，把一次 API 转成内部任务；
6. 增加 operation count；
7. 结束内部 group，并在必要时触发整个 group 的准备与 launch。

即使应用没有显式写 <code>ncclGroupStart</code>，单次调用内部也会形成一个 group 边界。显式 group 只是让多个调用共享更外层的事务：内层 end 不会立即 launch，直到最外层 <code>ncclGroupEnd</code> 才统一处理。

这个结构解释了为什么 group 里的单个调用返回后，不能立刻在其 stream 上同步。它可能只完成了“加入事务”，真正 enqueue 发生在最外层 group end。

### 2.31.2 同时存在两条 task 形成路径

阅读 2.31.2 源码时最值得警惕的是 <code>NCCL_ENQUEUE_REARCH_ENABLE</code>。在这个 tag 中，它的内部默认值仍是 0。

#### 默认路径：直接形成既有 task

默认情况下，<code>taskAppend</code> 会按操作种类进入 collective、P2P、RMA 等既有 append 路径。对经典 AllReduce，内部形成 <code>ncclTaskColl</code>，其中逐步填入：

- buffer、count、datatype 与 reduction op；
- algorithm 与 protocol；
- 允许的最大 channel 数；
- warp 数与 device function id；
- chunkSteps、sliceSteps；
- 注册 buffer、NVLS、CollNet 等状态；
- profiler 和清理资源。

这些 task 被放入 communicator 的 planner 数据结构，等待 group 边界统一组装。

#### 可选重构路径：raw task 再准备

若显式开启 enqueue rearchitecture，API 首先追加 <code>ncclRawTask</code>。固定版本的 [task_prep/task_prep.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/enqueue/task_prep/task_prep.cc) 依次执行：

~~~text
rawTaskQueue
    |
    v
pre-tuning
  生成 tuning input，识别 buffer/window/registration 等条件
    |
    v
ncclTuningCompute
  枚举有效候选并估算成本
    |
    v
classification
  分到 symmetric / legacy / p2p / RMA / CE 等队列
    |
    v
post-tuning
  将结果写回具体 task，并准备注册、连接和清理状态
~~~

这条路径把“原始请求、调优输入、执行类别、最终 task”分开，结构上更容易扩展。但 2.31.2 的 [task_sched/task_sched.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/enqueue/task_sched/task_sched.cc) 仍把计划中的 proper schedulers 整段注释掉并直接调用 legacy <code>doLaunches</code>；[src/group.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/group.cc) 也明确写明 scheduler/launcher 尚未实现，当前仍在用户线程上回落到单次 phased <code>doLaunches</code>。

所以，不能看到新目录就断言“2.31 默认已经采用新的多队列调度器”。源码恰好给出了相反证据：重构中的 prepare 已经出现，最终 launch 仍处于迁移边界。

### Raw task、task 与 plan 分别回答什么问题

可以把三种对象理解为三个逐渐具体化的问题：

| 层次 | 回答的问题 | 典型信息 |
| --- | --- | --- |
| Raw task | 用户请求了什么 | API、buffer、count、dtype、op、stream |
| Task | 这次请求准备怎样执行 | algorithm、protocol、channel 上限、warps、注册方式 |
| Kernel plan | 哪些工作由同一次 launch 承载 | channel mask、kernel function、work batches、proxy ops、cleanup |

task 仍是“逻辑工作项”。plan 才是 host launch 的物化单位。多个兼容 task 可能被聚合进同一 plan；一个很大的或受资源限制的 group 也可能产生多个 plan。

固定版本的 [src/include/comm.h](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/include/comm.h) 中，<code>ncclKernelPlan</code> 保存 kernel 函数、参数区、channel mask、每 block 线程数、work batch 数、collective 数、proxy op 队列和清理队列。它不是 CUDA Graph 的同义词，也不是算法图本身。

## 从 Planner 到 Stream 提交

### Planner 为什么要在 group 边界工作

planner 的价值来自“看到一组任务以后再做决定”。它可以：

- 收集 group 中出现的用户 streams；
- 按通信量粗略排序 collective task；
- 决定哪些任务可共享一次 kernel launch；
- 为不同 channel 建立 work batch 链；
- 只为真正需要 CPU 推进的连接生成 proxy ops；
- 维护普通 plan 与 CUDA Graph capture 下 persistent plan 的不同生命周期；
- 让多 GPU 单线程提交在统一阶段完成跨设备 launch。

聚合不是无限融合。kernel 参数空间、work metadata、channel 工作量、协议兼容性、注册资源、profiler 状态和不同专用路径都会形成边界。把 group 理解成“所有 API 必然只发一个 kernel”并不正确。

### Stream 收集：用户 stream 与内部 launch stream

一个 task 带着用户传入的 stream，但 planner 可能要处理一个 group 内的多个 streams。2.31.2 使用内部 strong stream 和事件依赖来保存顺序关系：

1. kernel 启动前，汇聚 group 中所有用户 streams 的先行依赖；
2. communication kernel 在选定的 launch stream 上运行；
3. kernel 结束后，再让参与的用户 streams 等待通信完成。

公开文档给出的可观察结果是：同一 group 混合多个 streams 时，所有 streams 的前置工作都完成后 NCCL kernel 才开始，而这些 streams 在 NCCL kernel 完成前都被阻塞。内部到底用了几个 event、哪条 strong stream，是版本实现细节。

这也解释了一个常见误判：多 stream 不等于多个 collective 自动并行。group 语义可能主动汇合它们。

## 从硬件拓扑到通信 Graph

### 拓扑发现发生在 communicator 初始化阶段

在 [src/init.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/init.cc) 的固定版本路径中，初始化大体按以下顺序推进：

~~~text
discover / load XML
        |
        v
build topology system
        |
        v
compute GPU<->GPU and GPU<->NIC paths
        |
        v
trim inaccessible GPUs and unused NICs
        |
        v
recompute paths
        |
        v
search Ring / Tree / CollNet / NVLS graphs
        |
        v
initialize channel metadata and connection capability
        |
        +-- eager connect when runtime-connect is unavailable or disabled
        \-- defer required connectors to first use when runtime-connect is enabled
~~~

这里的连接时机有版本条件。2.31.2 的 <code>NCCL_RUNTIME_CONNECT</code> 默认值为 1，但只有 communicator 的 CUDA memory capability 满足要求时，<code>runtimeConn</code> 才实际启用；此时初始化主要建立 graph、channel 与连接能力，具体 connector 可在 collective 准备阶段通过 <code>ncclCollPreconnect</code> 等路径按需建立。runtime-connect 不可用或设为 0 时，Ring、Tree、PAT 等连接则在初始化阶段预先建立。

拓扑信息来自本机设备发现、CUDA/NVML、PCI 层级、NVLink、CPU/NUMA、network plugin 暴露的 NIC 属性，以及多 rank 交换后的合并结果。若配置了 <code>NCCL_TOPO_FILE</code>，NCCL 还会先加载 XML 描述；当前文档也说明系统可能默认读取 topologyd 生成的 virtual topology。

“探测到了某条链路”只说明硬件图里存在这条边。它不自动意味着某个 collective 一定走它，后面还有路径约束、graph search、transport 能力与 cost model。

### 物理拓扑图里有什么

NCCL 的拓扑系统不是一张只有 GPU 的邻接表。固定版本会建模 GPU、CPU、PCI、NIC/NET、NVSwitch 等节点与带宽链接。可以把一台双路机器简化为：

~~~text
CPU0
  |
  +-- PCI switch A -- GPU0
  |                \- GPU1
  |
  +-- NIC0

CPU1
  |
  +-- PCI switch B -- GPU2
  |                \- GPU3
  |
  +-- NIC1

GPU0/1 <---- NVLink or NVSwitch ----> GPU2/3   (if platform provides it)
CPU0   <---------- UPI/C2C ----------> CPU1
~~~

如果只看 <code>nvidia-smi topo -m</code> 的 GPU 对矩阵，会漏掉 NIC、network plugin 设备视图、带宽属性和 NCCL 自己的发现结果。<code>NCCL_TOPO_DUMP_FILE</code> 是 NCCL 合并后的 topology discovery 输入证据，却不是“最终执行拓扑”：2.31.2 在 path 计算、trim 和 graph search 之前生成它，dump 模式下的 network device 处理也可能与正常运行的 plugin virtual device 视图不同。最终判断还要结合 GRAPH 日志、运行时 transport 选择与 <code>NCCL_GRAPH_DUMP_FILE</code>。

### Path type 不是简单的 hop count

[src/graph/paths.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/graph/paths.cc) 会为关键节点对预计算路径，并按类别描述距离。常见标记包括：

| 标记 | 直观含义 |
| --- | --- |
| LOC | 本地节点本身 |
| NVL | 直接 NVLink |
| NVB | 经过中间 GPU 的 NVLink 路径 |
| PIX | 经过单个 PCIe switch |
| PXB | 经过多个 PCIe switch |
| P2C | PCIe 与 C2C 组合路径 |
| PXN | 通过中间 GPU 到 NIC |
| PHB | 经过 CPU host bridge |
| SYS | 跨 NUMA/CPU 系统互连 |
| DIS | 不可达 |

这些类别不只是日志标签。它们会影响 P2P 可用性、GDR 条件、LL128 是否安全启用、graph search 的路径上限以及 NIC 选择。数值枚举本身属于内部实现；运维配置应使用文档支持的字符串标识，而不是把某个版本的整数长期写进脚本。

### 从路径到通信 graph：搜索在优化什么

固定版本的 [src/graph/search.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/graph/search.cc) 不是简单做一次最短路。它要寻找多条可并行 channel，并在候选路径上预留带宽，约束路径类型、跨 NIC 行为、GPU 访问顺序和 pattern。

可以抽象成：

$$
\text{graph quality}
\approx
\text{channels}\times\min(BW_{intra}, BW_{inter})
-\lambda\cdot\text{hops}
$$

这不是源码的完整打分公式，只是帮助理解的近似。真实搜索会尝试离散带宽档位、不同 GPU 顺序与 path type，受到超时限制；比较候选时会考虑 channel 数乘带宽，以及 hop 等信息。搜索失败时某些 pattern 还能回落到简单顺序。

关键点是：多 channel 不能各自贪心选择“最短路径”。若所有 channel 都抢同一条 PCIe 上行或同一 NIC，逻辑上有很多 channel，物理瓶颈仍只有一个。graph search 必须考虑共享边的容量。

### 通信 graph 与运行时算法选择是两个阶段

初始化阶段会分别尝试构造 Ring、balanced Tree、CollNet Chain/Direct、NVLS 等 graph。它们记录可用 channel、rank 顺序、intra/inter bandwidth、路径类型和 NIC 端点。

单次 AllReduce 到来后，cost model 才在“已有且对当前操作有效的 graph”之间选择 algorithm/protocol。两者关系是：

~~~text
topology graph search:
  这台机器能构造哪些通信结构？

per-call tuning:
  对这次消息，哪个有效结构预计更快？
~~~

所以看到日志里打印了 Ring graph，不代表此次 AllReduce 使用 Ring；看到 Tree 被构造，也不代表 Tree 一定进入候选；最终要看 <code>TUNING</code> 或 collective launch 相关日志。

## Channel 与数据流水粒度

### Channel 到底是什么

Channel 是 NCCL 内部的一条并行通信轨道。它把三类信息绑定在一起：

1. 一组 rank 邻接关系，例如 Ring 的 prev/next、Tree 的 parent/children；
2. 每个 peer 对应的 send/recv connector 与 transport buffer；
3. kernel 中由一个 CTA 负责推进的 work batch 链。

它不是：

- 一条独占 NVLink；
- 一个 CUDA stream；
- 一个 NIC queue pair；
- 一个 rank；
- 一个 chunk。

不同 channel 可以映射到不同 NVLink 或 NIC 以聚合带宽，也可能共享某段物理路径。一个 channel 内还会连续处理很多 chunk。把日志里的 “16 channels” 直接解释成“使用 16 条物理链路”没有依据。

### Channel、CTA 与 grid 的对应关系

在经典 general communication kernel 的固定版本实现中，[<code>ncclLaunchKernel</code>](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/enqueue/enqueue.cc) 根据 plan 的 <code>channelMask</code> 统计激活 channel 数，并设置：

~~~text
grid.x  = number of set bits in channelMask
block.x = plan.threadPerBlock
~~~

设备端 [src/device/common.h](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/device/common.h) 再把 <code>blockIdx.x</code> 映射到 channel mask 中第 n 个置位 bit。因此在这条路径上，可以近似认为：

$$
\text{one active channel} \leftrightarrow \text{one CTA}
$$

但不要把这个关系提升为所有 2.31 操作的通则。symmetric kernel、copy-engine collective、RMA、device-initiated 路径和未来实现可以有不同的资源组织。即便在 general kernel 内，一个 CTA 也可能依次执行多个 work batch，而不是只处理一个 collective。

### Chunk、slice、step 是三个尺度

这三个词经常被混用，可以按从大到小理解：

#### Chunk：算法分工单位

整个 tensor 先在多个 channel 之间分区，再被算法拆成可沿 Ring 或 Tree 推进的 chunk。以 Ring AllReduce 为例，不同 chunk 在不同阶段分别经历 reduce-scatter 和 all-gather。

#### Slice：流水细粒度

一个 chunk 可以继续拆成 slice。发送方不必等整个 chunk 完成后才通知下一跳，较小 slice 可以让复制、归约、发送与网络进度更早重叠。

#### Step：连接 FIFO 的槽位推进

connector 的 Simple/LL/LL128 buffer 通常按 <code>NCCL_STEPS</code> 划分循环槽。生产者和消费者通过 head/tail、flag 或 step counter 协调槽位复用。step 是协议同步进度，不等于算法的一轮。

关系可以粗略画成：

~~~text
logical tensor
  +-- channel 0 part
  |     +-- chunk 0
  |     |     +-- slice 0 -> FIFO step s
  |     |     \-- slice 1 -> FIFO step s+1
  |     \-- chunk 1 ...
  |
  \-- channel 1 part ...
~~~

具体 chunk 大小会受协议 buffer、数据类型、rank 数、channel 数、算法和尾部对齐共同影响，不应从一个 benchmark 的日志推导成固定常数。

## Algorithm：Ring、Tree 与硬件 Offload

### Ring：带宽友好，但启动步骤会随规模增长

固定版本的 [src/device/all_reduce.h](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/device/all_reduce.h) 清楚展示了 Ring AllReduce 的设备端结构：

1. 首步把一个 chunk 推给 next rank；
2. 中间若干步执行 recv + reduce + send；
3. 完成 reduce-scatter 后，把已归约 chunk 沿 Ring 继续转发；
4. 最后一步接收自己的完整结果块。

大消息时，Ring 能让每条边长期处于流水状态，且每 rank 的理想收发量接近带宽最优。但其算法步骤与 rank 数相关，小消息或超大规模时，逐步延迟更容易显现。

多 channel Ring 会为不同数据分片建立多条逻辑环，从而利用多条可用物理路径。是否真的提升带宽，取决于 graph search 是否把 channel 放到不冲突的瓶颈上。

### Tree：用对数深度换取不同的数据流

Tree AllReduce 通常先向根归约，再从根广播。理想关键路径深度约为：

$$
2\lceil \log_2 p\rceil
$$

固定版本的设备代码同时包含 up/down 与 split tree 处理，线程还可能在 reduce 与 broadcast 两部分之间分配。Tree 的价值不是“永远比 Ring 低延迟”，而是它在特定消息大小、rank 规模和拓扑下减少串行步骤；代价是根附近 fan-in/fan-out、双向阶段、每 channel 峰值与协议效率不同。

在 2.31.2 的 general cost model 中，Tree 候选只用于 AllReduce。这是当前源码事实，不应泛化成 collective 理论的限制，也不保证未来版本不变化。

### PAT：针对大规模 AllGather/ReduceScatter 的另一条轴线

NVIDIA 在 [NCCL 2.23 官方介绍](https://developer.nvidia.com/blog/new-scaling-algorithm-and-initialization-with-nvidia-collective-communications-library-2-23/) 中引入 Parallel Aggregated Trees（PAT），目标是让 AllGather 与 ReduceScatter 在大规模、小到中等消息上获得对数级扩展特性。2.26 又把 PAT step 的计算与执行拆到不同 warp，使多棵并行树可以更好地并发，详见 [NCCL 2.26 官方介绍](https://developer.nvidia.com/blog/improved-performance-and-monitoring-capabilities-with-nvidia-collective-communications-library-2-26/)。

固定版本的 [src/tuning/pat.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/tuning/pat.cc) 给出更窄的实现边界：

- 只为 AllGather 和 ReduceScatter 建模；
- 只支持 Simple protocol；
- 需要满足计算能力、网络设备类型等约束；
- 多 rank per node 的 PAT 会利用 NVLS 处理节点内阶段，而且在该版本 cost model 中仍是 opt-in；
- 它仍保留线性开销成分，不应把“对数算法”理解为端到端时间只剩纯粹的 $\log p$。

PAT 不是 Ring 的通用替代品，更不是 AllReduce 算法名称。它解决的是另一组规模与 collective 组合。

### CollNet：网络 collective 能力进入 NCCL

CollNet 通过 <code>ncclCollNet</code> plugin 接口让网络侧参与 collective，而不仅是普通点对点 send/recv。2.31.2 区分 CollNet Direct 与 CollNet Chain，两者的节点内组织不同。

根据固定版本的 [src/tuning/collnet.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/tuning/collnet.cc)，候选是否有效取决于：

- CollNet plugin 是否加载并声明对应操作；
- collective 是否在对应实现的支持矩阵内：CollNet Direct + Simple 支持 AllReduce、AllGather、ReduceScatter，CollNet Chain + Simple 只支持 AllReduce，二者都没有 LL/LL128 实现；
- reduction op 与 datatype 是否受网络能力支持；
- 节点内 local rank 数、head 数和 NVSwitch 等条件；
- Direct/Chain 的实现与 arity 限制；
- communicator 配置是否允许 CollNet。

“集群有 InfiniBand”并不等于“必然使用 CollNet”，“网卡支持 RDMA”也不等于“支持 in-network reduction”。前者只说明普通 NET transport 可能可用，后者需要 plugin 和硬件 collective 能力的共同证据。

### NVLS 与 NVLSTree：把 NVLink SHARP 放进候选

NVLS 指 NCCL 对 NVLink SHARP 能力的使用，借助 NVSwitch 的 multicast/reduction 机制减少 GPU SM 上的部分数据搬运和归约工作。固定版本的 [src/tuning/nvls.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/tuning/nvls.cc) 显示：

- NVLS 与 NVLSTree 只配 Simple protocol；
- NVLS 支持 AllReduce、AllGather 与 ReduceScatter；
- NVLSTree 在该版本只支持 AllReduce，且单节点时被禁用；
- 多节点 NVLS 还与 CollNet/网络 collective 支持、head 数等条件交织；
- graph 必须拥有足够 channel，操作与 dtype 必须满足 NVLS 能力。

因此，NVLS 不是一个与 transport 完全平行的“新网络”。它既是算法候选，也依赖 NVLink fabric、multicast 资源、注册条件和多节点网络阶段。当前 [环境变量文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) 把 NVLS/NVLSTree 列为可选择算法，并明确它们启用 NVLink SHARP offload。

### 五类算法的边界放在一起

| 算法族 | 主要优势 | 2.31.2 中需特别注意的边界 |
| --- | --- | --- |
| Ring | 大消息流水与带宽利用 | 步骤随 ranks 增长；路径冲突会削弱多 channel |
| Tree | 较少关键路径层数 | general model 当前只给 AllReduce；根附近流量形态不同 |
| PAT | 大规模 AG/RS 的对数扩展方向 | 当前只 AG/RS + Simple；多 RPN 路径仍有额外限制 |
| CollNet Direct/Chain | 网络 collective/offload | 必须有 plugin、op/dtype/head/拓扑支持；Direct 支持 AR/AG/RS，Chain 只支持 AR，当前都只配 Simple |
| NVLS/NVLSTree | NVLink SHARP offload | 需要相应 NVSwitch/NVLS 能力；当前只 Simple，支持矩阵不同 |

表中的“优势”是选择理由，不是强制阈值。最终选择由当前版本 cost model、硬件图与消息共同决定。

## Protocol：Simple、LL 与 LL128

### Protocol 与 Algorithm 是正交维度

Algorithm 决定“谁在第几步和谁通信”；protocol 决定“每条连接上的数据怎样编码、同步和流水”。同一个 Ring 可以配 Simple、LL 或 LL128；同一个 Tree 也有多个协议候选。某些算法只实现 Simple，说明正交组合会被实现能力裁剪，并非所有笛卡尔积都存在。

一个简化的选择空间是：

$$
\mathcal{C}=
\{(a,p,c,w)\mid
a\in Algorithms,\,
p\in Protocols,\,
c\in Channels,\,
w\in Warps,\,
\text{constraints valid}\}
$$

cost model 不是只选 Ring/Tree，而是在有效的 algorithm × protocol × resource 组合中找预计时间更小的一项。

### Simple：有效载荷效率与大消息吞吐

Simple protocol 使用较常规的数据 buffer 与 head/tail 同步，payload 不需要像 LL 那样为每个小数据单元携带同等比例的 flag。它通常更适合带宽主导的大消息，也是 PAT、NVLS、CollNet 等多种专用算法在 2.31.2 中唯一实现的 protocol。

固定版本的 [src/device/prims_simple.h](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/device/prims_simple.h) 展示了更具体的角色分工：不同线程负责 input/output、wait recv/send、post recv/send，发送侧还可留出 warp，把数据 copy 与 threadfence 协调重叠。它并不是“调用 cudaMemcpy 的简单协议”，而是一套设备端流水 primitive。

Simple 的潜在代价是需要更明确的同步与较大粒度的数据推进；消息很小时，启动和同步可能压过 payload 时间。

### LL：用更高的元数据比例换低延迟

LL（Low Latency）把数据与 ready flag 紧密放在 FIFO line 中。固定版本的 [src/device/prims_ll.h](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/device/prims_ll.h) 中，每条 16-byte line 只有一半是有效数据，另一半用于重复 flag；接收方轮询 flag，看到预期 step 才消费数据。

这种设计的收益是生产者写出一小份数据时就能让消费者识别其就绪，不必再等待较粗粒度的独立通知。代价也直接可见：

- wire/buffer 有效载荷比例较低；
- flag 轮转需要清理逻辑，避免旧 flag 被误判为新数据；
- 大消息会为低延迟机制支付过多带宽成本。

所以 LL 的“快”主要是低延迟维度，不是任何消息大小下的最高 GB/s。

### LL128：更高 payload 比例，但依赖链路与平台正确性

LL128 把 128-byte line 中的一部分 word 用作 flag，其余承载数据。固定版本定义中，每行 16 个 64-bit word，其中 15 个为 data、1 个为 flag，理论 payload 比例为：

$$
\frac{15}{16}=93.75\%
$$

它试图在 LL 的细粒度就绪与 Simple 的有效带宽之间取中间点。但 LL128 对内存访问原子性、路径与 GPU 架构更敏感。2.31.2 的 cost model 会根据计算能力、intra/inter path type 等条件决定默认是否启用。

当前官方文档对强制 <code>NCCL_PROTO=LL128</code> 给出非常严厉的提醒：在不支持的平台上启用可能导致数据损坏。这个风险说明 protocol 不只是性能旋钮，也携带正确性前提。

### 协议不是固定的“小、中、大消息三段表”

把 LL、LL128、Simple 记成“小、中、大”是有帮助的第一近似，却不足以预测实际选择。影响结果的还有：

- algorithm 与 collective；
- rank、node、每节点 rank 数；
- NVLink/PCIe/NET path type；
- GPU 架构与混合架构；
- channel 数、thread threshold；
- 注册 buffer、NVLS、CollNet；
- network plugin 的 latency 与 bandwidth；
- 当前 NCCL release 的修正因子。

NVIDIA 的 [NCCL tuning 官方文章](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/) 也把算法、protocol、CTA/channel 和 chunking 放在同一个动态调优问题中，而不是给出永久不变的尺寸分界。

## Plan 与 GPU Kernel 如何执行

### Plan 如何选择 generic 或 specialized kernel

NCCL 在构建时会为 collective、datatype、reduction op、algorithm 和 protocol 的组合生成 device function 映射。固定版本的 [src/device/generate.py](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/device/generate.py) 负责生成表与 kernel 变体。

运行时 plan 保存 <code>kernelFn</code> 和 device function id：

- 若整次 plan 与某个已生成组合匹配，可以走 specialized kernel；
- 若一个 plan 聚合了不同 work，generic kernel 可读取 work batch 中的 function id，再通过 device function table 分派；
- 两者都使用相同的核心 work metadata 与 channel 映射思想，只是分派开销和编译特化程度不同。

因此，profile 中看到不同 kernel symbol 不一定代表不同通信算法；也可能只是 generic/specialized 的 host 选择差异。

### GPU kernel 内部怎样消费 work batch

固定版本的 [src/device/common.cu](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/device/common.cu) 定义 generic kernel，[src/device/common.h](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/device/common.h) 的 <code>ncclKernelMain</code> 展示公共主循环：

1. 把 kernel arguments 搬到 shared memory；
2. 根据 channel mask 把 block 映射到 channel id；
3. 加载该 channel 的第一个 work batch；
4. 调用 specialized runner 或 device function table；
5. 若存在 next batch，重新加载并继续；
6. 检查 abort/profiler 状态，最后退出。

~~~text
CTA for channel 3
   load batch #3
      -> AllReduce Ring Simple work A
      -> nextJump
   load batch #17
      -> another compatible work B
      -> end
~~~

这正是“一个 launch 可以承载多个 task”的设备端落点。kernel 不是只收到 tensor 指针，它收到的是带 channel 链接与 work list 的小型解释程序输入。

### “Persistent kernel”最容易产生的误解

NCCL communication kernel 在一次 collective 执行期间会在设备端轮询 peer/proxy 进度，属于会等待数据到达的 blocking kernel。它不是一个跨整个训练进程永久驻留、不断接受任意新请求的 daemon kernel。

更容易混淆的是 2.31.2 <code>ncclKernelPlan::persistent</code> 的源码注释直接写着 “aka captured in a graph”。这里的 persistent 主要表示：

- plan 被 CUDA Graph capture；
- work metadata 和资源必须活到 graph executable 不再使用；
- 重放 graph 时不能依赖 capture 后已经释放的临时 host 状态；
- communicator 维护 persistent reference 与 graph destructor。

所以应分别说：

1. **communication kernel 的运行形态**：一个操作内长时间推进/等待多个 step；
2. **persistent plan 生命周期**：CUDA Graph 捕获导致计划与资源跨多次 replay 保留。

两者不是同一个概念。

### 2.31.2 已经存在“不是传统 communication kernel”的分支

固定版本的 plan 结构含有 <code>isSymColl</code>、<code>isCeColl</code>、<code>isRma</code> 等标志，enqueue 与 tuning 也能把任务分到 symmetric、copy engine、RMA、GIN 相关路径；例如 group launcher 遇到 <code>isCeColl</code> 时会调用 <code>ncclLaunchCeColl</code>，而不是把它伪装成 general communication kernel。NVIDIA 在 [NCCL 2.28 官方文章](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/) 中公开介绍了 device API、GPU-initiated networking 和 copy-engine collective。

2.31 还增加了 Compute Fabric Transport（CFT）相关能力：公开 <code>ncclConfig_t.hostCftMode</code> 控制 host-side CFT 查询支持，[src/cft_dev_runtime.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/cft_dev_runtime.cc) 则管理 logical endpoint 与 device runtime 集成。在 v2.31.2 的公开支持范围内，CFT 要求 Blackwell GPU 与 CUDA Toolkit 13.3+，并且 host-side CFT 默认关闭；必须显式配置且成功创建 logical endpoints。它与 registered window、device communicator 和平台能力相连，不是 Ring/Tree 的同义词，更不能因为源码里存在字段就断言普通 host-initiated AllReduce 已改走 CFT data path。

因此本文的 kernel/channel/proxy 主线最适合解释经典 AllReduce general path，但要保留两个判断：

- 当前一次 AllReduce 是否走 general communication kernel，应以 TUNING/launch/profile 证据为准；
- 未来版本可能让更多 collective 进入 symmetric、CE、CFT 或 device-initiated 路径，不能用 2015 年的 monolithic kernel 描述覆盖整个 2.31.2。

## Transport 与 CPU Proxy

### Transport 选择发生在连接级

Algorithm graph 决定 rank 邻接，transport 决定某一对相邻 rank 怎样交换数据。固定版本 [src/transport.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/transport.cc) 的 transport 尝试顺序是：

~~~text
P2P -> SHM -> NET -> CollNet
~~~

每个 transport 先调用 <code>canConnect</code>，第一个满足条件者负责 setup/connect。这个顺序不能解读成全局优先级，例如跨节点 P2P/SHM 本来就不可用，会自然落到 NET；CollNet 则用于特定 collective/offload 连接。

同一次多节点 AllReduce 可以同时出现多种 transport：节点内相邻 rank 用 P2P，节点间边用 NET；NVLS/CollNet 路径还会添加专用连接。不存在“一次 collective 只能选一个 transport”的规则。

### P2P：GPU 直接访问与 CUDA IPC

[src/transport/p2p.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/transport/p2p.cc) 处理同节点 GPU 间的直接路径。常见情况包括：

- 同进程、CUDA P2P 可用时交换直接 device pointer；
- 跨进程时通过 CUDA IPC 或 VMM shareable handle 映射远端内存；
- 某些平台允许 P2P read，某些使用 write；
- 特殊情况下可通过中间 GPU 或 copy engine 路径。

P2P 的“direct”表示 GPU 对 GPU 地址可直接 load/store，不意味着物理上不经过 PCIe switch 或 NVLink fabric。真正路径仍由 topology 决定。

日志里的 <code>via P2P/direct pointer</code>、<code>P2P/IPC</code> 能帮助判断地址共享方式，但它们不是端到端带宽证明。

### SHM：不是普通 pageable 内存兜底

当同节点 GPU 之间不能使用 NCCL P2P，SHM transport 可以使用进程间共享的 pinned host memory 作为 staging buffer。[src/transport/shm.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/transport/shm.cc) 会为各 protocol 布置 buffer，并给 GPU 映射可访问的 host pointer 与 head/tail 控制区。

路径可以简化为：

~~~text
GPU A -> pinned shared host buffer -> GPU B
~~~

它通常受 PCIe、CPU/NUMA 与 host memory bandwidth 限制。容器中的 <code>/dev/shm</code> 尺寸、NUMA 绑定或 cuMem host allocation 能力异常，都可能让 SHM setup 失败或性能异常。此时 NCCL 还可能退到 NET/Socket，即使两个 ranks 位于同一台机器。

所以“同机通信却看到 NET”往往是值得调查的信号，而不一定是 NCCL 的预期最优路径。

### NET：GPU kernel 与 CPU proxy 各做一半

传统 NET transport 的数据面需要 network plugin 提供 <code>isend</code>、<code>irecv</code>、<code>test</code>、register memory 等操作。GPU kernel 负责在 connection FIFO 中生产或消费 slice，CPU proxy thread 负责把这些 step 映射成网络请求并持续调用 plugin 推进完成。

发送方向可简化为：

~~~text
GPU kernel
  produce slice and advance tail
        |
        v
send proxy observes ready step
  plugin.isend(...)
        |
        v
NIC / network
        |
        v
remote recv proxy
  plugin.irecv/test/optional flush
        |
        v
publish completion for remote GPU kernel
~~~

如果使用 GDR，payload 可以直接位于 GPU memory；如果不使用，proxy 可能在 host staging buffer 与 GPU 之间协调搬运。无论哪一种，传统 NET 路径中的 CPU proxy 都不是“复制所有数据的 CPU 算法”，它主要负责无法由普通 GPU load/store 完成的网络控制与进度。

### Proxy op 怎样从 plan 到 progress thread

planner 为真正需要 proxy 的 connector 生成 <code>ncclProxyOp</code>，包含 protocol、nsteps、sliceSteps、chunkSize、channel、peer 与 transport connection 等信息。plan launch 前后，这些 op 被追加到 proxy subsystem；progress thread 把可合并的 sub-ops 组成 args，循环调用 transport 的 <code>proxyProgress</code>。

[src/proxy.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/proxy.cc) 与 [src/transport/net.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/transport/net.cc) 中可以看到两级状态机：

- NCCL proxy 负责队列、append、sleep/wakeup、连接与 op 生命周期；
- NET send/recv progress 负责 posted、transmitted/received、done 等 step，并调用 plugin。

一个 channel 卡住时，GPU kernel 可能表现为长时间占用 SM 并轮询；真正根因却可能是 proxy 没得到 CPU 时间、NIC CQ 不推进、远端 rank 没提交匹配工作，或 GDR flush 没完成。只看 GPU timeline 无法区分这些情况。

## 数据路径优化：Registration、GDR 与 PXN

### Buffer registration：省掉的不是“注册调用”，而是中间 copy

[User Buffer Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html) 将目标说得很明确：让 NCCL 直接在用户 buffer 上发送、接收或操作，减少内部 copy 和资源占用。用户可以显式 <code>ncclCommRegister</code>，也可以在满足条件时利用 CUDA Graph registration。

但注册不是通用的“调用一次必然加速”：

- source 与 destination 通常都要满足注册条件；
- communicator 内若有 rank 使用注册 buffer，其他 rank 也必须遵循文档要求，混用可能是 undefined behavior；
- allocator、VMM granularity、共享 handle、GPU Direct RDMA capability 都会影响是否真正启用；
- NVLS、IB SHARP、P2P IPC 与普通 NET 的注册路径不同；
- cleanup 生命周期必须覆盖所有异步使用。

注册的主要收益是 data path 能绕过 NCCL internal staging，而不是消除 NIC memory registration 的所有成本。若 buffer 生命周期短、频繁 register/deregister，控制面开销可能抵消收益。

### GDR：NIC DMA 到显存，而不是“绕过所有 CPU”

GPU Direct RDMA（GDR）允许 NIC 对 GPU memory 直接 DMA。它省掉 payload 经 host memory staging 的一次或多次 copy，但仍可能需要 CPU proxy 提交网络请求、轮询 completion 和更新控制状态。

2.31.2 在 NET setup 时根据 GPU-NIC path 调用 GDR 检查，并在日志中打印 <code>/GDRDMA</code>。官方 [NCCL_NET_GDR_LEVEL 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) 按 LOC、PIX、PXB、PHB、SYS 描述允许距离，默认由 NCCL 根据架构与环境选择。

强制更远距离使用 GDR 不一定更快。跨 CPU socket 的 DMA、IOMMU、ACS、PCIe read 特性和 GPU 型号都可能使 host staging 反而更合适。正确方法是先看自动选择和链路计数，再做对照实验，而不是把 <code>SYS</code> 当作“最高性能等级”。

### PXN：让邻居 GPU 代接更合适的 NIC

PXN 可读作 PCI × NVLink。若 GPU A 到本地 NIC 的路径不理想，但它能经高速 NVLink 到 GPU B，而 GPU B 更靠近该 NIC，NCCL 可以让 GPU B 作为 network proxy rank：

~~~text
GPU A -- NVLink --> GPU B -- PCIe --> NIC B === network === remote
~~~

NVIDIA 在 [NCCL 2.12 PXN 官方文章](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/) 中还说明，PXN 可以汇聚节点内多个 GPU 的消息，减少网络侧小消息与连接压力。2.31.2 的 topology path 中使用 <code>PATH_PXN</code>，NET setup 日志会在 NIC id 后打印实际 proxy rank。

PXN 也有边界：

- 增加一次节点内转发；
- 依赖 GPU 间高速可达与 proxy GPU 的资源；
- 当前文档说明 network buffer registration 与 PXN 不兼容，PXN 开启时即使显式注册，网络注册优化也不会启用；
- <code>NCCL_PXN_DISABLE=1</code> 可以做诊断对照，却可能丢失聚合与更优 NIC 路径。

这里不存在“registration 一定优于 PXN”的统一答案，必须结合消息、拓扑和网络连接规模测试。

## Cost Model 与自动调优

### Cost model：选择预计时间最短的有效候选

固定版本的 [src/tuning/tuning.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/tuning/tuning.cc) 会枚举 tuning id，过滤无效项，运行模型，再选择 <code>timeUs</code> 最小的候选。概念上可以写成：

$$
T(a,p,c)
\approx
L_{base}(a,p)
+N_{steps}(a)\cdot L_{link}
+\frac{Bytes}{BW(a,p,c)}
+T_{overhead}
$$

其中 $a$ 是 algorithm，$p$ 是 protocol，$c$ 是 channel/CTA 资源。真实 2.31.2 模型比这个式子复杂得多：它区分 intra/inter hardware、GPU 架构、per-channel 上限、tree correction、ring plateau、network posting overhead、CollNet/NVLS 能力、注册状态与专用 kernel。

重要的是理解模型性质：

- 它是经验模型，不是链路物理定律；
- 预测值用于相对选择，不等于实际 kernel duration；
- 候选先被正确性与实现约束过滤，再比较性能；
- 初始化得到的 graph bandwidth 只是输入之一；
- 新硬件、新 driver、新 network plugin 都可能让旧阈值失准。

### Channel/CTA 数不是越多越好

增加 channel 通常能提高并行度，却会同时增加：

- GPU CTA 与 warp 占用；
- connection buffer 和 work metadata；
- proxy/NIC 队列压力；
- 小消息被切得过碎后的固定开销；
- 与训练计算争抢 SM、L2、HBM 与链路的程度。

因此，调优结果还会根据 message bytes 与 thread threshold 收缩 channel。对大消息，更多 channel 可能更快；对小消息，一个或少量 CTA 往往足够。对 compute/communication overlap，单看 NCCL microbenchmark 的最低延迟也不够：一个使用 16 CTA 的方案可能通信快一点，却使关键 compute kernel 更慢。

评估目标应该是：

$$
\min T_{step}
\quad\text{而不只是}\quad
\min T_{collective}
$$

### Tuner plugin 能改成本，不会创造不存在的能力

NCCL tuner plugin 可以读取 NCCL 给出的 algorithm/protocol cost table，修改候选成本和 channel 上限。它适合把平台实测、特定拓扑或工作负载经验注入选择。

但 plugin 不能安全地绕过核心约束：

- 没有 CollNet plugin，就不能凭调低成本制造 CollNet；
- 平台不支持 LL128，不能把它当成普通性能候选；
- NVLS graph 不存在，不能只改时间表启用 NVLS；
- buffer 未满足注册条件，不能假设 zero-copy。

好的 tuner 以“在有效候选中校准排序”为目标，并按 NCCL 版本、GPU/NIC 型号、node shape、collective 和消息区间做验证。

### 强制 NCCL_ALGO / NCCL_PROTO 的正确用途

当前环境变量文档支持按函数选择算法或协议，例如只对 AllReduce 限定候选。但 NVIDIA 同时明确提醒，debug/tuning 类变量不应长期固化在生产脚本；新版本自动选择变化后，旧配置可能造成性能下降、功能缺失，甚至 hang 或数据损坏。

强制变量最适合三类实验：

1. **归因**：比较 Ring 与 Tree，判断回归来自算法选择还是 transport；
2. **规避**：临时排除怀疑有 bug 的 protocol；
3. **建模**：扫 algorithm × protocol，采集平台 cost surface。

不适合的做法是只测一个 1 GiB AllReduce，然后永久写入：

~~~bash
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
~~~

这会忽略真实训练里的小 bucket、不同 collective、节点规模变化、NVLS/CollNet 能力和 NCCL 升级。自 2.24 起，如果限制后没有有效候选，NCCL 会严格失败，而不是偷偷回退到 Ring；这种失败反而比静默执行错误假设更安全。

## Stream、Group 与多 Communicator 顺序

### Group ordering：聚合不改变 collective 匹配顺序

[Group Calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html) 要求不同 GPU 上的操作保持相同发起顺序，即使它们被放进同一个 group。假设两个 ranks 顺序不一致：

~~~text
rank 0: AllReduce(A) -> AllReduce(B)
rank 1: AllReduce(B) -> AllReduce(A)
~~~

buffer shape 恰好相同时，它也不是“自动交换顺序”，而是两个 ranks 对 operation count 的解释发生分歧，可能 hang 或产生错误结果。NCCL 没有跨 rank 读取 Python 变量名来匹配 collective。

group 可以聚合 launch，也可让单线程管理多 GPU 时避免第一个调用阻塞等待尚未提交的其他 GPU；它不放松全局顺序约束。

### 同一 group 混合多个 stream 会建立汇合屏障

官方 stream 文档说明，同一 group 使用多个 streams 时：

- NCCL kernel 等待所有参与 stream 的前置工作；
- 所有这些 stream 又等待 NCCL kernel 完成。

~~~text
stream A prework --\
                    >-- NCCL kernel --+--> stream A continues
stream B prework --/                 \--> stream B continues
~~~

这比“每条 stream 各自排一个 kernel”更强。若期望两个独立 collective 并行，先把它们塞进一个 mixed-stream group 可能适得其反。是否能安全重叠还受 communicator、资源占用和全局 launch order 约束。

### 多 communicator 的 launch order

同一 GPU 上并发多个 communicator 是最容易形成循环等待的场景。当前 [communicator 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html) 要求所有设备以一致的 host 顺序发起操作。

从 2.26 起，<code>NCCL_LAUNCH_ORDER_IMPLICIT=1</code> 可以依据 host program order 插入跨 communicator 顺序；CUDA runtime/driver 12.3+ 时仍可允许重叠，旧版本可能序列化。2.31 还增加 per-communicator <code>launchOrderImplicit</code> 配置。

它不能修复不确定的多线程发起顺序。如果两个 host thread 竞态提交 comm A/comm B，各 rank 观察到的顺序可能不同。<code>NCCL_LAUNCH_RACE_FATAL</code> 尝试把这类竞态变成错误，但设计上仍应让每个 device 的 host launch 顺序确定。

### CUDA Graph capture 让 plan 生命周期变长，也让顺序成为集体属性

NCCL 从 2.9 起支持 CUDA Graph capture。当前 [CUDA Graph 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html) 强调：

- 某次操作是否 capture，必须在参与 ranks 间一致；
- graph launch 本身仍是 collective，所有 ranks 必须发起对应 graph；
- 多 GPU 单进程 capture/launch 存在阻塞与 deadlock 风险；
- 多 communicator 的 capture 顺序和 graph launch 顺序都要全局一致。

capture 时，plan、work buffer、registration handle 与清理 callback 不能按普通 launch 立即释放，这正是内部 persistent plan 的来源。修改 buffer 地址、注册状态或 profiler mask 后，已有 graph 不一定自动采纳；RAS 2.31 文档也说明 profiler mask 在 enqueue/capture 时采样，已 capture work 要 recapture 才更新。

## 故障语义与 RAS

### 异步错误：stream 卡住时不能只等 CUDA

网络失败或远端 rank 崩溃后，GPU kernel 可能一直等待不会到来的 step。若应用只调用阻塞的 <code>cudaStreamSynchronize</code>，host thread 也会被困住，失去执行恢复逻辑的机会。

当前 [communicator API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html) 建议在等待时同时：

1. 用 <code>cudaStreamQuery</code> 检查 GPU 工作；
2. 用 <code>ncclCommGetAsyncError</code> 检查 communicator；
3. 配置应用自己的 deadline；
4. 一旦出现 fatal async error，停止向该 communicator 提交新工作；
5. 调用 <code>ncclCommAbort</code>，再由上层决定重建或终止。

发生 communicator error 后，不能假设已 enqueue 操作完成了某个安全前缀，文档明确说其完成与正确性都不可再推断。训练框架必须在更高层以 checkpoint、iteration boundary 或幂等 step 恢复，而不是复用可疑 tensor。

### Revoke、Shrink 与“自动容错”的边界

2.31.2 的公开 API 已包含 <code>ncclCommRevoke</code> 和 <code>ncclCommShrink</code>。Revoke 可以让 communicator 停止接受新 collective，并使其进入可安全 destroy/split/shrink 的 quiesced 状态；<code>NCCL_SHRINK_ABORT</code> 可在排除失败 ranks 时处理未完成操作。

这并不意味着 AllReduce 可以在一个 rank 消失后自动给剩余 ranks 返回“部分和”。collective 的成员与数学结果已经变化，上层必须明确重建 group、恢复数据 placement，并决定本次训练 step 是重放还是丢弃。

NCCL 提供的是 communicator 资源与成员重构工具，不替应用定义容错语义。

### RAS：为 hang 提供全局视角

[NCCL RAS](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html) 从 2.24 起默认启用。每个进程启动一个低开销 RAS thread，通过初始化时建立的 OOB TCP 控制网络交换 keepalive 和 communicator 状态；<code>ncclras</code> 默认连接 localhost:28028 查询。

基础 RAS status 特别适合回答：

- 哪个进程已不可达或被认为 dead；
- communicator ranks 的状态是否一致；
- 各 rank 已发起的 collective count 是否长期不一致；
- 初始化、运行、finalize 中哪一阶段停滞；

2.31 的 RAS Diagnostics 才进一步检查 NCCL 环境变量、GPU inventory、driver、ECC、NVLink 等节点状态。Diagnostics 不会默认随 communicator 初始化运行：可以设置 <code>NCCL_RUN_RAS_DIAGNOSTICS=1</code>，或在需要时执行 <code>ncclras -D</code>。因此看到普通 <code>STATUS</code> 可用，不能推断这些硬件与配置检查已经执行。

使用 RAS 判断 hang 时要做时间序列，而不是只看一次 snapshot。工作负载不均衡会短暂出现 operation count mismatch；若多次查询计数都不前进，才更像真正 stuck。

#### 2.31 的 CONTROL namespace

2.31 新增 <code>CONTROL</code> 命令空间。当前首个控制项可在整个 job 动态切换 profiler event mask，例如：

~~~bash
ncclras CONTROL PROFILER_MASK coll,kernelch
~~~

它是 out-of-band 观测控制，不是改变 algorithm、重排 channel 或修复 deadlock 的远程调参接口。并且 RAS diagnostics 不实际压测 NCCL data path；官方文档明确说它不是全面 cluster health assessment。

## 从日志和拓扑文件建立证据链

### 怎样读 NCCL_DEBUG：先按层过滤

推荐从：

~~~bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,COLL,P2P,SHM,NET,PROXY,REG,RAS
NCCL_DEBUG_FILE=/path/nccl.%h.%p.log
~~~

开始。<code>%h</code> 和 <code>%p</code> 让每台主机、每个进程写独立文件，避免多 rank 覆盖或交错。当前 [NCCL_DEBUG_SUBSYS 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) 还列出 ENV、ALLOC、PROFILE、DESTROY、NVLS 等子系统。

按以下顺序阅读比全文搜索 “WARN” 更有效：

1. **VERSION/INIT**：每个 rank 是否加载相同 NCCL、CUDA、network plugin；
2. **GRAPH**：系统拓扑、paths、Ring/Tree graph 和 channel 是否一致；
3. **NET/P2P/SHM**：每条 connector 选择了什么 transport，是否出现意外 fallback；
4. **TUNING**：INFO 级别查看当前消息的 algorithm、protocol 与 channel 范围；若要精确确认 warps、chunking 与 work metadata，再使用 TRACE/COLL、profiler 或该版本源码辅助核对；
5. **COLL**：opCount、count、datatype、stream 是否在 ranks 间匹配；
6. **PROXY**：网络 step 是否提交、完成或停在某阶段；
7. **REG**：用户 buffer registration 是否成功，是否因 PXN/GDR/allocator 退回；
8. **RAS**：进程与 communicator 的全局状态。

日志行是证据链，不是单行结论。例如 “NET/IB” 说明 plugin/transport 路径，不代表 GDR 已启用；还要寻找 <code>GDRDMA</code> 和实际 data path。

### Topology dump 与 graph dump 解决不同问题

<code>NCCL_TOPO_DUMP_FILE</code> 输出 path 计算与 trim 之前的 discovery XML：设备、层级、link、NIC 属性等。它回答“NCCL 合并了哪些拓扑发现输入”，不包含随后算出的 paths、trim 结果或最终 graph。

2.31.2 源码还支持内部 <code>NCCL_GRAPH_DUMP_FILE</code>，把搜索得到的多个 communication graph 输出为 XML：pattern、channels、intra rank 顺序、inter NIC 端点、bandwidth/path type 等。它回答“在这个拓扑上搜索出了什么”。

建议对比三份资料：

~~~text
lspci / nvidia-smi topo -m / ibdev2netdev   -> OS/hardware view
NCCL_TOPO_DUMP_FILE                         -> NCCL topology view
NCCL_GRAPH_DUMP_FILE + GRAPH logs           -> searched communication view
~~~

要特别谨慎使用 <code>NCCL_TOPO_FILE</code> 回灌修改后的 XML。错误的 link、bandwidth、GPU/NIC id 可能让 graph search 做出不可达选择。它适合可复现实验或平台集成，不应成为掩盖驱动、容器设备和 network plugin 探测故障的长期补丁。

## 用 nccl-tests 验证执行机制

### 不只是追一个峰值

官方 [NVIDIA/nccl-tests（本文固定到 commit <code>b4d5bee</code>）](https://github.com/NVIDIA/nccl-tests/tree/b4d5beebca8a76cf01335f724d154b9b9d394d96) 同时检查 correctness 与 performance。单机基础扫描可以从：

~~~bash
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
~~~

开始；这个命令用于 AllReduce，因此可对照 Ring、Tree、CollNet 与 NVLS/NVLSTree。要验证只支持 AllGather/ReduceScatter 的 PAT，应分别运行对应测试，而不是给 <code>all_reduce_perf</code> 强塞 PAT：

~~~bash
./build/all_gather_perf -b 8 -e 128M -f 2 -g 8
./build/reduce_scatter_perf -b 8 -e 128M -f 2 -g 8
~~~

多节点需要用 MPI 构建并启动，每个进程/线程/GPU 的乘积构成总 ranks。一次可靠实验至少记录：

- NCCL、CUDA、driver、GPU/NIC firmware 版本；
- 节点数、每节点 ranks、rank 到 GPU/NIC 映射；
- in-place 与 out-of-place；
- dtype、reduction op、消息范围与迭代数；
- 自动选择的 algorithm/protocol/channel；
- transport、GDR、PXN、registration 状态；
- CPU affinity、NUMA、GPU clocks 与网络背景负载；
- correctness error count。

只贴一行最大消息带宽，几乎无法解释机制。

### algbw 与 busbw 应怎样解释

[nccl-tests PERFORMANCE（commit <code>b4d5bee</code>）](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/doc/PERFORMANCE.md) 定义：

$$
algbw=\frac{S}{t}
$$

对基于点对点 Ring 模型的 AllReduce，工具使用：

$$
busbw=algbw\cdot\frac{2(p-1)}{p}
$$

<code>algbw</code> 更接近应用关心的“多大 tensor 花多少时间”；<code>busbw</code> 用 collective-specific factor 做归一化，便于和传统链路通信量比较。

但对 NVLS、CollNet、分层或硬件 offload 算法，这个 factor 不再等于某一条真实总线上的实测流量。工具在 API 层并不知道 NCCL 最终内部算法，因而不能把 busbw 高于单条 NVLink/NIC 额定值直接判为错误。此时应以 algbw、硬件 counters、NCCL 选择日志和端到端 step time 共同判断。

## 一套可复现的调优实验

### Algorithm/protocol 对照矩阵

为了区分 cost model、algorithm 与 transport，可以按以下顺序实验：

#### 第一轮：保留自动选择

扫完整消息范围，保存 INFO/TUNING/GRAPH 日志，得到基线。

#### 第二轮：只固定 algorithm

对 <code>all_reduce_perf</code> 分别允许 Ring、Tree；在支持的平台再测 CollNet、NVLS/NVLSTree。PAT 改用 <code>all_gather_perf</code> 与 <code>reduce_scatter_perf</code>，并分别核对该 collective 的候选矩阵。每次都保留协议自动选择，观察算法本身的 crossover。

#### 第三轮：固定 algorithm × protocol

只在文档和日志确认候选有效时测试。LL128 必须先确认平台支持；PAT、NVLS、CollNet Direct/Chain 在当前版本都只配 Simple，而且各自支持的 collective 不同。

#### 第四轮：改变资源

扫描 CTA/channel 上限，记录 standalone collective 与真实训练 overlap。不要只保留 nccl-tests 峰值。

#### 第五轮：改变 data path

分别对照 P2P、SHM、GDR、PXN、registration，但每次只改变一个变量，并确认日志确实采用预期路径。

这样得到的是一个带版本和平台条件的 decision surface，而不是一条无法迁移的“最佳环境变量”。

## 常见症状怎样沿层次定位

### 小消息延迟突然升高

先看 protocol 是否从 LL/LL128 变成 Simple，再看 channel/CTA 是否变多、group 是否混合多 stream、是否意外进入 NET/SHM。不要先调整 socket 线程数。

### 大消息带宽只有预期一半

检查 graph channel 是否集中到同一 NIC/PCI root、GDR 是否启用、PXN proxy rank 是否合理、CPU NUMA 与 NIC affinity、是否发生 P2P→SHM/NET fallback。再比较 Ring/Simple 对照。

### GPU timeline 中 NCCL kernel 很长

长 kernel 可能是正常的大消息，也可能是 device polling。结合 proxy 日志、NIC counters、远端 operation count 与 RAS；仅看 kernel duration 不能判断 GPU 端慢还是远端未到。

### 所有 ranks 没有 WARN 但作业不前进

比对每个 communicator 的 opCount、collective type、count/dtype、group 顺序和 CUDA Graph launch 次序。collective mismatch 往往不会在最初调用处立即变成可读错误。

### 强制协议后性能好却偶发数据错

立即撤销强制项，确认 LL128 平台支持与链路 path type，运行 correctness test，并把 NCCL/CUDA/driver 版本纳入复现。吞吐提升不能抵消正确性失败。

### 注册后没有任何收益

检查 REG 日志是否真实成功、source/destination 是否都注册、buffer allocator 是否 RDMA capable、PXN 是否启用、算法是否支持 user buffer，以及 register/deregister 成本是否进入测量区间。

## 版本迁移与源码阅读

### 哪些结论可以跨版本，哪些必须重新验证

相对稳定的认知包括：

- NCCL 操作遵守 CUDA stream 异步语义；
- communicator 定义 collective 匹配范围和顺序；
- topology、algorithm、protocol、channel、transport 是不同层；
- 多节点传统网络路径通常需要 GPU kernel 与 CPU/network progress 协作；
- 强制调优必须以 correctness 与端到端性能验证。

必须绑定版本重新验证的内容包括：

- task/plan 的结构名与文件路径；
- enqueue 重构是否默认启用；
- algorithm × collective × protocol 支持矩阵；
- cost model 常数、阈值和修正因子；
- channel/CTA 上限与 specialized kernel；
- registration、PXN、GDR 的组合约束；
- RAS 输出字段、timeout 和 CONTROL 能力。

这也是为什么本文给出 tag/commit，而不是只写“看 NCCL 源码”。

### 推荐的源码阅读顺序

若要在 2.31.2-1 上继续追一次 AllReduce，按下面顺序最不容易迷路：

1. [src/collectives.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/collectives.cc)：公开 API 到 <code>ncclInfo</code>；
2. [src/enqueue/enqueue.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/enqueue/enqueue.cc)：检查、task append、plan 与 launch；
3. [src/include/comm.h](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/include/comm.h)：task、planner、plan、channel 的数据结构；
4. [src/tuning/tuning.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/tuning/tuning.cc) 与 [cost_model.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/tuning/cost_model.cc)：候选过滤与选择；
5. [src/graph/topo.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/graph/topo.cc)、[paths.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/graph/paths.cc)、[search.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/graph/search.cc)：拓扑、路径和 graph search；
6. [src/device/all_reduce.h](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/device/all_reduce.h) 与三个 primitive header：设备端算法与 protocol；
7. [src/transport.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/transport.cc)、[p2p.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/transport/p2p.cc)、[shm.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/transport/shm.cc)、[net.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/transport/net.cc)：连接与 data path；
8. [src/proxy.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/proxy.cc)：CPU progress 与 proxy op 生命周期；
9. [src/ras/](https://github.com/NVIDIA/nccl/tree/v2.31.2-1/src/ras)：运行时全局诊断。

阅读时始终带着一个具体 case：rank 数、node shape、消息大小、dtype、algorithm/protocol 日志和 transport。脱离实例通读模板代码，很容易被大量编译期组合淹没。

## 从一次 AllReduce 回到整条因果链

现在可以把标题中的过程更准确地复述一遍：

1. <code>ncclAllReduce</code> 先生成 host 侧请求描述，并在 group 事务中完成检查和 task append；
2. communicator 初始化时已经把硬件发现结果转成 topology system、paths 与多个 algorithm graph；
3. per-call tuning 根据操作、消息、注册状态和 graph 能力过滤 algorithm/protocol 候选，估算时间并决定 channel/CTA 与线程；
4. planner 把 task 组装成一个或多个 plan，为每个 active channel 编排 work batch，并生成需要的 proxy op；
5. general kernel launch 时，grid 中的 CTA 映射到 channel，依次执行 work batch；设备 primitive 按 chunk、slice、step 推进 Ring/Tree 等算法；
6. P2P/SHM 连接可由 GPU 直接或经 host buffer 交换，NET 连接通常由 GPU kernel 与 CPU proxy/network plugin 协作；
7. GDR、PXN 与 user buffer registration 改变 payload 经过哪块内存和哪张 NIC，却不取消 stream/order/failure 语义；
8. API 返回只代表提交边界，完成由 stream 观察，故障由 async error、日志和 RAS 联合判断。

真正的性能问题很少只属于某一层。算法选择、channel 并行、PCIe/NVLink 争用、proxy CPU 调度、NIC/GDR、stream 排序和训练 overlap 会共同决定结果。最可靠的调优方法也因此不是背环境变量，而是让每个结论都能在 topology、TUNING、transport、timeline、hardware counter 与 correctness 中找到相互印证的证据。

## 主要资料

- [NCCL 2.31.2 User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html)
- [NCCL v2.31.2-1 release 与固定源码 tag](https://github.com/NVIDIA/nccl/releases/tag/v2.31.2-1)
- [NVIDIA NCCL GitHub：v2.31.2-1](https://github.com/NVIDIA/nccl/tree/v2.31.2-1)
- [Understanding NCCL Tuning to Accelerate GPU-to-GPU Communication](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/)
- [New Scaling Algorithm and Initialization with NCCL 2.23](https://developer.nvidia.com/blog/new-scaling-algorithm-and-initialization-with-nvidia-collective-communications-library-2-23/)
- [Networking Reliability and Observability at Scale with NCCL 2.24](https://developer.nvidia.com/blog/networking-reliability-and-observability-at-scale-with-nccl-2-24/)
- [Improved Performance and Monitoring Capabilities with NCCL 2.26](https://developer.nvidia.com/blog/improved-performance-and-monitoring-capabilities-with-nvidia-collective-communications-library-2-26/)
- [Doubling All2All Performance with NCCL 2.12：PXN](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/)
- [NVIDIA/nccl-tests（commit b4d5bee）](https://github.com/NVIDIA/nccl-tests/tree/b4d5beebca8a76cf01335f724d154b9b9d394d96) 与 [固定版本的 performance metric 说明](https://github.com/NVIDIA/nccl-tests/blob/b4d5beebca8a76cf01335f724d154b9b9d394d96/doc/PERFORMANCE.md)
