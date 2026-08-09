---
layout: post
title: "NIXL 与 KV Connector：把推理引擎和传输后端解耦"
subtitle: "从内存注册、元数据握手到异步 P/D KV Cache 交接"
date: 2026-07-05 09:00:00 +0800
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: kv-cache-memory
series_order: 50
technology_year: 2025
tags: [分离式推理, KV Cache, 分布式推理]
---

Prefill/Decode 分离后，一条请求会跨过两个推理实例：P worker 计算 prompt 并生成 KV Cache，D worker 接收这些 KV，再继续逐 token decode。架构图里只需要画一根从 P 到 D 的箭头，runtime 中却必须回答一串具体问题：KV 在哪几块 GPU memory 中，远端如何访问这些地址，什么时候开始传，何时确认完成，P 端何时可以释放 block，D 端又怎样把外部 KV 映射进本地 paged cache。

如果每种网络、存储和推理引擎组合都实现一套专用路径，系统会迅速被绑死在具体设备和 backend 上。NIXL（NVIDIA Inference Xfer Library）与 KV Connector 处在两个互补的抽象层：

- **NIXL** 把 HBM、DRAM、文件和其他存储中的数据段统一描述，通过插件选择合适的数据移动 backend，并返回可异步查询的传输状态；
- **KV Connector** 把“移动一组 buffer”翻译成 serving engine 能理解的 KV 生命周期，包括请求匹配、block 分配、layer-wise load/save、完成通知和失败回退。

二者的边界很重要。NIXL 不知道一块 tensor 是否属于第 17 层的 K，也不知道哪个 token 前缀可以复用；Connector 也不应该重新实现 RDMA connection、memory key 与不同存储协议。一个负责数据面，一个把数据面嵌入推理语义。

## 先把一次 P/D 交接展开

假设 prompt 已在 P worker 完成 prefill。理想状态下，D worker 下一步就能在这些 KV 上做 decode：

```text
P worker                                          D worker
────────                                          ────────
allocate KV blocks
run prefill
write layer K/V
       │
       ├──── publish transfer metadata ─────────────►
       │                                             allocate local blocks
       │◄──────────── ready / descriptors ──────────┤
       │                                             post read or await write
       ├════════════ KV data movement ═════════════►│
       │                                             verify completion
       │◄──────────── consumed / release ───────────┤
free or reuse source blocks                         start decode
```

箭头中混合了两种完全不同的信息：

1. **控制面**：request ID、模型身份、block 数、远端地址描述、连接 metadata、完成/失败和生命周期；
2. **数据面**：真正占带宽的 K/V bytes。

控制消息很小，却决定数据应该从哪里到哪里；KV payload 很大，应该走高吞吐、低 CPU 开销的路径。若把两者都塞进同一个普通 RPC，或者让大数据路径反复承担连接协商，通信开销就难以与计算重叠。

## 为什么不能直接传一个 GPU 指针

GPU pointer 只在所属进程和设备的虚拟地址空间中有意义。远端进程拿到十六进制地址，并不自动拥有：

- 访问该地址的权限；
- 对应 NIC/RDMA backend 的 memory registration key；
- 源 device、NUMA 和可用 transport 信息；
- buffer 长度、数据类型与布局；
- 地址仍然存活的保证。

即使在同一台机器上，CUDA IPC、NVLink P2P、host staging 和跨进程共享也有不同建立方式。跨节点时还可能选择 InfiniBand/RoCE、TCP 或存储路径。

因此，传输系统需要比裸指针更完整的描述：

```text
segment = {
  memory_type,
  local_address_or_object_id,
  length,
  device / storage identity,
  backend-specific registration metadata
}
```

NIXL 将这些可传输地址范围组织为 Memory Section/segments，并由对应 backend 完成注册。远端得到的是足够建立访问的数据描述，而不是随意读取本进程内存的通行证。

## NIXL 的位置：通用 point-to-point 数据面

NIXL 面向分布式 AI inference 的点对点数据移动。它试图提供一个统一 API，让上层不用为每种源/目标组合直接耦合底层协议：

```text
inference runtime / KV connector / weight loader
                         │
                         ▼
                  NIXL Transfer Agent
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    Memory Section   Metadata Handler   Backend Interface
          │              │              │
          └──────────────┼──────────────┘
                         ▼
             UCX / GDS / other plugins
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
         HBM            DRAM       file / storage
```

这使同一套上层流程可以用于 KV Cache、模型权重、activation 或其他 tensor 对象。NIXL 名称里有 Inference，但它本身不包含 Transformer-specific 的 attention 与 sampling 逻辑。

## Transfer Agent 抽象的三个核心对象

### Memory Section

Memory Section 是交给 agent 管理的一组 address segments。它可以表示 GPU memory、host memory 或存储对象；每个 segment 包含本地访问所需信息，以及 backend 为远端访问生成的标识。

上层仍负责实际分配内存。NIXL 负责注册和描述，不是替推理引擎决定 KV block pool 应该多大。

### Transfer Backend Interface

Backend plugin 知道怎样在某类 memory/storage 上执行传输。相同内存区域可以注册给多个 backend；当本地和远端都支持多条路径时，agent 可以根据源/目标 memory type 与共同 backend 选择执行方式，也允许上层显式指定。

例如，GPU 到远端 GPU 可能选择 UCX/RDMA 路径，GPU 与文件系统之间可能由 storage-oriented backend 处理。具体可用插件随 NIXL 版本、构建选项与硬件变化，不应把示例列表当作永久兼容矩阵。

### Metadata Handler

两个 agent 要通信，必须交换 backend connection 信息和已注册 segment 的远端标识。Metadata Handler 负责序列化、加载、缓存和失效这些信息。

NIXL 不强制上层使用哪一种 metadata service。serving system 可以通过 side channel 直接交换，也可以使用集中式服务。控制面负责身份、权限、发现和心跳，NIXL 负责理解与目标 backend 相关的那部分 metadata。

## 内存注册为什么应该离开请求热路径

高性能网络通常要求 memory registration。若每次请求到来后才注册一组临时 GPU buffer，再交换 metadata，固定开销可能吞掉异步传输的收益。

Paged KV manager 往往预先分配大块 KV arena，再把它切成逻辑 block。更合适的流程是：

```text
engine startup
  allocate stable KV arena
  register arena with NIXL backends
  export agent / segment metadata
  exchange or publish metadata

request hot path
  select subranges / blocks in registered arena
  build descriptors
  post transfer
  poll completion while computation continues
```

一次注册覆盖长期存活的内存池，请求只传递 offset/length 等 descriptor。这样既减少注册开销，也让 connection 和 metadata 有机会预热。

但长期注册会带来资源约束：registered/pinned memory、NIC memory key、endpoint 与 cache 都不是无限的。扩缩容、GPU reset 或 block arena 重建时，旧 metadata 必须失效；否则远端可能继续引用已经释放或重新分配的地址。

## Metadata Exchange 不等于 KV Cache Metadata

“metadata”在这套系统里至少有两层含义，很容易混淆。

### 传输层 metadata

NIXL agent metadata 描述：

- agent identity；
- backend connection 信息；
- 注册 memory segment 的远端访问标识；
- backend type 与能力。

它回答“怎样访问远端内存”。

### KV 语义 metadata

Connector/serving scheduler 还要描述：

- request、prompt 或 cache key；
- 哪些 token/block 已经计算；
- layer 与 block 对应关系；
- source/destination block IDs 与 offsets；
- KV layout、dtype、tensor parallel shard；
- model/tokenizer/adapter compatibility；
- 本轮采用 push 还是 pull，以及完成回执。

它回答“应该访问哪些 bytes，以及这些 bytes代表什么”。

两个 agent 即使成功握手，也不表示某个 KV Cache 可被语义安全地使用。相反，KV block key 完全匹配，也仍可能因为没有可用数据通路、远端 memory 已失效而无法取回。

## 一次 NIXL 传输的抽象流程

在 NIXL 的基本模型中，runtime 为本地和远端 buffer 准备 descriptor list，指定目标 agent 和读/写方向，再创建并提交 transfer request：

```text
local_descs  = [segment A subrange, segment B subrange, ...]
remote_descs = [remote X subrange, remote Y subrange, ...]

handle = create_transfer(
    operation = READ or WRITE,
    local_descs,
    remote_descs,
    remote_agent
)

post(handle)

while status(handle) is pending:
    do other serving work
```

非阻塞 API 的价值不在于 `while` 循环写得更漂亮，而是让 runtime 能把数据移动与下列工作重叠：

- P worker 继续执行后续 layer/chunk；
- D worker 处理其他请求的 decode step；
- CPU scheduler 准备下一 batch；
- 其他 transfer 在不同 NIC/path 上推进。

若调用后立刻同步等待，底层即使是异步 RDMA，端到端仍是一条串行关键路径。

## Pull 与 Push 是两种不同所有权关系

### D 端 Pull

P 完成 prefill 后保留源 KV，发布可读 descriptors。D 分配目标 block，再对 P 发起 read：

```text
P: source KV [READY, PINNED]
                ▲
                │ NIXL READ initiated by D
                │
D: target KV [ALLOCATED] → [LOADING] → [READY]
```

优势是 D 知道自己何时有空间，可以控制读入节奏。P 必须一直保留源 block，直到 D 明确完成或 lease 超时。

### P 端 Push

D 先发布目标地址，P 发起 write：

```text
D: target KV [ALLOCATED, EXPOSED]
                ▲
                │ NIXL WRITE initiated by P
                │
P: source KV [READY] → [SENDING] → [RELEASABLE]
```

优势是 P 可以在 layer 产生时主动发送，适合流式 handoff；但 D 必须提前分配容量并安全暴露 descriptor，P/D control plane 也要处理目标变化。

实际 connector 可能支持单向、双向或混合策略。无论方向如何，transfer initiator 与 KV owner 不是同一概念：谁发起网络操作、谁拥有源 block、谁决定释放，需要分别定义。

## KV Connector 为推理引擎补上了什么

以 vLLM 的 connector 抽象为例，scheduler-side connector 和 worker-side connector 分担不同职责。

### Scheduler 侧

Scheduler 关心请求能复用多少外部 KV、需要预留多少本地 block，以及某请求是否应等待异步 load。它生成本轮 connector metadata，随着 scheduler output 发送给 worker。

概念上它执行：

```text
lookup external KV availability
        │
        ▼
decide matched token count
        │
        ▼
allocate local paged blocks
        │
        ▼
emit request/block transfer metadata
```

如果外部状态还没查清，connector 可以让 scheduler 之后再询问；如果 KV 不再可用，就必须返回实际可加载的最大前缀，不能把潜在命中当作已完成计算。

### Worker 侧

Worker 拿到具体 tensor 与 block mapping，注册 KV arena，启动 load/save，并在 attention layer 需要数据之前等待对应传输完成。

它处理的对象更接近：

```text
layer name
local paged KV tensor
block table / offsets
remote descriptors
async transfer handles
```

Scheduler 不应该直接操作 CUDA pointer，worker 也不应该独立改变全局请求顺序。两侧 connector 通过每个 engine step 的 metadata 对齐决策与执行。

## Layer-wise Load/Save 怎样隐藏通信

如果 connector 只暴露 `load_all()` 与 `save_all()`，目标模型必须等完整请求的所有层 KV 到齐才能 forward，P 端也要等 prefill 完整结束才开始发送。

Layer-wise interface 允许更细的 pipeline：

```text
P compute:     L1 ─ L2 ─ L3 ─ L4
P save/send:      S1 ─ S2 ─ S3 ─ S4

D load:               R1 ─ R2 ─ R3 ─ R4
D forward:                 L1 ─ L2 ─ L3 ─ L4
```

D 在进入 attention layer $i$ 前只等待第 $i$ 层 KV load 完成，而不是一开始等待所有层。vLLM connector base 因此包含 start load、wait for layer load、save KV layer 与 wait for save 一类生命周期点。

可隐藏程度取决于：

- 每层计算时间与对应 KV transfer 时间；
- P/D 是否同时占用相同 NIC/PCIe 路径；
- tensor layout 是否能连续传输；
- layer buffer 是否提前注册；
- 是否出现某一层或某一 shard 的尾部 straggler。

只展示 aggregate GB/s 无法证明 layer pipeline 有效，还要测 attention 因等待 KV 而停顿的时间。

## KV Layout 是 Connector 的硬契约

同一个逻辑 KV 可以有不同物理布局，例如：

```text
[num_blocks, block_size, num_kv_heads, head_size]
[num_blocks, num_kv_heads, block_size, head_size]
cross-layer contiguous arena
per-layer independent tensors
```

再乘上 tensor parallel 后，每个 rank 只持有部分 KV heads；pipeline parallel 只持有部分 layers；某些 attention backend 还会使用特定 packing、alignment 或 quantized scale。

如果 P 和 D 的布局不同，直接逐字节复制可能“传输成功、推理错误”。Connector 必须：

- 要求两端使用相同 layout；或
- 显式执行 layout conversion；
- 为每个 rank/layer 建立正确 descriptor mapping；
- 验证 dtype、shape、stride 与 block size；
- 在 heterogeneous path 中记录 conversion 的算力和临时内存成本。

这也是 connector compatibility matrix 必不可少的原因。模型能在 P 和 D 单独启动，不代表其 KV layout 能被当前 connector 组合正确交接。

## Block 生命周期必须是一个状态机

异步 transfer 最大的正确性风险不是网络失败，而是源/目标 block 被过早复用。

### P 端源 Block

```text
ALLOCATED
   │ prefill writes KV
   ▼
READY
   │ publish descriptor / grant lease
   ▼
PINNED_FOR_TRANSFER
   │ remote read complete or push completion
   ▼
RELEASABLE
   │
   ▼
FREE / PREFIX-CACHED
```

如果 scheduler 在 `PINNED_FOR_TRANSFER` 时把 block 分给另一请求，D 可能读到一半旧 KV、一半新 KV。Connector 的异步完成状态必须反向约束 block allocator。

### D 端目标 Block

```text
RESERVED
   │ start transfer
   ▼
LOADING
   │ completion + validation
   ▼
READY_FOR_ATTENTION
   │ request finishes / evicted
   ▼
FREE
```

目标 block 在 completion 前不能进入正常 prefix match，也不能被 attention 读取。load 失败时，scheduler 要么重新计算缺失 token，要么取消/重试；不能把未写满的 block 标成 ready。

## Lease 解决“完成消息永远没来”的问题

Pull 模式中，P 为 D pin 住 KV。若 D 崩溃、请求取消或 control message 丢失，P 可能永远等不到 release，最终耗尽 KV pool。

租约为 pin 状态设置有界生命周期：

```text
grant lease(TTL)
      │
      ├─ D heartbeat/renew ─► extend expiry
      ├─ transfer complete ─► explicit release
      └─ no renewal until expiry ─► reclaim
```

TTL 太短，慢传输或排队中的正常请求会被提前回收；TTL 太长，故障 worker 会长期占住 HBM。更可靠的设计用 heartbeat 续租，并把 request cancel、worker failure 与 transfer completion 接入同一释放路径。

租约只能保护生命周期，不能替代 generation/version。block 地址重新分配后，即便旧 completion 迟到，也不能释放新 owner 的 block；回执应携带 request/transfer epoch 或唯一 ownership token。

## 动态扩缩容首先是 Metadata Cache 问题

新增 worker 时，它要创建 agent、注册内存并把 metadata 交给需要通信的 peer。下线或故障时，其他 agent 必须失效相关 metadata 与 connection：

```text
scale out:
new agent → register → publish metadata → peers cache → ready

scale in/failure:
mark draining/dead → stop new transfers → invalidate metadata
                   → finish or abort in-flight → deregister memory
```

如果 service discovery 已删除一个 endpoint，但 NIXL/connector 仍缓存旧 remote segment，下一次传输会访问失效地址。反过来，如果先 deregister memory，再让控制面停止分配请求，in-flight transfer 也可能失败。

因此，扩缩容不是简单地更新负载均衡器地址。至少需要 draining、metadata version、in-flight accounting 与 timeout。NIXL 提供 agent metadata 的加载/失效机制，何时调用以及如何与 scheduler membership 保持一致，仍由上层平台负责。

## NIXL 不等于 KV Store

NIXL 可以在 memory 与 storage 之间移动数据，但不自动提供完整的分布式 KV Cache 语义。一个 Store 还要负责：

- 根据 token prefix/model identity 查找对象；
- 全局 key 到副本位置的目录；
- 容量、淘汰、热度与复制；
- 多租户隔离和配额；
- 对象写入的可见性与失败恢复；
- cache admission 与调度策略。

可以用一句分层关系区分：

```text
KV Store decides WHAT exists and WHERE replicas live.
KV Connector decides WHEN serving needs it and HOW it maps to blocks.
NIXL moves the BYTES between selected source and destination.
```

Mooncake Store、LMCache 或其他存储系统可以选择 NIXL/其他 backend 作为数据通路；NixlConnector 也可以在 P/D worker 之间直接搬运 KV，而不必先构建长期分布式 store。

## NIXL 与 NCCL 也不是同一种通信抽象

NCCL 擅长一组 rank 之间的 collective，例如 all-reduce、all-gather 和 all-to-all。训练 tensor parallel 或 MoE expert parallel 常以固定 communicator、相对规则的 buffer 执行这些操作。

P/D KV 交接更像动态 point-to-point：

- 每个请求的源、目标和 payload 长度不同；
- worker 会弹性加入和离开；
- 数据可能在 HBM、DRAM、文件或远端存储；
- 上层希望异步读/写某些 address segments；
- 生命周期与 cache object/request 绑定。

NIXL 针对后者提供统一 transfer descriptor 与 backend plugin。某些 backend 内部仍可能利用已有通信库或硬件能力，但应用层语义不是 collective group。

## 配置 Connector 时真正要核对什么

以当前 vLLM 文档中的结构为例，NixlConnector 通过 `kv-transfer-config` 选择，并可指定 NIXL backends：

```bash
vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_both",
    "kv_buffer_device": "cuda",
    "kv_connector_extra_config": {
      "backends": ["UCX", "GDS"]
    }
  }'
```

字段和兼容性会随版本变化，复制命令前应查看对应 release 的文档。比配置是否成功解析更重要的是：

- P/D model、tokenizer、KV dtype 与 block size 相同；
- tensor/pipeline parallel 拓扑受支持；
- 两端 rank mapping 与 network interface 正确；
- 所需 NIXL plugin 实际加载，而非静默走慢路径；
- memory registration 与 metadata handshake 完成；
- firewall、RDMA device、IOMMU/peer access 等环境成立；
- scheduler/router 能把 P 端 transfer parameters 带给正确 D 请求；
- prefix cache、chunked prefill、CUDA Graph、quantization 等组合在兼容矩阵内。

启动日志至少要打印 connector、backend、agent identity、registered bytes 和 peer discovery。仅看到 HTTP 服务可用，不能证明数据面已经走预期路径。

## 正确性验证要故意制造竞态

### Byte 与 Logits 对照

对固定 prompt：

1. 在 colocated baseline 完整 prefill；
2. 在 P 生成 KV，经 connector 传到 D；
3. 比较各层/各 rank 的 KV checksum 或抽样元素；
4. 比较 D 首个 decode logits 与 baseline；
5. 继续生成多个 token，确认序列/分布在允许误差内。

只比最终文本可能漏掉少量错误，因为 argmax 未必立刻变化。

### 生命周期竞态

主动测试：

- transfer 尚未完成时 P block allocator 尝试复用源 block；
- D 在 load 中被 preempt；
- 请求取消发生在 descriptor 发布前、传输中和完成后；
- completion 消息重复、乱序或延迟；
- lease 到期与 heartbeat 同时发生；
- peer 重启后沿用相同 host/port，但 agent epoch 已变；
- 某一 layer/shard 失败，其他部分已经完成。

每种情况下都要检查 block reference、in-flight handle、registered memory 和请求状态最终归零，没有 silent partial KV。

### Layout 与兼容性

扫描不同 attention backend、KV dtype、TP size 和 block size。对不支持的组合应在握手/启动阶段明确失败，而不是等第一条线上请求产生错误 logits。

## 性能测试要分开 Post 与 Transfer

异步 API 有两类时间：

$$
T_{post}=\text{准备 descriptor 并提交到 backend 的同步耗时}
$$

$$
T_{xfer}=\text{从提交到 backend 报告完成的时间}
$$

高 `post` latency 会阻塞 scheduler/worker thread，即使网络带宽很高也会影响每步执行；高 `xfer` latency 则可能来自 payload、网络拥塞、路径或远端 memory。

还应计算：

$$
BW_{effective}=\frac{\text{successful payload bytes}}
{\sum T_{xfer}}
$$

以及真正进入关键路径的：

$$
T_{visible}=T_{request\ waits\ for\ KV}
$$

若 $T_{xfer}=20\text{ ms}$，但其中 18 ms 与计算重叠，用户感知的代价只有约 2 ms；反过来，传输峰值很高但 D 每层都等尾部 shard，visible latency 仍会很差。

## 需要长期观察的指标

### NIXL 数据面

- successful/failed transfer count；
- post 与 completion latency 的 P50/P90/P99；
- bytes/transfer 与 effective bandwidth；
- backend/path 选择分布；
- registration、connection 与 metadata cache 命中；
- retry、timeout、remote invalidation。

### Connector

- external matched tokens 与实际 loaded tokens；
- P pinned blocks、lease renew/expiry；
- D loading/ready blocks；
- layer wait time 与 transfer overlap ratio；
- fallback recompute、request cancel 和 partial failure；
- scheduler 因等待外部 KV 暂停的请求数。

### Serving SLO

- TTFT、ITL/TBT 与 E2E；
- P/D queue time；
- HBM block utilization；
- tokens/s 与 SLO goodput；
- colocated/chunked-prefill baseline 的交叉点。

vLLM 文档特别提醒，disaggregated prefilling 的直接目标是隔离 prefill、控制 tail ITL，而不是自动提高吞吐。完整平台可以通过资源配比、独立扩缩容和缓存复用改变有效容量，但必须用自己的 workload 测量，不能把“接入 NIXL”直接等同于“吞吐提升”。

## 常见故障怎样定位

### 首次请求很慢，后续正常

可能是 lazy connection、memory registration、metadata exchange 或 backend 初始化在第一条请求发生。把长期内存池注册、peer discovery 与 connection warm-up 移到 readiness 之前，并单独统计首传耗时。

### 网络带宽很高，TTFT 仍然没有下降

检查传输是否与 prefill/layer load 重叠、D 是否先在队列中等待、host staging 是否重复 copy，以及 KV payload 是否本来就比重算慢。还要比较 visible wait，而不是只看 NIXLBench。

### 偶发错误 token

优先检查 layout/dtype/rank mapping、源 block 过早复用、旧 agent metadata、partial transfer 和 completion 竞态。这类问题通常不是模型随机性。

### P 端 KV pool 逐渐耗尽

检查 cancel/failure 是否触发 release、lease 是否续期但不终止、`get_finished` 一类完成状态是否被 scheduler 消费，以及 transfer handle 是否在异常路径泄漏。

### 扩容后新节点传输失败

检查 agent metadata 是否发布到所有需要通信的 peer、backend 能力交集、connection endpoint 与网络策略，以及 router 是否在节点 readiness 前就发送请求。

## 何时应该使用更简单的路径

NIXL + Connector 的抽象很适合多节点、高速网络、异步 P/D 或分层 KV storage，但不是所有部署都需要：

- 同进程 P/D 不需要远端 metadata；
- 单机跨进程可能用 CUDA IPC/NVLink 的专用 connector 更简单；
- 小 prompt 重算比连接、descriptor 和搬运更快；
- 低速 Ethernet 下，host serialization 可能成为主要成本；
- 固定两节点 PoC 可以先用明确的 P2P path 验证系统收益，再引入插件化和弹性发现。

抽象的价值来自组合数量和动态性。如果部署只有一个稳定路径，先证明 P/D 分离的收益，往往比一开始搭建完整传输平台更可靠。

## 小结

NIXL 与 KV Connector 补上了分离式推理中最容易被一根箭头掩盖的工程层。NIXL 用 Memory Section、Transfer Backend 和 Metadata Handler 抽象异构内存与存储，建立可注册、可寻址、可异步完成的 point-to-point 数据面；KV Connector 则把这种 buffer transfer 对齐到请求、Paged KV block、layer 和 scheduler step。

系统正确性的核心不是“bytes 最终到达”，而是两端对这些 bytes 的身份、布局和生命周期达成一致：P 在远端读完前不能释放源 block，D 在所有必要数据完成前不能让 attention 读取，故障、取消和扩缩容必须让 lease、metadata 与 block owner 一起收敛。

沿着前一篇 Mooncake 的视角看，Store 和 scheduler 决定哪些 KV 值得保存、放在哪里，Connector 决定推理引擎何时加载或保存它们，NIXL 决定选中的数据怎样跨 HBM、DRAM 和存储高效移动。把这三层分开，才能替换 backend 而不重写 serving 语义，也能在性能下降时准确定位是缓存策略、Connector 状态机还是底层路径的问题。

## 参考资料

- [NIXL 官方仓库](https://github.com/ai-dynamo/nixl)
- [NIXL Core Concepts 与 Architecture](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md)
- [vLLM Disaggregated Prefilling 与 KV Connector](https://docs.vllm.ai/en/stable/features/disagg_prefill/)
- [vLLM NixlConnector Usage Guide](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)
- [vLLM KVConnector V1 Interface](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
