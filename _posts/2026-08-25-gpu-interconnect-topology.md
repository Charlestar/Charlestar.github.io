---
layout: post
title: "GPU 互联拓扑：数据怎样从 HBM 走到另一张 GPU 与远端 NIC"
subtitle: "从板内互联、PCIe 层级到跨节点 RDMA，建立可验证的数据路径模型"
date: 2026-08-25 09:00:00 +0800
last_modified_at: 2026-09-03
author: iStar
catalog: true
series: distributed-training
series_order: 10
technology_year: 2000
mathjax: true
tags: [分布式训练, GPU优化]
---

当一个 `AllReduce` 看起来“网络很慢”时，真正拥塞的地方可能根本不在网络：数据也许先跨过了 CPU socket，GPU 与 NIC 之间的 PCIe 事务也许被重定向到 Root Complex，某条 PCIe 链路也许只协商到了较窄的 width，或者本来能够走 NVLink 的 GPU 对实际上使用了 host staging。反过来，`nvidia-smi topo -m` 中看起来很近的两张卡，也不等于应用一定获得了高带宽。

理解 GPU 互联不能从一张产品带宽表开始，而要从一个更朴素的问题开始：

> 某一段 tensor 此刻归谁所有；下一步由谁发起读写；事务经过哪些端点、交换结构和共享截面；完成信号又怎样让下一段 GPU 工作安全地看到结果？

本文沿着一次数据移动的真实方向，依次走过 HBM、GPU 内部数据通路、NVLink/NVSwitch、PCIe switch/Root Complex、NUMA 互联、NIC，以及跨节点的 InfiniBand 或 RoCE fabric。目标不是背诵某一代设备的峰值，而是建立一套不会随着 SKU 更替而失效的分析方法：

```text
先画路径，再判断能力；
先定义带宽口径，再读取数字；
先做单段实验，再解释端到端结果。
```

文中会持续区分两类结论：

- **来源事实**：由 NVIDIA、Linux kernel、PCI-SIG、InfiniBand Trade Association 等官方资料直接说明的接口、约束或语义；
- **工程推导**：基于路径和共享资源做出的性能假设，必须由当前机器上的测量验证，不能拿拓扑标签直接当作结论。

本文的命令与接口范围主要是 Linux、NVIDIA data-center GPU、CUDA/NVML/DCGM、PCIe/NVLink，以及 InfiniBand/RoCE。Windows、AMD GPU、CXL/coherent SoC 与虚拟化平台有各自的枚举和一致性 contract，本文的图模型仍可复用，但不能直接照搬命令或能力结论。文章时间线取 2000 年，是因为本文最早的 scale-out 基础 InfiniBand Architecture 在这一时期形成；PCIe、GPUDirect、NVLink 与 NVSwitch 则在后续年代逐步进入这条数据路径。

## 把机器画成“端点—链路—交换结构”图

### 顶点不只是 GPU

可以把一台训练节点抽象成有向多重图 \(G=(V,E)\)。顶点至少包括：

- GPU 及其 HBM；
- GPU 上可发起或接收传输的执行/复制资源；
- NVLink 端口与 NVSwitch；
- PCIe endpoint、switch、Root Port 与 Root Complex；
- CPU socket、NUMA node 与 socket 间互联；
- NIC/HCA 的 PCIe function 和物理端口；
- 跨节点交换机及其上行、下行端口。

同一对顶点之间可以有多条物理链路，同一条端到端路径也可能经过多种协议。NVLink 是 GPU 互联；PCIe 是 I/O fabric；InfiniBand 与 RoCE 则把流量带出主机。把它们统称为“总线”会掩盖路由、隔离域和完成语义的差异。

边 \(e\in E\) 不应只有一个“带宽”属性，至少还要记录：

```text
capacity(e)       链路在指定方向、指定口径下的容量
latency(e)        传输或转发引入的固定与排队延迟
sharing(e)        哪些流共享该端口、上行或内存控制器
routable(e)       平台与软件是否允许目标事务通过
health(e)         链路当前是否 up，错误计数是否增长
observability(e)  能否从工具看到状态、计数或实际流量
```

这解释了为什么“能枚举”“能访问”“链路是 Up”“跑得快”是四个不同命题。

### Tensor ownership 决定流量方向

路径分析必须带着 tensor ownership。假设 GPU 0 的 HBM 中有 \(D\) 字节，GPU 1 需要一份副本：

```text
初始：
GPU0 HBM  [xxxxxxxx]   owner/source
GPU1 HBM  [        ]   destination

完成：
GPU0 HBM  [xxxxxxxx]
GPU1 HBM  [xxxxxxxx]
```

如果调用的是 bulk copy，某个 DMA/copy mechanism 会搬运数据；如果 GPU 1 上的 kernel 直接 load GPU 0 的 peer memory，则 GPU 1 是事务发起方，远端访问的粒度、缓存行为与 bulk copy 都可能不同；如果是 collective，通信库还可能把 tensor 分 chunk，用 GPU kernel 进行 reduction，再让不同 channel 走不同路径。

所以“GPU 0 发给 GPU 1”只是逻辑语义，不足以判定谁在 PCIe/NVLink 上发起 read 或 write。可靠的描述应包括：

1. buffer 位于哪一张 GPU 或哪一个 host NUMA node；
2. 是 copy、kernel load/store、RDMA read/write，还是 collective；
3. 发起方、目标方与完成通知分别是谁；
4. 路径是否允许 P2P，若不允许会怎样回退；
5. 哪些流在同一物理截面竞争。

### 控制路径与数据路径不要混为一谈

“Direct”不代表 CPU 消失。以 GPUDirect RDMA 为例，CPU 仍可能负责：

- 创建通信对象与 queue pair；
- 注册/映射 GPU memory；
- 提交 work request；
- 轮询或处理中断、completion queue；
- 处理连接、错误和重试。

所谓 direct data path，指 payload 不必先复制到普通 host bounce buffer，再由 NIC 从该 buffer 取走。控制面仍由 host 软件参与。区分这两条路径，才能解释“CPU 占用降低”与“CPU 完全不参与”为何不是同一件事。

## 一次 GPU 到 GPU 复制从 HBM 怎样出发

### HBM 是起点，不是可单独计时的一根链路

HBM 连接 GPU 的内存控制器。kernel、copy engine 或其他数据移动机制要从源 HBM 读取 cache line/transaction，再经 GPU 内部互联进入外部 I/O endpoint；目标侧完成相反过程，最终写入目标 HBM。

一个保守的抽象是：

```text
source HBM
  -> source GPU memory subsystem
  -> source egress / protocol endpoint
  -> NVLink or PCIe fabric
  -> destination ingress / protocol endpoint
  -> destination GPU memory subsystem
  -> destination HBM
```

这里故意没有把每个字节都画成“经过 copy engine”。[CUDA Programming Guide 的多 GPU 章节](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)说明 CUDA 在 peer copy 可用时会利用专用 copy engine 和 NVLink；但实际通信也可能由 SM 上的 kernel 发起 load/store，collective library 也可能运行自己的通信/归约 kernel。内部仲裁与具体 engine 随架构、操作和软件版本变化，不能仅凭 API 名称猜测。

端到端有效带宽的上界因此是多个资源的最小值，而不是 HBM 峰值：

$$
B_{\text{effective}}
\le
\min(
B_{\text{HBM-read}},
B_{\text{source-egress}},
B_{\text{fabric-path}},
B_{\text{destination-ingress}},
B_{\text{HBM-write}}
)
$$

若 source 与 destination 还同时运行计算，HBM、L2、copy/SM 资源竞争会进一步改变结果。这是工程推导，必须通过“空闲链路测试”和“计算重叠测试”分别验证。

### 路径 A：直连 NVLink

两张 GPU 若由一条或多条 NVLink 直接连接，peer traffic 可以沿这些 GPU 端口移动，而不需要把 payload 绕入 host memory。多条 link 连接同一对 GPU 时，软件和硬件可能把足够大的传输分散到多个 link；但“有 \(k\) 条 link”只说明物理连接关系，并不自动证明某次操作用满 \(k\) 条 link。

需要分开看四层：

1. **拓扑关系**：GPU A 与 GPU B 存在多少条直接 NVLink；
2. **P2P 能力**：运行时是否允许所需的 read、write 或 atomics；
3. **链路状态**：端口是 Up、Down、Disabled 还是不受支持；
4. **实际流量**：测试期间各 link 的 tx/rx 计数是否增长，应用有效带宽是多少。

[DCGM 的 Topology and NVLink 文档](https://docs.nvidia.com/datacenter/dcgm/latest/learn/core-services/topology-and-links.html)明确把这些视图分开：topology 描述关系，不保留资源、不测带宽；link status 是瞬时状态；counter 是被动证据；diagnostic 才是主动测试。这个边界对所有拓扑工具都适用。

### 路径 B：经 NVSwitch 交换

NVSwitch 系统不是“每两张 GPU 之间各有一根线”，而是 GPU 的多个 NVLink 端口进入交换 fabric：

```text
GPU0 HBM -> GPU0 NVLink ports -> NVSwitch fabric
                                      |
GPU1 HBM <- GPU1 NVLink ports <-------+
```

交换 fabric 可以为更多 GPU 对提供高连通性，并让多组传输并发。然而，端点注入/接收能力、GPU 到 switch 的 link 集合、switch 内部路径及共享截面仍会限制总吞吐。不能把“所有 GPU 可互访”推导成“任意数量的 GPU 对都能同时获得单对峰值”。

不同代际的 NVSwitch fabric 配置方式不完全相同。[NVIDIA Fabric Manager 文档](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html)说明，部分 NVSwitch 平台由 Fabric Manager 与内核驱动负责初始化、配置和监控；新平台也可能使用不同的 fabric 管理组件。若交换 fabric 尚未正确初始化，CUDA 应用可能失败。DCGM 负责观测，而不是替代 fabric manager 把链路拉起。

因此排障时应依次问：

```text
设备是否被发现？
fabric 管理服务是否完成初始化？
每个预期端口是否 Up？
错误计数是否在测试窗口内增长？
业务流量是否真的经过这些端口？
端到端测试是否达到该平台自己的健康基线？
```

### 路径 C：经 PCIe P2P

没有可用 NVLink 的 GPU 对，仍可能经 PCIe 进行 peer-to-peer：

```text
GPU0 endpoint
  -> PCIe switch upstream/downstream fabric
  -> GPU1 endpoint
```

如果两个 endpoint 位于同一 PCIe switch hierarchy，且路由控制允许，Transaction Layer Packet 可以在 switch 内转发，不必进入 CPU memory。若事务到达 host bridge/root port，路径可能在 Root Complex 内 hairpin，也可能需要穿过 CPU SoC，甚至跨 socket；这时支持性和性能都更依赖平台。

[Linux PCI P2PDMA 文档](https://docs.kernel.org/driver-api/pci/p2pdma.html)给出了重要边界：PCIe TLP 在到达 host bridge/root port 之前的路由有明确规则；跨 hierarchy domain 的转发并非 PCIe 规范普遍保证，Linux 对无法证明安全的组合采取保守策略。因此，“两块设备都插在同一台服务器”不是 P2P 可路由性的充分条件。

### 路径 D：退回 host staging

当 peer access 不可用或通信库判断 direct path 不合适时，常见回退是：

```text
GPU0 HBM
  -> PCIe
  -> pinned host buffer
  -> PCIe
  -> GPU1 HBM
```

这条路径至少多了一次落地与再读取，还可能占用 CPU memory controller、Root Complex、NUMA interconnect，并受 pinned memory、copy scheduling 和同步影响。它不一定“完全不可用”，但应与 direct P2P 分开测量。否则某些 GPU 对的异常会被平均值掩盖。

一个实用的验证方法是控制变量：对同一 GPU pair、同一消息大小、同一方向，分别测 P2P enabled 与显式禁用/不可用的路径；再结合 link counter 或 profiler 判断差异来自选路，而不是仅看总时间猜测。

## NVLink 与 NVSwitch 不是统一性能标签

### 同名互联跨代际并不等价

NVLink 的链路数、每 link 能力、方向定义、端口布局及平台布线都会随 GPU 和整机 SKU 改变；NVSwitch 的代际、层数和 fabric 管理方式也会变化。因此本文不列一张“NVLink 版本—带宽”表。这样的表很快过时，而且常把单向、双向总和、每 link 与每 GPU aggregate 混在一起。

若必须引用数字，应写全限定条件，例如：

```text
厂商：
GPU / NVSwitch / 整机 SKU：
互联代际：
每方向还是双向合计：
每 link、每 GPU 端点还是整机 aggregate：
line rate、协议 payload 还是应用有效带宽：
来源文档版本和日期：
```

缺少其中任何一项，数字都不适合用来判断当前机器。

### NVLink 可见不等于应用走 NVLink

`nvidia-smi topo -m` 中的 `NV#` 表示该 GPU 对的已知关系穿过一组 bonded NVLinks；它不是一次动态路由 trace。应用是否使用这些 link，还取决于：

- CUDA peer capability 与 access 是否启用；
- buffer 类型与分配方式；
- 操作是 peer copy、remote load/store 还是 library collective；
- 通信库选择的 algorithm、protocol 和 channel；
- link/fabric 当前健康状态；
- stream ordering 与并发工作；
- 虚拟化或隔离环境暴露出的拓扑。

因此，拓扑图是提出假设的依据，不是完成验证的证据。

## PCIe 层级决定事务能否在本地转弯

### Endpoint、Switch、Root Port 与 Root Complex

[PCI-SIG 对 PCI Express Base Specification 的说明](https://pcisig.com/specification-overview/pci-express-base)把 PCIe 定义为包含 architecture、interconnect attributes、fabric management 与 programming interface 的互联体系。分析 GPU/NIC 路径时，可以用以下简化层级：

```text
CPU / SoC Root Complex
  |
  +-- Root Port
        |
        +-- PCIe Switch
             +-- GPU endpoint
             +-- NIC endpoint
             +-- another switch ...
```

- **Endpoint** 发起或响应 PCIe transaction；
- **PCIe switch** 按地址与路由规则在端口间转发 TLP；
- **Root Port** 把一个 PCIe hierarchy 接入 Root Complex；
- **Root Complex/host bridge** 连接 CPU/SoC、memory 与一个或多个 PCIe hierarchy。

真实机器可能有多级 switch、多 root port、多 host bridge，或者把互联集成进 accelerator module。简图是推理工具，最终仍要回到 BDF 与平台文档。

### 同一 Switch 下也要检查共享上行

GPU 与 NIC 都在同一 PCIe switch 下，通常有利于 P2P 局部转发，但不代表任意流量都不经过上行。要看事务目标和 ACS 设置：

```text
GPU -> NIC       可能在 switch 内转发
GPU -> host RAM  必须走向 Root Complex / memory controller
GPU -> other RC  可能穿过 SoC 或不受支持
```

而且两个下行端口即使各自协商到完整 width，switch 的上行或内部共享资源仍可能被其他设备竞争。拓扑上的“一跳”不能代替交换芯片和整机的容量模型。

### Root Complex 不是一块无限快的黑盒

一旦路径触及 Root Complex，应进一步展开：

```text
PCIe Root Port
  -> CPU I/O die / host bridge
  -> memory controller or internal fabric
  -> [optional socket interconnect]
  -> another host bridge / Root Port
```

同一 CPU socket 内不同 Root Port 之间的 P2P 是否支持、怎样 hairpin，由 CPU/平台决定；跨 socket 还叠加 UPI、Infinity Fabric 或其他 SMP interconnect。Linux 文档强调跨 PCI hierarchy 的 routing 不是 PCIe 规范普遍保证的能力，因此不能用“物理上存在连线”代替厂商 qualification。

### Negotiated speed 与 width 比槽位规格更重要

“插在 x16 槽”只描述机械或设计能力，不代表当前 link 以目标代际、目标 width 运行。链路可能因为 riser、retimer、信号质量、固件策略或训练失败而降速/降宽。应查看 endpoint 及其上游 port：

```bash
lspci -D
lspci -D -t
lspci -s <BDF> -vv
```

重点比较 `LnkCap` 与 `LnkSta`，并沿上游 bridge 逐级检查。实际 path capacity 由最窄、最慢且被共享的那一段决定。不要只看 GPU endpoint 的一行。

## NUMA 把“同一台主机”分成多个局部世界

### PCIe 设备具有 CPU 与 memory affinity

多 socket 系统中，每个 GPU/NIC 通常更靠近某个 Root Complex、CPU 集合与 host memory node。`nvidia-smi topo -m` 同时展示 GPU/NIC relationship 和 CPU/memory affinity，正是因为 host 线程和 staging buffer 的放置会影响路径。

Linux 暴露的关键入口包括：

```bash
# 设备的 PCI NUMA node；-1 表示内核不知道，而不是 node 0
cat /sys/bus/pci/devices/0000:65:00.0/numa_node

# 每个 NUMA node 的 CPU 列表与距离
cat /sys/devices/system/node/node0/cpulist
cat /sys/devices/system/node/node0/distance

# 进程和内存 NUMA 状态
numactl --hardware
numastat -p <pid>
```

[Linux sysfs PCI ABI](https://www.kernel.org/doc/html/latest/admin-guide/abi-testing.html)说明 `numa_node` 来自 PCI device state，初值通常来自 ACPI `_PXM` 或类似固件信息；`-1` 的含义是 unknown。盲目把它写成某个 node 会使内核带上 firmware workaround taint，正确做法首先是核对固件与整机厂商信息。

### CPU affinity 影响的不只是 host staging

即使 payload 走 GPUDirect RDMA，通信仍有控制面工作：post work request、CQ polling、proxy/helper thread、连接管理、metadata 和 page registration。若这些线程运行在远端 socket，可能引入额外 NUMA latency，甚至让流量经过 socket interconnect。

一个更完整的 rank binding 是三元组：

$$
\text{placement(rank)}
=
(\text{GPU BDF},\ \text{NIC/HCA port},\ \text{CPU set})
$$

只设置 `CUDA_VISIBLE_DEVICES`，不绑定 CPU 和 NIC，无法保证可重复。

### HBM 与 host NUMA memory 不能混称

“GPU 靠近 NUMA node 0”通常表示 PCIe/CPU affinity，不代表 GPU HBM 就是普通 Linux node 0 内存。某些 CPU-GPU coherent platform 或 GPU NUMA 暴露模式会提供更丰富的语义，但必须按平台文档解释。通用排障中，至少分清：

- GPU device memory/HBM；
- pinned host memory 与 pageable host memory；
- host NUMA node；
- GPU 最近的 CPU/memory affinity；
- 特定平台额外暴露的 GPU NUMA ID。

把这些统称“NUMA 内存”，会导致错误的 placement 假设。

## 从 GPU 到 NIC 有 direct 与 staged 两条主路径

### Host-staged 路径

没有 GPUDirect RDMA，或当前组合不满足其条件时，发送路径常近似为：

```text
GPU HBM
  -> GPU PCIe endpoint
  -> Root Complex
  -> pinned host buffer
  -> NIC DMA read
  -> NIC wire port
```

接收方向则是 NIC DMA 写 host buffer，再由 GPU copy 进入 HBM。具体实现可以 pipeline 多个 chunk，CPU 也不一定逐字节 copy，但 payload 需要在 host memory 落地。此时 GPU↔host 和 host↔NIC 两段会竞争 PCIe、host memory bandwidth 与 NUMA fabric。

当 GPU 与 NIC 分属不同 socket，路径可能变成：

```text
GPU -> local Root Complex -> socket interconnect
    -> remote memory / remote Root Complex -> NIC
```

这正是 NIC affinity 和 host buffer NUMA binding 必须一起看的原因。

### GPUDirect RDMA 路径

[NVIDIA GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)定义了 GPU 与第三方 PCIe peer device 之间的 direct data path。经典 PCIe 模型下，GPU memory 的页被注册并映射给 peer device，NIC 可对相应 PCIe/BAR 地址发起 read/write：

```text
send-like direction:
GPU HBM -> GPU PCIe peer mapping -> NIC DMA read -> wire

receive-like direction:
wire -> NIC -> NIC DMA write -> GPU PCIe peer mapping -> GPU HBM
```

这是逻辑示意，不规定所有 NIC、verb 和 platform 都用完全相同的读写方向或内部 engine。重要的是 payload 不必先 bounce 到普通 host buffer。

这张图还特意限定在经典“离散 GPU + PCIe peer NIC”模型。Grace Hopper/Grace Blackwell、DirectNIC、多节点 NVLink 等新平台可能把 CPU-GPU、GPU-NIC 或跨节点互联组织成不同的数据通路，不能硬套传统双 socket PCIe 服务器的结论。[NVIDIA Grace Blackwell with ConnectX-8 GPUDirect RDMA 指南](https://docs.nvidia.com/multi-node-nvlink-systems/grace-blackwell-cx8-gpudirect-rdma-guide/index.html)就是一个平台专用例子：部署者应使用该整机的 reference topology、ACS/IOMMU 要求和验证程序，而不是使用别的代际经验。

建立 direct path 通常还需要：

- GPU 与 NIC/driver 组合受支持；
- GPU memory 可被 pin/register，并有正确生命周期；
- 使用 DMA-BUF 或 `nvidia-peermem` 等受支持机制；
- PCIe topology 与 peer routing 合格；
- IOMMU/ACS/虚拟化配置满足该平台要求；
- 通信库实际选择 GDR transport；
- completion 与 CUDA stream/memory ordering 正确衔接。

[NVIDIA GPU Operator 的 GPUDirect 说明](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html)列出了 DMA-BUF 与 legacy `nvidia-peermem` 两条内核支持路径及各自前提。部署时应以当前 driver、kernel、GPU、NIC 与平台 qualification 为准，不应只检查模块是否存在。

### “RDMA completion”不自动等于 GPU kernel 可安全读取

跨设备内存可见性需要明确同步。[GPUDirect RDMA 的 Synchronization and Memory Ordering 章节](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html#synchronization-and-memory-ordering)指出，CUDA 的同步与 work submission API 才为 GPUDirect RDMA 操作提供所需 ordering；GPU kernel 与第三方设备对同一 memory 的并发访问可能构成 data race。

因此，正确的数据流至少要定义：

1. NIC 何时报告 work completion；
2. host 或 device control path 何时把该完成转换为 CUDA 可见的依赖；
3. consumer kernel 进入前需要什么 stream/event/fence；
4. buffer 在最后一个 DMA 完成前不能 unpin、free 或复用；
5. error completion 如何阻止下游读取半成品。

“网卡已经收完”与“这个 stream 上的 kernel 可以消费”之间，必须有受支持的 ordering bridge。

## 数据出节点后：InfiniBand 与 RoCE

### RDMA 语义与物理网络是两层

RDMA 描述 remote memory 操作与 queue/completion 等语义；InfiniBand 和 RoCE 是承载这些语义的 fabric 选择。[IBTA 的 specification FAQ](https://infinibandta.org/ibta-specification/)说明，InfiniBand 是面向高性能数据中心的 switched fabric，RoCE 则让 RDMA 运行在以太网二层或三层网络上，RoCE 标准由 IBTA 定义在 InfiniBand Architecture 体系内。

不要把“使用 verbs”直接等同于“物理上是 InfiniBand”。同样的 verbs 风格应用可能绑定 InfiniBand HCA，也可能绑定 RoCE-capable Ethernet port。

### InfiniBand 端到端路径

一个典型的 GDR 发送路径可以画成：

```text
GPU A HBM
  -> local GPU/NIC PCIe P2P path
  -> local HCA port
  -> InfiniBand link
  -> leaf/spine switches and selected route
  -> remote HCA port
  -> remote GPU/NIC PCIe P2P path
  -> GPU B HBM
```

其中至少有三个独立拥塞域：

1. source GPU 到 local HCA 的 PCIe/accelerator I/O path；
2. network fabric 的端口、route 与共享截面；
3. remote HCA 到 destination GPU 的路径。

只测 host-memory `ib_write_bw` 可以验证 NIC 与 network，却不能证明两端 GDR path 健康；只测本地 GPU-NIC P2P 也不能证明交换 fabric 无拥塞。

### RoCE 端到端路径

RoCE 把 RDMA transport 带到 Ethernet fabric。原始 RoCE 面向二层域；[IBTA 对 RoCEv2 的公告](https://www.infinibandta.org/ibta-announces-new-roce-specification/)说明 RoCEv2 增加了基于 IP 的三层可路由能力。端到端路径仍有 GPU↔NIC 两段，但中间经过 Ethernet switch、VLAN/IP route、队列与拥塞控制：

```text
GPU -> NIC/CNA -> Ethernet L2/L3 fabric -> NIC/CNA -> GPU
```

RoCE 的问题不能只用“链路 Up”解释。PFC、ECN、queue mapping、MTU、路由不对称、拥塞与丢包恢复策略都会影响 tail latency 和 throughput；具体启用哪些机制应服从网络设计，不能照搬单机 GPU 参数。

### 两端 topology 都要对称检查

跨节点测试常只检查发送端 GPU 是否靠近 NIC，却忽略接收端。正确路径是两个局部路径加一个网络路径：

$$
P_{\text{end-to-end}}
=
P_{\text{GPU}_s\rightarrow\text{NIC}_s}
\oplus
P_{\text{network}}
\oplus
P_{\text{NIC}_d\rightarrow\text{GPU}_d}
$$

这里的 \(\oplus\) 表示串接而非简单加法。端到端容量近似受最小截面限制，延迟则会叠加固定开销、序列化和排队。两端任何一侧选错 NIC，都可能让网络指标看似正常、GPU buffer 测试却很差。

## Scale-up 与 Scale-out 是边界模型

### Scale-up：扩大一个紧耦合计算域

工程上常把节点内或一个紧耦合 accelerator fabric 内的互联称为 scale-up。它追求：

- 较低延迟；
- 高 GPU 注入/接收带宽；
- 细粒度 peer access 或 load/store 能力；
- 对 collective、tensor/sequence parallel 的高频通信友好。

NVLink、NVSwitch 和 PCIe P2P 都可能参与 scale-up，但能力不同。NVLink 也并不天然等于 cache-coherent shared memory；一致性和原子语义必须按 CUDA 与具体平台说明判断。

### Scale-out：把多个计算域通过网络连接

Scale-out 通常以 NIC/HCA 和 network switch 扩展节点数，数据被分组、路由并由 transport 管理。它需要处理：

- 网络寻址与 route；
- congestion、queue 与多租户隔离；
- link/port failure；
- 跨节点进程编排；
- 较大的 latency-bandwidth trade-off。

Data Parallel、Expert Parallel 往往更容易跨 scale-out 边界，Tensor Parallel 则倾向留在更强的 scale-up 域；这只是常见映射，不是算法定律，应由通信频率、消息大小和实际拓扑共同决定。

### 新平台会模糊边界

多节点 NVLink 或 CPU-GPU coherent interconnect 会让“节点内等于 scale-up、节点间等于 scale-out”不再严格成立。因此更稳妥的定义是：

- scale-up 域由较紧耦合的 accelerator fabric 与地址/访问语义界定；
- scale-out 域由可扩展网络、路由和 transport failure boundary 界定。

文章后续仍使用这两个词，但不把机箱边界当作绝对技术边界。

## 容量、超售率与有效带宽口径必须先说清楚

### Per-link bandwidth：一条链路、一个方向

最基础的口径是单条物理 link 在某一方向的速率。对 lane-based interconnect，可抽象为：

$$
B_{\text{link,payload}}
\le
R_{\text{lane}}
\times N_{\text{lane}}
\times \eta_{\text{encoding}}
\times \eta_{\text{protocol}}
$$

其中 line rate、编码效率和协议开销不是应用 payload。若厂商写“bidirectional aggregate”，还可能把两个方向相加。比较前必须统一：

- bit/s 还是 byte/s；
- GB/s 还是 GiB/s；
- 单向还是 tx+rx；
- line rate 还是 payload；
- 单 link 还是多个 bonded links。

### Endpoint aggregate：一张 GPU/NIC 的总注入或接收能力

若一张 GPU 有多条外部 link，可以定义理论 aggregate 为各 link 容量之和：

$$
B_{\text{endpoint,theory}}^{\text{out}}
=
\sum_{e\in E_{\text{out}}} C_e
$$

但应用能达到的 aggregate 还受 HBM、内部 crossbar、copy/SM engine、协议、destination 和 route 限制。多 link 之和只有在流量能被均匀分配、链路不共享更窄上行且端点能持续注入时才有意义。

### Bisection bandwidth：把通信端点切成两半的最小截面

对于同时发生的多对通信，单 link 峰值通常不够。前文的 \(V\) 还包括交换机、Root Port 等内部顶点，不能按全部顶点数量机械平分。先定义需要通信的终端集合 \(T\subseteq V\)，例如参与本次 collective 的 GPU 或节点；对任意顶点划分 \(S\subset V\)，只要求两侧终端数量近似相等，内部转发顶点可以落在任一侧。

对有向、全双工链路，先只计算一个方向的 cut 容量：

$$
C^{\rightarrow}(S,\bar S)
=
\sum_{e\in\delta^{+}(S)} C_e
$$

于是从 \(S\) 侧终端流向另一侧的 bisection bandwidth 可写成：

$$
B_{\text{bisect}}^{\rightarrow}
=
\min_{S\subset V,\ |S\cap T|\approx |T|/2}
C^{\rightarrow}(S,\bar S)
$$

反向容量应以 \(C^{\rightarrow}(\bar S,S)\) 单独计算；只有明确报告 `TX+RX aggregate` 时，才把两个方向相加。这个量能解释：每个 GPU 到本地交换机的 link 都很快，但大量 GPU 同时跨组通信时仍在共同 uplink 或 spine layer 堵塞。真实网络还要考虑方向、ECMP 路由、故障降级和 oversubscription，公式只是容量模型。

### Oversubscription：需求峰值与共享截面容量之比

本文约定：

$$
\rho_{\text{over}}
=
\frac{\sum B_{\text{edge-facing peak}}}
{B_{\text{uplink/cut available}}}
$$

当 \(\rho_{\text{over}}>1\) 时，所有下行不能同时以各自峰值穿过该截面。有些资料使用相反的“uplink:downlink”写法，看到 `2:1` 时必须先确认定义。

分子与分母必须使用相同方向、相同的 line-rate 或 payload 口径；分子也只统计确实需要同时穿过该 cut 的需求，不能把留在 cut 同侧的流量算进去。

Oversubscription 可能出现在：

- PCIe switch 多个下行共享一个上行；
- 多 GPU 共享一个 NIC port；
- 多 NIC 共享同一个 leaf uplink；
- leaf/spine 之间的 fabric；
- 多租户共享的队列或 traffic class。

### Effective、algorithm 与 bus bandwidth 不是同一个数字

应用最关心的通常是：

$$
B_{\text{effective}}
=
\frac{D_{\text{useful}}}{T_{\text{wall}}}
$$

collective benchmark 还可能报告 algorithm bandwidth 与经过算法因子换算的 bus bandwidth。[NVIDIA nccl-tests 的性能说明](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)定义 `algbw=S/t`，并按 collective 类型把它换算为 `busbw`，用于帮助解释硬件利用率。这个 `busbw` 是规范化指标，不是某个物理端口计数器，也不一定等于线缆速率。

报告结果时应同时保留 operation、rank 数、每 rank size、算法、in-place/out-of-place、时间统计方式，以及指标公式。

## 拓扑发现先建立稳定身份

### GPU index 会变，UUID 与 PCI BDF 才能对表

`GPU0`、`cuda:0`、local rank 0 都是某一层软件的临时编号。容器的 device visibility、启动顺序、MIG 配置或硬件维护都可能改变编号。做拓扑快照时，至少关联：

```text
framework rank
  <-> CUDA visible index
  <-> GPU UUID（或 MIG UUID）
  <-> physical GPU UUID
  <-> PCI domain:bus:device.function (BDF)
  <-> physical slot / module position
  <-> nearest NIC BDF and RDMA device/port
```

可以先取得 GPU 身份：

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv
nvidia-smi -L
```

MIG 环境需要再加一层：多个 logical device 可以共享同一块物理 GPU 的 BDF，各自使用 MIG UUID，并关联 GPU Instance ID 与 Compute Instance ID。此时 `--query-gpu` 主要给出物理 GPU 清单，不能单独恢复 rank 到 MIG device 的映射；快照应保存 `physical GPU UUID/BDF -> GI -> CI -> MIG UUID -> CUDA visible index`，再与进程实际可见设备对表。

PCI BDF 最好保留 domain，例如 `0000:65:00.0`，不要只记 `65:00.0`；多 domain 或复杂虚拟化环境中，省略 domain 可能产生歧义。

### 物理 slot 也不能替代 BDF

slot label 便于现场运维，但固件枚举变化后 BDF 可能变化；BDF 适合本次系统快照，却也不是跨换卡操作永久不变的设备身份。因此资产记录最好同时保存 GPU serial/UUID、BDF、slot 与平台拓扑图，启动时再生成 rank mapping，而不是把 `rank 3 -> 65:00.0` 永久写死。

## 正确读取 nvidia-smi topo

### `topo -m` 展示的是关系矩阵

[NVIDIA System Management Interface 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html#topology)给出的常见标记是：

| 标记 | 文档定义的路径关系 |
| --- | --- |
| `X` | 自身 |
| `NV#` | 经过由若干 NVLinks 组成的连接 |
| `PIX` | 最多一个 PCIe switch/bridge |
| `PXB` | 多个 PCIe switch/bridge，但不经过 host bridge |
| `PHB` | 经过 PCIe host bridge |
| `NODE` | 经过同一 NUMA node 内 host bridge 间互联 |
| `SYS` | 经过 PCIe，并跨 NUMA node 的 SMP interconnect |

矩阵还会给 GPU 的 CPU affinity、memory affinity，以及识别到的 NIC/data-direct device 候选。候选项未必是当前可用的 RDMA port，仍要用 RDMA device、port、driver 与 link layer 复核。矩阵回答“系统认为两者怎样连接”，不回答：

- 当前 link 是否健康；
- 实际 negotiated width/speed；
- P2P read/write 是否被允许；
- 应用是否选择这条路径；
- 同时有多少流竞争；
- 此时真实带宽和 tail latency。

尤其不要把 `NV# > PIX > PXB > PHB > NODE > SYS` 当作跨所有平台固定、可量化的性能排序。标签代表 traversed components；不同 CPU、switch、link generation、traffic direction 与并发负载会改变结果。排序只能作为待测假设。

### 用 `-mp` 分离 PCIe 关系

如果默认矩阵优先显示 NVLink，排查 PCIe fallback 时还应查看：

```bash
nvidia-smi topo -m
nvidia-smi topo -mp
nvidia-smi topo -gpu
nvidia-smi topo -nic
```

`topo -mp` 排除 NVLink，显示 PCI-only relationship。它能回答：“若 NVLink 不可用，PCIe 层级会怎样连接这对设备？”这对设计 fallback 基线很有用。

`-gpu` 与 `-nic` 是较新的 NVSMI 子命令。先运行 `nvidia-smi topo -h` 核对当前驱动支持的语法；旧版本不支持时，保留 `-m/-mp` 矩阵并用 PCI/RDMA 枚举补全身份映射，不要把“unknown option”误判为拓扑故障。

### 用 `-p2p` 查 capability，不要拿 `-m` 代替

`nvidia-smi topo -p2p` 可以按 capability 输出 GPU pair 状态：

```bash
nvidia-smi topo -p2p p   # PCIe P2P
nvidia-smi topo -p2p n   # NVLink P2P
nvidia-smi topo -p2p r   # read
nvidia-smi topo -p2p w   # write
nvidia-smi topo -p2p a   # atomics
```

不同 capability 可能得到不同矩阵；“可以 copy”不能自动推导“支持所需的 remote atomics”。而且 [NCCL GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)特别提醒：即使 `topo -p2p` 显示 `OK`，Linux bare-metal IOMMU 或 ACS 配置仍可能导致 peer access 异常、性能下降甚至 hang。capability probe 是必要条件检查，不是系统 qualification 的终点。

## 用 PCI tree 与 sysfs 还原层级

### `lspci -t` 先看树，`lspci -vv` 再看每条边

推荐保留以下只读输出：

```bash
lspci -Dnn
lspci -D -tv
lspci -s <GPU_BDF> -vv
lspci -s <NIC_BDF> -vv
```

分析时从 endpoint 沿父 bridge 向上走，标出：

- 每级 bridge/root port 的 BDF；
- `LnkCap` 与 `LnkSta`；
- 是否存在 ACS capability/control；
- endpoint 是否落在同一 switch、同一 root port、同一 root complex；
- AER counter 或 kernel log 是否在测试窗口增长。

`lspci -t` 是枚举树，不是流量 trace。两个 endpoint 在树上有共同祖先，不代表事务必然在最低共同祖先转弯；ACS 与平台路由规则仍可能把它送向上游。

### sysfs 给脚本一个可复现入口

Linux 为每个 PCI device 建立 sysfs 目录：

```bash
GPU_BDF=0000:65:00.0
readlink -f /sys/bus/pci/devices/$GPU_BDF
cat /sys/bus/pci/devices/$GPU_BDF/numa_node
cat /sys/bus/pci/devices/$GPU_BDF/current_link_speed
cat /sys/bus/pci/devices/$GPU_BDF/current_link_width
cat /sys/bus/pci/devices/$GPU_BDF/max_link_speed
cat /sys/bus/pci/devices/$GPU_BDF/max_link_width
```

并非所有 kernel 都暴露完全相同的属性，所以采集脚本要记录“文件不存在”，不要用零值代替 unknown。`readlink -f` 展开的目录层级可帮助定位父 bridge；生产脚本则应避免依赖某个固定层数。

Linux 还为支持 AER 的设备提供错误统计；[sysfs ABI 文档](https://www.kernel.org/doc/html/latest/admin-guide/abi-testing.html)提醒，错误可能由 link partner 观察并计在上游 port，而不一定出现在真正发起问题的 endpoint 上。排障要同时比较 endpoint 和相邻 bridge 的计数增量。

### 发现 RDMA device、netdev 与 PCI function 的映射

NIC 可能有多个 port、多个 PCI function，RDMA device name 也不等于 Linux netdev name。应建立：

```text
PCI BDF
  <-> RDMA device (例如 mlx5_N)
  <-> RDMA port
  <-> netdev
  <-> InfiniBand LID/GID 或 RoCE GID
  <-> physical switch port / rail
```

常用只读命令包括：

```bash
rdma link show
ibv_devices
ibv_devinfo
ibdev2netdev
ip -d link show
ethtool -i <netdev>
```

工具是否安装取决于发行版；命令输出要与 `/sys/class/infiniband/<device>/device`、`/sys/class/net/<netdev>/device` 的 symlink 交叉核对。不要根据 `mlx5_0` 的编号猜它靠近 GPU0。

## NVLink/NVSwitch 发现还需要状态与计数

### topology、status、counter、diagnostic 分四层保存

`nvidia-smi topo` 提供关系；DCGM 可以进一步观察 link entity：

```bash
dcgmi discovery --list
dcgmi topo --gpuid 0
dcgmi nvlink --link-status --show-entity-ids
dcgmi nvlink --errors --gpuid 0
```

在有 NVSwitch 的系统上，还要核对 Fabric Manager 或该代平台对应 fabric service 的状态和日志。链路 port 编号是其 parent GPU/NVSwitch 的局部编号；不要从另一台 SKU 猜测端口含义。

`--errors` 的聚合层级和字段语义也随 GPU 代际及 DCGM 版本变化：Hopper 及更早平台常见 per-link 输出，较新平台可能给 GPU-level error detail。采集前应按已安装 DCGM 的 field catalogue 确认字段是 per-link 还是 GPU-level、COUNT 还是 RATE，并保留 unsupported/N/A；空字段不能按零处理。

### 错误总数要看时间窗口中的增量

许多 link error counter 是累计值。一个非零值可能发生在数月前，单次截图不能证明当前故障。正确做法是：

```text
T0: 保存 topology、link state、counter
    运行唯一、可复现的测试
T1: 再次保存 link state、counter
    计算每个 endpoint/port 的 delta
```

若某个 counter 增长，仍需结合 peer endpoint、platform log、温度/功耗状态和测试流量解释；若不增长，也只能说明该计数器没有观察到对应错误，不能证明性能正常。

## P2P capability 与实际 routing 是两张表

### CUDA capability probe 是有方向的

[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#multi-device-peer-to-peer-transfers-and-memory-access)规定可以用 `cudaDeviceCanAccessPeer()` 查询设备能否访问另一个设备的 memory，并用 `cudaDeviceEnablePeerAccess()` 启用 peer access。启用关系由“访问方 device”指向“被访问方 device”，应分别检查 \(i\rightarrow j\) 与 \(j\rightarrow i\)。

一个完整的 capability matrix 不是对角对称表的假设，而是实测：

$$
P_{ij}^{(c)}
=
\begin{cases}
1,& \text{device } i \text{ 具备 capability } c \text{ 到 } j\\
0,& \text{otherwise}
\end{cases}
$$

其中 \(c\) 可以是 copy/read/write/atomic 等应用实际需要的能力。若软件只测试一个方向，然后默认反向等价，可能在异构或受限环境中埋下错误。

### P2P copy 与 remote load/store 是不同工作负载

`cudaMemcpyPeerAsync` 适合测 bulk copy，可能利用专用 copy engine；kernel 直接访问 peer pointer，则由 SM 发起远端 load/store，访问粒度和 outstanding transaction 不同；NCCL 等库可能用 SM kernel 在搬运过程中做 reduction。NIC 的 GPUDirect DMA 又是 RNIC 发起 PCIe peer transaction，不由 GPU copy engine 搬运。

因此实验至少分三类：

| 类型 | 事务发起/执行模型 | 回答的问题 |
| --- | --- | --- |
| peer bulk copy | CUDA copy path，可能使用 copy engine | 大块连续复制上限 |
| peer kernel access | SM load/store | 远端访问与通信 kernel 行为 |
| GPU↔NIC GDR | NIC DMA + GPU peer mapping | 网络 direct buffer 路径 |

不能用第一类结果为后两类直接背书。

### Stream ordering 决定“数据什么时候算到达”

同步 `cudaMemcpyPeer` 与异步 copy 的 ordering 不同。CUDA 文档说明，不同 device 有各自 default stream；跨 device event 可以通过 `cudaStreamWaitEvent()` 建立依赖。异步 peer copy 可以与其他 stream 的 kernel/copy 重叠，所以测量与正确性都必须明确：

- producer kernel 属于哪个 device/stream；
- copy 排在哪个 stream；
- destination consumer 等待什么 event；
- 计时 event 是否与被测 device/stream 匹配；
- benchmark 是否在读表前同步了所有工作。

若计时只覆盖 enqueue，得到的是提交开销，不是数据移动时间；若 consumer 未等待 completion，则问题是数据依赖错误，不是互联“偶发错误”。

### 通信库可能主动绕开“最短”路径

topology-aware library 会综合 message size、collective algorithm、channel、NIC 数量、rail、GPU direct 能力与并发，实际 route 不必等于人眼认为的最少 hop。某些节点内流量可能走 shared memory/host path，某些跨节点 collective 会先在本地聚合再进 NIC，某些多 rail algorithm 会刻意选择并行路径。

验证 actual routing 时应结合：

- library debug log 中选择的 transport、channel、NIC；
- profiler 的 kernel/copy timeline；
- NVLink、PCIe/NIC port counter 的测试窗口增量；
- 对照实验中显式限制某一 transport 后的变化。

环境变量适合用来做受控实验，不宜长期堆成“调优秘方”。升级库版本后，内部选择策略可能改变。

## NIC affinity：最近的 NIC 只是候选

### “最近”至少包含四层含义

给 GPU 选择 NIC 时，常说“选最近的”。至少要问：

1. GPU 与 NIC 是否在同一 PCIe switch 或 root complex；
2. direct peer routing 是否被平台和 driver 支持；
3. NIC port 接到哪个 network rail、leaf 与 traffic class；
4. 为它服务的 CPU/helper thread 是否 NUMA-local。

本地 PCIe hop 最少的 NIC 若接到了拥塞 rail，端到端可能不如稍远但网络路径健康的 NIC。反过来，仅看网络 ECMP 也会忽略 GPU↔NIC 的跨 socket 代价。

可以把 rank \(r\) 绑定到 NIC \(n\) 的估价写成多目标函数。不同量纲不能直接相加，因此先把观测映射为同方向、无量纲的归一化 penalty：

$$
\operatorname{cost}(r,n)
=
w_p \hat L_{\text{PCIe}}(r,n)
+w_s \hat \rho_{\text{shared}}(r,n)
+w_n \hat L_{\text{network}}(n)
+w_c \hat L_{\text{CPU-NUMA}}(r,n)
$$

其中 \(\hat L\) 越大表示归一化延迟 penalty 越高，\(\hat\rho_{\text{shared}}\) 越大表示共享截面的预计竞争越强。归一化基线和权重随 workload 改变：小消息对 latency 更敏感，大 collective 更关心共享截面和 path diversity。这是工程模型，不是厂商公式，必须用当前集群的测量标定。

### Rank mapping 要配合并行组

假设每节点 8 GPU、2 NIC，不应只按 local rank 顺序轮流绑 NIC。先看通信组：

- Tensor/Context Parallel 通信频密，优先放在强 scale-up 子图；
- Data Parallel 主要跨节点，可让各本地代表 rank 靠近对应 NIC；
- Expert Parallel 的 all-to-all 既需要本地交换，也可能打满跨节点 bisection；
- Pipeline Parallel 的相邻 stage 应尽量避免跨最弱 cut。

一个抽象映射例子：

```text
node A                           node B

GPU0--GPU1--GPU2--GPU3          GPU0--GPU1--GPU2--GPU3
  \ local group /                \ local group /
   NIC0 ===== rail 0 ============ NIC0

GPU4--GPU5--GPU6--GPU7          GPU4--GPU5--GPU6--GPU7
  \ local group /                \ local group /
   NIC1 ===== rail 1 ============ NIC1
```

图中连线只是设计目标；真正部署前要用 BDF、`topo -m` 和 physical cabling 清单确认，不要假设每台节点编号相同就代表布线相同。

### Rail-optimized 网络要求跨节点保持 NIC 一致性

多 NIC 节点常把每张 NIC 接入独立 rail。如果 rail 之间连接较弱，同一 ring/tree 在不同节点切换 NIC 会产生 cross-rail 流量。[NCCL 环境变量文档中的 `NCCL_CROSS_NIC`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-cross-nic)说明了这种取舍：有的网络应尽量在同一 rail 使用相同 NIC，有的网络允许 cross-NIC 以获得更好路径。

因此 rank mapping 不只是“GPU0→NIC0”；还要保证：

- 各节点的 NIC physical rail 标识一致；
- 故障替换后 mapping 被重新生成；
- communicator 缺少对称 rank 时，算法仍有受控 fallback；
- network team 与 compute scheduler 对 rail 命名使用同一事实表。

### CPU helper thread 也属于 mapping

[NCCL performance and tuning 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/performance_and_tuning.html)提醒，不正确的 process/thread placement 会显著影响性能。应在拓扑基线中记录 launcher 的 CPU binding、通信 helper/proxy thread 的可用 CPU set，以及 container cpuset。

如果 scheduler 给 rank 的 CPU 集合与 GPU affinity 完全不交叠，`nvidia-smi topo` 甚至可能显示 affinity 为 `N/A`。此时首先修复资源分配，不应直接调整 collective channel 参数。

## ACS：隔离功能也会改变 P2P 路由

### ACS 的核心不是“开了就慢”

PCIe Access Control Services 为 peer request 的验证、重定向和隔离提供控制。某些 ACS 设置会把原本可以在 switch 内完成的 P2P transaction 重定向到上游 Root Complex：

```text
without redirection:
GPU -> PCIe switch -> NIC

with upstream redirection:
GPU -> PCIe switch -> Root Complex
                    -> PCIe switch -> NIC
```

后者增加路径、占用上行，也可能遇到 Root Complex 不支持相应 peer forwarding 的限制。[Linux PCI P2PDMA 文档](https://docs.kernel.org/driver-api/pci/p2pdma.html)说明，switch 内路径是否保持在 hierarchy 中取决于 ACS 等配置；[NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html#pci-access-control-services-acs)给出了用 `lspci -vvv` 检查 `ACSCtl` 的方法。

### 不要在生产机器上盲目关 ACS

ACS 同时承担隔离和虚拟化边界。VM 直通通常需要 ACS，某些平台还通过 ATS 等能力实现正确、高效的地址转换；新 GPU/CPU/NIC 平台也可能要求特定 ACS bit 组合。于是：

> “裸机某代 x86 平台的 GDR P2P 受 ACS upstream redirect 影响”不能被扩写为“所有机器都应该关闭 ACS”。

安全流程应是：

1. 只读记录 bridge 的 `ACSCap/ACSCtl`；
2. 对照整机厂商、GPU/NIC vendor 与虚拟化方案的 qualification；
3. 在维护窗口做受控 A/B，并保留恢复方案；
4. 同时验证 P2P correctness、isolation 和性能；
5. 不用 `setpci` 临时写寄存器作为长期配置。

本文不会给出通用“关 ACS”命令，因为错误的全局修改可能破坏设备隔离或让平台进入未支持状态。

## IOMMU：地址翻译既是安全边界也是兼容边界

### 经典 GPUDirect RDMA 依赖 peer 可理解的地址

GPUDirect RDMA 需要 GPU 与 peer device 对被映射页的 I/O address 形成一致理解。NVIDIA 文档指出，经典 GPUDirect RDMA 模型与非 1:1 IOMMU translation 不兼容，要求 disable 或 pass-through；CUDA/NCCL 文档进一步限定，Linux bare-metal 的 PCIe P2P 不应在不受支持的 translated IOMMU 模式下运行，否则风险不只限于变慢，还可能出现 silent data corruption。

检查当前状态可以从：

```bash
cat /proc/cmdline
dmesg | grep -i -E "iommu|dmar|default domain"
find /sys/kernel/iommu_groups -maxdepth 3 -type l
```

开始，但这些输出只能描述当前 kernel 视图，最终仍需按 platform support matrix 判断。

### 虚拟机不是裸机规则的简单复制

VM passthrough 需要 IOMMU 提供隔离，ACS 也常是把 device function 安全分配给 guest 的基础。此时“禁用 IOMMU/ACS”通常不是选项；正确方案可能依赖 VFIO、ATS、guest/host driver 和 hypervisor 配置。容器虽共享 host kernel，也可能因 device cgroup、mount namespace 或虚拟 `/sys` 看到不完整拓扑。

[NCCL topology detection 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html#topology-detection)说明 NCCL 依赖 `/sys` 发现 GPU 与 NIC 的 PCI topology；虚拟或不完整的 `/sys` 会导致次优选择。排障时要比较：

```text
host 上看到的 BDF/tree/NUMA
guest 或 container 里看到的 BDF/tree/NUMA
runtime 看到的 GPU UUID 与 NIC
scheduler 实际授予的 cpuset/device set
```

不能因为 host 图正确，就假设 container 内 library 获得了同一事实。

### 正确性验证必须覆盖 payload

IOMMU/peer mapping 问题可能表现为错误数据而非清晰异常。实验不能只报告 GB/s，还应：

- 用可识别 pattern 初始化 source；
- 每次传输后在正确同步点校验 destination；
- 覆盖不同 size、alignment 与方向；
- 在压力、并发和长时间运行下重复；
- 把 mismatch、timeout 与 transport error 作为失败，而不是丢弃样本；
- 在 unpin/free 前确认最后一个 DMA 已完成。

[linux-rdma/perftest](https://github.com/linux-rdma/perftest)提供多种 verbs bandwidth/latency 测试，当前版本还包含 CUDA memory 与数据校验相关选项；具体参数会演进，运行前应以安装版本的 `--help` 和项目 man page 为准。

## 一个路径的性能由最小截面和排队共同决定

### 大消息近似看 bottleneck bandwidth

对大小为 \(S\) 的单次数据移动，可先用 latency-bandwidth 模型：

$$
T(S)
\approx
\alpha_{\text{setup}}
+
\sum_{e\in P}\alpha_e
+
\frac{S}{B_{\text{bottleneck}}}
+
T_{\text{queue}}
+
T_{\text{sync}}
$$

其中：

- \(\alpha_{\text{setup}}\)：API、registration、launch 或 protocol setup；
- \(\alpha_e\)：每段固定转发/协议成本；
- \(B_{\text{bottleneck}}\)：路径上当前可用的最小共享容量；
- \(T_{\text{queue}}\)：与其他流竞争产生的排队；
- \(T_{\text{sync}}\)：等待 producer、completion 与 consumer ordering。

对足够大的 \(S\)，容量项占主导；对小消息，launch、doorbell、round trip 和 synchronization 更重要。单个大 buffer 的 GB/s 不能预测小 collective 的 latency。

### 多流并发要看共享边

若 \(k\) 条 flow 同时经过一条容量 \(C_e\) 的共享边，理想公平分配也只有：

$$
\sum_{i=1}^{k} b_{i,e}\le C_e
$$

实际还会受 packet scheduling、priority、head-of-line blocking 与 flow hashing 影响。两个单独测试都能跑满的 GPU-NIC pair，同时运行时可能各自下降；这不是峰值数据造假，而是实验从单流变成了共享截面问题。

### 双向结果不能随意相加

某些互联支持全双工，但端点内部 engine、HBM read/write 或交换共享资源仍可能让 tx/rx 互相影响。双向 benchmark 应分别报告：

```text
TX payload bandwidth
RX payload bandwidth
TX + RX aggregate（明确只是求和）
single-direction baseline
```

“双向总和翻倍”只在两方向独立且端点不受其他瓶颈限制时成立，是待验证结果而非默认事实。

## 可复现实验从问题矩阵开始

### 先写假设，再选择 benchmark

不要从“跑一下 AllReduce”开始。先把问题写成可以证伪的假设：

```text
H1: GPU A -> GPU B 的 direct peer copy 可用，且未经过 host staging。
H2: GPU A -> NIC 0 的 GDR 路径优于 GPU A -> NIC 1，差异来自 PCIe/NUMA。
H3: 单节点 collective 的拐点来自 NVLink/PCIe path，而不是 kernel launch。
H4: 多节点下降只在并发跨某个 fabric cut 时出现。
H5: 某个 GPU pair 的异常与 ACS/IOMMU/降宽有关，而非 GPU 时钟。
```

每个假设都要配对照组。例如验证 H1 时，不能只有一次 direct 测试；还要有 P2P capability、禁用 P2P 的 staged baseline、反向 copy、不同 size，以及 link counter 变化。

### 实验矩阵至少覆盖六个维度

下面的矩阵不给出虚构结果，只规定应采集的组合：

| 维度 | 建议取值 | 目的 |
| --- | --- | --- |
| 路径类别 | NVLink direct、NVSwitch、PIX/PXB、PHB/NODE/SYS、GPU↔NIC、跨节点 | 隔离不同物理层级 |
| 方向 | \(i\rightarrow j\)、\(j\rightarrow i\)、双向 | 发现不对称与共享资源 |
| size | 从小消息到超过 cache、进入 steady-state 的大消息 | 分离 \(\alpha\) 与 bandwidth |
| 并发 | 单 pair、多 pair、与 compute overlap | 定位 bisection/oversubscription |
| memory path | GPU peer、host pinned、GDR、host staged | 验证 direct/fallback |
| binding | 不同 GPU-NIC-CPU 三元组 | 验证 affinity |
| network | 单 port、多 port、不同 rail/route | 发现 fabric cut 与拥塞 |
| operation | copy、kernel remote access、RDMA verb、collective | 区分 initiator 和协议 |

pairwise 测试不要只抽 GPU0。最少生成全 pair matrix，因为相同机型中不同 GPU 对也可能属于不同 switch group：

$$
M_{ij}(S,d,c)
=
\operatorname{measure}
(\text{GPU}_i\rightarrow\text{GPU}_j,
\text{size}=S,
\text{direction}=d,
\text{concurrency}=c)
$$

结果应保留矩阵，不要过早压成一个平均值。异常 pair 正是拓扑问题最有价值的信号。

### 分层基准避免“一次 AllReduce 猜全部”

推荐从底向上：

1. **GPU 本地基线**：HBM/compute 状态、时钟、温度、ECC 与健康；
2. **GPU↔GPU pair**：CUDA sample `p2pBandwidthLatencyTest` 或等价的受控 copy/kernel test；
3. **NIC host-memory 基线**：`ib_write_bw`、`ib_read_bw` 等验证 port 与 network；
4. **GPU-memory RDMA**：同一 NIC/route 下切换 host memory 与 CUDA memory；
5. **collective 基线**：`nccl-tests`，先单节点再多节点；
6. **应用 replay**：保留真实 message size、communicator 与 overlap；
7. **主动诊断与 counter**：在唯一测试窗口前后采集。

[CUDA Samples 的 domain-specific 示例](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific)包含 `p2pBandwidthLatencyTest`；[NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)提供 collective 性能测试。工具只是探针，仍需记录源码版本、构建选项和命令行。

### 一份可复现的只读采集骨架

下面脚本片段只展示采集范围，实际运行前应根据权限和工具版本调整；涉及环境变量或设备名时不要照抄：

```bash
date --iso-8601=seconds
uname -a
cat /etc/os-release
cat /proc/cmdline

nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version \
  --format=csv
nvidia-smi topo -m
nvidia-smi topo -mp
nvidia-smi topo -p2p p
nvidia-smi topo -p2p n

lscpu
numactl --hardware
lspci -Dnn
lspci -D -tv

rdma link show
ibv_devinfo
ip -d link show
```

随后为每个 GPU/NIC BDF 采集 `lspci -vv` 与 sysfs link/NUMA 属性。若使用 container，应在 host 与 container 内各采一次，并记录 cgroup cpuset、device visibility 与 mount 情况。

### 动态状态要与静态清单一起保存

实验记录至少分三类：

| 类别 | 示例 |
| --- | --- |
| 静态身份 | 整机 SKU、GPU/NIC 型号、UUID、BDF、slot、cabling、switch/rail |
| 软件配置 | BIOS/firmware、kernel、driver、CUDA、NCCL、OFED/rdma-core、container image |
| 动态状态 | negotiated link、GPU clocks、power/temperature、port state、error counter、网络拥塞计数 |

没有 firmware/driver/kernel 版本的结果很难复现；没有动态 link state 的结果，则可能把一次降宽或错误恢复误认为架构特征。

### Warm-up、计时与统计方式写进报告

互联测试常受首次 context 创建、memory registration、page mapping、JIT、connection establishment 影响。报告要说明：

- warm-up 是否计入；
- 每个 size 迭代次数与持续时间；
- 使用 mean、median、p95 还是 minimum；
- 是否每轮重新注册 memory；
- 是否同步整个 device；
- 是否有 compute/background traffic；
- 单向还是双向；
- error/timeout 是否计入失败率。

只挑“最好一次”适合估计无干扰上限，不适合描述生产稳定性；只给平均值又会掩盖 tail latency。两者应分别报告。

## 从实验结果反推瓶颈

### 小消息慢而大消息正常

更可能与固定成本有关：

- kernel/copy launch；
- queue/doorbell；
- connection setup；
- synchronization；
- software proxy；
- collective algorithm 的轮数。

这时继续增加 link 数未必有效，应画出 latency-vs-size 曲线并寻找进入线性区的拐点。

### 大消息提前平台化

可能是某个容量上限：

- PCIe negotiated width/speed；
- endpoint aggregate；
- switch shared uplink；
- HBM read/write 或内部 egress；
- NIC port；
- network cut；
- destination ingress。

用单 pair、并发 pair 和反向测试比较：单 pair 就低通常是局部 path；单 pair 正常而并发下降更像共享截面。

### 只有某些 GPU pair 异常

按 BDF 把异常 pair 标回 PCI tree：

```text
同一 NVLink/NVSwitch group 内异常？
只跨 PCIe switch 时异常？
只跨 Root Complex 时异常？
只跨 socket 时异常？
正向、反向是否一致？
```

若边界与 `PIX/PXB/PHB/SYS` 分组吻合，拓扑假设增强；若不吻合，检查 link health、降宽、device state、background traffic 和 mapping，不要硬套标签。

### Host-memory RDMA 正常，GPU-memory RDMA 异常

网络本身未必有问题，应优先检查：

- GPU/NIC 是否在 qualified peer topology；
- DMA-BUF 或 `nvidia-peermem` 路径；
- memory registration/pinning；
- ACS 与 IOMMU；
- CUDA/driver/NIC driver compatibility；
- GPU/NIC BDF 是否选错；
- completion 后的数据校验与 ordering。

反过来，host 与 GPU memory 都异常，更像 NIC port、cable、switch、route、MTU/queue 或通用 RDMA stack。

### 单节点正常，多节点异常

先区分“跨节点就异常”与“节点数增大后才异常”：

- 两节点单 flow 异常：检查 port/link、route、GID/LID、MTU、firewall/partition；
- 单 flow 正常，多 flow 异常：检查 ECMP、rail、oversubscription、queue/congestion；
- 点对点正常，collective 异常：检查 rank mapping、algorithm、collective ordering；
- 只有某节点异常：比较该节点的 BDF/firmware/link/cabling 快照；
- 只有某方向异常：检查两端 local GPU↔NIC path 与对应 port counter。

“多节点慢”不是自动等于“网络带宽不够”。

## 一套从静态到动态的故障诊断顺序

### 第 0 层：先确认测试语义

记录 tensor size、dtype、direction、operation、rank group、buffer location 与同步点。最常见的“性能异常”之一是 bytes 计算错了，或者把 per-rank size 与 total size 混用。

### 第 1 层：身份与可见性

确认：

- rank 到 GPU UUID/BDF 的映射；
- NIC RDMA device/port 到 BDF/netdev 的映射；
- host 与 container/VM 的映射是否一致；
- CPU cpuset 是否覆盖预期 affinity；
- 没有因 `CUDA_VISIBLE_DEVICES` 重排而选错 GPU。

### 第 2 层：静态 topology 与 capability

采集：

- `nvidia-smi topo -m/-mp`；
- `nvidia-smi topo -p2p` 的所需 capability；
- `lspci -t` 和 sysfs NUMA；
- RDMA port/link state；
- 平台预期 NVLink/NVSwitch layout。

此层只判断“应该怎样连、声称允许什么”，不宣布性能合格。

### 第 3 层：当前 link 与 fabric health

检查：

- PCIe `LnkSta` 是否降速/降宽；
- NVLink/NVSwitch port 是否 Up；
- Fabric Manager/对应服务是否完成初始化；
- NIC port state、physical state、错误；
- AER、Xid、driver/fabric log；
- IOMMU/ACS 当前模式。

如果 link health 不对，先修复基础状态，再调应用。

### 第 4 层：单段与受控 A/B active test

依次测：

```text
GPU<->GPU
host memory<->NIC<->network<->NIC<->host memory
GPU<->NIC<->network<->NIC<->GPU
```

每段都要做 payload validation。普通 verbs/perftest 的 GPU-buffer 测试仍需要 QP 对端，不能天然把本地 GPU↔NIC 从网络与远端路径中单独计时。若平台没有专用 loopback/diagnostic，应固定远端 endpoint、route 和 buffer，在本端做 host buffer/GPU buffer A/B，交换两端角色，再结合 PCIe/NIC counter 与不同 GPU-NIC pair 推断本地段。只有低层路径健康后，才值得解释 collective。

### 第 5 层：并发、collective 与应用

再增加：

- 多 pair 并发；
- 多 rail；
- collective algorithm；
- compute overlap；
- 原始应用 communicator 与 message trace。

若低层测试健康而应用仍慢，才把重点转向调度、chunk/channel、stream ordering 和工作负载不均衡。

## 常见误判逐条拆开

### 误判一：`NV#` 越大，实际带宽就按比例增长

`NV#` 表示 bonded link relationship，不证明一次操作使用全部 link，也不证明端点/HBM/算法能按比例扩展。用流量 counter 和 active test 验证。

### 误判二：`PIX` 一定快于 `PHB`，`SYS` 一定最慢

这些是 traversal 标签，不是固定性能等级。不同平台的 switch、Root Complex、socket interconnect 与 traffic sharing 差异很大；甚至 direction 和操作类型也会改变结论。标签用于分组实验，不用于直接填性能数字。

### 误判三：`cudaDeviceCanAccessPeer=true` 就证明没有 host staging

它证明 runtime 报告 capability，不证明 access 已正确启用、buffer/operation 使用了它、ACS/IOMMU 没有干扰，也不证明 library 最终选路。要观察 transport 与 counter，并做 P2P on/off 对照。

### 误判四：GPUDirect RDMA 表示数据经过 GPU copy engine

GDR 的关键是 NIC 对 GPU memory 的 peer DMA path；RNIC 是 PCIe transaction 发起方之一，不是让 GPU copy engine 把数据主动送进网卡。peer copy、SM remote access 与 NIC DMA 要分别建模。

### 误判五：Direct 意味着 CPU 不参与

payload 可绕过 host bounce buffer，但 connection、registration、queue submission、completion 与 error handling 仍可能由 CPU/control thread 完成。CPU affinity 仍然重要。

### 误判六：所有“网卡”都能做 RDMA

`nvidia-smi topo -m` 可能显示 bonded NIC 或 data-direct device，文档也提醒某些显示项未必是可用 RDMA port。必须用 RDMA device/port、driver、link layer 与 platform support 交叉确认。

### 误判七：同一 Root Complex 就必然高性能

NVIDIA GPUDirect RDMA 文档指出，同 root complex 是经典路径的重要条件，但某些 CPU/IOH 路径仍可能性能受限，跨 CPU interconnect 更可能严重受限或不可靠。支持性还取决于具体 chipset/platform qualification。

### 误判八：关闭 ACS/IOMMU 总能解决问题

这会忽略隔离、VM/VFIO/ATS 与新平台的特定要求。错误配置可能导致安全边界破坏、device 不可用或数据损坏。先判定环境，再遵循整机与软件栈支持矩阵。

### 误判九：`numa_node=-1` 等于 NUMA node 0

Linux ABI 的定义是 unknown。把 unknown 当作 local 会让 binding 脚本稳定地做错事；应查 ACPI/firmware、平台文档与实际 latency。

### 误判十：单 pair 峰值能预测 AllReduce

collective 同时受 algorithm、channel、reduction kernel、bisection、rank mapping 和 synchronization 影响。单 pair 是路径上界探针，不是 collective performance model。

### 误判十一：`busbw` 就是物理链路带宽

`nccl-tests` 的 bus bandwidth 是按 collective 数据移动因子规范化的指标。它有助于跨 rank 数比较，不是交换机端口的 tx counter；硬件 offload 或层次化算法下更要谨慎解释。

### 误判十二：错误计数为零就证明路径健康

counter 只覆盖特定错误类型；无增长不能证明 route 正确或没有拥塞。topology、link state、counter、active diagnostics 和 payload validation 缺一不可。

## 把 rank mapping 做成可审计产物

### 输入、算法与输出都要保存

一个可维护的 mapper 应读入：

```text
GPU UUID/BDF 与 peer matrix
GPU-GPU / GPU-NIC topology relationship
NIC BDF、port、rail 与 network health
CPU/NUMA affinity 和 scheduler cpuset
parallel group definition
message size / frequency / direction
```

输出则包括：

```text
global rank -> host -> local rank -> GPU UUID/BDF
rank -> CPU set
rank/group -> NIC/HCA port
group -> expected scale-up/scale-out cut
fallback and degraded-mode policy
```

不要只把最终 `CUDA_VISIBLE_DEVICES` 列表塞进启动脚本。保存输入 snapshot 和 mapping reason，故障时才能回答“为什么 rank 17 绑定到这个 NIC”。

### 映射应有健康前置条件

如果某条预期 NVLink Down、某个 NIC port degraded、PCIe link 降宽或 topology discovery 不完整，mapper 不应默默按旧模板启动满规模任务。至少有三种策略：

- fail closed：关键高带宽训练拒绝启动；
- degraded placement：缩小 communicator 或换 rail，并明确标记；
- quarantine：把异常 GPU/NIC/node 移出资源池。

选择哪种策略由业务容错需求决定，但不能把未知 topology 当作健康 topology。

## 从拓扑到 collective 的接口

本文没有展开 Ring、Tree、channel 与每个 collective 的通信量，那属于集合通信算法层。连接关系是：

```text
tensor ownership / collective semantics
        |
algorithm chooses logical peers and chunks
        |
transport maps logical edges to NVLink / PCIe / NIC
        |
physical topology supplies capacity, latency and failure domain
```

同一个 Ring 逻辑边在单节点内可能落到 NVLink，也可能跨 PCIe；跨节点边还会经过 GPU↔NIC local path 和 network。后续分析 collective 性能时，应把逻辑 channel 投影回本文建立的物理图，而不是只数通信量。

同样，本文只建立 NCCL 等库所需的 topology 基础，没有提前展开其完整算法选择、protocol、channel 调优和 RAS 语义。这样可以避免把某个版本的实现细节误写成硬件定律。

## 上线前的路径检查清单

### 静态清单

- GPU UUID、BDF、slot 与整机 SKU 对得上；
- NIC BDF、RDMA device、port、netdev 与 rail 对得上；
- PCI tree、NUMA 与平台设计一致；
- NVLink/NVSwitch 预期 layout 可被发现；
- driver、firmware、kernel、CUDA、communication library 属于受支持组合；
- container/VM 暴露的 `/sys` 与 device set 足够完整。

### 能力与正确性

- 按方向检查需要的 P2P capability；
- direct 与 staged path 都有明确、可观察的选择；
- GPU↔NIC memory registration 机制正确；
- ACS/IOMMU/ATS 配置符合当前裸机或虚拟化平台；
- completion 到 CUDA consumer 有明确 ordering；
- benchmark 校验 payload，不只报带宽；
- buffer 生命周期覆盖最后一次 DMA。

### 性能与容量

- 统一 per-link、aggregate、bisection 与 oversubscription 口径；
- 区分单向、双向总和、line rate、payload 和 collective `busbw`；
- 检查每级 PCIe negotiated width/speed；
- 用全 pair matrix 识别不对称；
- 用多 pair 验证共享截面；
- GPU-NIC-CPU 与 network rail 一起映射；
- 保存无背景流量和带真实并发的两套基线。

### 运维与诊断

- topology、link state、counter、active diagnostic 分开采集；
- counter 以前后 delta 解释；
- Fabric Manager/对应服务纳入健康检查；
- 异常 topology 触发 fail/degrade/quarantine，而非静默继续；
- 变更 ACS/IOMMU/BIOS 前有厂商依据、维护窗口和回滚；
- 基线按 node/SKU/software version 分组，不跨平台套阈值。

## 一条数据路径的完整复盘

最后把跨节点 GPU tensor 发送完整走一遍。假设 GPU A 的 HBM 持有 source，GPU B 的 HBM 将接收结果：

```text
1. producer kernel 在 GPU A 的某个 CUDA stream 生成 tensor
2. 通信栈通过受支持的 CUDA/transport bridge 等待 producer stream/event 完成；只创建 event 不会让 NIC 自动遵守它
3. source buffer 被正确注册并在整个 DMA 期间保持有效
4. local NIC 按 GPU A<->NIC A 的合格 peer path 读取数据
5. NIC 将 payload 分组并从指定 port/rail 发出
6. InfiniBand 或 RoCE fabric 按当前 route 穿过交换网络
7. remote NIC 收包并按 transport 语义完成远端写入
8. NIC B<->GPU B 的 peer path 把 payload 写入 GPU B memory
9. completion 通过受支持的同步机制进入 GPU B 的执行依赖
10. consumer kernel 在结果完整、可见后开始读取
```

这十步中：

- 1、2、9、10 属于 execution ordering；
- 3、4、8 属于 memory registration 与 PCIe peer path；
- 5、6、7 属于 network transport/fabric；
- 所有步骤都受 identity、topology、health 与 failure handling 约束。

若不能指出某次失败落在哪一层，就还没有得到根因；若只能说“网络慢”，也还没有建立足够精确的模型。

## 结语

GPU 互联拓扑不是一张静态距离表，而是一组会共同决定结果的约束：

$$
\text{observed behavior}
=
f(
\text{ownership},
\text{initiator},
\text{capability},
\text{routing},
\text{topology},
\text{health},
\text{ordering},
\text{contention}
)
$$

从 HBM 到另一张 GPU，数据可能走直连 NVLink、NVSwitch fabric、PCIe P2P，也可能回退到 host staging；从 HBM 到远端 NIC，数据还要经过 GPU↔NIC peer path、InfiniBand/RoCE fabric 和远端 ingress。每段都有自己的能力、带宽口径、隔离边界和故障语义。

最可靠的工作方式是：用 UUID/BDF 建立身份，用 topology 工具形成假设，用 capability probe 判断可行性，用 link state/counter 观察健康，用分层 active test 验证路径，最后才用 collective 与应用解释端到端表现。这样换一代 GPU、一个新整机 SKU 或一套新网络时，分析框架仍然成立。

## 参考资料

- [NVIDIA CUDA Programming Guide：Programming Systems with Multiple GPUs](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)
- [NVIDIA GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)
- [NVIDIA System Management Interface：Topology](https://docs.nvidia.com/deploy/nvidia-smi/index.html#topology)
- [NVIDIA NCCL User Guide：GPU Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
- [NVIDIA NCCL User Guide：Performance and Tuning](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/performance_and_tuning.html)
- [NVIDIA NCCL User Guide：Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA DCGM：Topology and NVLink](https://docs.nvidia.com/datacenter/dcgm/latest/learn/core-services/topology-and-links.html)
- [NVIDIA DCGM：NVSwitch and ConnectX](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/nvswitch.html)
- [NVIDIA Fabric Manager User Guide](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html)
- [NVIDIA GPU Operator：GPUDirect RDMA and GPUDirect Storage](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html)
- [NVIDIA Grace Blackwell with ConnectX-8 GPUDirect RDMA Guide](https://docs.nvidia.com/multi-node-nvlink-systems/grace-blackwell-cx8-gpudirect-rdma-guide/index.html)
- [Linux Kernel：PCI Peer-to-Peer DMA Support](https://docs.kernel.org/driver-api/pci/p2pdma.html)
- [Linux Kernel：ABI Testing Symbols](https://www.kernel.org/doc/html/latest/admin-guide/abi-testing.html)
- [Linux Kernel：NUMA Memory Performance](https://docs.kernel.org/admin-guide/mm/numaperf.html)
- [PCI-SIG：PCI Express Base Specification Overview](https://pcisig.com/specification-overview/pci-express-base)
- [InfiniBand Trade Association：Specification FAQ](https://infinibandta.org/ibta-specification/)
- [InfiniBand Trade Association：RoCEv2 Announcement](https://www.infinibandta.org/ibta-announces-new-roce-specification/)
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)
- [NVIDIA nccl-tests：Performance Metrics](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
- [linux-rdma/perftest](https://github.com/linux-rdma/perftest)
