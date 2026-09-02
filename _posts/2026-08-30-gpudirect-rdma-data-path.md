---
layout: post
title: "GPUDirect RDMA：网卡如何直接访问 GPU Memory"
subtitle: "从 PCIe BAR、内存注册与 rkey 到有序完成和可恢复数据面"
date: 2026-08-30 09:00:00 +0800
last_modified_at: 2026-09-02
author: iStar
catalog: true
series: distributed-training
series_order: 3
technology_year: 2012
mathjax: true
tags: [分布式训练, GPU优化]
---

两台服务器上的 GPU 要交换一个张量，最直观的实现并不是“GPU 直接连上网络”。GPU Memory、系统内存和网卡分别属于不同设备与地址空间；CUDA stream、RDMA queue pair、PCIe 路由和内核驱动也各自维护状态。若没有额外机制，数据通常先从显存复制到 pinned host memory，再由网卡读取；接收端反向走一遍，最后才回到显存。

GPUDirect RDMA（下文简称 GDR）改变的是其中的 **I/O 数据路径**：允许支持该能力的 NIC/HCA 对已经注册的 GPU Memory 发起 PCIe peer DMA，省掉 host bounce buffer。它没有让 CPU 从系统里消失，也没有让字节停止移动。CPU 仍可能负责地址注册、连接建立、work request 提交、completion progress、错误恢复和资源回收；NIC 仍要通过 PCIe 读取或写入 GPU Memory，数据仍要穿过链路、交换机和远端 PCIe 路径。

真正的问题因此不是“有没有 GPUDirect”这个布尔值，而是：

1. 哪一段 GPU 虚拟地址被稳定地映射成 NIC 能 DMA 的地址；
2. 哪个 memory key 授权本地或远端访问；
3. GPU kernel、CUDA stream 与 NIC DMA 之间如何建立 happens-before；
4. topology、IOMMU、ACS 和 NUMA 是否允许这条路径高效、可靠地成立；
5. completion 到底证明了什么，buffer 又何时才能复用或释放；
6. 上层库是否真的选择了 GDR，而不是静默回退到 host staging。

本文沿着一次传输的生命周期回答这些问题。讨论以离散 NVIDIA GPU、Linux、RDMA verbs 和常见 InfiniBand/RoCE 部署为主；统一内存架构、C2C、一体化 SoC、虚拟机和不同厂商 NIC 可能采用不同映射机制，必须以对应平台支持矩阵为准。

## 先把“Direct”解释准确

“Direct”最容易被压缩成三个不准确的结论：

- **不是没有 CPU**：CPU 可以离开 bulk payload 的复制路径，但通常仍在 control path 上；
- **不是没有数据移动**：NIC DMA engine 仍把字节从 GPU Memory 搬到网络，或从网络搬回 GPU Memory；
- **不保证只有一次物理跳转**：PCIe switch、root port、NUMA interconnect、NIC 内部缓冲与网络交换都可能参与。

更准确的定义是：**NIC 能把 GPU Memory 注册成可 DMA 的内存区域，数据传输不必先落到系统内存 bounce buffer。**

从端到端时延看，GDR 主要删掉的是显式 D2H/H2D staging 及其调度、host memory bandwidth 和额外 buffer 生命周期。长度为 $S$ 的数据，其理想下界仍至少受三段能力约束：

$$
T_{\text{gdr}}(S)
\gtrsim
\max\left(
\frac{S}{B_{\text{gpu-nic}}},
\frac{S}{B_{\text{network}}},
\frac{S}{B_{\text{nic-gpu}}}
\right)
+T_{\text{queue}}+T_{\text{protocol}}
$$

这里使用 `max` 是为了表达流水化后的瓶颈，而不是声称三段一定完全重叠。实际系统还会受到 PCIe payload、读写方向、queue depth、拥塞、registration、同步与软件 progress 的影响。

### GPUDirect 是一组技术，不是一条万能通路

NVIDIA 将 GPUDirect 定义为一个家族。P2P、RDMA 与 Storage 都可能避开 host bounce buffer，但参与设备、上层 API 和完成语义不同。

#### 为什么时间线从 2012 年算起

这里把 GPUDirect RDMA 的技术年份标为 2012，而不是后来生态逐步成熟的年份。NVIDIA 在 **2012 年 10 月 14 日**发布 CUDA 5 production release 时，已经把“GPUDirect technology support for RDMA”列为正式功能，并明确说明它允许网卡直接访问 GPU、减少经系统内存中转的瓶颈；CUDA 5.0 归档页也保留了对应 production release。当前 CUDA 13.3 文档则从实现视角把它描述为随 Kepler-class GPU 与 CUDA 5.0 引入的 PCIe peer-device direct path。

这个时间点只表示功能进入 CUDA 正式版本，不表示 2012 年的任意 GPU、NIC、驱动和平台组合都已经即插即用。可部署性始终取决于 GPU 与 HCA 支持、peer-memory 或 DMA-BUF 接口、内核与驱动版本、PCIe 拓扑以及上层通信栈；本文后续所有“支持”都应按这一整组条件理解。

| 技术 | 典型两端 | 核心路径 | 不负责什么 |
| --- | --- | --- | --- |
| GPUDirect P2P | 同一主机内 GPU ↔ GPU | CUDA peer access，经 PCIe、NVLink 或 NVSwitch | 跨节点网络连接、RDMA rkey |
| GPUDirect RDMA | GPU Memory ↔ NIC/HCA 等 PCIe peer | NIC 对已注册 GPU Memory 做 DMA，再经过 IB/RoCE 等网络 | collective 语义、拥塞控制策略、张量布局 |
| GPUDirect Storage | GPU Memory ↔ 本地或远端存储 | cuFile/GDS 组织 NVMe、文件系统或 storage NIC 的直接 DMA 路径 | 普通网络消息语义、模型缓存一致性 |

#### P2P 的边界

CUDA P2P 解决同一系统中两个 GPU 的 peer memory access。`cudaDeviceCanAccessPeer`、`cudaDeviceEnablePeerAccess`、CUDA IPC 或 VMM 决定另一个 GPU/进程能否访问某段显存。它可以使用 NVLink，也可以使用合适的 PCIe 路径，但不建立跨节点 RDMA QP，更不产生远端 `rkey`。

#### RDMA 的边界

GDR 让第三方 PCIe 设备得到对 GPU Memory 的 DMA 映射。跨节点时，网络仍由 RDMA transport 处理：QP 状态、packet、重传、拥塞、CQ 和 memory protection 都没有被 CUDA 替代。GDR 是“RDMA 的 buffer 位于 GPU Memory”这一能力，不是另一套网络协议。

#### Storage 的边界

GDS 面向文件、块设备和远端存储。某些远端文件系统底层也经过 NIC/RDMA，但应用看到的是 cuFile、文件偏移和 I/O completion，而不是普通 `ibv_post_send`。把“远端 NVMe 到 GPU”全部称为 GDR 会掩盖文件系统、`nvidia-fs`、PCI P2PDMA 与 cuFile 的额外语义。

### 普通 CPU Staging 路径怎样搬一块张量

先看没有 GDR 的典型发送与接收：

```text
sender                                                   receiver
──────                                                   ────────
GPU kernel produces tensor                              NIC receives packets
        │                                                       │
        ▼ D2H via GPU copy engine / PCIe                         ▼ DMA
pinned host staging buffer                              pinned host staging buffer
        │ DMA read by NIC                                        │
        ▼                                                       ▼ H2D
       NIC ─────────────── RDMA / network ───────────────► GPU Memory
```

这条路径仍然可以很高效。D2H/H2D 往往由 GPU copy engine 执行，NIC 与 host memory 之间由 DMA 执行，CPU 未必用 `memcpy` 搬每一个字节。通过双缓冲和 chunking，D2H、network、H2D 还可以形成流水线。

但 staging 有三个不可忽略的成本：

1. 每端都需要一块可 DMA 的 pinned host buffer；
2. payload 额外消耗 PCIe/host memory controller 带宽；
3. 应用要管理“GPU buffer → staging slot → NIC → staging slot → GPU buffer”的多阶段所有权。

若各段无法充分重叠，一个简化模型是：

$$
T_{\text{staging}}
\approx T_{\text{D2H}}+T_{\text{network}}+T_{\text{H2D}}
+T_{\text{queue}}+T_{\text{copy\_sync}}
$$

这不是宣称 GDR 在所有消息尺寸上都更快。很小的消息可能被 eager protocol、doorbell 与同步开销主导；拓扑很差时，NIC 直接读取 GPU Memory 也可能不如从本地 host buffer 读取。正确比较必须同时测方向、尺寸、并发和 topology。

### GDR 数据路径省掉了哪两个 Buffer

在可用的直接路径中，sender NIC 读取源 GPU Memory，receiver NIC 把数据写到目标 GPU Memory：

```text
sender node                                             receiver node
───────────                                             ─────────────
GPU Memory                                               GPU Memory
   ▲  │                                                     ▲
   │  └── NIC DMA read through peer mapping                  │ NIC DMA write
   │                                                         │
GPU kernel          NIC ═══════ fabric ═══════ NIC           GPU kernel
   │                  ▲                         │                 ▲
   └─ CUDA ordering ──┘                         └─ completion ───┘

bulk data: GPU Memory <-> PCIe/NIC <-> network <-> NIC/PCIe <-> GPU Memory
control:   allocation, registration, QP, WR, CQ, metadata, timeout, teardown
```

host staging buffer 消失了，但两类工作依旧存在：

- **数据面**：NIC DMA、PCIe TLP、网络 packet、远端 DMA；
- **控制面**：识别 GPU pointer、注册 memory region、交换 remote address/rkey、post work request、推进 CQ、发布完成、处理撤销与错误。

“CPU bypass”应限定为 bulk data 不经系统内存中转。许多实现依然由 CPU 向 NIC doorbell 写入 work request，并在 CPU 上 poll completion；只有 GPUDirect Async、IBGDA、DOCA GPUNetIO 等更专门的路径才可能进一步把部分控制推进交给 GPU 或设备，而且它们有独立的硬件、软件与安全边界。

### Send/Recv、RDMA Write 与 RDMA Read 看到的路径不同

GDR 不改变 verbs 的基本语义。

#### Two-sided Send/Recv

接收端先 post 一个指向已注册 GPU buffer 的 receive WR。发送端 post send 后，NIC 读取发送 buffer，经网络把数据放入匹配的接收 buffer。双方依靠 send/receive CQE 观察完成。

#### One-sided RDMA Write

initiator 持有远端地址和 `rkey`，本地 NIC 读取本地 source buffer，并让远端 NIC 写入 target GPU buffer。远端 CPU 不需要为每次 payload 预先执行 receive，但应用仍要解决“远端如何知道写完”。常见做法是 Write with Immediate、额外 control message、doorbell/notification 或上层状态机。

#### One-sided RDMA Read

initiator 请求远端 NIC 从远端 GPU Memory 读出数据，再由本地 NIC 写入本地 buffer。read path 对 peer read 性能与 topology 更敏感，不能用 RDMA Write 的结果代替它。

因此，`GPU → NIC` 与 `NIC → GPU` 应分别测量。PCIe peer read 和 peer write 的实现、read request credit、root complex 行为可能不对称；“链路是 400 Gb/s”也不代表每个方向、每个 GPU-NIC pair 都能达到同一有效带宽。

## 一个 GPU Pointer 为什么不能直接交给 NIC

`cudaMalloc` 返回的是当前 CUDA context/UVA 中可用的 GPU virtual address。它首先是一个虚拟地址，不是：

- NIC 能直接发出的 PCIe bus address；
- 远端进程可解引用的普通指针；
- 自动附带生命周期与访问权限的 capability；
- 在 free/reallocate 后仍代表原 allocation 的永久标识。

同一个数值地址在 allocation 释放后可能被另一个 GPU、另一个 context 或另一段物理内存复用。NVIDIA 文档对旧 token 路径特别强调：地址本身不足以唯一标识 GPU VA space；registration cache 也必须用 buffer identity/generation 或 allocator hook 识别这种复用。

远端真正需要的是一份受约束的 descriptor：

```text
{
  endpoint / QP identity,
  remote IOVA or registered virtual address,
  length,
  rkey,
  allocation generation,
  memory type and device identity,
  protocol-level object/version metadata
}
```

其中 address + rkey 只够 NIC 定位并授权一段 MR。它不知道这段 bytes 是梯度、KV block 还是 checkpoint shard；dtype、shape、stride、layer、checksum 与对象版本仍由上层协议负责。

### 从 GPU VA 到 NIC DMA Address 要经过哪些步骤

一次典型注册可拆成六层，不应把它们全部叫“pin”：

1. **allocation**：CUDA allocator 创建 GPU allocation，并返回进程可见 VA；
2. **pointer classification**：通信库查询 pointer 属性，确认它属于 GPU Memory、哪个 device、是否支持所需导出路径；
3. **pin/export**：GPU driver 固定 allocation 对应的 backing，并提供 peer pages 或 DMA-BUF handle；
4. **DMA mapping**：kernel/IOMMU/DMA API 根据 importer NIC，把 scatter-gather 或 peer MMIO 映射为该 NIC 可使用的 `dma_addr_t`；
5. **MR registration**：RDMA provider 在 HCA 中创建 translation/protection state，生成 `lkey/rkey`；
6. **metadata publication**：应用只把必要的 remote address、length、rkey 与 generation 交给获授权 peer。

可以把转换关系画成：

```text
CUDA VA
  │ pointer attributes / allocation identity
  ▼
GPU allocation + page/segment description
  │ nvidia-peermem callbacks or DMA-BUF export/import
  ▼
scatter-gather / peer bus mappings for this NIC
  │ dma_map_* + RDMA provider registration
  ▼
HCA translation/protection entries
  │
  ├── lkey: local SGE validation
  └── rkey: remote RDMA authorization
```

任一层改变，旧 descriptor 都可能失效。尤其是 CUDA VMM remap、memory pool 地址复用、GPU reset、driver revoke、NIC reset 与 MR deregistration，都会破坏“相同十六进制地址仍可访问”的假设。

### BAR1 是窗口，不是第二份显存

PCIe 设备通过 BAR（Base Address Register）向系统声明 MMIO aperture。经典离散 GPU GDR 路径中，GPU driver 可让 GPU Memory 的部分 backing 经 GPU BAR aperture 暴露到 PCIe 地址空间，NIC 再对这些地址发起 peer DMA。NVIDIA 文档把 BAR1 视为 GDR mapping 的主要资源，并提供 `nvidia-smi -q` 或 NVML 查询 BAR1 使用量。

BAR1 的角色容易被误解：

- 它不是把整块 tensor 复制到另一份“BAR 内存”；
- 它更像一扇可重映射的地址窗口，让 PCIe peer 能触达显存 backing；
- mapping 通常按固定粒度建立，历史文档与 API 示例常见 64 KiB 对齐；
- BAR1 总量不等于全部可供应用注册的容量，driver 会保留一部分；
- 大 BAR 能降低窗口压力，但仍受 BIOS、地址空间、GPU/driver 与平台实现约束。

现代 DMA-BUF、VMM、coherent interconnect 或 integrated GPU 平台不应被强行解释成完全相同的 BAR1 细节。当前 CUDA Driver API 还提供 `CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE`，用于在支持的平台上明确请求经 PCIe BAR1 的 DMA-BUF mapping；未设置该 flag 或不同平台时，应由 driver/provider 的实际 contract 决定。

### Pin、DMA Map 与 MR Register 不是同一件事

这三个动作解决的问题不同。

#### Pin：让 backing 在访问期间保持稳定

NIC DMA 不能接受“传输到一半物理页被迁移或回收”。pin/export 建立了 backing 与生命周期约束。GPU Memory 不是普通用户页，因此旧路径由 NVIDIA driver 提供 peer-memory API，不能简单用 host memory 的 `get_user_pages` 代替。

#### DMA Map：得到特定设备可用的地址

同一组 backing pages 对不同 PCIe device 未必有相同 DMA address。DMA API 会考虑 topology、IOMMU domain、scatter-gather 合并和 device constraints。`phys_addr_t`、CPU virtual address、GPU virtual address 与 NIC 使用的 `dma_addr_t` 不是同义词。

#### MR Register：建立 NIC 的翻译与保护状态

`ibv_reg_mr`、`ibv_reg_dmabuf_mr` 等 verbs 把范围与 protection domain、access flags 绑定，provider 在 NIC/driver 中建立 translation。注册结果中的：

- `lkey` 用于本地 SGE，证明本地 NIC 可按指定权限访问 buffer；
- `rkey` 交给远端 peer，在 provider、memory type 与 access flags 都支持时，用于 RDMA Read/Write/Atomic 的远端授权；
- `addr/IOVA + length` 决定 key 的可访问范围；
- access flags 决定 remote read、remote write、atomic 等能力。

`rkey` 是 NIC memory protection capability，不是应用身份、租户认证或端到端加密。泄露有效的 address/rkey/endpoint metadata 可能扩大 DMA 攻击面，控制面必须把它当敏感短期凭证管理。

## `nvidia-peermem` 路径如何协作

`nvidia-peermem` 是 NVIDIA GPU driver 包提供的 peer-memory client。对支持的 NVIDIA HCA/OFED 组合，它把 RDMA core/HCA driver 的 memory registration 请求接到 NVIDIA GPU driver：

```text
userspace ibv_reg_mr(GPU pointer)
            │
            ▼
       RDMA core / mlx5 provider
            │ peer-memory callbacks
            ▼
       nvidia-peermem
            │ NVIDIA peer-memory API
            ▼
       NVIDIA GPU driver
            │ page table / DMA mapping
            ▼
       HCA MR + lkey/rkey
```

旧的自建驱动路径可能显式调用 `nvidia_p2p_get_pages`、`nvidia_p2p_dma_map_pages` 与相应 unmap/put API。NVIDIA 文档还要求为普通 `nvidia_p2p_get_pages` mapping 提供 invalidation/free callback：当应用提前 `cuMemFree`、销毁 context 或进程退出时，GPU driver 会同步进入回调，第三方驱动必须先阻止新 DMA、停止并等待 outstanding DMA，再释放对应引用。回调内必须调用 `nvidia_p2p_free_page_table()`，不能调用 `nvidia_p2p_put_pages()`；NVIDIA driver 要等回调返回后才真正 unmap 对应区域。

三个细节很关键：

1. revoke callback 不是“投递任务后立即返回”的普通异步通知。耗时的非关键清理可以交给 worker，但回调返回前必须满足 API 要求的旧 DMA 停止、引用释放与 page-table free；同时不能在 driver lock 下等待新的 GPU 工作，否则可能死锁；
2. CUDA 12.2 起还提供 `nvidia_p2p_get_pages_persistent` / `nvidia_p2p_put_pages_persistent`。persistent mapping 不注册 invalidation callback，因此调用方必须按该接口自己的显式生命周期收回，不能把两种路径的 teardown 规则混用；
3. NVIDIA 在 CUDA 13.x 文档中的 nv-p2p deprecation 说明主要针对 Tegra/Blackwell SoC 演进，不能据此宣称所有 x86 离散 GPU 上的 `nvidia-peermem` 已被 CUDA 13 全面移除。

不过在当前 GPU Operator 文档中，`nvidia-peermem` 已明确被称为 legacy 路径；新部署应先评估 DMA-BUF。

## DMA-BUF 路径为什么成为当前优先选项

DMA-BUF 是 Linux 内核用于跨驱动共享 buffer 的通用框架。GPU driver 作为 exporter，把一段 GPU allocation 导出为 fd；RDMA/NIC driver 作为 importer，attach 并为自己的 device 建立 DMA mapping。userspace 可通过 `ibv_reg_dmabuf_mr` 把 fd 对应的范围注册为 MR。

```text
CUDA allocation
     │ cuMemGetHandleForAddressRange(... DMA_BUF_FD ...)
     ▼
DMA-BUF fd
     │ pass/import into RDMA provider
     ▼
dma_buf_attach + map_attachment
     │ sg_table mapped for NIC
     ▼
ibv_reg_dmabuf_mr
     ▼
MR / lkey / rkey
```

这条图不是说任意 GPU pointer 都能直接导出。以 CUDA 13.3 的 `cuMemGetHandleForAddressRange` 为例，至少要先确认 `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED`；`dptr` 与 `size` 都按 host page size 对齐；range 来自 `cuMemAlloc`，或来自 `cuMemAddressReserve` 且已经被 `cuMemMap` 完整映射。VMM range 的所有 backing allocation 必须在同一 device 上并具有一致属性；底层 physical allocation 一旦改变，就要重新获取 handle。进入 verbs 时，`ibv_reg_dmabuf_mr` 还要求 `iova` 与 DMA-BUF `offset` 具有相同 page offset，并只接受该 API 支持的 access flags；remote write/atomic 同时要求 local write 权限。

相较私有 peer-memory callback，DMA-BUF 有几个系统层优势：

- exporter/importer 生命周期与引用关系由上游内核框架表达；
- fd 是可传递、可关闭、可审计的显式 handle；
- importer-specific DMA mapping 与 invalidation 进入标准协议；
- Linux PCI P2PDMA 文档也建议在没有普通 `struct page` 的 MMIO peer memory 上使用 DMA-BUF 建立 revoke/shutdown。

`close(dma-buf fd)` 只释放当前进程持有的这个 fd 引用，不会自动撤销已经建立的 MR/attachment，也不保证 outstanding DMA 已停止。真正的 revoke 仍需按 `stop admission -> drain -> deregister MR -> detach/unmap` 的顺序完成。

但 DMA-BUF 不是自动获得 GDR 的开关。当前 NVIDIA GPU Operator 文档列出的典型前提包括 Linux kernel 5.12+、CUDA 11.7+、NVIDIA Open GPU Kernel Modules 以及支持的 GPU；UCX 官方 FAQ 同样给出 UCX 1.14+、kernel 5.12+、CUDA 11.7+ 与 open kernel module 这组条件。具体 driver、NIC、rdma-core、发行版和虚拟化组合仍要按支持矩阵锁定。

截至本文修改日期，GPU Operator 明确推荐 DMA-BUF 而不是 `nvidia-peermem`。这是一条部署方向，不是“任何旧集群都应立刻卸载 peer-memory module”的命令。升级必须先用 raw verbs、UCX/NCCL 与真实 workload 验证功能、性能和回退行为。

### 两条内核路径不能只靠 `lsmod` 判断

`lsmod | grep nvidia_peermem` 只能证明 module 已加载，不能证明：

- 当前进程注册成功；
- RDMA provider 实际使用 GPU peer mapping；
- topology 允许 P2P；
- middleware 没有选择 host staging；
- DMA-BUF exporter/importer 组合正确；
- 数据方向与消息尺寸使用的是 direct protocol。

同理，没有 `nvidia-peermem` 也不意味着 GDR 不可用：DMA-BUF 路径本来就不需要它。应同时检查：

1. CUDA/device 的 DMA-BUF capability；
2. open/proprietary GPU kernel module 类型；
3. kernel 与 rdma-core/provider 是否支持 `ibv_reg_dmabuf_mr`；
4. GPU buffer registration 是否产生成功的 MR；
5. UCX/NCCL/NIXL 日志或 trace 中实际 backend/protocol；
6. host staging counters、copy engine 与 host memory traffic 是否符合预期。

## Topology 决定“能走”和“值得走”是两回事

物理层的完整发现与量化方法见同系列的[《GPU 互联拓扑：数据怎样从 HBM 走到另一张 GPU 与远端 NIC》]({% post_url 2026-08-25-gpu-interconnect-topology %})。本节只保留会直接改变 GDR eligibility、距离 cutoff 与安全边界的部分。

NVIDIA GDR 文档把 GPU 与 third-party device 共享 upstream PCIe root complex 视为最重要的经典条件，并把路径粗分为：

1. 只经过 PCIe switches：通常最佳；
2. 经过单个 CPU/IOH：可能可用，但 peer read 尤其可能受限；
3. 跨 QPI/UPI/HT 到另一个 socket/root complex：可能严重降速，甚至不可靠。

截至 NCCL 2.31.2，`NCCL_NET_GDR_LEVEL` 接受五个距离字符串：

```text
LOC   从不使用 GDR
PIX   GPU 与 NIC 在同一 PCIe switch
PXB   经过多个 PCIe switches
PHB   GPU 与 NIC 在同一 NUMA node，流量经过 CPU
SYS   跨 NUMA/SMP interconnect
```

`NCCL_NET_GDR_LEVEL` 用这些字符串控制允许使用 GDR 的最远距离；整数值曾发生语义变化，官方文档建议使用字符串。`NODE` 可能出现在其他 topology 表示中，但不是该变量当前接受的值；不能把 `nvidia-smi` 或 NCCL 内部图中的所有 path label 都当成这个配置接口的枚举。默认策略会随 NCCL 与平台演进，生产配置应记录版本和最终 topology graph，而不是只保存环境变量。

### 先画物理树

最基本的证据包括：

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p p
lspci -D -t
lscpu --extended=CPU,NODE,SOCKET,CORE
```

要为每个 rank 建立 `GPU → closest NIC → CPU NUMA node` 映射。多 rail 系统不能把所有 GPU 都固定到 `mlx5_0`；某个 NIC 物理上更快，不代表它的 IP/GID、RoCE PFC/ECN、IB partition 或 container device exposure 也配置正确。

### NUMA 仍然影响“直接”路径

即使 payload 不进入 DRAM，CPU/NIC locality 仍影响：

- QP/CQ 创建与 userspace doorbell；
- CQ polling/progress thread 的 cache 与 MMIO；
- control message、metadata 与 fallback staging buffer；
- NIC interrupt、completion vector 与 memory registration；
- 多 rail 调度和 PCIe root complex 选择。

一个常见错误是 GPU 与 NIC 配对正确，却把 communication progress thread 绑在远端 socket；另一个错误是 rank 的 cpuset 根本不包含 closest CPU cores，库只能接受 launcher 留下的 affinity。应同时检查进程、线程、GPU、NIC 和 host memory 的 NUMA placement。

### ACS 可能把 P2P 绕回 Root Complex

PCIe Access Control Services 可以为了隔离而改变 peer transaction 的路由。Linux PCI P2PDMA 文档指出：TLP 在同一 hierarchy 内如何转发与 ACS 设置有关；跨 host bridge 的转发没有被 PCIe 规范统一定义，因此内核默认保守阻止，只有已知兼容的硬件才放行。

在 bare-metal GPU Direct 场景，ACS redirect 可能把本应在 switch 内完成的流量送到 root complex，导致带宽下降甚至 hang。NCCL 官方故障文档建议用 `lspci -vvv` 检查 `ACSCtl` 与 `SrcValid`。

但不能把“关闭 ACS”写成跨环境通用命令：

- VM/SR-IOV 往往依赖 ACS 做隔离；
- 错误修改 bridge config 可能破坏设备隔离和稳定性；
- 某些平台需要 ATS/VFIO/厂商固件配合；
- BIOS、switch 与 hypervisor 的支持条件不同。

应采用 OEM/CSP 验证过的拓扑配置，并把安全模型与性能目标一起评审。

### IOMMU 是地址隔离与 P2P 兼容性的交叉点

经典 GDR 文档要求不同 PCIe device 看到一致的物理地址，因此不兼容任意非 1:1 translation，通常要求 IOMMU disabled 或 pass-through。当前 CUDA Programming Guide 进一步明确：

- Linux bare metal 的 CUDA PCIe P2P 不支持 translated IOMMU，需禁用以避免潜在 silent corruption；
- VM passthrough 则应启用 IOMMU，并通过 VFIO 建立设备隔离；
- Windows 不受同一限制。

这不是让所有集群全局关闭 IOMMU 的建议。对多租户环境，DMA isolation 是核心安全边界；如果安全需求与某条 GDR 路径冲突，应该选择受支持的 VM/ATS/SR-IOV 组合、pass-through domain、host staging fallback，或者更换平台，而不是为了跑通 microbenchmark 牺牲隔离。

## Memory Ordering 才是最容易出现静默错误的地方

注册成功只说明 NIC **能访问**；它没有说明 GPU kernel 与 NIC **何时访问**。NVIDIA GDR 文档明确指出这是一个独立数据流路径，GPU 的 relaxed memory model 不会自动把任意并发 kernel 与第三方 PCIe writes 排成正确顺序。

若 kernel 与 RDMA 同时访问同一范围，可能出现：

- NIC 读到 producer 尚未写完的旧数据；
- kernel 看到 NIC 只写入一部分的新数据；
- relaxed ordering 下分片以不同顺序可见；
- buffer 已被 allocator 复用，CQE 却晚到；
- control flag 已更新，payload 尚未对 consumer kernel 完整可见。

因此 correctness 应写成显式依赖图，而不是“stream 最后同步一下”。

### GPU 作为发送源：producer 必须先于 NIC Read

设 kernel $K_p$ 产生 buffer $B$，NIC 随后读取：

$$
K_p(B)\;\prec\;\text{RDMA\_POST}(B)\;\prec\;\text{NIC\_READ}(B)
$$

典型 CPU-progress 实现是：

1. 在 producer stream 记录 CUDA event；
2. CPU 查询/等待 event；
3. event 完成后才 post 指向该 MR 的 WR；
4. NIC 完成后再发布 transport completion。

`CU_POINTER_ATTRIBUTE_SYNC_MEMOPS` 是旧 peer mapping 路径中的重要一致性属性。它让相关 CUDA memory operation 对 host 采用更保守的同步行为，避免 API 已返回而 D2D copy 仍未完成、NIC 随即读到旧值。但它不是一个“给任意 running kernel 加 fence”的 API，也不能修复 kernel 与 NIC 对同一地址的并发数据竞争。

高层库可能把 CUDA stream dependency 与 network progress 封装起来；使用者仍应确认该 API 的 stream contract。若库要求调用前 buffer ready，就不能把尚在另一个 stream 生产的数据直接传入。

### GPU 作为接收目标：CQE 之后还需要 CUDA 边界

接收方向需要：

$$
\text{NIC\_WRITE}(B)\;\prec\;\text{transport completion}
\;\prec\;K_c(B)
$$

NVIDIA 文档提醒，即使第三方设备已经发完 PCIe transaction，并发运行的 GPU kernel 仍可能看到 stale、partial 或 out-of-order data；访问被 NIC 覆盖的同一范围构成数据竞争。安全的传统流程是：

1. NIC/transport 完成写入；
2. CPU 观察到相应 CQE 或可靠的完成状态；
3. CPU 通过 CUDA work submission/synchronization 边界启动 dependent kernel；
4. consumer stream 才读取 buffer。

不能让一个 persistent kernel 盲目轮询普通 GPU buffer，并假设 NIC 写入天然符合 CUDA memory model。GPU-initiated networking、stream memory operations、system-scope atomics 或专用 semaphore 可能提供更强机制，但必须使用对应 API 和硬件 contract，不能从普通 GDR 自动推导。

#### `cuFlushGPUDirectRDMAWrites` 解决的是可见范围，不是并发竞争

当前 CUDA Driver API 还提供 `cuFlushGPUDirectRDMAWrites(target, scope)`。应用应分别查询 `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING` 与 `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS`：前者说明硬件原生保证到哪个 consumer scope；若它已经覆盖目标 scope，flush 是 no-op，可以省略。否则后者要按 bitmask 解读：包含 `CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST` 才能走 host API，包含 `..._MEMOPS` 才表示 stream wait/memop flush 路径可用；`NONE` 表示两种机制都不受支持，不能反过来把它解释成“可以调用 flush”。

它不能改变前述基本依赖。调用者仍要先证明目标 transfer 已完成；flush 也不能把一个正在并发读取同一范围的 kernel 变成安全程序。更细的一点是，支持多条 GPUDirect RDMA 硬件路径的平台可能把它们视为不同 ordering domain；CUDA 文档指出，在这类情况下即便属性报告 `ALL_DEVICES` 且 flush 是 no-op，同时混用不同路径的先后关系仍需由应用在 CUDA 之外建立。因而 descriptor 最好记录实际 mapping/path，不能把“同一 GPU 的所有远端写”默认合并成一个全局顺序。

### Completion 不等于远端应用已消费

不同 completion 回答不同问题：

- local send CQE：本地 WR 达到 transport 定义的完成条件，本地 source buffer 何时可复用需看 opcode/provider contract；
- receive CQE：two-sided receive buffer 已按 transport 规则完成；
- RDMA Write initiator CQE：不自动给远端应用生成事件；
- Write with Immediate/额外 SEND：可以触发远端 CQ/通知，但 payload 与通知的 ordering 仍需遵守 QP、fence 与 provider 规则；
- NIXL/Mooncake handle `DONE`：只代表该 backend 请求完成，不代表 KV block 已挂入正确 request 或 consumer kernel 已结束。

因此至少要区分三个时刻：

```text
TRANSFER_DONE   NIC/transport 不再访问 source，target bytes 已按协议放置
PUBLISHED       上层元数据确认 target object/version 可见
CONSUMED        GPU kernel 或应用已用完，buffer 可以回收
```

把第一个时刻当成第三个，会造成早期复用；把第一个时刻当成第二个，又可能让 scheduler 读到未校验或错误版本的对象。

## Buffer Lifetime 应该写成状态机

一段可远程访问的 GPU buffer 可以使用如下状态：

```text
ALLOCATED
    │ register/export succeeds
    ▼
REGISTERED ── publish addr/rkey/generation ──► EXPOSED
    │                                             │
    │ post transfer                               │ remote may issue
    ▼                                             ▼
IN_FLIGHT ── CQ / backend completion ───────► TRANSFER_DONE
    │                                             │
    │ error / revoke                              │ consumer finishes
    ▼                                             ▼
INVALIDATING ◄────────────────────────────── CONSUMED
    │ stop admission + drain + unpublish
    ▼
DEREGISTERED
    │
    ▼
FREED / RETURNED_TO_POOL
```

必须满足：

1. `REGISTERED` 前不能发布 rkey；
2. `IN_FLIGHT` 时不能 deregister、free 或让 allocator 把同一地址交给另一个对象；
3. 停机先停止新请求，再撤销 metadata，随后 drain outstanding WR；
4. 只有 NIC 与 GPU consumer 都释放引用，才可回到 pool；
5. GPU reset、process exit、dmabuf revoke 或 free callback 会把状态异步推向 `INVALIDATING`。

### `cudaMallocAsync` 与 Memory Pool 增加了一层 Generation

stream-ordered allocator 允许 free 先进入 stream，而底层 allocation 继续留在 pool。相同 VA 可能快速分配给另一个逻辑 buffer。若 registration cache 只以 `base address + length` 为键，就可能把旧 MR/rkey 错配给新的对象。

安全做法包括：

- 从长期注册的 arena 中 suballocate，arena 的物理 backing 在整个服务期稳定；
- 每个 suballocation 携带 generation，并让远端 metadata 同时携带 generation；
- allocator free hook 先阻止新 transfer，再等待 in-flight 归零；
- CUDA VMM remap 后重新导出 DMA-BUF handle，不能沿用旧 handle；
- 使用 `CU_POINTER_ATTRIBUTE_BUFFER_ID` 或等价 allocator identity 检测地址复用。

地址相同不等于 allocation 相同；MR 存活不等于其上层对象仍合法。

### Registration 为什么必须缓存

NVIDIA 文档指出，GPU peer pinning/registration 可能达到微秒到毫秒级，远高于 hot-path WQE posting。若每个消息都执行：

```text
register -> transfer -> deregister
```

小消息时注册成本会完全淹没 direct path 的收益。设 registration cache hit rate 为 $h$，一次请求发生 eviction 的概率为 $p_{\text{evict}}$：

$$
\mathbb{E}[T]
=T_{\text{xfer}}
+(1-h)T_{\text{register}}
+T_{\text{lookup}}
+p_{\text{evict}}T_{\text{evict}}
$$

miss 不一定导致 eviction；只有 cache 资源压力触发淘汰时才支付后者成本。当 $T_{\text{register}}\gg T_{\text{xfer}}$ 时，提升 $h$ 往往比微调 packet size 更重要。常见策略是启动时注册大 arena，运行期只传 subrange；或对 allocation 采用 lazy unpin + LRU。

### 一个可用的 Registration Cache 要缓存什么

cache entry 不应只有 pointer：

```text
{
  allocation_base,
  aligned_base,
  aligned_length,
  buffer_id / generation,
  gpu_device,
  exporter handle or peer page mapping,
  importer NIC / protection domain,
  MR, lkey, rkey,
  access flags,
  in_flight_refcount,
  last_used,
  state
}
```

NVIDIA 旧 API 示例建议按 64 KiB 边界对齐，因为同一 GPU page 内的两个子范围可能共享 mapping。真正的 cache key 还必须包含 NIC/PD/access：同一 GPU allocation 对两个 NIC 的 DMA mapping 和 MR 不一定相同；只读 MR 与 remote-write MR 也不能混为一项。

#### Eviction 不是直接 `ibv_dereg_mr`

正确 eviction 是：

1. 标记 entry 为 draining，拒绝新引用；
2. 从 remote metadata/lookup 中撤销该 generation；
3. 等待所有 WR completion 和 consumer reference；
4. deregister MR，detach/unmap DMA-BUF 或 peer mapping；
5. 最后才允许 allocation free/remap。

超过 BAR1、NIC MTT/MR、locked-memory 或 fd 限额时可以驱逐 LRU，但不能驱逐仍在 `IN_FLIGHT` 的 entry。应监控 hit/miss、register/deregister latency、pinned bytes、BAR1 usage、MR count、eviction reason 和 invalidation wait。

### Unified Memory 通常不是普通 GDR Buffer

Managed Memory 可以迁移或由 UVM 改变驻留位置，与“NIC DMA 期间 backing 稳定”的前提冲突。NVIDIA GDR 文档警告，peer page table 在某一时刻存在不等于 GPU copy 始终是最新可写副本；对被 UVM 迁移的范围做 DMA 可能读到 stale data 或被后续迁移覆盖。

因此常规 GDR arena 应使用明确驻留、受支持的 device allocation，例如当前平台文档允许的 `cuMemAlloc/cudaMalloc` 或 VMM allocation。不要因为某个 perftest 提供 managed-memory flag，就推断生产 middleware 对其有完整一致性支持。

## Security：`rkey` 是能力，不是身份

RDMA 追求绕过远端 CPU 的快速访问，这也意味着授权材料必须更谨慎。

### 最小权限

只申请必要的 `IBV_ACCESS_REMOTE_READ` 或 `REMOTE_WRITE`。不需要 remote atomic 就不要开放；不需要远端访问的 local scratch 不应发布。用独立 PD、QP、MR/MW 和 NIC function 划分租户，而不是让整个 GPU arena 共用一个长期 rkey。

### Metadata 保护

远端 address、rkey、endpoint、generation 通过认证控制面交换，并绑定 job/tenant/session。不要把完整 descriptor 写入普通日志或 metrics label。连接关闭、worker 被驱逐或 buffer 回收时，要先让 metadata 过期。

### fd 与旧 token

DMA-BUF fd 本身是一个可传递的 kernel capability，跨进程传递要限制 Unix socket peer 与 fd 生命周期。关闭 fd 只减少引用计数，不等于撤销已注册的 MR、停止 DMA 或使远端 rkey 失效；必须先停止新请求、drain、deregister，再 detach/unmap。旧 `p2pToken/vaSpaceToken` 路径在 NVIDIA 文档中有明确安全说明：知道 token 即可能映射对应 VA space，token 已长期处于 deprecated 状态。现代路径也不能因为“不再显式看到 token”就忽略授权边界。

### 多租户与 IOMMU

IOMMU、ACS 与 SR-IOV/VFIO 是设备 DMA 隔离的关键边界；container device ACL、RDMA cgroup/namespace 则限制设备暴露和部分资源使用，但不能替代 IOMMU 对错误或恶意 DMA 的约束。为了性能关闭其中一层必须经过威胁模型评审。GDR 的目标是删掉 bounce copy，不是删掉 access control。

## NCCL、UCX、NIXL 与 Mooncake 分别在哪一层

它们都可能最终使用 GDR，但解决的问题不同：

```text
training / inference semantics
        │
        ├── NCCL: collective + rank/topology scheduling
        │
        ├── KV Connector / Mooncake Store: object, cache, ownership, placement
        │
        ▼
point-to-point transfer abstraction
        ├── UCX: transports, protocols, endpoint, memory type
        ├── NIXL: heterogeneous segments, metadata, backend selection
        └── Mooncake Transfer Engine: segment/batch transfer + transports
        ▼
RDMA provider / verbs / QP / MR / CQ
        ▼
DMA-BUF or nvidia-peermem
        ▼
GPU driver + NIC driver + PCIe topology
```

### NCCL：Collective 与拓扑选择

NCCL 接收 GPU buffers 和 collective 操作，构建 ring/tree/CollNet/NVLink/network path，并根据 GPU-NIC distance 决定是否使用 GDR。`NCCL_NET_GDR_LEVEL`、`NCCL_NET_GDR_READ` 等是诊断和调优接口，不是开启后就保证更快的魔法开关。

NCCL 还负责 collective ordering、chunking、channel 与多 rail；GDR 本身不知道 AllReduce。若 NCCL 日志显示 network transport 使用 GDR，只能证明某条 channel 的 buffer path，不能说明算法、拥塞与端到端 step time 已最优。

### UCX：Transport 与 Protocol

UCX 允许 tag、stream、active message 等 API 接收 GPU pointer，并在支持时选择 CUDA copy、GDR、shared memory、TCP 或 RDMA transport。官方 FAQ 说明，大消息通常通过 rendezvous 使用 GPU zero-copy RDMA；小消息可能使用 eager/staging 路径。因而“UCX 支持 GDR”不等于每个消息都走 GDR。

UCX 还维护 memory-type detection 与 registration cache。`ucx_info -d | grep cuda` 可以检查构建能力，但实际协议仍要结合 `UCX_LOG_LEVEL`、`UCX_TLS`、message size 和 topology 验证。官方文档也提醒 GPU RMA API 支持程度与 tag/stream/AM 不完全相同，设计时应锁定 UCX 版本与所用 API。

### NIXL：异构 Memory Section 与 Backend

NIXL 把 VRAM、DRAM、file、block/object storage 描述为 segment/descriptor list，由 Transfer Agent 选择 UCX、GDS 等 backend，并管理连接与 remote metadata。它返回异步 transfer handle，但不替上层分配 KV block，也不理解 tensor 的 layer、dtype 与 request ownership。

NIXL 推荐在初始化阶段注册长期使用的 memory segments，把 metadata exchange 放在 control path。worker 动态移除时，应先 invalidate remote agent metadata，再 drain/deregister；否则其他 agent 的缓存 descriptor 仍可能指向已失效 rkey。

### Mooncake：Segment、Store 与 KV 语义

Mooncake Transfer Engine 用 Segment/Buffer 和 BatchTransfer 组织 DRAM/VRAM/NVMe-oF 数据移动，当前文档列出 TCP、RDMA、cuFile、NVLink 等 transport，并默认可走 DMA-BUF；设置 `WITH_NVIDIA_PEERMEM` 才选择对应 legacy registration 路径。

Mooncake Store/Conductor 位于更上层：决定 KV block 是否缓存、放在哪里、何时复制和淘汰。Transfer Engine 报告 RDMA 完成，不代表 Store 已提交对象 metadata；Store 命中也不代表传输 path 一定是 GDR。分层监控必须分别给出 cache decision、transfer backend 与底层 MR/CQ 状态。

## GDR 不是 GDRCopy

GDRCopy 使用 GPUDirect RDMA 相关映射能力，让 CPU 可以通过 userspace mapping 低开销地读写 GPU Memory。它特别适合很小的 CPU-driven copy 或 control data，但 bulk payload 仍由 CPU load/store 驱动。

GDR 则让第三方 device DMA engine 访问 GPU Memory。二者可能都消耗 BAR mapping，也可能被 UCX/NCCL 同时使用，但数据 mover 不同：

```text
GDRCopy: CPU instructions  <-> GPU BAR mapping
GDR:     NIC DMA engine   <-> GPU peer mapping
```

看到 `libgdrapi` 或 `gdrdrv` 不能单独证明跨节点 payload 已走 GPUDirect RDMA。

## 一套可恢复的 Failure State Machine

性能路径如果没有失败语义，只能算 benchmark demo。建议把 endpoint 与 buffer 两类状态显式组合：

```text
endpoint:
INIT -> CONNECTING -> READY -> DEGRADED -> DRAINING -> CLOSED
                         │          │
                         └-------> FAILED

buffer:
ALLOCATED -> REGISTERED -> EXPOSED -> IN_FLIGHT -> DONE -> CONSUMED
                 │            │          │
                 └-------> INVALIDATING <-┘
                                  │
                                  ▼
                            DEREGISTERED -> FREED
```

### Registration 失败

可能来自不支持的 allocation、BAR1/MTT/locked-memory 资源不足、DMA-BUF export/import 失败、module/driver 不匹配或 topology 不兼容。策略应明确为：

- 对允许回退的请求切到预先验证过的 pinned-host path；
- 对要求 direct path 的 job fail closed；
- 记录精确 backend、errno、GPU/NIC BDF、allocation 类型和注册阶段；
- 不把部分创建的 MR 或 fd 留在 cache。

### Remote access error

stale rkey、错误 address/length、权限不匹配或远端已 deregister 会产生 work completion error，并可能把 QP 推入 error state。处理时要使该 endpoint 上受影响的 in-flight requests 全部进入 terminal/unknown 状态，而不是只重试最后一个 WR。

### Timeout 与 Unknown Outcome

本地超时不能证明远端完全没写。RDMA Write 可能已经修改目标一部分或全部，只是 completion/notification 丢失或 progress 停止。安全重试需要：

- 每次 transfer 使用 `transfer_id + object_generation`；
- 写入独立 target slot，校验后再原子发布 metadata；
- 对重复 write 保证幂等，或让旧 generation 永远不可见；
- checksum/length/schema 验证通过后才进入 `PUBLISHED`；
- endpoint reset 后不继续使用旧 rkey cache。

### GPU Reset、进程退出与 revoke

GPU reset/context destroy 会使一批 allocation 同时失效。`nvidia-peermem` free callback 或 DMA-BUF exporter lifecycle 必须让 importer 停止 DMA。普通 nv-p2p callback 在安全条件满足前不能返回：先阻止新提交、停止并等待旧 DMA，再执行规定的 page-table free；可以把回调后的非关键销毁交给 worker，但不能只“唤醒 worker”就把仍可 DMA 的 mapping 交还给 exporter。DMA-BUF/P2PDMA 的 `move_notify()` shutdown 同样要求 exporter remove 前同步完成 importer DMA-unmap。这里应避免的是在 driver lock 下等待新的 GPU 工作，不是跳过对既有 DMA 的 drain。

### Link/QP 故障

RoCE loss、IB port down、CQ overrun、RNR、retry exceeded 或 device fatal 都可能让请求无法继续。恢复不是简单创建新 QP：还要交换新的 endpoint metadata，确认远端 buffer generation、重新注册或 reload rkey，并让上层决定未提交对象是重传还是丢弃。

## 监控要证明路径，而不只是证明吞吐

建议把证据分成五组。

### 能力与版本

- CUDA runtime/driver、GPU kernel module flavor；
- kernel、rdma-core、OFED/DOCA-OFED、NIC firmware；
- UCX/NCCL/NIXL/Mooncake commit/tag；
- DMA-BUF capability、`nvidia-peermem` 状态；
- GPU/NIC BDF、NUMA、IOMMU/ACS mode。

### 注册与资源

- registration cache hit/miss/eviction；
- register、deregister、DMA map 延迟直方图；
- active MR、pinned bytes、DMA-BUF fd 数；
- BAR1 used/free 与异常增长；
- failed registration 按 errno/backend/topology 分类。

### 传输与完成

- bytes/messages 按 direct、staging、TCP、CUDA copy backend 分类；
- RDMA Read/Write/Send 分方向吞吐与 P50/P95/P99；
- outstanding WR、CQ depth、CQ polling latency；
- completion timeout、remote access error、retry exceeded、QP reset；
- transfer completion 到 consumer kernel launch 的等待。

### 硬件路径

- NIC port counters、retransmit、ECN/PFC、congestion；
- PCIe link width/speed、AER error；
- GPU/NIC PCIe throughput；
- GPU copy-engine activity 与 host memory bandwidth，用于识别 staging；
- progress thread CPU 与 NUMA migration。

### 上层正确性

- object generation mismatch、checksum failure、短写；
- remote metadata cache age 与 invalidation lag；
- buffer premature reuse；
- fallback ratio；
- job 级 step time、collective tail、KV handoff latency 与 SLO goodput。

只看网络带宽会漏掉 registration thrash；只看 GPU utilization 会漏掉 stale rkey；只看 `nvidia-peermem loaded` 会漏掉 silent staging。

### Benchmark 第 0 阶段：固定实验合同

每次实验先保存：

- 两端 GPU/NIC 型号与 BDF；
- PCIe tree、NUMA mapping、link width/speed；
- kernel、driver、firmware、CUDA、rdma-core 与 library 版本；
- IB/RoCE link、MTU、GID、PFC/ECN/路由；
- IOMMU/ACS/virtualization mode；
- direct path、DMA-BUF/peer-memory 与 fallback 配置。

测试矩阵至少覆盖：

```text
direction: GPU->GPU write, GPU<-GPU read, send/recv
size:      small / medium / large, not only one payload
depth:     QD1 and steady-state queue depths
topology:  closest pair and deliberately distant pair
mode:      host, nvidia-peermem, DMA-BUF
cache:     cold registration and warm registration
load:      isolated, multi-flow, multi-rail, application contention
```

不锁定这些变量，两个“同样启用 GDR”的结果不可比较。

### Benchmark 第 1 阶段：先验证 CUDA 与 PCIe

先用 `nvidia-smi topo`、CUDA P2P sample、`nvbandwidth` 等验证：

- GPU allocation、kernel 与 copy 正常；
- GPU-GPU P2P matrix 符合预期；
- 本地 PCIe/NVLink 路径无明显异常；
- IOMMU/ACS 没有制造 silent corruption 或意外绕路。

这一步不经过网络，用来排除 GPU 与本地 topology 问题。若本地 peer path 已异常，直接跑 NCCL 多节点只会把故障隐藏在更复杂的层中。

### Benchmark 第 2 阶段：先测 Host RDMA 基线

使用 `ib_write_bw`、`ib_read_bw`、`ib_send_bw`、对应 latency tests 或平台认可工具，先对 pinned host memory 测：

- 单 pair link 是否达到合理区间；
- MTU、GID、QP、routing 与 congestion 是否正确；
- 单向、双向、多 QP 是否稳定；
- 错误计数与 tail latency 是否可接受。

这一步若失败，GDR 不会修复网络。host baseline 也为“direct 是否真的减少 staging 成本”提供参照。

### Benchmark 第 3 阶段：Raw GPU Memory RDMA

linux-rdma/perftest 当前支持 `--use_cuda=<gpu>`，DMA-BUF 路径还需要 `--use_cuda_dmabuf`；具体构建选项和参数随 release 演进，应以所用版本 man page 为准。

需要分别测：

1. legacy peer-memory 与 DMA-BUF（若两者均被支持）；
2. `ib_write_bw` 与 `ib_read_bw`；
3. cold register 与 warm MR；
4. 每个 GPU-NIC pair，而不是只测 GPU0/NIC0；
5. correctness/data validation 模式与纯性能模式。

raw test 的目标是证明 `GPU buffer → MR → NIC DMA → remote GPU buffer` 成立。它仍不代表 UCX/NCCL 会选择相同 protocol。

### Benchmark 第 4 阶段：Middleware 分层验证

#### UCX

用 `ucx_info -d` 检查 CUDA transport，再用 `ucx_perftest` 或应用最小复现测 host/GPU buffer、eager/rendezvous 阈值与日志中的 lane。不要强制一组 `UCX_TLS` 后就把结果当默认自动选择。

#### NCCL

运行 nccl-tests 的 point-size sweep 与真实 rank 数，使用 `NCCL_DEBUG=INFO` 和 `NCCL_DEBUG_SUBSYS=GRAPH,NET` 保存相应日志。比较自动选择、限制 GDR distance、禁用 GDR 的差异，并观察 collective bus bandwidth 与 tail，而不是只看某个 channel 的峰值。debug 变量只用于受控诊断，不应长期固化到生产配置。

#### NIXL / Mooncake

使用 NIXLBench、Mooncake Transfer Engine benchmark 或最小 descriptor transfer，确认 backend、registered segment、metadata exchange、async status 与 teardown。随后再测真实 KV/weight buffer layout，避免 benchmark 使用一个连续大 buffer，而业务由大量碎片小 block 构成。

### Benchmark 第 5 阶段：回到真实 Workload

训练场景应观测：

- AllReduce/AllGather/ReduceScatter 在 step 中的 exposed time；
- compute-communication overlap；
- registration 是否发生在 steady-state hot path；
- 多 job 竞争下的 collective tail；
- checkpoint、optimizer offload 或 data loader 是否争用同一 PCIe/NIC。

推理场景应观测：

- KV/activation/weight transfer 按实际 size distribution 的延迟；
- TTFT、TPOT、P99 与 SLO goodput；
- P/D worker 等待 transfer 的 visible time；
- cache hit 后节省的计算是否大于传输与排队；
- scale-in、worker crash 与 stale metadata 下的正确恢复。

microbenchmark 给出能力上限，application benchmark 才回答收益是否落在关键路径上。

### 怎样判断真的绕开了 Host Staging

至少形成交叉证据：

1. GPU buffer 能直接注册为 MR；
2. middleware log 明确选择 GDR/DMA-BUF 对应 lane；
3. 禁用 GDR 后性能、copy engine 或 host memory traffic 出现符合预期的变化；
4. direct 模式下没有额外 D2H/H2D buffer 生命周期；
5. GPU/NIC PCIe counters 与 payload 大小、方向一致；
6. CPU memory bandwidth 没有按 payload 双端增加；
7. correctness validation 在长时间、并发与故障注入下通过。

单个指标都不够。CPU usage 低可能只是异步 copy；吞吐高可能来自 host pipeline；日志写着 CUDA support 可能只说明 library 编译了插件。

## 常见错误判断

### “网卡和 GPU 都支持，所以一定能 GDR”

缺少 driver、DMA-BUF/peer-memory、root topology、IOMMU/ACS、MR capability 或容器设备权限时仍会失败。

### “用了 RDMA 就是 GPUDirect RDMA”

RDMA buffer 可以在 host memory。只有 GPU Memory 被正确注册并由 NIC 直接 DMA，才是本文讨论的 GDR path。

### “Zero-copy 就完全没有 Copy”

这里通常指没有 host bounce copy。NIC 仍移动 bytes，协议栈还可能为 eager、alignment、encryption、compression 或 fallback 使用内部 buffer。

### “CQE 到了，GPU 可以随时读”

还要满足 CUDA ordering，并确保 consumer kernel 没有与 NIC 写入并发。

### “长期注册整个 HBM 一定最快”

它可以提高 cache hit；在 BAR1-based mapping 上会占用 BAR1，而所有路径都会占用相应的 MR/provider/cache/fd 等资源，并降低 allocator flexibility、扩大 rkey 泄露影响范围。应按 arena、权限、租户和生命周期分区。

### “DMA-BUF 一定比 `nvidia-peermem` 快”

DMA-BUF 的主要优势是上游接口、生命周期与部署方向。性能取决于 driver/provider、mapping、topology 与版本，必须实测；迁移也要验证是否发生 silent fallback。

### “GDR 一定比 Staging 快”

小消息、远 topology、peer read 受限、registration miss、进度线程错绑或 NIC/GPU contention 时，staging 可能更合适。成熟 middleware 的价值之一就是按协议与距离选择路径。

## 一个生产化数据面的不变量

无论上层是 NCCL collective、UCX message、NIXL descriptor 还是 Mooncake KV block，都应维持以下不变量：

1. **地址不变量**：descriptor 的 allocation generation 与当前 backing 一致；
2. **权限不变量**：rkey/PD/access 仅覆盖本次会话需要的范围与操作；
3. **拓扑不变量**：选择的 GPU-NIC pair 在当前 namespace/VM 中仍可见且受支持；
4. **顺序不变量**：producer completion 先于 NIC read，NIC write completion 先于 consumer launch；
5. **生命周期不变量**：任何 in-flight DMA 都持有 allocation、MR、endpoint 与 metadata 引用；
6. **提交不变量**：transport completion 与应用对象发布分离，校验后才提交；
7. **恢复不变量**：timeout/endpoint failure 不复用未知状态的 target，旧 generation 不再可见；
8. **可观测不变量**：每次 transfer 能回答 backend、direct/fallback、GPU/NIC、MR generation 与 terminal state。

这些约束比某个环境变量更能决定 GDR 是否可长期运行。

## 版本敏感项

本文依据 2026 年 9 月可访问的当前文档整理，以下内容尤其容易随 release 改变：

- CUDA 13.3 GPUDirect RDMA 文档同时保留传统 peer-memory API 说明，并为 DMA-BUF capability/PCIe mapping 增补 Driver API；其中 nv-p2p deprecation 的平台范围必须按原文理解；
- NVIDIA GPU Operator 当前推荐 DMA-BUF，列出的 kernel、CUDA、open module 与 GPU 最低条件是该部署栈的当前支持表，不是所有独立 verbs 应用的永久定律；
- UCX 从 1.14 起记录 DMA-BUF 支持条件，但各 API 的 GPU memory 支持和 protocol 选择仍会演进；
- NCCL 的 GDR distance、默认值、变量名称与 topology 选择会变化，字符串距离比旧整数更稳妥；
- NIXL 与 Mooncake backend/plugin 列表持续更新，必须固定 tag/commit；
- GDS 从 CUDA 12.8 起在部分 NVMe 场景更多采用上游 PCI P2PDMA，不能用旧 `nvidia-fs` 架构图覆盖全部新路径。

部署文档应把“事实”写成 `hardware + firmware + kernel + driver + library version` 的组合，而不是一句“已开启 GPUDirect”。

## 小结

GPUDirect RDMA 的核心不是一个更快的 `cudaMemcpy`，而是让 NIC 成为 GPU Memory 的受控 PCIe peer。CUDA pointer 先经过 allocation identity、pin/export、DMA mapping 与 MR registration，最终变成 NIC 能验证的 address、lkey/rkey；BAR1 在经典离散 GPU 路径中提供显存的 PCIe aperture，DMA-BUF 或 `nvidia-peermem` 则负责跨驱动共享映射与生命周期。

数据绕开 host bounce buffer 后，系统反而更需要清晰的 control path。GPU kernel 与 NIC DMA 必须通过 CUDA/transport completion 建立顺序；MR、allocation、remote metadata 和 consumer 对 buffer 的引用必须共同结束；ACS、IOMMU、NUMA 与 PCIe tree 决定路径能否成立；registration cache 决定它能否在 steady state 保持性能；rkey、fd 与设备隔离决定它能否安全地开放。

NCCL、UCX、NIXL 与 Mooncake 并不是 GDR 的替代品。它们分别在 collective、transport、异构数据面和 KV/Store 语义层使用这项底层能力。只有把每层的完成、失败和 fallback 分开观察，才能确认一条“GPU 到 GPU”的箭头究竟走了哪条路，以及省下的复制是否真正缩短了训练 step 或推理关键路径。

## 参考资料

- [NVIDIA Newsroom：CUDA 5 Production Release（2012-10-14）](https://nvidianews.nvidia.com/news/nvidia-releases-cuda-5-making-programming-with-world-s-most-pervasive-parallel-computing-platform-even-easier-6622749)
- [NVIDIA Developer：CUDA Toolkit 5.0 Archive](https://developer.nvidia.com/cuda-toolkit-50-archive)
- [NVIDIA GPUDirect RDMA 13.3 Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [CUDA Programming Guide：Multi-GPU Systems、P2P、IOMMU 与 ACS](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)
- [CUDA Driver API：Memory Management 与 DMA-BUF Handle](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
- [CUDA Driver API：Device Management 与 `cuFlushGPUDirectRDMAWrites`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html)
- [CUDA Programming Guide：Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)
- [NVIDIA GPU Operator：GPUDirect RDMA and GPUDirect Storage](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html)
- [NVIDIA GPUDirect Developer Overview](https://developer.nvidia.com/gpudirect)
- [NVIDIA GPUDirect Storage Design Guide](https://docs.nvidia.com/gpudirect-storage/design-guide/)
- [Linux Kernel Documentation：PCI Peer-to-Peer DMA Support](https://docs.kernel.org/driver-api/pci/p2pdma.html)
- [Linux Kernel Documentation：DMA-BUF Buffer Sharing and Synchronization](https://docs.kernel.org/driver-api/dma-buf.html)
- [Linux rdma-core：`ibv_reg_mr` / `ibv_reg_dmabuf_mr` Manual](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_reg_mr.3)
- [linux-rdma/perftest：GPUDirect 与 DMA-BUF 测试参数](https://github.com/linux-rdma/perftest)
- [NCCL 2.31 Documentation：GPU Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
- [NCCL Documentation：`NCCL_NET_GDR_LEVEL`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-net-gdr-level)
- [OpenUCX FAQ：GPU Memory 与 DMA-BUF](https://github.com/openucx/ucx/blob/master/docs/source/faq.md)
- [NIXL Architecture](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md)
- [NIXL Backend Plugin Guide](https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md)
- [NIXLBench](https://github.com/ai-dynamo/nixl/tree/main/benchmark/nixlbench)
- [Mooncake Transfer Engine Design](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html)
- [NVIDIA GDRCopy Official Repository](https://github.com/NVIDIA/gdrcopy)
