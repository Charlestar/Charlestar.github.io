# 网络长文独立全文审阅记录（2026-09-09）

本文件记录额外分配的 2 篇文章，不并入 `2026-09-09-full-training.md` 的 12 篇计数。两篇均从 front matter 到参考资料逐行完整读完，共 2,125 行原文；核查不局限于上一轮评审意见。未执行真实 CUDA、RDMA 或 NCCL 作业。

## 2026-08-30-gpudirect-rdma-data-path.md

- 阅读状态：原文 973 行完整读完。
- 全文核查点：host staging 与 direct path、Read/Write 方向、PCIe read 不对称、GPU VA/MR/rkey 身份、BAR1 与 pin/export/map/register 区分、nv-p2p callback、persistent API、DMA-BUF 对齐/导出/引用、拓扑/ACS/IOMMU、CUDA 可见性、allocation 与 suballocation 生命周期、注册缓存期望成本及驱逐、租户权限、NCCL/UCX/NIXL/Mooncake 分层、GDRCopy、失败重试、诊断实验与结尾不变量。
- 修改：
  1. 普通 RDMA Write 无须接收 WQE，但常规 RC Write with Immediate 消耗远端 RQ/SRQ receive WQE；补上 RNR 与通知信用边界。没有泛化到 provider 的 unsolicited receive 扩展。
  2. 补充 generation/目录撤销与 NIC rkey 权限的区别，以及迟到 Write 覆盖复用 slot 的反例。修改状态图、free hook、eviction、timeout 和结尾不变量，要求覆盖远端持有 descriptor 的访问，完成 quiesce/drain 或受验证的撤权与隔离；不把 CPU fence、`IBV_SEND_FENCE`、关闭 fd 当成全局撤权。
  3. 修正 Mooncake 默认 DMA-BUF 的错误。v0.3.12 及本次查阅的 main 配置源码均将 `WITH_NVIDIA_PEERMEM` 默认设为 true；显式 0 才选择 DMA-BUF。指出设计文与源码不一致，绑定 release 与实际运行配置。
  4. `last_modified_at` 更新为 2026-09-09；保留发布时间、URL、标题与文章深度。
- 一手核验来源及位置：
  - [NVIDIA GPUDirect RDMA 13.3](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)：§2.1 Tegra nv-p2p 弃用范围、§2.2 CUDA 12.2 persistent API，§3.1–3.3 pinning/cache/callback，§3.4 topology，§3.7 synchronization。核验 callback 返回前等待 DMA、不把该义务转交后台线程；未把 Tegra 弃用泛化为所有 x86。
  - [CUDA Driver API Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)：`cuMemGetHandleForAddressRange` 的 host-page 对齐、同设备一致 backing、remap 后重新导出、PCIe mapping flag。
  - [CUDA Driver API Device Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html)：`cuFlushGPUDirectRDMAWrites` 的目标/scope、硬件 ordering 已覆盖时 no-op，以及多硬件路径的独立 ordering domain。维持正文限定，不把 flush 当作等待尚未完成的网络传输。
  - [NVIDIA RDMA Aware Networks Programming Manual v1.7](https://docs.nvidia.com/rdma-aware-networks-programming-user-manual-1-7.pdf)：§3.3.2（PDF 页 22）和术语表（PDF 页 13）明确 Write with Immediate 消耗 RR，缺信用可触发 RNR。
  - [rdma-core ibv_post_send(3)](https://man7.org/linux/man-pages/man3/ibv_post_send.3.html)：WR 的 remote address/rkey 字段、FENCE 适用范围、completion 前 buffer 不可复用。
  - [rdma-core ibv_reg_mr(3)](https://man7.org/linux/man-pages/man3/ibv_reg_mr.3.html)：MR/rkey/access 与 DMA-BUF iova-offset 同页偏移；remote write/atomic 需要 local write；relaxed ordering 不取消 completion 语义。
  - [rdma-core ibv_bind_mw(3)](https://man7.org/linux/man-pages/man3/ibv_bind_mw.3.html)：函数返回只表示 bind 已提交，不能当作绑定完成；type 2 MW 走直接 bind WR。正文没有宣称某个通用 API 可替代所有 provider 的撤权协议。
  - [GPU Operator RDMA/GDS 文档](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html)：DMA-BUF/legacy prerequisites 表及推荐路径；[UCX FAQ](https://openucx.readthedocs.io/en/master/faq.html)：GPU zero-copy rendezvous 与 DMA-BUF 的 UCX 1.14、kernel 5.12、CUDA 11.7/open module 条件。
  - [Mooncake v0.3.12 environ.cpp](https://github.com/kvcache-ai/Mooncake/blob/v0.3.12/mooncake-common/src/environ.cpp)：`Environ::Environ()` 中 `GetBool("WITH_NVIDIA_PEERMEM", true)`；[main environ.cpp](https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-common/src/environ.cpp) 同为 true；[已合并 PR #2192](https://github.com/kvcache-ai/Mooncake/pull/2192) 和 release changelog 交叉确认变更。与[设计文](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/index.md) 的 unset→DMA-BUF 表述存在冲突，采用配置源码为准。
- 数值/状态复核：注册期望成本按 miss 与 eviction 分开计数，未发现量纲错误；脚本枚举旧 DMA、slot 复用、新 DMA 的 3 种有效顺序，其中 1 种虽然拒绝旧 generation 发布仍损坏 B，排除该顺序需要物理访问生命周期约束。
- 未验证边界：没有复现任一 NIC/provider 的 MR/MW 失效竞态，也没有运行 BAR1、CUDA memory ordering、RDMA atomic、ACS/IOMMU 或多租户隔离实验。NIXL/UCX 的所有 backend/version 支持组合未逐一执行；正文仍要求锁定实际版本、driver、allocator、memory type 与 transport。

## 2026-09-02-nccl-internals-topology-channel-protocol.md

- 阅读状态：原文 1,152 行完整读完。
- 全文核查点：API/comm/per-call config、enqueue/task/plan、重构开关、初始化拓扑与 graph search、channel/CTA、chunk/slice/step、Ring/Tree/PAT/CollNet/NVLS 候选矩阵、LL/LL128 编码、kernel work loop 与 CUDA Graph 生命周期、CE/CFT/device API、transport/proxy、registration/GDR/PXN、cost model、stream/group/多 communicator 顺序、revoke/shrink/async error、RAS、日志证据、nccl-tests 指标与实验矩阵。
- 修改：
  1. 图中 `slice 1 → s+1` 改为 `s+sliceSteps`。说明 Simple 每 slice 可推进多个 FIFO step；增加 chunkSteps=4、sliceSteps=2 的可复算例子与空尾 slice 仍可能推进同步计数的边界。
  2. 补上非阻塞 `ncclCommRevoke` 返回 `ncclInProgress` 不是完成，必须查询直到成功；错误不是安全回收状态。区分 `NCCL_SHRINK_DEFAULT` 的无 outstanding 前提与 `NCCL_SHRINK_ABORT` 的中止/不共享资源语义。
  3. `last_modified_at` 更新为 2026-09-09；保持固定 tag、发布时间与 URL。
- 一手核验来源及位置（源码均固定 `v2.31.2-1`）：
  - [collectives.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/collectives.cc)：公开 collective wrapper/请求描述；[enqueue.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/enqueue/enqueue.cc)：`ENQUEUE_REARCH_ENABLE=0`、`finishPlan`、`ncclPrepareTasks`、`ncclLaunchKernel` 中 channelMask 位数决定 grid.x、`calcCollChunking` 的 slice/chunk 与 proxy 计数。
  - [group.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/group.cc)：raw-task scheduler/launcher 尚未实现、回落到用户线程 phased `doLaunches` 的注释与实际调用。未把新目录存在当作默认全面启用重构。
  - [prims_simple.h](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/device/prims_simple.h)：`waitPeer`、`postPeer` 的 `step += StepPerSlice`，`genericOp` 对剩余空 slice 的同步循环；[device.h](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/include/device.h) 的 `ncclLLFifoLine` 与 `NCCL_LL128_LINESIZE/LINEELEMS/DATAELEMS`，以及 [prims_ll128.h](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/device/prims_ll128.h) 的 flag/payload 分工。LL 50%、LL128 15/16=93.75% 是布局比例，不是实测链路效率。
  - [PAT cost model](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/tuning/pat.cc)：AG/RS+Simple、SM60、net device、multi-RPN NVLS 与 opt-in、线性延迟项；[CollNet model](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/tuning/collnet.cc) 与 [NVLS model](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/tuning/nvls.cc)：候选过滤、heads、op/dtype、single-node NVLSTree 禁用，保留正文支持矩阵和限定。
  - [topo.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/graph/topo.cc)：discovery XML fusion/dump 与 topology tracking 分离；[transport.cc](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/transport.cc)：P2P→SHM→NET→CollNet 列表及逐连接 `canConnect`。
  - [User Buffer Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html)：注册混用限制、PXN 不兼容、allocator 与 zero-CTA 条件；[CFT](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cft.html)：2.31 的 device-side 能力、CUDA 13.3/SM100 例程要求，不能等同默认 AllReduce data path。
  - [Communicator API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html)：`ncclCommRevoke`、`ncclCommShrink`、`ncclCommGetAsyncError` 的完成、资源与错误边界；[communicator usage](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)：并发和 host 顺序。
  - [RAS](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html)：CONTROL PROFILER_MASK、capture 时采样、Diagnostics 非默认且不执行 data-path 压测。
- 数值/状态复核：sliceSteps 累计、空 slice、协议布局比例、非阻塞 revoke 状态；Tree 对数深度作为理想模型、algbw/busbw 作为工具归一化保留，没有将其写成硬件计数或性能保证。
- 未验证边界：没有编译整个 NCCL、执行所有 generated kernel/algorithm×protocol 组合、验证 CFT/CE/NVLS/GPU 互联硬件或 profiler 输出。`task_sched/task_sched.cc` 本轮浏览抓取失败，重构结论通过同 tag 的 enqueue 默认值和 group fallback 源码独立印证，未把抓取失败误当作文件不存在。NCCL 真正选择的 channel/CTA 与算法仍需平台 profile 证明。

## 验证方式

`node scripts/reviews/full-network-examples.mjs`：7 个零依赖算术/状态例子，全部通过。它们是可复算反例，不是 CUDA/verbs 的可运行替身，也不是 GPU 性能测试或 provider 一致性证明。`git diff --check` 用于补丁空白检查；站点构建由主审集成执行。
