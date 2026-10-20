# PyTorch 显存分配与快照调研交接

日期：2026-10-21。范围固定 PyTorch 2.8 文档与 v2.8.0 native CUDA allocator。按调研 → 写作 → 独审依赖执行；此处不是全后端正确性证明或 GPU 性能报告。

## 实际读取的来源

调研代理读取 native allocator 开头设计注释、free/free_block、recordStream、release_cached_blocks 和事件等待路径；memory.py 的统计/快照/历史接口；官方文档中 snapshot 两种视图、非 PyTorch 分配、history 环形限制、memory management/backend 调参，以及 2023 年官方工具介绍。写作代理再次读取并对照 cpp 的上述生命周期/回收片段、memory.py 对应函数，及快照/record_stream/empty_cache 文档。

- https://github.com/pytorch/pytorch/blob/v2.8.0/c10/cuda/CUDACachingAllocator.cpp
- https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/memory.py
- https://docs.pytorch.org/docs/2.8/torch_cuda_memory.html
- https://docs.pytorch.org/docs/2.8/notes/cuda.html
- https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.record_stream.html
- https://docs.pytorch.org/docs/2.8/generated/torch.cuda.memory.empty_cache.html
- https://pytorch.org/blog/understanding-gpu-memory-1/

2023 年锚定该官方文章介绍 PyTorch 2.1 experimental Memory Snapshot 能力，不表示 allocator 起源于 2023。get_allocator_backend 实际路径是 torch.cuda.memory.get_allocator_backend，不使用不存在的 get_memory_backend。

## 本篇大纲与关键边界

1. Python 名字/存储所有权 → 释放请求 → stream-safe 可复用 → 缓存 → 整段归还；free_completed 是复用安全，不是所有旧 kernel 在物理时间上全部完成。同 stream 顺序足够时无需等待所有工作实际完成。
2. A/P/I/R 账本：active_allocated、active_awaiting_free、inactive 与 segment 总量，reserved-allocated 不是单一 fragmentation。block size、原始请求与 tensor logical bytes 也不相同。
3. requested_bytes 在 native free_block 才减少，因此可含 pending 请求；toy requested_live 只数 active_allocated，明确不是该统计字段。
4. 两段固定 segment 的例子解释整段 inactive 与可复用 split 余量。empty_cache 真实路径可能先同步 outstanding events，toy whole_inactive 只静态分类，不模拟 API 的最终释放量。
5. 另一布局有 12 MiB inactive、最大现成 block 4 MiB，6 MiB 请求不可直接复用，但可能向 CUDA 新申请或回收重试，不能据此预测必 OOM。此处是虚拟区间，不是物理 HBM 连续性模型。
6. 有界 history，在 OOM 现场首先 dump，不预先同步/清理改变现场；私有 API 版本需复核。Active timeline 找生命周期，allocator state 找事件时布局与重试，地址加分配代次而非仅 pointer。
7. 按 live 保留、生命周期重叠、尺寸/布局变化分别修复；native 配置不套 cudaMallocAsync，experimental expandable_segments 不按固定 segment 解释全部行为。
8. Snapshot 不覆盖 direct CUDA/NCCL 所有分配；device_memory_used 是 device-level 范围，不把减本进程 reserved 的差精确命名 NCCL 占用。

## 代码与验证范围

- 自定义 CPU ledger 使用整数字节，检查状态、正数 block、请求不超过大小、segment 内大小和、segment 互不重叠。inactive 请求量在此格式清零，不声称真实 snapshot inactive requested_size 必为零。
- 24/6/2/16 MiB 例子与 8 MiB whole-inactive、0.5 MiB live rounding 差异可复算；不同布局/事件完成转换用独立按字节所有权 oracle 与反例检查。
- 写作阶段对 GPU wrapper 只做 Python AST 语法解析，不导入 torch，不执行分配、真实 snapshot、CUDA 或 OOM。独审阶段另用显式 stub 验证包装器的异常传播，不将其称为 PyTorch 运行。包装器限定 history 事件数并在 finally 尝试关闭；dump 或关闭 history 失败时保留原始 workload 异常，正常路径的关闭失败则仍抛出。

独立审核、主代理的构建/渲染与提交发布另行记录。CPU 模型不复刻真实池、取整、allocator 调度、driver 申请失败或性能。
