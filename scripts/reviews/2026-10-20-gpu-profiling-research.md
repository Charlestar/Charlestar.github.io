# GPU profiling 研究交接

日期：2026-10-20。职责分离：research_oct20 调研与大纲，writer_oct20 根据交接撰写，独立 auditor 另写审核记录。此记录是取材范围，不是硬件实测证明。

## 文章边界

已有 Roofline 文章覆盖算术强度、单位、事件计时与计数条件。本篇补充 normal baseline → NVTX 语义标注 → Systems 时间线和依赖 → 具体实例 → Compute 定向计数 → 无工具重测的工作流。主机标注不等于 GPU 执行时间；累计时长、时间轴并集、依赖图关键路径分别讨论。

## 实际查阅的主要来源

- https://docs.nvidia.com/nsight-systems/UserGuide/index.html ：Focused Profiling、Marking and Labeling Regions、capture-range/capture-range-end、CLI/stats。cudaProfilerApi 控制采集，NVTX 标注 CPU 语义并可关联 GPU 工作。
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html ：replay、section 定义、cache control、serialization、clock control。Range replay 是保留范围内并发的特定模式，不能把单 kernel 与范围指标混同。
- https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html ：NVTX Filtering、list/query、launch-count。单个 push/pop range 名末尾需要 `/`；count 1 不承诺获取整个高层算子。
- https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html#timing ：CPU 完成时间与 CUDA events 区间的区别。
- https://docs.pytorch.org/docs/2.8/generated/torch.cuda.nvtx.range.html 与 https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/profiler.py ：NVTX push/pop 及 CUDA profiler start/stop 接口。
- https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html#updates-in-2019-1 ：2019.1 新增 NVTX include/exclude，作为系列时间锚点，不作为工具起源。独审改用已打开并核对的官方现存历史章节，原 archive.docs.nvidia.com 链接未能访问。

## 可复算关系与待验范围

纯假设 DAG：prepare 2，compute 6 与 communication 9 并行，finalize 1；wall 12、两分支累计 15、区间并集 9；compute 减半 wall 仍 12，communication 减半 wall 9。持续时间固定且无资源竞争，不是 GPU 预测模型。

正文 GPU 示例提供可保存执行的 workload，预热后同步，再启用采集与 NVTX，末端同步后停止。原始无 profiler 模式测串行 step 的主机完成延迟。没有安装 torch/nsys/ncu，不执行 GPU、生成 trace、收集 counters 或宣称性能收益；本地只执行纯 CPU 模型并对 GPU 示例做 Python 语法解析。版本依赖的 sections 与 metrics 要查询本机，不把网站当前版本冒充已安装版本。
