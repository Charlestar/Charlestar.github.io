# 训练数据管线调研交接

日期：2026-10-21。本文按调研 → 写作 → 独审的依赖顺序推进。此记录描述选材、固定版本和可复算模型，不替代独审、构建、浏览器检查及发布验证。

## 实际读取的主要来源

调研代理读取 PyTorch 2.8 文档的 Dataset、自动 batching、多进程、pinning、随机性与 DataLoader API 部分，并读取以下 v2.8.0 源码的对应逻辑。写作代理再次读取 distributed.py、fetch.py、worker.py，以及 dataloader.py 的预取、返回补任务和 pin-memory 线程路径；CUDA stream 部分另行对照官方文档。

- https://docs.pytorch.org/docs/2.8/data.html
- https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/data/distributed.py
- https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/data/dataloader.py
- https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/data/_utils/fetch.py
- https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/data/_utils/worker.py
- https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/data/_utils/pin_memory.py
- https://docs.pytorch.org/docs/2.8/notes/cuda.html
- https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.record_stream.html
- https://pytorch.org/blog/a-year-in/

2017 年仅锚定 PyTorch 的公开发布，来源为 2018-01-19 官方一周年回顾，不作为 DataLoader/预取概念的发明时间。采用旧版明确接口，不将 2.8.0 行为宣称为之后所有版本的行为。

## 本篇独立结构

沿一个 batch 追踪取样、worker 取数与 collate、结果传递、pinning、H2D 和计算；然后分别解释生产能力与缓冲、提前派发窗口、两层 drop_last、IterableDataset 的全局消费者、持久 worker 状态、三类传输/存储寿命、消费游标恢复，以及验证正确性后的调优步骤。避免复述已有分布式并行策略或 Roofline。

## 关键语义

- 常规多进程 collate 在 worker fetch 中完成；单进程路径不创建 worker 或后台 pinning 线程。map-style 可选 __getitems__ 负责批量取样，随后仍走 collate。
- W × prefetch_factor 描述提前派发任务的窗口，不保证有同等数量 ready/pinned batch，更不构成 GPU 预取。
- DistributedSampler 先在全局顺序上补齐或截断，再按 rank 抽取；DataLoader 再处理本地尾批，两者的 drop_last 不是同一个操作。
- IterableDataset 每 worker 有独立副本，必须同时区分 rank 与 worker；丢尾批也按 worker 副本发生。即使样本分片无重复，rank 的 batch 数仍可能不同。
- v2.8.0 worker 启动时已初始化 Python random、torch 及可用时 NumPy 全局随机状态。自建生成器/第三方状态另管；persistent_workers 不在每 epoch 重跑 worker_init_fn。
- 非默认 stream 的数据 ready 依赖、CUDA 分配器复用安全、pinned host buffer 内容不被过早改写是三个问题。record_stream 不能替代 copy event 等待，也不保护 host 内容。
- Checkpoint 输入进度应与模型/优化器相同消费边界一致，派发游标领先于消费游标；改变 world size/worker 数时，局部 offset 不能直接冒充新的全局恢复位置。

## 算例与执行范围

1. 两阶段确定性模型：6 批、准备 12 ms、计算 6 ms，串行 108 ms、理想独立资源流水 78 ms。非真实 trace，不包含有限队列或设备资源竞争。
2. 两层 drop_last：N=10、R=3、B=2，在 no-shuffle 前提下分别计算补齐/截断与本地组批。覆盖 N<R、N=0 和参数拒绝，不用 Python shuffle 冒充 torch.randperm。
3. IterableDataset：N=23、R=2、W=2，消费者长度 6/6/6/5，B=3 丢各 worker 尾批后总 21 样本，rank 步数 4/3。

`dataloader-examples.mjs` 提取正文三个实际 Python 块，并使用独立逐 tick 两服务器模拟、全局矩阵转置分配 oracle、逐消费者轮流投递 oracle 检查。写作阶段该专项、102 篇元数据检查与 diff-check 通过。未导入/运行 PyTorch，未执行 DDP、H2D、CUDA 或真实存储吞吐实验；这些 CPU 检查不证明设备性能或跨版本后端行为。
