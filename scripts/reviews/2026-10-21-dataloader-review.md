# 训练数据管线独立全文审核

日期：2026-10-21。对象：`2026-10-21-training-data-pipeline-dataloader.md` 全文、调研交接记录与 `dataloader-examples.mjs`。写作代理交接冻结后，审核代理接管修改，未与写作代理并发编辑。本记录不替代主代理的完整构建、浏览器、CI 与线上检查。

## 独立读取与逐段核对

独立读取 PyTorch 2.8 DataLoader 文档中 Dataset、自动组批、多进程、IterableDataset、随机性、pinning、API 参数和 DistributedSampler 段落，以及 v2.8.0 `distributed.py`、`_utils/fetch.py`、`_utils/worker.py`、`_utils/pin_memory.py`。另行读取 `dataloader.py` 的构造参数检查、单/多进程迭代器、reset/预派发、按序/非按序返回与补充任务路径；CUDA 2.8 语义及 `Tensor.record_stream` 官方文档用于核对传输依赖与分配器寿命。公开发布年份按官方 2018-01-19 一周年回顾确认，2017 不是数据加载概念的起源。

核对既有 sequence packing 与 distributed checkpoint 文章的上下文：新文不把 batch 布局当作训练目标，也不把 sampler 超前进度当作模型快照对应的数据消费状态。

- 多进程的读取、变换与常规 collate 在 worker fetch 中完成；可选 `__getitems__` 返回样本列表，随后仍 collation。单进程路径没有额外 pin-memory 线程。
- `prefetch_factor` 属于派发任务窗口，而不是 ready batch 数量、GPU 预取深度或 host 内存的严格上界。按序返回可能队头阻塞；改变返回顺序可能改变有限窗口内训练样本分布，未宣称普遍加速。
- 两阶段公式在正阶段时长、至少一批、独立资源和理想重叠条件下正确；6 批的 108/78 ms 数字及 12 ms 发车间隔一致。48 ms 缓冲耗尽是注明相位限制的连续流量近似，不冒充离散 trace。
- DistributedSampler 全局补齐/截断后步进抽取，各 rank 的 Loader 再丢本地尾批。10/3/2 表格四行与源码一致，重复样本不会因本地完整批次自动消失。
- Iterable 的消费者编号在统一 W、统一稳定源流前提下形成全局划分；实际对象应尽量按所有权读取。23/2/2/3 例中每 worker 丢尾批共丢 2 个样本，rank 步数 4/3，能暴露同步 DDP 不均匀输入问题。
- v2.8.0 worker 启动已 seed Python random、torch 与可用的 NumPy 全局 RNG，普通 persistent Dataset 复用时不重跑初始化；文章没有照搬过时的 NumPy 默认不 seed 说法，也未承诺仅靠 seed 跨设备逐 bit 一致。
- Pinned memory 是 host memory；non_blocking 的 host 返回、设备 copy/compute 重叠、数据就绪、释放后存储复用及显式 buffer 覆写各有不同条件。新增说明避免把 `record_stream` 当作原地覆写互斥锁。
- 恢复边界与模型/优化器一致；不仅数据游标，预取期间超前推进的随机变换也要回到一致状态。未宣称普通 DataLoader 自带完整中途快照或 exactly-once 训练协议。

## 审核修正

1. 开头加“假设”，使 6/12 ms 明确为问题设定而非本机测量。
2. 流水公式明确 `K >= 1` 和正阶段时长，避免将含空批数分支的教学函数与正批数公式混同。
3. 将 stream 表中的“GPU 存储复用”限定为释放后的分配器复用；补充固定 CUDA 双 buffer 再写入前需要等待旧计算读完，`record_stream` 不阻止程序显式 `copy_`。
4. 补充 worker RNG 也可能因预取超前：从消费位置重读时，保存当前 worker RNG 不一定复现原增强，需按样本随机数、确定性重放或一致保存完整 pending 状态。

## 独立补充测试与实际执行

保留写作阶段的逐 tick 双服务器 oracle、矩阵转置 sampler oracle、消费者轮流投递 oracle 和参数拒绝测试。独审新增：

- 对长度 1–6 的全部 873 个排列、rank 数 1–4 以及 sampler 补齐/截断两种策略，共 6984 组配置，以 Counter 验证共享全局排列的覆盖及重复计数。不把 Python 排列当作 torch.randperm 的输出。
- 构造各 rank 分别排列导致 `[0,2,4]` 与 `[2,4,0]` 重复、遗漏奇数样本的反例。
- 只按本地 worker 分片、忘记 rank 的反例：整个数据源被每个 rank 各消费一次。
- 同一精确分片在保留各 worker 尾批时有 4/4 步、丢尾批时为 4/3 步；验证丢失的 2 个样本。
- 100 批已消费、116 批已派发的纯游标模型：从派发边界恢复会跳过 101–116。这不是 worker RNG、队列快照或完整训练恢复的运行测试。

实际运行 `node scripts/reviews/dataloader-examples.mjs`、`node scripts/audit-posts.mjs` 和 `git diff --check`，均通过；元数据检查覆盖 102 篇、10 个系列、24 个既有标签。专项执行的是从正文提取的 Python 标准库算例和独立 oracle；没有导入 Torch、启动真实 DataLoader/DDP、加载模型或执行 CUDA。没有把数学调度模型换算成实际速度收益。

未发现剩余阻止发表的概念、逻辑、公式或教学代码合同问题。构建、最终渲染、匹配提交的部署状态与线上正文仍由主代理独立确认。

## 主代理发布前检查

重新运行正文提取专项、102 篇元数据检查、生产 Jekyll 构建、生成站点与数学标记验证，均通过：118 个 HTML 页面、143 个生成文件；95 篇数学文章中 1549 个原生行内、1058 个块级公式。浏览器读取本篇最终本地页面：18 个 MathJax 容器、0 个公式错误、无裸露 `$$` 或页面横向溢出；另检查流水公式段落截图，公式与正文排版正常。既有 35 个 JavaScript 专项及 full-agent-examples.py 也通过。这里只记录发布前的本地结果；本提交的 CI 和线上结果须在推送后另行确认。
