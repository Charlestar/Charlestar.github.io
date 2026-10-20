# PyTorch 显存分配与 OOM 快照独立全文审核

日期：2026-10-21。对象：`2026-10-21-pytorch-cuda-memory-allocator-snapshot.md` 全文、调研记录与 `allocator-snapshot-examples.mjs`。写作代理交接冻结后才修改本文，未并发编辑。

## 独立读取与核验

独立读取 PyTorch 2.8 快照文档、CUDA memory management/stream 文档、`record_stream`、`empty_cache`、`device_memory_used` API 说明及 2023-12-14 官方 Memory Snapshot 介绍。固定源码为 PyTorch v2.8.0：读取 `torch/cuda/memory.py` 统计与 history/snapshot 接口，以及 `CUDACachingAllocator.cpp` 的策略注释、分配/释放计数、snapshot block 汇总、recordStream、事件处理、release_cached_blocks、同步后回收与 graph 私有池边界。

逐段核对结论：

- `del` 删除名字而非强制释放共享存储；view、容器、图与其他所有者可延长存储寿命。detach 不等于搬到 CPU 或释放被保留的 tensor。
- `free_completed` 是分配器安全复用边界，不是所有设备工作物理完成的全局屏障；同 stream 后续写入仍排在旧使用之后。跨 stream 生命周期必须正确记录或手动同步，显式 buffer 覆写另有依赖。
- Snapshot 状态按 block size 分区时 `R=A+P+I`、`active=A+P` 正确。allocated 不等于所有逻辑 tensor 字节之和，reserved-allocated 不是纯碎片，更不是立即可用于任意请求的容量。
- Native free() 已减少 allocated_bytes，但 requested_bytes 直到 free_block() 才减少；pending 请求可以仍计入 requested current。教学格式的 requested_live 只统计 active_allocated，名称与原生统计明确分开。
- 24/6/2/16 MiB、18 MiB 差值、8 MiB 当前整段 inactive 与 0.5 MiB live block/request 差值正确；16 MiB 第一段仍有 live block，pending 即使安全完成也不能使它整段归还。
- 12 MiB 空闲、最大现成 4 MiB 的布局解释不能直接容纳 6 MiB 请求，但不预测必 OOM。stream/pool、driver 新申请、缓存回收和重试仍影响结果；解释的是虚拟区间而非物理 HBM 连续分配。
- Native 普通全局池 empty_cache 路径会同步 outstanding events 后再归还符合条件的整块；文章没有把静态 whole_inactive 统计当作 API 最终释放量，也没有断言 pending 永远无法被回收。
- 有界 history 的条数不是文件大小；两种视图、地址代次、OOM device_free 的含义与文档一致。采集先于故障；事后先同步/清缓存会改变现场。
- Snapshot 不包含所有 direct CUDA/NCCL 分配；device_memory_used 是设备全局量，减本进程 reserved 不能得到精确 NCCL 占用。不同 allocator backend、expandable segments 与 graph 私有池不能直接套普通固定段 toy。
- 2023 仅锚定 PyTorch 2.1 experimental 工具的公开介绍，不冒称 allocator 或快照概念起源。

## 审核修正

采集包装器原先保护了 OOM 分支中的 dump 失败，但 finally 关闭 history 若失败，仍可能遮盖原始异常。现以 `sys.exc_info()` 判断是否已有错误：已有 workload/dump 错误时保留原错误并报告 stop 失败；正常成功路径的 stop 失败则继续抛出，不伪装成完整成功。正文另明确包装器自行控制进程 history，不应嵌套或接管已有采集会话。没有为此加入后端或复杂采集框架。

## 独立补充测试与实际范围

保留从正文提取的真实 CPU 账本块，以及写作阶段的 81 组 block 状态组合逐字节所有权 oracle、pending 完成转换、布局/区间和参数拒绝测试。独审新增：

- 各状态组合的 `R-A=P+I` 与 `requested_live<=A` 恒等/约束；地址平移与字节缩放不改变分类，全部汇总按倍数变化。
- 只抽取采集函数 AST，以显式 fake torch/memory/Path 对象执行 Python 控制流，不导入 Torch、不操作文件、不执行 CUDA。核对成功结果返回、history 上限、成功/失败快照名与最终关闭调用。
- dump/stop 失败的四种组合都保留同一个原始 OOM 对象；普通 workload 错误在 stop 成功/失败时也保留身份且不冒充 OOM dump；原本成功的工作在 stop 失败时仍暴露错误。

实际运行 `node scripts/reviews/allocator-snapshot-examples.mjs`、103 篇的 `node scripts/audit-posts.mjs` 与 `git diff --check`，均通过。未执行 PyTorch 分配器、真实 snapshot、CUDA Graph、模型或 GPU OOM；stub 控制流并不是官方 API 兼容性或设备行为测试。

未发现剩余阻止发表的概念、逻辑、公式或教学示例合同问题。完整构建、页面渲染、提交/推送、匹配提交的 CI 与线上正文由主代理分别确认。

## 主代理发布前检查

主代理全文复核后，重跑正文提取专项与 103 篇元数据检查；生产 Jekyll 构建、站点及数学标记验证通过：119 个 HTML 页面、144 个生成文件，96 篇数学文章含 1559 个原生行内、1059 个块级公式。浏览器读取最终本地正文，确认异常清理修正版已进入页面，11 个 MathJax 容器、0 个公式错误、无裸露 `$$` 或页面横向溢出；账本公式截图排版正常。未执行真实 Torch/CUDA；本提交的 CI 和线上状态须在推送后确认。
