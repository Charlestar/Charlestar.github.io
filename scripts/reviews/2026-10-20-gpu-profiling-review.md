# GPU 性能剖析独立审核

日期：2026-10-20。对象：`2026-10-20-gpu-profiling-nsight-systems-compute.md` 全文、对应 research 记录与 `gpu-profiling-examples.mjs`。调研、写作和本次独审由不同代理承担；本记录不替代主代理的构建、页面和发布检查。

## 逐段核对

- 开篇与基线：CPU API 返回、GPU 完成、设备 event 区间、吞吐与请求延迟没有混为一谈。示例每个 step 同步，正文明确其串行主机完成延迟口径，没有把它扩为服务吞吐。计时前完成预热，baseline 不插入 NVTX。
- NVTX：push/pop 标注 CPU 范围；GPU 投影用于来源关联，不把 host range 长度当 kernel 执行时间。异常退出有 finally pop，采集 start/stop 前后有对应同步。
- Systems：按当前官方 User Guide 核对 capture-range、capture-range-end、sample 与 cpuctxsw；按 Post-Collection Analysis Guide 核对 cuda_gpu_kern_sum、cuda_api_sum、nvtx_sum。CPU sampling 关闭不代表无 CPU 问题，空白区也不直接等同 Python 瓶颈。
- DAG：2 + max(6, 9) + 1 = 12 ms；分支累计 15 ms 与区间并集 9 ms 不等同 wall time；计算减半仍 12 ms，通信减半为 9 ms、加速 4/3。条件是固定持续时间、只受显式依赖限制和可完全重叠，不是从 stream 数量推断真实并行，也不是 GPU 调度预测。
- Compute：按官方 CLI 核对 push/pop 的 target/ 语法、NVTX 开关和匹配 launch-count；一层算子可能发起多 kernel，第一项不承诺代表整个矩阵乘。没有混拼同名不同 shape/实例；说明了 launch skip 匹配顺序的区别。
- 计数解释：occupancy、eligible warp、发射率与峰值利用率保持分离；warp stall 高占比不必然处在未隐藏等待上；MemoryWorkloadAnalysis 的不同层流量未一律称作 HBM。
- Replay：核对 kernel replay、range replay、cache control、serialization 与 clocks 的范围；没有把 range replay 的整体计数归给单个 kernel。冷计数与暖耗时不能拼成同一实测点，NCCL mandatory concurrency 另列采集限制。
- 收尾：从假设到改变机制、数值语义检查、局部计数与正常运行重测的因果关系清楚，没有宣称已取得实际硬件收益。

本次独立打开并阅读了 NVIDIA Systems User/Analysis Guide、Compute Profiling Guide/CLI/Release Notes 的相关章节、CUDA 12.8 timing、PyTorch 2.8 NVTX 文档与 v2.8.0 profiler.py。没有以搜索摘要代替上述关键接口定义。

## 修正

1. 原 2019.1 archive.docs.nvidia.com 链接无法打开，改为现存官方 Release Notes 的 2019.1 历史章节，核对新增 nvtx-include/nvtx-exclude 的原始条目。2019 只是本篇选定的技术年锚，不宣称是任一工具的起源。
2. 原脚本只 ast.parse，正文及 research 的“静态编译 GPU 示例”改为“Python 语法解析”，避免把 Python AST 检查当成 CUDA 或后端编译。
3. CPU 时间验证补上超大整数转换失败与有限 duration 相加溢出；不会返回无限完成时间后继续声称合法结果。

## 独立反例与实际执行

保留写作阶段已有的 endpoint partition 并集 oracle 与路径枚举 oracle。另加入离散时间格占用集合 oracle，穷举 3375 组 3 区间输入；另用事件推进模拟（非最长路径递推、非路径枚举）检查全部 64 个四节点递增边 DAG 与 81 组含零时长的持续时间，共 5184 组。专门比较两个独立 6 单位任务与显式串行依赖的 6/12 区别，确认程序不擅自推断资源排斥。

已实际运行 `node scripts/reviews/gpu-profiling-examples.mjs`，9 个 CPU 语义组通过，并提取正文 GPU 块进行 Python AST 解析；`node scripts/audit-posts.mjs` 的 99 篇结构/元数据/系列/标签检查及 `git diff --check` 通过。上述穷举输入数量不是独立人工审阅文章数量。

没有导入或执行 PyTorch，没有执行 CUDA kernel、运行 nsys/ncu、生成 trace、采集硬件计数器或测量 GPU 性能。当前审核未发现剩余需要阻止发表的概念、公式或代码合同问题；构建、浏览器呈现、CI 与线上正文仍由主代理分别验证。
