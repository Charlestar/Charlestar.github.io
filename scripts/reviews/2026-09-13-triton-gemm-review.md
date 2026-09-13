# Triton GEMM 独立最终审核记录

- 审核日期：2026-09-13。
- 正文：`_posts/2026-09-13-triton-gemm-tiling-masking.md`。
- 角色与范围：独立审核者在 writer 停笔交接后完整阅读正文，不仅审阅 diff；核对元数据、全文概念与推导、Python kernel/wrapper、全部手算例、引用版本及性能边界，并直接补修。构建、浏览器、提交与部署由主进程另行验证。
- 结论：下述补修及 CPU 检查完成后，未发现尚未处理的发布阻断项。解释性 kernel 未在真实 Triton/CUDA 环境编译运行，不作生产资格或性能保证。

## 实际独立阅读的一手来源

1. [IBM 论文元数据与摘要](https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations)：核对标题、MAPL、2019 年与语言而非推理服务器的背景。[原论文](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)实际读 §3（语法、tile/broadcast 语义、SPMD）与 §5.2（分层 tiling、coalescing、shared memory 分配及同步），并查看首页元数据；不声称通读全部论文或复现其实验。
2. [官方编程模型说明](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)：阅读 blocked-program 与 CUDA/thread 映射解释。该页为动态资料；正文代码的版本核对以以下 v3.1.0 源码为准。
3. [v3.1.0 矩阵乘教程](https://github.com/triton-lang/triton/blob/v3.1.0/python/tutorials/03-matrix-multiplication.py)：完整阅读教程文件的算法、stride 地址、M/N modulo、K mask、grouped mapping、kernel、wrapper、精度检查及 benchmark 定义；仅阅读，未执行官方 benchmark。
4. [v3.1.0 core.py](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/core.py)：针对 `program_id`、`arange`、形状创建、`load`、`store`、`where`、`dot`、`tensor.to` 阅读相关定义和接口说明；核对 mask 的保护位置、store dtype 转换、where 两分支求值。
5. [v3.1.0 semantic.py](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/semantic.py)：阅读 `program_id`/`arange` 返回 int32、整数提升与二元检查、`add`/`mul`，以及完整 `dot` 实现。核对乘法前提升索引、块维最小限制、输入类型组合与 FP32 accumulator 合同。
6. [v3.1.0 autotuner.py](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/runtime/autotuner.py)：阅读 `run` 缓存键构建与选择、`Config` 和 `autotune` 说明；核对 tensor dtype 参与 key、stride 不自动全部参与、warp/stage 与重复运行的边界。
7. [NVIDIA Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：阅读矩阵乘背景、算术强度、tile 效率/并行度、tile/wave quantization 的相关章节（§1–§3.2）。未把历史设备示例的速度当作本文环境测量。

## 审核中实际修改

- 修复示意 kernel 的大 stride 整数偏移风险：在乘法前将 program 编号、M/N/K 索引提升为 int64。K 索引每轮增加 BK，再从固定基址构造指针，避免循环编号先做 int32 乘法；同步改写说明以排除重复推进基址。
- 新增 `65536 * 65536` 在 32 位回绕为零的地址反例，说明 mask 无法修复已错误的地址，以及 int64 仍受合法 view、可表示地址和设备 launch 限制约束。
- 新增配套 CPU 复算入口与验证边界。没有把纯 JS 参考说成 Python/Triton kernel 测试。
- 保留并复核：一维 program 的唯一输出 ownership，三轴独立 mask/零贡献，正 stride 转置，FP16 输入/FP32 累加/FP16 输出，同设备 CUDA context、输出无别名、无 backward 的限制，以及 grouped mapping 不保证执行顺序。
- 重算 stage 载荷/FLOP 比、BK 相消、accumulator 16/64 KiB、主例有用 FLOP 与规则 tile FLOP；未发现需更改的算术。

## 已执行检查

在仓库根目录运行 `node scripts/reviews/sep13-triton-gemm-examples.mjs`，最终 8 组通过：

1. 多组 M/N/K 和多种 tile 的尾部 mask，逐元素对照朴素矩阵乘，并检查每个合法输出恰好写一次。
2. A/B 普通转置与带间隙正 stride 的地址参考；正文 `B[1,0] = 11` 与错误连续解释的 12。
3. 只 mask store 不足以阻止无效 K 读污染，K 取模错误得到 36 而非 32。
4. 32 位乘积回绕及“事后转 64 位不能恢复”的整数模型，无大内存分配。
5. 主例 130×70×65 对照完整朴素结果：9 个 program、3 轮 K、右下角 2×6 个有效输出、末轮仅 1 个合法 K。
6. 有用与规则 tile FLOP：1,183,000 与 3,538,944，约 33.4%；另含一组非整齐形状。
7. 64/128 方形 tile 的 stage 算术强度、BK 相消与逻辑 accumulator 大小。
8. `Math.fround` 串行 FP32 累加舍入反例，明确不把 FP32 当作实数精确求和。

## 未执行与限制

- 未安装或运行 PyTorch、Triton、CUDA，未编译正文 kernel，未做 GPU 内存检查、Tensor Core 数值对照、autotune、计时或 cuBLAS 比较。
- JS GEMM 使用 CPU 数值参考，不模拟 GPU 指令布局、并行累加顺序或实际 FP16 load/store。FP32 单独反例也不是 Tensor Core 模拟器。
- 引用核查按上述实际阅读章节/函数记录，不声称通读每个源码仓库。Jekyll 页面与部署验证不属于本记录的执行证据。

## 主审发布前复核

主审另行完整阅读正文，抽核 v3.1.0 `semantic.dot`、`core.where`、autotuner 缓存键与 Config，重读修改后的 64 位索引循环和合同说明。本篇 8 组 CPU 复算、83 篇 metadata/taxonomy 检查通过；一个 Python 代码块以 UTF-8 输入通过 AST 语法解析，没有导入或编译 Triton。

生产模式 Jekyll 构建、97 个 HTML 页面/122 个生成文件检查通过；原生公式检查保留 422 个行内及 849 个显示公式，覆盖 76 篇有公式的文章。计数对应当时工作区（含尚未发布的第三篇），不是已经在线的文章数量。

使用 computer-use 在本地构建页面观察到 49 个 MathJax 容器、零数学错误节点、无页面横向溢出；目视检查文章标题、stage 输入载荷/算术强度公式和长 kernel 的代码高亮。仅检查默认桌面视口，不声称已覆盖全部设备。提交仅包含本篇正文与对应 research/review/复算脚本，线上部署另行验证。
