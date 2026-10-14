# 模型量化系列写作与验收计划

制定日期：2026-10-15。系列 ID：`model-quantization`。

## 目标与边界

完成一个连贯的模型数值量化系列，共 12 篇：新增 8 篇，重新审核并整合既有 4 篇。不是金融量化，也不是只列方法名称的综述。每篇针对自己的问题完整展开，并提供可核对的推导、例子与来源。

系列顺序按核心技术首次公开时间安排，不修改旧文发布日期或 URL。基础篇以 Jacob 等人的 2017 年公开论文为教学锚点，不将其称为量化起源；收尾篇是 2026 年的跨方法工程综合，不宣称其中的评估方法始于 2026 年。旋转篇以 QuaRot 的 2024 年 3 月公开为锚，后来的 SpinQuant 仅作有时间标记的比较；NVFP4 篇以 2025 年为锚，并单独交代 OCP MX 的更早规范时间。

## 全系列大纲

| 顺序 | 核心技术时间 | 文章主题 | 具体内容与关键例子 | 操作 | 本次状态 |
| --- | --- | --- | --- | --- | --- |
| 10 | 2017-12-15 | 从实数到整数的量化数值契约 | 仿射与对称量化、scale/zero point、ties-to-even、clipping、粒度、GEMM 修正、fake quant/STE；手算与标准库代码 | 新增 | 已上线，见下方发布证据 |
| 20 | 2022-08-15 | LLM.int8 异常值分解 | vector-wise 量化、按输入特征维拆分 INT8/FP16、阈值、近似误差与实际系统成本；小矩阵分解反例 | 新增 | 已上线，见下方发布证据 |
| 30 | 2022-09-12 | FP8 格式与缩放 | E4M3/E5M2、正常/非正规数、current/delayed scaling、scale 方向与 K 维约束；编码边界与等比例缩放反例 | 保留旧 URL，逐篇校订 | 全文复审与本地验收完成，待发布验证 |
| 40 | 2022-10-31 | GPTQ 层重建与误差补偿 | 校准输入、二次目标、近似二阶信息、逐列补偿、分块/Cholesky、damping；独立二次目标算例 | 新增 | 调研完成，逐篇写作待开始 |
| 50 | 2022-11-18 | SmoothQuant 等价通道缩放 | alpha 的真实作用、输入维与输出维、folding 分叉、W8A8 的运行时边界；缩放等价性与量化后误差 | 保留旧 URL，逐篇校订 | 已迁系列，待本轮全文复审 |
| 60 | 2023-05-23 | QLoRA 与 NF4 | 非均匀码本、分块 absmax、double quant、冻结基座与 LoRA 梯度、存储/计算精度、合并再量化；小码本与梯度数据流 | 新增 | 待调研 |
| 70 | 2023-06-01 | AWQ 保护显著权重 | activation salience、等价缩放的 group 耦合、clipping、W4A16 packing；权重 MSE 与输出误差不同的反例 | 保留旧 URL，逐篇校订 | 已迁系列，待本轮全文复审 |
| 80 | 2024-02-05 | KIVI 的非对称 KV 量化 | K/V 不同量化轴、group 与残余高精度区、增量写入、Attention 误差；实际布局和容量计算 | 新增 | 待调研 |
| 90 | 2024-03-30 | QuaRot 的正交旋转 | Hadamard 与离群值、图不变性、norm/残差/非线性/RoPE 边界、离线 folding 与在线开销；等价及不等价算例，比较 SpinQuant | 新增 | 待调研 |
| 100 | 2024-05-07 | W4A8 与 QServe | 两级整数表示、protected range、nibble 解包、scale 与零点修正、计算布局；枚举边界与 unpack reference | 保留旧 URL，逐篇校订 | 已迁系列，待本轮全文复审 |
| 110 | 2025，NVFP4 专题 | NVFP4 与块缩放 | E2M1、MXFP4 与 NVFP4 区别、块/全局 scale、metadata、转置和 kernel 契约；枚举 FP4 levels 与真实字节成本 | 新增 | 待调研 |
| 120 | 2026，工程综合 | 量化模型的校准与上线验收 | 校准/选择/独立评估的数据隔离、数值与真实 kernel parity、任务质量、P/D 性能、fallback、manifest/回滚；可运行的小型验收工具 | 新增 | 待调研 |

日期依据来自原论文 arXiv 的 v1 submission history；详细来源、日期及范围由相邻 `2026-10-15-quantization-series-research.md` 记录，未完成研究的条目不假装已阅读全文。文章标题可以在不改变主题与交付范围的前提下精修。

## 已有文章的处理

迁移以下四篇的系列 metadata，保留文件名、发布日期与 URL，修正与新导航冲突的“下一篇”口吻。迁移不等于完成本轮内容复审。

- `_posts/2026-08-01-fp8-formats-scaling-llm-inference.md`
- `_posts/2026-08-02-smoothquant-w8a8-activation-outliers.md`
- `_posts/2026-08-03-awq-activation-aware-weight-quantization.md`
- `_posts/2026-08-04-w4a8-progressive-quantization-kernel.md`

NVFP4 KV Cache、QuantSpec、FP4 Attention 等既有文章继续保留在原主系列，通过相关内容链接衔接，不为凑数量重复发表同一篇。GPU Runtime 系列仍保留 Roofline、Triton、CUDA Graph、编译器与运行时等主题。

## 每篇的完成门槛

1. 调研者读取原论文/官方规范/相关实现的实际段落，提供明确大纲和 claim/source 对照；来源中有实验数字时，保留模型、硬件、任务与版本条件。
2. 写作者按本篇问题组织完整推导和例子，不贴“新手”等读者标签，不批量套同一结构，不提前写所有文章。
3. 独立审核者核对概念、逻辑、公式、单位、维度、边界和参考资料；执行正文真实代码，并使用不同 oracle 或反例检查，而不是只复制同一公式自证。
4. 明确区分：数学等价、fake quant 误差、真实 packing/kernel 正确性、真实模型质量、目标硬件性能。只声称实际验证过的层次，不将 CPU 教学算例称为 GPU/训练复现。
5. 站点元数据、链接、代码与数学检查通过；实际构建，并在浏览器检查公式与排版。
6. 每新增一篇，独立提交并推送；旧文内容修改按单篇形成提交。核对对应提交的 CI/部署与线上实际页面，之后才登记“上线”。

## 最终完成审计

必须逐条核对上表 12 篇均有完整正文、本轮独立审核记录、可执行验证和已发布页面；四篇旧文也不能仅凭旧测试通过而跳过。检查系列完整顺序、没有重复位置/倒退年份、所有计划链接有落点、相关旧文 URL 不变，并汇总尚未执行的真实模型/GPU 实验边界。完成审计不能由“计划写完”“元数据通过”或工作流绿色单独替代。

## 执行记录

### 系列建立与旧文导航迁移

独立审核者核对了 12 篇范围、四篇旧文 metadata 和末段、十项论文原始公开日期；未发现迁移阻断。原文件名、发布日期、技术年份与 URL 均未变。

本地执行：90 篇文章、10 个系列、24 个标签的结构检查通过；20 个 JavaScript 审核算例套件和既有 Python 套件通过；生产构建生成 104 个 HTML 页面和 129 个文件；83 篇数学文章中的 538 个行内公式和 896 个独立公式通过结构检查。浏览器实查量化系列四篇按 FP8、SmoothQuant、AWQ、W4A8 排列，原 URL 正确，1265 像素视口没有整页横向溢出。

本段只证明计划、迁移和上述检查，不证明四篇旧文的本轮全文审核已经完成；后者继续按上表逐篇推进。

发布证据：迁移提交 `03dc6f3f9acdbd299a09a5a80abe671fac12ba9f` 已推送，匹配的 [工作流 38612046754](https://github.com/Charlestar/Charlestar.github.io/actions/runs/38612046754) 的 build/deploy 均成功；浏览器已从线上 `/series/#model-quantization` 读回四篇标题与原链接。

### 第一篇 数值基础

正文 `_posts/2026-10-15-quantization-numerical-foundations.md`；三角色研究、写作与独立审核完成。正文三块代码直接执行，9 组专项检查通过，包含 1,395 个精确有理数格点用例；root 补充修复 GFM 误解析行内绝对值的排版问题和构建回归检查。

文章提交 `d9a40f8071985322a4d7dc65ac723626576c7cdd` 已推送，匹配 [工作流 38613320553](https://github.com/Charlestar/Charlestar.github.io/actions/runs/38613320553) 的 build/deploy 均成功。浏览器已读回 [线上文章](https://charlestar.github.io/2026/10/15/quantization-numerical-foundations/)：97 个 MathJax 容器、0 个渲染错误、无双美元残留，实际带有最终编码范围 guard。此证据不扩展为真实模型/GPU/QAT 实验。

### 第二篇 LLM.int8

正文 `_posts/2026-10-15-llm-int8-outlier-decomposition.md`；两块正文代码直接执行，7 组独立专项检查通过，包括行/列 scale、整列并集、阈值/空集合/零向量、数值中间溢出与下溢、固定版本实现差异和误差抵消反例。全部 22 个 JavaScript 套件、既有 Python 套件、生产构建及本地页面检查通过。

提交 `15ea4dea0d69e7b5f4b06a9afb2426590b5bcd82` 已推送，对应 [工作流 38614546451](https://github.com/Charlestar/Charlestar.github.io/actions/runs/38614546451) 的 build/deploy 均成功。浏览器实际读回 [线上文章](https://charlestar.github.io/2026/10/15/llm-int8-outlier-decomposition/)：46 个 MathJax 容器、0 个错误、无双美元残留；最终 scale_product guard 和非单调改进边界都在正文中。这里没有执行真实 bitsandbytes、模型或 GPU benchmark。

### 第三篇 FP8 旧文全文校订

保留原日期与 URL，完成独立原论文、固定 TE v2.3 源码与 API 核对，补充 FN/FNUZ、特殊值、current/delayed 与 block/MX 区别。8 组专项含 512 个格式/byte 组合的独立有理数码本和全部相邻格点边界；全部 23 个 JavaScript 套件、Python、元数据、生产构建与页面/数学检查通过。本地浏览器读到 63 个 MathJax 容器、0 个错误、无单双美元残留，格式表截图排版正常。发布证据待推送后单独登记；没有真实 TE、CUDA 或模型实验。
