# LLM.int8 独立审核记录

审核日期：2026-10-15。正文：`_posts/2026-10-15-llm-int8-outlier-decomposition.md`。

本篇在 writer 静止后全文审核并修订，覆盖概念、矩阵维度、公式、数值算例、正文代码、引用与实验边界。

## 独立阅读的一手来源

- [LLM.int8 arXiv v2](https://arxiv.org/html/2208.07339v2)：实际阅读 §3.1–3.4 的逐向量量化、列集合分解、质量实验与结果；核对 §4 的异常维经验观察与 §6 的对象/限制；逐段阅读附录 D.1–D.2 以及表 5/6，区分 raw GEMM、Q/DQ/decomposition 与多 GPU 端到端推理。未完整复读训练/微调附录，也未复现论文性能与模型质量。
- [论文版本历史](https://arxiv.org/abs/2208.07339)：首发 2022-08-15 与采用的 v2 版本 2022-11-10 分开；`technology_year: 2022` 没有被写成所有相关技术起源。
- [bitsandbytes 0.47.0 functional API](https://huggingface.co/docs/bitsandbytes/v0.47.0/en/reference/functional)：实际核对 `int8_linear_matmul`、`int8_mm_dequant`、`int8_vectorwise_dequant` 和 `int8_vectorwise_quant` 的参数、返回与 `threshold=0` 禁用规则。
- [0.47.0 default/ops.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/default/ops.py#L30)：实际阅读混合支路、scale 恢复、矩阵乘与 vectorwise quant。混合支路从原始 A 取 selected columns，却从 CB 反量化相应权重；stat 计算前仅临时屏蔽逐元素 outliers，计算后才整列清零 CA。本文没有将教学重算 R 尺度方案称为这一实现的逐位复制。
- [0.47.0 cuda/ops.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/cuda/ops.py#L23)：实际阅读 matmul、dequant 与 vectorwise quant；确认内部 K 不被 4 整除时用 FP32 matmul 后转 INT32、正阈值用 `>=` 检测列、多个 token 行时整列清零。没有推断所有版本/硬件都有相同 fallback。
- [0.47.0 autograd/_functions.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/autograd/_functions.py#L146)：实际跟读 MatMul8bitLt.forward 中输入 FP16 转换、activation quant、weight CB 缓存与 mixed/scaled 分支，核对模块 B[N,K] 和文中 W[K,N] 的转置。未运行该 autograd 路径。

## 本轮实修与保留的边界

1. 初稿 `scale_x * scale_w * acc` 可能发生静默中间下溢。例：输入与权重均为 `1e-160`，最终点积约 `1e-320` 在 Python 中仍可表示，但两个 scale 先相乘成为 0。修为明确检查 scale product；非零 acc 遇到零乘积时拒绝，同时拒绝非有限乘积。代码旁注明这是受限的教学参考，不伪装成任意动态范围的数值库。
2. 阈值先受控转换为 float，不能转换或过大整数溢出时也返回清楚的 `ValueError`，再检查有限、非负条件。保留 threshold 0 禁用拆分，而不是按 `>=0` 选择所有列。
3. 正文补明拆分后输出误差不保证单调改善。实际程序给出反例：`X=[0.125,0.125,0.125,0.375,100]`、`W=[1,1,1,-1,1]^T`，原始和不拆分结果都是 100；重算常规尺度后输出约 99.9970472441。它说明误差抵消可能改变，不是模型质量结论。
4. 保留并独立核查：行/列 scale 的外积、整列 O 的多行并集、空 R/O、零向量、原始浮点权重与 CB 反量化权重的差异、default row stats 与教学统计集合不同、INT8 payload 不代表必然用 INT8 Tensor Core、FP16 支路仍有舍入、projection 不等于 attention function/KV quantization。

## 已执行的正文专项验证

命令：`node scripts/reviews/oct15-llm-int8-examples.mjs`。Windows 通过 `BLOG_PYTHON` 指向当前 Python；脚本只依赖标准库，不导入 PyTorch 或 bitsandbytes。

7 个不同语义的验证组通过，没有将每个矩阵元素或参数组合计成独立组：

| 验证组 | 独立证据 |
| --- | --- |
| 两块真实正文代码 | 根据唯一 marker 从最终文章提取 single-row 与 matrix-decomposition，各自独立执行自身断言，解析真实打印结果，确认没有遗漏其他 Python 块。 |
| 向量与外积缩放 | 在二进制精确 scale 的正负 tie、零向量和空向量上，用 Fraction 枚举全部 255 个重建 level 的距离/偶数优先级核对实际 quant_vector；另由重建后的矩阵做独立 rectangular GEMM，核对逐行/逐输出列缩放。 |
| 列并集与矩阵分解 | 多行、多输出、由不同行触发不同列，检查 selected column 内未超阈值值也归入浮点支路；对同一矩阵的全部 8 种列划分，在精确有理数域核对关闭量化的完整矩阵恒等式。 |
| 阈值与空支路 | 覆盖正负阈值等号、`nextafter` 相邻值、0/负零禁用、全部 O/全部 R、K=1、全零行和全零权重输出列、追加 token 行时并集不缩小。 |
| 输入与数值保护 | 覆盖空/不矩形/不匹配矩阵、非有限或不能转换的元素/阈值、最小 subnormal 导致 scale 下溢、scale 乘积下溢、scale 乘积/最终恢复溢出、浮点支路累加和两路相加溢出。 |
| 具体实现差异反例 | 构造被 CB 量化后选中 weight 0.1 重建为约 13/127，异常 activation 即使不量化仍留下 weight 误差；并核对第二行在源码元素屏蔽统计下 absmax 0.5、教学整列移出下 absmax 0.3，code 因此可不同。这里只执行数值构造，没有执行或模拟认证 bitsandbytes kernel。 |
| 不保证单调改进 | 原始三维例子确实改善但保留误差；上述五维 cancellation 反例却变差，阻止将一个教学例子推广成所有矩阵误差/任务质量保证。 |

`node scripts/audit-posts.mjs` 当时通过 92 篇文章、10 个系列、24 个标签。`git diff --check` 没有空白错误。静态通过不等于其他 90 篇完成本轮复审。

## 未验证事项及交接

未运行 bitsandbytes、PyTorch、原生 FP16/INT8/FP32 GPU kernel、真实模型或生产推理；未复现论文质量、显存或速度。源码阅读支持对应固定版本的描述，不能证明当前设备实际命中该路径。Python 任意精度整数也没有模拟 INT32 overflow。

本文内容审核与标准库专项检查完成；root 接手后继续实际 Jekyll 构建、页面链接/公式检查、浏览器排版、逐篇提交推送、对应提交工作流与线上页面验证，并按实际结果补记。此处不预先宣称已经发布。

## Root 的最终本地验证

移交后的静止稿直接提取执行，7 组专项再次通过。全部 22 个 JavaScript 审核套件和既有 Python 套件通过；92 篇文章、10 个系列、24 个标签的结构检查通过。

生产 Jekyll 构建成功；107 个 HTML 页面和 132 个生成文件的站点校验通过。数学检查覆盖 85 篇生成文章中的 653 个行内公式和 924 个独立公式，并通过双美元泄露检查。

本地浏览器实际读回本文：46 个 MathJax 容器、0 个数学错误、无双美元残留，1265 像素视口下没有整页横向溢出。实际查看异常列集合、互补列集合与矩阵分解公式的截图，段落、公式与来源链接显示正常。线上发布仍须在本篇单独提交推送后另外核验。

后续线上核验：提交 `15ea4dea0d69e7b5f4b06a9afb2426590b5bcd82` 的 [工作流 38614546451](https://github.com/Charlestar/Charlestar.github.io/actions/runs/38614546451) 的 build/deploy 均成功。root 从线上读回真实标题、最终 scale_product guard 和输出误差不保证单调下降的段落；46 个 MathJax 容器、0 个错误、无双美元残留。审核用五维数值反例完整保存在专项脚本，正文用相应原理限定结论，不将其称为模型质量实验。
