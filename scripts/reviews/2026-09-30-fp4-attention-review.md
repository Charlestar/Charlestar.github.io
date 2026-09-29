# FP4 Attention 新文独立全文审核

- 审核日期：2026-09-30。
- 对象：`_posts/2026-09-30-fp4-attention-precision-dataflow.md`。
- 流程：research → writer 完稿并停止 → auditor 全文独立审阅、补修与复算 → 主审构建和发布。

## 实际阅读与证据边界

独立通读本篇全部正文、公式、表格、流程与数值示例。独立打开并核对以下一手材料：

- [Hardware-Aware FP4 FlashAttention-4 v1](https://arxiv.org/html/2609.04105v1)：实际读取 §1–§5、§6.1–§6.2、§7–§8、Appendix E，以及相关正文表格。重点核对 E2M1 code map、幅度 `/6` 和产品 `/36` 的约定、represented normalizer、FTZ 特例、N32/K64 所有权、前向/反向训练路线以及四种计时范围。未声称完整精读所有 56 页、全部附录和每个模型评测。
- [该预印本提交历史](https://arxiv.org/abs/2609.04105v1)：Robert Hu 单作者，首次提交 2026-09-03。
- [原 FlashAttention-4 v1 metadata/abstract](https://arxiv.org/abs/2603.05451v1)：Ted Zadouri 等作者，首次提交 2026-03-05。仅此范围用于核对两篇论文身份不同，不声称本轮全文复审原 FA4。
- [NVIDIA PTX ISA Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)：本轮页面显示 9.4，读取 §9.7.18.1、寻址与分配子节。只用来核实 SM100 的 128 行、512 列、32-bit cell 及按列分配，不把新文档的所有新指令能力归给论文二进制。
- [作者 RELEASE_STATUS.md](https://github.com/MrHuff/fp4-fa4/blob/main/RELEASE_STATUS.md)：2026-09-30 读取 main 页面，文件状态日期为 2026-09-03。读取 Current state、Current release checks、source pins、experiment coverage、supported training boundaries。它自报审计 snapshot 为 `5926d20188ec7a8a033e4efc7075f4c40325e3e8`；尝试固定该 commit 的 raw/blob 页面未获取成功，不能声称独立审计该 commit。未逐行审计 kernel 源码，也未运行其 verifier 或 CPU/GPU suites。

论文报告与公开 release 自报必须分别归因。主文保留 not fully GPU-validated、部分历史原始记录缺失、receipt-only 等边界，不能把离线重建图表当作重新测出 GPU 数据。

## 审核发现与修订

1. 核心公式与原创数值算例正确。`P` 与未归一化 `p`、block scale 与特殊 amplitude、非因果前向与 causal training、近似算子自洽与原始 Attention 等价均已区分。
2. 完稿前独立预审提醒已由 writer 纳入：affine 分类不是精确指数；`/6` 与 `/36` 不是所有 MXFP4 固有定义；K/V 联动排列不能在保留三角 mask 时直接推广到 causal attention；同量化 payload 重建不等于离散量化函数拥有精确普通梯度。
3. 依主审反馈，将技术 contract 的生硬译法“合同”替换为适合语境的“数值约定 / 编码约定 / 输入约定 / 精度与布局约定”。
4. 在完整单 GPU 更新性能段补充初始 logits 相对 BF16 已明显不同。短 synthetic timing 不只是观察时间不够，不能默认为质量一致；§7.5 原文将其明确限定为 performance-only。
5. 确认四种计时边界没有混用：预量化的非因果 kernel；反向 core 与加入 publisher；完整含低精度 projection 的 8B 更新；独立长期训练轨迹。2.023 倍几何平均、0.356/0.501/0.508 ms 和 854.516/751.722 ms 均保留各自条件。
6. 长期训练没有被包装成“质量相同”：两路线各一次、验证 loss 差 0.0900；不能隔离归因于 Attention；所有被测 MXFP4 P/V 发散也不能推广成任何 FP4 方法不可能训练。

## 已执行 CPU 检查

`node scripts/reviews/sep30-fp4-attention-examples.mjs`：11 组全部通过。

- E2M1 正值七个中点与显式 mathematical nearest-even codebook。正文示例均避开 tie；测试增加 tie 参考但不冒称执行硬件转换器。
- affine 在 `x=0` 相同 code、`x=1` 不同 code，以及 log-base 变换。
- `alpha*q/6` 与双操作数 `/36`。
- 常量 V 的一致 normalizer 得 2，错误分母得 `40/21`；零分母不可直接相除。
- 非常量 V 的 exact `12/7`、represented `1.5` 和 mismatch `10/7`。
- block scale 不可省略，以及逐 tile 独立 softmax 不能代替整行 softmax。
- 明确假设 output-FTZ 的 FP32 数学模型中，`(2^-126/6)*12` 与 `2^-126*(12/6)` 的不同中间范围；不是本机 GPU 测量。
- 非因果成对 K/V 排列保持加权和、只排 K 或 causal mask 不变的反例。
- TMEM `128*512*4=256 KiB`、四 bank 容量、两片 N32 发布和简化 ownership 不变量。
- Amdahl `0.55/0.65`、扩展 backward 计时后收益反转、完整更新比率。
- `cos(v,2v)=1` 但 relative L2 为 1、global batch 1024、validation gap 0.09。

## 未实测

没有运行 Direct-P converter、CUDA/PTX kernel、TMEM 并发、模型投影、反向传播、optimizer 更新或训练收敛。CPU 参考只验证正文指定数学和简化状态，不构成实现级正确性证明，更不构成 Blackwell 性能基准。页面公式和生产构建由主审另行记录。

## 主审发布前验证

主审通读全文及研究、独立审核和 11 组 CPU 脚本，重新执行脚本通过，并核对主论文 §2–§4、§7–§8 相关推导与证据。85 篇文章、9 个系列、24 个标签的元数据检查通过，未新增标签或改动既有文章。

production Jekyll 构建成功；站点检查通过 99 个 HTML 页面、124 个生成文件；公式检查通过 450 个原生行内公式、864 个展示公式，覆盖 78 个含公式的构建页面。

本地真实浏览器检测到本篇 23 个 MathJax 容器、零数学错误标记，无页面横向溢出。目视检查标题和前向/训练精度对照表，目录跳转可用。该验证不扩大上述 GPU、训练和复现证据边界。
