# Roofline 新文独立全文审核

- 审核日期：2026-09-09。
- 对象：`_posts/2026-09-09-roofline-gpu-performance-model.md`。
- 顺序：research → writer 完稿移交 → 独立 auditor 全文核对并修改。

## 实际覆盖与来源

完整审读资源下界与吞吐上界的推导、GEMM FLOP/byte 账本、三行数值表、潜在瓶颈与实际饱和的区别、occupancy、分层存储、有用与实际执行运算、量化路径、CUDA 计时、profiling 扰动、严格串行阶段及汇总下界、结论和参考资料。不是只查看差异。

独立核对以下原始资料中的章节或定义：

- [Williams 等 Roofline CACM 2009 原文 PDF](https://aiichironakano.github.io/cs596/Williams-Roofline-CACM09.pdf)：印刷页 66–69 的操作强度定义、DRAM 流量边界、可持续带宽、`min` 上限、ridge point 与 ceilings。原作者站点访问失败后使用原论文教学镜像；未声称整篇所有后续实验都已复现。
- [Roofline 原作者技术报告 §3](https://digital.library.unt.edu/ark:/67531/metadc934195/m1/1/)：实际打开文章所引的第 1 页，核对缓存与 DRAM 之间、经过缓存过滤后的流量定义。
- [NVIDIA Matrix Multiplication Background §1–3](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：GEMM alpha/beta、FMA 按 2 FLOP、`2MKN`、算术强度模型的局限、tiling 与 tile/wave quantization。
- [NVIDIA GPU Performance Background §3–4](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)：并行度、延迟隐藏、计算/内存/延迟限制与模型前提。
- [NERSC Roofline](https://docs.nersc.gov/tools/performance/roofline/)：performance envelope、Empirical Roofline Tool、hierarchical Roofline、runtime/FLOP/读写字节与存储边界。
- [CUDA C++ Best Practices Guide 12.8.0 §8.1、§10.1](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html)：CPU/GPU timing、event 的等待与 elapsed time、occupancy 的活跃 warp 比例定义以及高 occupancy 不保证高性能。
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：本次页面显示为 13.3；核对 §2.9 Roofline、§2.6 reproducibility 下的序列化/时钟/缓存控制、kernel/range replay 差异，以及 metrics/occupancy/scheduler 描述。未将特定 replay 模式的行为推广成所有模式必然如此。
- [Time-Based Roofline v1 §I、§II A–C](https://arxiv.org/html/2009.04598v1)：计算复杂度变化会使 FLOP/s 不能单独代表完成时间的动机；文章没有引用该论文的硬件实验数值。

上述记录只覆盖实际读过的关键章节、定义与机制，不宣称逐字精读全部来源，也没有运行其 GPU 实验。

## 发现与修正

1. 将较泛的 occupancy 描述改为 SM 上活跃 warp 数占最大允许活跃 warp 数的比例；补充 active 不等于可发射指令、理论可达值与运行时采样平均值应分开，添加固定 CUDA 12.8 文档近旁引用。
2. 正文增加可执行的标准库复算命令，明确没有执行 CUDA、硬件计数或真实 GPU 时间测量。
3. 配套脚本逐项匹配正文 120 TFLOP/s、2 TB/s、M=1/32/128，检查强度和两类时间的取整列；将最后一个工作量示例同步为正文的 `100/10 → 40/5`，并按正文 A/B 顺序列出串行阶段。
4. 脚本的流量变量改称 `modelBytes`，避免把冷缓存账本误认为任意缓存状态下的实测最低流量；补查非零 beta 额外读取旧输出的字节数。
5. 原文上界/下界方向、FMA 口径、冷缓存假设、三组 GEMM 数字、分层 `B·I` 对应、200/101 μs 串行例均正确，独立复算后保留。没有为了增加改动量而重写正确推导。
6. 保留经验屋顶不是不可越过物理定律、左/右区域不等于实际资源饱和、有用工作与执行工作不能混用、profiling 条件不能随意与正常运行时间拼接等关键限定。

## 已执行验证

`node scripts/reviews/new-roofline-examples.mjs`：8 组检查通过。

- 计算/带宽 ridge 的量纲，速率上界与时间下界的一致性。
- FP32 SAXPY 的两次读、一次写和 FMA 操作数。
- 冷缓存 GEMM/GEMV 的输出写入及非零 beta 旧输出读取。
- 正文三种 GEMM 行数的精确 FLOP/byte、强度、取整后的计算与搬运时间以及最大值。
- 一组无量纲严格串行反例，检查分阶段下界比汇总下界更紧。
- 正文 A/B 阶段各 100 μs、总下界 200 μs，而总量 Roofline 仅给出 101 μs。
- 分层流量各自匹配带宽，速率与时间表达式对应。
- 正文工作量降低后速度翻倍但 FLOP/s 从 10 降至 8 的示例。

## 未实测边界

脚本仅执行 JavaScript 标准库算术断言。没有 CUDA/GPU、Nsight、实际 kernel、真实缓存流量、时钟、精度误差或端到端 serving 性能验证。假想设备的上限和冷缓存账本不是任何产品的规格或 benchmark。脚本通过只支持上述有限算例及模型关系，不证明任意 GPU 实现的效率。

## 主审发布前验证

主审完整阅读正文及审核记录，在 auditor 完成正文修订后执行最终验证：

- `audit-posts.mjs` 通过：80 篇文章、9 个系列、24 个标签；新文未引入新标签，也未修改旧文章。
- `check-coverage.mjs` 通过原 77 篇历史审查记录的范围校验；另外 3 篇由各自的 `new-*-review.md` 记录覆盖，不冒充历史审核已包含它们。
- `run-examples.mjs` 的 10 个 JavaScript 套件、共 92 组检查通过，其中本轮新增 RoPE / Activation Checkpointing / Roofline 分别为 9 / 8 / 8 组；`full-agent-examples.py` 通过。
- 另以 Python AST 解析重计算文章中的 1 个 Python 示例块，语法检查通过；没有导入或运行 PyTorch。
- 生产构建成功，验证 93 个 HTML 页面和 118 个生成文件；73 篇含对应原生公式的文章中，306 个行内式与 819 个展示式保留检查通过。这一静态检查不替代浏览器排版验证。
- 真实浏览器重新载入 Roofline 最终页面，确认新的 occupancy 定义与复算命令已显示；52 个 MathJax 容器，无 `merror` 或 `data-mjx-error`，当前桌面视口无整页横向溢出。
- 目视检查双对数 Roofline 示意图、三行 GEMM 表、串行阶段表与 200/101 微秒公式，未发现排版异常。
- 前两篇的逐篇提交已推送，并分别确认 GitHub Actions 构建/部署成功：RoPE `326f16a`，Activation Checkpointing `e662839`；正式页面分别显示 90 / 63 个无错误的 MathJax 容器。第三篇部署状态需在本记录提交后另行确认，不提前声明。
