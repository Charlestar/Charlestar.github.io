# 模型量化系列：覆盖差距、边界和原始来源索引

调研日期：2026-10-15。主协调已确认范围为神经网络模型量化，不扩展到金融量化。本文是研究过程记录，不是发布状态或审核通过声明。

## 现有覆盖与整合决策

实际检查了 `_posts` 的标题、元数据、章节和相关内容：

| 现有文章 | 覆盖 | 整合原则 |
| --- | --- | --- |
| `2026-08-01-fp8-formats-scaling-llm-inference.md` | FP8格式、缩放、精度路径、硬件/验证 | 迁入量化系列，逐篇核对，不重复新增FP8文章；原URL不变 |
| `2026-08-02-smoothquant-w8a8-activation-outliers.md` | 等价缩放、alpha、图融合、W8A8工程 | 同上，保留原URL |
| `2026-08-03-awq-activation-aware-weight-quantization.md` | activation-aware缩放、group、packing、质量 | 同上，保留原URL |
| `2026-08-04-w4a8-progressive-quantization-kernel.md` | QServe/QoQ、两级整数、INT8 protected range、kernel | 同上；须区分整数QServe和其他FP8 W4A8，保留原URL |
| `2026-06-02-nvfp4-kv-cacheblackwell.md` | NVFP4 KV缓存、K/V误差、部署 | 保留KV系列，作为格式篇后的专题阅读，不搬成重复一般格式文 |
| `2026-09-30-fp4-attention-precision-dataflow.md` | FP4 attention、P、scale、流水交接 | 保留Attention系列，交叉链接，不声称低位浮点通识已完全覆盖 |
| `2026-05-21-QuantSpec-...md` | 分层KV量化与推测解码缓存事务 | 保留推测解码系列；其中KIVI引用不等于已有KIVI完整原理文 |

全文检索未发现 LLM.int8()、GPTQ、QLoRA/NF4、KIVI、QuaRot/SpinQuant 的独立完整文章。QAT在既有文章只是术语和选项，不等于完整教材。

## 经主协调确定的有界十二篇

排序依据是本篇核心技术的首次公开时间；同篇可以解释明确标注日期的后续分支。数值基础和工程验收是统一铺垫/综合，不冒称某年才“发明量化/验证”。既有四篇迁移而不是再造URL。页面出版日期与技术公开年份分别保存；不回填历史出版日期。

| 顺序 | 技术时间锚 | 主题与核心问题 | 本次动作 |
| --- | --- | --- | --- |
| 10 | Jacob arXiv 2017-12-15（CVPR 2018） | 数值基础：code、scale、zero point、rounding、clipping、granularity、integer dot、fakequant/STE边界 | 新增 |
| 20 | LLM.int8() 2022-08-15 | INT8为何被activation outlier破坏，vector-wise normalization与混合精度分解怎样协作 | 新增 |
| 30 | FP8 Formats 2022-09-12 | E4M3/E5M2、表示范围、scale生命周期、operand/accumulator/output | 改造既有FP8 |
| 40 | GPTQ 2022-10-31 | 用calibration input重建层输出，逐列量化与二阶误差补偿、Hessian/damping/ordering | 新增 |
| 50 | SmoothQuant 2022-11-18 | 在图等价前提下迁移量化难度，alpha折中与W8A8执行 | 改造既有SmoothQuant |
| 60 | QLoRA 2023-05-23 | NF4 codebook、double quant、frozen base与LoRA梯度、paged optimizer、merge边界 | 新增 |
| 70 | AWQ 2023-06-01 | 激活幅值与显著权重通道、等价缩放、group交互、clipping及W4A16 packing | 改造既有AWQ |
| 80 | KIVI 2024-02-05 | Key与Value不同误差轴、channel/token非对称策略、streaming分组和residual buffer | 新增 |
| 90 | QuaRot 2024-03-30；篇内SpinQuant 2024-05-26 | 随机/学习正交旋转如何保持函数并重分布outlier；RMSNorm、残差、非线性和在线变换边界 | 新增 |
| 100 | QServe 2024-05-07 | W4 storage供应A8 compute，progressive quant、保护范围、重排和epilogue | 改造既有W4A8 |
| 110 | NVFP4详细官方格式说明2025-06-24；前史MX规范Sep2023 | E2M1、MX32/E8M0与NV16/E4M3+FP32、scale量化和真实存储成本 | 新增 |
| 120 | 2026工程综合，无首创主张 | 从参考Q/DQ、packing、kernel到任务质量与serving SLO的验收；可复现实验设计 | 新增 |

第110篇不能写“MXFP4出现在2025年”，也不把2025-06-24细节介绍页当作已证明NVFP4首次提名日期。第90篇按QuaRot锚排在QServe之前，即使后半SpinQuant晚于QServe。若要求严格每篇所有讨论内容都在下一篇之前，任何含后续对比的文章都无法满足；此处按核心技术公开顺序组织并清楚标注篇内分支日期。

## 核心原始日期与当前实际读取范围

| 来源 | 核验日期/内容 | 本轮已读范围 |
| --- | --- | --- |
| [Jacob1712.05877v1](https://arxiv.org/html/1712.05877v1) | 2017-12-15；Affine integer-only和simulated quant | 元数据；§2.1–2.4、§3–3.2；详见第一篇research |
| [LLM.int8()2208.07339](https://arxiv.org/abs/2208.07339) | 2022-08-15 | 目前元数据、摘要；方法正文待第二篇逐段研究 |
| [FP82209.05433](https://arxiv.org/abs/2209.05433) | 2022-09-12 | 目前元数据、摘要；格式/缩放正文待改造该篇时复核 |
| [GPTQ2210.17323](https://arxiv.org/abs/2210.17323) | 2022-10-31（会议ICLR2023不改锚） | 目前元数据、摘要；算法正文待第四篇研究 |
| [SmoothQuant2211.10438](https://arxiv.org/abs/2211.10438) | 2022-11-18（会议ICML2023不改锚） | 目前元数据、摘要；具体公式待第五篇复核 |
| [QLoRA2305.14314](https://arxiv.org/abs/2305.14314) | 2023-05-23 | 目前元数据、摘要；NF4/block quant/反向/内存待该篇研究 |
| [AWQ2306.00978](https://arxiv.org/abs/2306.00978) | 2023-06-01 | 目前元数据；正文待该篇逐节复核，不把2026修订当首发 |
| [KIVI2402.02750](https://arxiv.org/abs/2402.02750) | 2024-02-05 | 目前元数据；K/V误差与算法正文待该篇研究 |
| [QuaRot2404.00456](https://arxiv.org/abs/2404.00456) | 2024-03-30 | 目前元数据；切勿误用其他ID |
| [QServe2405.04532](https://arxiv.org/abs/2405.04532) | 2024-05-07 | 目前元数据；§4两级与protected范围待该篇复核 |
| [SpinQuant2405.16406](https://arxiv.org/abs/2405.16406) | 2024-05-26 | 目前元数据；学习rotation参数化/约束待该篇研究 |
| [OCP MX v1.0 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) | 规范封面Sep2023 | 搜索返回规范原始片段E8M0/运算；尚未逐页读取完整规范，不作全文已读声明 |
| [OCP联盟公告](https://www.opencompute.org/blog/amd-arm-intel-meta-microsoft-nvidia-and-qualcomm-standardize-next-generation-narrow-precision-data-formats-for-ai) | 2023-10-17联盟公开公告 | 实际读公告开头，发布MX v1.0 |
| [NVIDIA NVFP4详细介绍](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) | 2025-06-24 | 已读日期、摘要；正式格式段落待第11篇研究；不使用AI-generated summary当格式规范 |

以上“待研究”不是写作许可。每篇正文开写前必须新建对应research，真正阅读方法相关段落和必要源码，再交writer；不得把此索引当“所有论文全文读过”。

## 质量门槛和建议可执行反例

- 数值基础：nearest-even位置与z奇偶、clipping界、全零scale、轴广播、zero-point dot展开、真实阶梯导数对STE；第一篇详细research已有算数结果。
- LLM.int8()：一个大activation通道拉大全row scale；分离该维后普通维分辨率恢复；FP16支路与INT8主路相加，不是所有元素永久转INT8。阈值不是跨模型定理。
- FP8：枚举格式有限值/特殊码；payload缩放和双operand rescale；current与delayed scale突变时的溢出；decode weight-only与GEMM FP8不混。
- GPTQ：二维正定H中逐列补偿最小化error；H=XX^T取决于样本方向，不能只看weight MSE；rank deficient与damping。
- SmoothQuant：关闭量化验证Xdiag(s^-1)diag(s)W等价；加量化后不同alpha不保证保持相等；s沿input-channel轴广播。
- QLoRA：codebook索引不是INT4均匀值；frozen W无梯度仍可对x/adapter反向；double quant多级metadata预算；重新量化merge不是严格同模型。
- AWQ：scale改变group范围，不能声称任意scale都等价于降低误差；activation salience和weight幅值不同；不能把GPTQ与AWQ只写标签差别。
- KIVI：构造K固定channel outlier与V token outlier；error进入QK与PV位置不同；残留buffer/padding/scale仍占内存。
- Rotation：正交Q保持线性输出；量化算子一般不rotation-equivariant；错误穿过elementwise非线性破坏等价；归一化Hadamard与未归一化幅值区别。
- W4A8：UINT4到INT8边界和protected范围；按group scale/zero不能任意挪到epilogue；storage4bit不推出INT4MMA。
- MX/NVFP4：E2M1有限codebook、E8M0无零scale与零block策略、非整组尾部、blockscale再量化、payload外bits；公式不可假装bit-exact所有库。
- 工程验收：一变一消融；基线tokenizer/template相同；PPL、KL、token一致率与任务成功分开；shape/fallback coverage；同步计时、warmup、TTFT/TPOT/goodput边界。

每篇都需独立审稿记录、直接提取执行文中代码/反例并检查公式、站点与数学渲染。没有真实模型/GPU运行时明确写未实测，不拿CPU算例替代论文复现；提交推送与线上部署成功分别取证。
