# FP4 Attention 调研与写作交接

调研日：2026-09-30。此记录为研究角色交接，不是独立审核或 GPU 复现声明。

## 一手材料与实际阅读

1. [Robert Hu: Hardware-Aware FP4 FlashAttention-4, 2609.04105v1](https://arxiv.org/html/2609.04105v1)，[abs](https://arxiv.org/abs/2609.04105v1) 核实首次公开 2026-09-03。已阅读 §1–§5.3、§6.1–§6.2、§7 全部、§8 全部、Appendix C、E、F。没有声称全文 56 页及全部附录逐字审计，未运行 GPU。
2. [原始 FlashAttention-4, 2603.05451v1](https://arxiv.org/abs/2603.05451v1)，核实 v1=2026-03-05，读 abstract/metadata。9 月主论文是 Robert Hu 基于 FA4/HAO 的独立研究，不是原 FA4 团队发布新版本的同一论文。
3. [HAO 官方 FP4 branch README](https://github.com/hao-ai-lab/flash-attention-fp4/tree/fp4/flash_attn/cute)，读两种 quantization 模式、结果和计时设置；作者主论文引用 comparator commit `9b0abefdbbbe4d0da1d4e0c7aa128e3338c4b247` 加 patch。对 README 当前分支的结果不可直接与论文跨硬件数据相除。
4. [作者 RELEASE_STATUS](https://raw.githubusercontent.com/MrHuff/fp4-fa4/main/RELEASE_STATUS.md)，读 §Current state、source pins、experiment coverage。状态日期 2026-09-03；作者发布说明自报已审计的 public snapshot 为 `5926d20188ec7a8a033e4efc7075f4c40325e3e8`。我们实际阅读的是 main 上这份说明，未独立审计该 commit 全部源码，固定 commit 文件的 web 读取未成功。它明确 source-complete **不等于**全部 clean-clone GPU 路线已验证；部分历史 data order/checkpoints/raw captures 不可恢复。正文如谈开源复现必须保留这一点。
5. [NVIDIA PTX ISA: Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)，当前页显示 9.4；只核对 §9.7.18.1 与相邻 shape table：SM100 TMEM 128 rows ×512 columns ×32 bit；动态以 columns 分配，32-column 单位、数量为 2 的幂。不要把新 PTX 最新指令能力反写为论文当时实测能力。

## 适合本篇的主线

拟题：`FP4 Attention：矩阵乘快了以后，概率、Scale 与数据交接为什么成了瓶颈`。既有博客讲过 FP8、NVFP4 KV Cache 与 FA3；本篇不重写格式科普，而从 QK → softmax → PV 中的概率 P 既是数值结果、又是下一条硬件指令的 operand 讲起。

## 必须正确区分的两条路线

| 边界 | Direct-P 非因果前向 | 保留的 causal training route |
| --- | --- | --- |
| QK 输入 | NVFP4 Q/K | NVFP4 Q/K |
| P/V | MXFP4 | FP8，不是 MXFP4 |
| 主要改动 | 分数直接转 E2M1 code，分母来自同一 represented P | 保存 quantized Q/K、scales、LSE，在 backward 重构概率，梯度用合适 FP8 views |
| 证据 | 单算子时延/误差，固定输入模型替换 | isolated backward、带投影子层、完整短 update、长期轨迹各自单独比较 |

“full FP4”仅指 attention 内 Q/K/P/V payload，**不是**所有中间累加均 FP4，更不是整个模型/训练均 FP4。Projection weights/activations 是另一个数值边界。论文所有被测试 MXFP4 P/V 训练轨迹发散，不代表数学上证明任何 FP4 Attention 永不可训练。

## 原理重点与符号

- `Z=QK^T/sqrt(d)`，P 为归一化概率；online softmax 内部常用未归一化 `p_j=exp(z_j-m)` 再累计分母。正文必须把这两个 P/p 语义说清楚，免得读者以为每个 tile 已独立 softmax。
- E2M1 非负值集合 `{0,.5,1,1.5,2,3,4,6}`，相邻 nearest-rounding 阈值 `{.25,.75,1.25,1.75,2.5,3.5,5}`。说明 examples 避开 ties；要写 ties 必须定义舍入规则，不能把同值距离比较当硬件 RNE。
- 论文 NVFP4 使用 block16 的 E4M3 scale，MXFP4 使用 block32 的 E8M0 power-of-two scale；不能把 payload4 bit 当全部存储4 bit。一般块量化写 `xhat=s*q` 即可。
- Direct-P 的特殊约定：stored amplitude `alpha=2^e`，真实重构 `ptilde=alpha*q/6`，而不是通用 MXFP4 定义必有 `/6`。硬件实际乘 alpha*q，故 P/V 两侧均用该约定时 product epilogue 要 `/36`。把 encoded amplitude、effective step 与 global factor 分清。
- `x=(z-m)*log2(e)-e+log2(6)`，理想 payload `Q_E2M1(2^x)`。作者用 affine+converter近似分类，不是精确指数：generic `max(0,1.5x+1.2)`；为了避免重抄全部代码，正文只需解释“目标是分桶结果而非高精度exp”。代码一致性不是原 softmax 数学等价。
- `Otilde=sum ptilde_j*Vhat_j / sum ptilde_j`，分子分母同一 represented P。若分子量化、分母仍是真正 exp sum，算的是另一算子。统一 normalization 只恢复一致性，不恢复丢失精度、不保证模型质量。
- masked key 必须得到 0；全 mask row 不能默认除0，走 kernel 约定。scale 为0/denominator为0/饱和等需要显式处理，CPU toy 不替代GPU数值边界。
- 小概率 individually小不等于总质量小。降低payload并非无需scale；E4M3 block scale下溢可能让整块消失。

## 数据流与所有权

论文沿用 HAO 的双 query-stage schedule；D128 下两块 score bank +两块 output bank =4×128=512 TMEM columns。QK→读取score片段→其存储被P/scale覆盖→PV读取→释放，旧P还被消费时不能向同一bank写下一QK。多一个 barrier 不会多出存储。

4 个 N32 producer fragments 组成 N128 score tile；scaled-FP4 PV 的 K64 消费粒度需等待两片（Q0+Q1、Q2+Q3）。quartering不代表QK被拆成四次小MMA；它减少fragment的live registers并允许早发布。其他query stage仍可交错，不要画成整个CTA完全串行。

主 forward benchmark Q/K/V 预先量化；QK scales折叠、K/V anchor排列在计时外，不能用其kernel speedup声称完整layer一样快。所谓 accuracy mode 也不是等同FP8/BF16：主结果仍有明显operator error；cosine会隐藏幅度误差，应同时看relative L2/任务质量。

## 可复算原创例子

1. 串行时间模型：原总时间1，矩阵乘占.6，其他占.4，矩阵乘4x→`.4+.6/4=.55`，上限1.818x；若新增quantize cost .1→1/.65=1.538x。这是 Amdahl toy，不是实际 pipeline benchmark；真实overlap需要critical path。
2. 取未归一化权重 `[1,.4]`，固定 `alpha=1,delta=1/6`，E2M1 nearest codes `[6,2]`，represented权重 `[1,1/3]`。V全为2时，一致分母结果2；错用原分母1.4得到`40/21≈1.90476`。同例V=`[0,6]`：精确结果`12/7≈1.71429`，represented正确归一结果1.5，错配分母`10/7≈1.42857`。证明一致不等于无误差。
3. 固定blockscale α=1 的输入 `[.01,.04,.10,.20,.40,1]`，乘6后量化并除6；测试只做协议/阈值，不声称复现affineconverter。
4. x=1 时精确 `2^x=2` → code2，而 generic affine=2.7→code3；所以Direct-P只是code agreement优化，不是exact conversion。x=0时精确1和affine1.2都round到1，说明数值近似误差不必等于payload错误。
5. `cos(v,2v)=1` 但relativeL2=1：只报cosine可完全漏掉幅度翻倍。
6. 128×512×4=262144 bytes=256KiB，四bank全部占满；若把FP4payload减半，但分配列数/合法scorebank不变，不能由bytes算出多一个合法重叠stage。
7. 1/.55与1/.65等CPU数只检验解析model；GPU TFLOP/s、training convergence均未本地复现。

## 引用数字的边界

- 主 forward：GB200九个D128 shape几何均值2.023x，峰值约2.13x均是noncausal内核；不是整模型/训练倍数。fast平均relativeL2约.3366，不能描述成无损。
- isolated backward core .356ms vs BF16 .501ms；加E5M2 dO/stat publisher总.508ms，优势被吞掉。这一对是很好例子，必须保留相同计时scope。
- 完整8B短update：B4 FP8 P/V bracket .854516s/.751722s=1.137x；MXFP4 bracket另有自己的BF16 anchor，不能混用baseline。包含低精度QKV/O projections，10warmup+21measured fixed synthetic updates，不是收敛证据。
- matched64GPU训练、B4、GA4、effective global batch1024，FP8路线与BF16各一条轨迹。论文报告100Btokens范围内稳定，末共同validation=2.3948vs2.3048，差.0900；不能写“质量相同”，差距不能归因attention单独，没run-to-run统计置信。
- 作者源代码当前release明确没通过全GPU freshclone matrix，部分历史receipt不能重取原始数据。正文至少一句：性能/训练结论均为论文报告，本地只验证算例，不作独立复现实验担保。

## 建议结构

1. QK/PV快而softmax不等比例快的问题；
2. P是在线生产的下一条MMA输入，因此依赖比位宽更关键；
3. E2M1+scale与Direct-P特殊重构约定；
4. 从算exp转为预测code，以及代表值归一化的两个小反例；
5. D128双stage、TMEM lifecycle、两N32才凑齐K64；
6. 非因果前向不能直接等同因果训练，精度契约表；
7. 从kernel、projection-inclusive、update到long trajectory四层证据；
8. 可执行验证清单与研究结论边界。

避免强行写完整CUDA/Triton实现；没有GPU执行条件时，一段数值toy和可复算反例比伪装可直接部署的kernel可靠。若推导attention backward，务必保留`1/sqrt(d)`：原文简式dQ=dS*K可能隐含scale约定，不应直接照抄成与前文Z定义矛盾的公式。量化是不可微的，保存quantized payload重建不等于自动得到离散量化函数的精确导数。
