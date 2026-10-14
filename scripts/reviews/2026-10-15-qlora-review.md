# QLoRA / NF4 独立审核记录

审核日期：2026-10-15。正文：`_posts/2026-10-15-qlora-nf4-frozen-base.md`。

writer 静止后独立全文审阅概念、维度、公式、数值、编码、梯度、预算和交付边界；只修改本篇及配套专项/审核记录。发布日期仍为 `2026-10-15`，作者、文件名/URL 和系列 metadata 不变。

## 实际独立阅读的一手来源

- [QLoRA arXiv v1](https://arxiv.org/html/2305.14314v1)：实际阅读 §2–3 的 LoRA、NF4、double quantization、paged optimizer 与计算 dtype；核对 §4 的比较对象，§8 的 33B/65B 质量限制，附录 E 的全部 16 值、F 的正态性观察、G 的显存图限定，以及 B.2 的部分训练设置。没有完整复读人工/GPT-4 聊天评分协议，不借历史 99.3% 比较证明普遍质量。
- [bitsandbytes 0.47.0 utils.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/utils.py)：独立核对完整 NF4/FP4 表。另读 [functional.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/functional.py) 的 `create_normal_map`、`create_dynamic_map`、blockwise、4-bit Q/DQ、nested scales、GEMV 恢复。`compress_statistics` 对一级 absmax 减全局均值 offset，以 blocksize 256 调用默认 signed dynamic 8-bit 映射，恢复后加回 offset，不是对 scale 作硬件 E4M3 cast。
- [0.47.0 default/ops.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/default/ops.py)：独立阅读 4-bit 与 blockwise Q/DQ，码表转相应 dtype、nearest lookup、高 nibble 优先的打包。教学零块/奇数尾部合同明确独立规定，不把所有 backend 的特殊值处理和布局当成一致。
- [0.47.0 modules.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/nn/modules.py)：独立读 Params4bit、Linear4bit、LinearNF4、compute dtype 和 forward。量化 weight 显式 `requires_grad=False`，并不自动冻结 bias、norm 或其他基座参数；Linear4bit 默认 fp4，不是默认 NF4。
- [0.47.0 autograd/_functions.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/autograd/_functions.py)：独立读正常非空 `MatMul4Bit.forward/backward` 的 DQ、`F.linear` 和 grad input 路径；packed weight grad 为 None，不需要为 frozen indices 采用 STE。源码 A/B 是输入和 packed weight，与正文 LoRA A/B 不是一套变量。

源码阅读不是执行对应框架或设备 kernel 的证明。

## 结论与本轮实修

1. `X[M,K] W_hat[K,N] A[K,r] B[r,N]` 下，`Y=XW_hat+cXAB` 的 dA、dB、dX 维度和转置正确。冻结基座参数仍可能需要 `dX=G W_hatᵀ+c G Bᵀ Aᵀ`；不能用全基座 `no_grad` 替代参数冻结，也无需对不更新的 NF4 code 求 STE。
2. NF4 是不对称的 16 值非均匀码表，7 个负值、一个零和 8 个正值；code 7 是精确零。它不是 signed INT4，也不是 E2M1。正态分位构造是设计动机，不是每个 absmax 归一化 block 都正态、或任意分布最小 MSE 的保证。
3. 编码参考的全零 scale 1、tie 取低索引、奇数尾补 7 和逻辑长度均已明示。修复 `unpack_codes` 允许 `[256]` 等非字节输入进入位操作的问题：只接受 bytes/bytearray，再检查长度与 padding，避免得到 >15 的伪 code。
4. double quant 的渐近值 4.126953125 bit/weight、节省 0.373046875 均正确。有限账单需要一级/二级 ceil、nibble 尾和每张量 offset；补明 P>0，避免把空 tensor 当作可以计算均值 offset 的量化对象。账单未假定包含码表、headers、对齐、未量化模块或训练 activation。
5. 标量反向示例确由同一 NF4 block 的端点 `[15,0]`、scale .5 生成基座 `[.5,-.5]`，不是随意选择一组该 block 无法表示的权重。五坐标梯度与数值吻合。B=0 时首步 dA=0、dB 可非零，不代表输入链被切断。
6. 修正 merge 段的时间顺序：明确使用梯度更新前的 A/B，`cAB=[.03,.06]` 是 LoRA 增量，不是代码已经更新后的 `updated_a/updated_b`。重新量化使用新的 absmax .53，得到 `[15,1]`，不是固定旧 scale .5 偷换比较。
7. checksum 仅覆盖这个小例子的 packet 和 binary64 scale。它不能证明完整模型的 bias/norm/所有参数均冻结，也不能证明 optimizer 没有错误管理其他参数；正文已有训练准备范围说明，专项和本记录不把它提升为全模型证据。
8. Paged optimizer 管相应优化器状态迁移，不是无限 activation 内存或无代价性能保证。论文 48GB/65B 设置、附录 G 的 batch1/len512/checkpointing 和不含 attention 的输入梯度分账均有限定；§8 未建立 33B/65B 全参 16-bit 等效的直接证据，不能用摘要口号覆盖该限制。

## 已执行的正文专项

命令：`node scripts/reviews/oct15-qlora-examples.mjs`，使用 `BLOG_PYTHON` 指定本机 Python。两块真实代码直接提取后独立执行；7 个不同语义组，不按码值、候选或坐标数量膨胀组数。

| 验证组 | 独立证据 |
| --- | --- |
| 真实正文代码 | 唯一 marker、完整 Python 块和自身断言独立执行；解析真实输出，核 NF4 `793f`、输出 -0.35 与梯度。 |
| 码表本身 | 用标准正态逆 CDF 和非对称概率网格独立构造 16 个值，容差反映 binary64 与原 FP32 linspace 的差异；再用从固定 utils.py 独立读出的全部 16 个 FP32 位型严格核对，避免近似容差掩盖一 ULP 笔误。验证顺序、正负计数、零、端点和与 E2M1 的区别。 |
| NF4 最近值 | 在重建 weight 单位中用 Fraction 中点和 bisect 区间作 oracle，不复制正文距离搜索；覆盖全部中点、前后相邻浮点、多种二进制精确 scale、每个 code、全零及非有限拒绝。端点还覆盖 binary64 最小 subnormal 和最大 finite，但不把其他 subnormal 乘法说成精确实数重建。 |
| 字节打包 | 独立 4-bit 字符串拼接核 256 个 code pair，核 `793f`、奇数尾 `0127`、bytearray、空 serializer；拒绝非法 code、容器、逻辑长度和 padding。不是所有 backend packing 的认证。 |
| Double quant 预算 | 枚举一级/二级 block records 与实际教学 packet 长度，核 P=1/2/63/64/65 等边界以及 16384/16385 二级尾；包含 offset，说明小张量甚至可能比 FP32 scales 更贵。用有理数核中心化/加回 offset 和 `delta w=C_j delta a`，未执行 bnb 二级量化器。 |
| 梯度与初始化 | 对正文五个标量以三个不同 dyadic 步长中央差分；另用 M2/K3/r2/N2、c=2/3 的真实 NF4 重建基座，对全部 16 个 X/A/B 标量作独立 loss 差分。核 B0 后 dA0/dB非0、B更新后 dA可非0，以及 c0 时基座仍有 dX。没有训练模型或执行自动微分框架。 |
| Merge/冻结证据 | 两支路与高精度 merge 输出一致；实际量化器重算 scale .53 后产生 >.1 的输出差。再构造一个能量化到同一 frozen base 的原始 W，证明直接 merge 到原始 W 仍是另一函数。hash 和一次 adapter 步骤 loss 下降只支持这个算例，不代表全模型冻结或优化收敛保证。 |

本轮 `node scripts/audit-posts.mjs` 通过 94 篇、10 个系列、24 个标签；目标文件 `git diff --check` 无空白错误。结构检查不等于其他文章的本轮内容复审。

## 未验证与移交

未运行 bitsandbytes/PyTorch、真实 NF4/BF16 GPU kernel、paged optimizer、65B 加载/48GB 峰值、LLM calibration 或微调质量；未模拟认证所有 backend 的码表舍入、kernel packing 和梯度 dtype。

正文已静止，交 root 继续生产构建、公式和浏览器检查、单篇提交推送、对应 commit 的工作流与线上核验，并据实补记。此处不提前宣称已发布。

## Root 本地验收

全文、完整专项及最终 review 已读回；7 组在官方 FP32 位型断言补充后复跑通过。全部 26 个 JavaScript 套件、既有 Python、94 篇元数据与历史 coverage 检查通过。实际生产构建生成 109 个 HTML 页面、134 个文件，站点检查通过，87 篇中的 899 个行内和 955 个独立公式通过数学检查。浏览器读回本篇最终 merge 参数时点与严格 byte-container guard：55 个 MathJax 容器、0 个错误、无美元残留，1265 像素视口无横向溢出；实际截图核对 dA/dB/dX 推导的排版。没有据此声称实际模型/GPU/optimizer 实验。
