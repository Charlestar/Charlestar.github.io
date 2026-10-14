# FP8 既有文章逐篇复核：来源、范围与改造交接

日期：2026-10-15。对象 `_posts/2026-08-01-fp8-formats-scaling-llm-inference.md`，URL保持不变。逐段读了该文现稿（含2026-10-15元数据迁移和数值边界修正），不只检查front matter。

## 本轮真正阅读及固定版本

1. [FP8 Formats for Deep Learning v2](https://arxiv.org/html/2209.05433v2)：实际逐段读 §2、§3/表1、§4开头（simulated FP8实验边界）、§4.1相关、§4.2–4.3。首发2022-09-12，v2修订2022-09-29；现稿不需引用历史任务表作为性能承诺。未复现训练/推理。
2. Transformer Engine **v2.3**，GitHub tree/commit核验为 **`5de3e1481cc1f5d6f7da6ac8938870b791b796db`**：
   - [docs/examples/fp8_primer.ipynb](https://github.com/NVIDIA/TransformerEngine/blob/5de3e1481cc1f5d6f7da6ac8938870b791b796db/docs/examples/fp8_primer.ipynb)：通过官方GitHub工具取JSON、只提取Markdown，实际读完所有Markdown（格式、mixed precision、delayed、MX、transpose、autocast、shape、backward）。代码未执行；图未声称视觉核验。旧release文档URL返回错误后以这个不可变源码补足。
   - [recipe/__init__.py](https://github.com/NVIDIA/TransformerEngine/blob/5de3e1481cc1f5d6f7da6ac8938870b791b796db/transformer_engine/common/recipe/__init__.py)：实际读Format、DelayedScaling、Float8CurrentScaling、MXFP8BlockScaling和Float8BlockScaling的定义/参数/notes。
   - [pytorch/fp8.py](https://github.com/NVIDIA/TransformerEngine/blob/5de3e1481cc1f5d6f7da6ac8938870b791b796db/transformer_engine/pytorch/fp8.py)：实际读check_fp8_support、check_mxfp8_support、check_fp8_block_scaling_support、get_default_fp8_recipe、get_fp8_torch_dtype。仅这是该版本支持判定，不是今天所有库的统一矩阵。
3. [ONNX float8技术文档](https://onnx.ai/onnx/technical/float8.html)：实际读E4M3FN/E5M2和FNUZ对照、bias、zeros/special值、cast saturation表、E8M0节。文档当前页版本1.24.0；仅用于定义，不作库实现兼容性保证。
4. [CUDA Math API 12.8 FP8 conversion](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html)：实际读`__nv_cvt_float_to_fp8`签名与RNE、saturation参数说明，constructor依赖SATFINITE段。没有执行CUDA。
5. 打开当前PTX ISA 9.4页面但未完整读取具体MMA段，**不能**据此声称精读当前PTX每种FP8指令accumulator细节。现稿对accumulator只维持“需查具体路径”的范围。

## 格式定义逐项核验

| 项目 | 核验结论 | 证据/修改意见 |
| --- | --- | --- |
| E4M3 sign1/exponent4/fraction3 | 正确；normal有效significand另含隐含1 | 原论文§3表1；用fraction术语避免把隐含位算进3 |
| E5M2 sign1/exponent5/fraction2 | 正确 | 同上 |
| E4M3 max finite448 | 正确，仅指原论文/NVIDIA E4M3FN | 增加variant界定，不泛指FNUZ |
| E5M2 max finite57344 | 正确 | 原表1 |
| E4M3 minnormal2^-6，minpositive subnormal2^-9 | 正确 | 原表1；有signed zero，NaN，无Inf |
| E5M2 minnormal2^-14，minsubnormal2^-16 | 正确 | 原表1；有Inf/NaN/signedzero |
| E4M3FNUZ | bias8，无negative zero，与FN bias7不兼容 | ONNX；不必长篇扩展，正文一短段明确范围即可 |
| E4M3的normal相对分辨率 | binade内gap=2^(e-3)，RNE误差≤半gap；相对界在正常范围成立 | 直接数学推导，不能说放大正常值必增有效精度 |
| 原始FP8论文的rounding | 编码定义与rounding正交，允许RNE/stochastic | §2；文中算例需明确选RNE+有限饱和，不把此约定冒充所有FP8cast |
| CUDA12.8转换 | float-to-fp8使用RNE并显式saturate参数 | 官方API；无Inf格式不表示所有cast都自动饱和 |

建议格式表适度扩成最大有限值、最小正常数、最小subnormal、Inf/NaN，帮助读者理解下溢例。不要把原文正确的relative-precision修正删回“outlier导致任何小值精度差”的泛泛句。

## Scaling、recipe、kernel claim mapping

| 现稿主张 | 研究判断 | 建议调整 |
| --- | --- | --- |
| `s=amax/Fmax`, `X8=cast(clip(X/s)),Xhat=sX8` | 合法重建scale教学约定 | 将clip范围显式写±Fmax；仅finite/amax>0，全零正scale例外保留 |
| 正常数用2幂缩放不增加有效位 | 正确，且已有1.1例 | 保留并执行枚举验证 |
| 更细granularity“通常提高精度” | 不应读成逐张量单调定理；scale网格也可改变rounding | 改“可减轻这类组间动态范围差异，是否降低误差须验证”更准 |
| TE“当前支持current/delayed/block” | 正文现有时态不稳 | 固定TE v2.3及commit；以recipe源码为范围 |
| Hybrid默认forwardE4M3/backwardE5M2 | DelayedScaling/Float8CurrentScaling默认正确 | 不可扩大成所有TE默认；v2.3 Blackwell默认recipe是MXFP8(E4M3) |
| MXFP8为连续32数一scale | 正确，E8M0/power-of-two，沿reduction轴 | 明确与generic blockFP8不同；若正文简短，可补一表而不重写为MX专文 |
| generic Float8BlockScaling | v2.3用FP32容器scale，默认约束2幂，可显式放宽 | 不是MX的E8M0/32；block选1×128或128×128，需保持与MX分开 |
| 当前amax得到当前scale（current） | 正确；reduction在cast前 | TE v2.3 Float8CurrentScaling；删除“当前文档称最简单”这类无必要动态引用 |
| history给未来step scale（delayed） | 正确 | DelayedScaling默认history1024、max/most_recent、margin与reduce_amax；无需堆默认参数，核心是过去amax不是当前amax |
| distributed amax一定全局同步 | 不一定 | v2.3 reduce_amax默认true但可false，各rank保持local；要讲选定group与重建约定，不是所有shard必须同scale |
| delayed collect amax“asynchronously” | 算法不保证独立异步 | 可写“生成本轮amax供后续更新”，只说可融合，避免默认async执行事实 |
| rawscale `sx sw`可拿出GEMM | 只同一dot内不沿K变化成立 | 现稿已修正标量、row/column与K-block的边界，保留 |
| fp8_autocast保护所有上下文代码 | 错误的潜在读法 | 范围限定TE modules的safe操作，不是任意torch算子自动转FP8；现稿基本正确 |
| TE Linear dims16 | v2.3 primer确有bothdims divisible16 | 写版本限定；不当作所有GPU所有FP8MMA必要条件 |
| GPU/Ada | v2.3 check_fp8_support支持CC>=8.9；Ada需CUDA>=12.1、cuBLASLt>=12.1.3 | 不必列全硬件型号；勿把“Ada之后”写成所有设备都支持所有recipe |
| MX支持 | v2.3check_mxfp8_support要求CC>=10.0 | 指这个TE路径；不是OCP格式只能在Blackwell解码 |
| generic block支持 | v2.3check_fp8_block_scaling_support：9.0<=CC<10.0且CUDA>=12.9 | 此限制看似反直觉，更说明版本必须写明；可只在对比表脚注说明，无需泛化 |
| communication DeepEP FP8dispatch/BF16combine | 与FP8GEMM独立这个结论正确，但现稿无近旁固定来源 | root/auditor若保留具体DeepEP例须另查其版本readme；否则改一般可选双向dtype案例，避免无来源细节 |
| FP8训练=全部张量FP8存储 | 不成立 | 原论文§4本身simulatedFP8且higher-precision arithmetic；正文已有master/output边界需保留 |

## 建议编辑范围

这是一篇已相当充实的旧文，不需全篇重写。必要修改：

1. 开头说明格式采用原论文E4M3FN/E5M2；实现事实以TE v2.3为例、非“当前所有软件”。
2. 表格补全normal/subnormal与特殊值，对照INT8时指向系列数值基础，而非再复制完整scalar Q/DQ教程。
3. 分清普通per-tensor FP8、generic block与MXFP8，固定默认/支持条件，修正“全部TE默认hybrid”的潜在泛化。
4. current/delayed用具体数值突变解释；删多余“async保证”和未读最新文档评价。
5. 保留normal相对精度、下溢、全零scale、K轴不能提scale的已有修正。
6. 删/缩本篇尾部与最终验收篇重复的长检查表，保留FP8特有saturation/underflow/history/scale-layout风险；不是把内容精简到摘要。
7. 附一段stdlib格式枚举/nearest-rounding代码，使格式表与数值例可直接核验，不声称hardware cast bit-exact。
8. 参考链接改不可变TE源码、原论文、ONNX类型、CUDA12.8转换。原有current/block当前文档若保留，仅额外阅读，不支撑未固定的当前支持断言。

## 已执行的参考数值

已用Node自行写E4M3FN/E5M2 code解码、枚举非负finite与nearest-even tie选择（不是PyTorch/CUDA）：

- E4M3FNcode0x7e=448，code1=0.001953125=2^-9，code8=0.015625=2^-6；含±zero的finite bitpatterns254。
- E5M2code0x7b=57344，code1=0.0000152587890625=2^-16。
- RNE(1.1)到E4M3FN为1.125；RNE(1.1×256)/256仍为1.125。
- 1e-4/(400/448)=0.000112，nearest E4M3FN=0，确实低于2^-10半步阈值。
- 旧amax1给s=1/448，下一tensor出现10时，饱和再重建只有1；current s=10/448则max可重建10。这个反例不是说所有delayed必然质量差，只揭示distribution jump风险。
- 若要测group-K缩放错误，可令X8=[1,1],W8=[1,1]、两group sproducts1与10，正确sum11；错误统一拿1乘dot得到2。

完整审核应由独立agent建立并运行最终文中代码测试：256codes、specialcodes、signedzero、ties、normal/subnormal、scale=0/NaN/Inf拒绝策略、全零block、两operand尺度、history jump。没有运行TE、GPU kernel、原始模型训练或real-world speed benchmark，不能声称这些已实测。
