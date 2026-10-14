# LLM.int8()：调研与写作交接

日期：2026-10-15。只研究本篇，不代表已发布或审核通过。建议文件 `_posts/2026-10-15-llm-int8-outlier-decomposition.md`，标题候选 `LLM.int8()：把异常通道留给高精度，为什么能挽救 INT8`。技术公开锚 2022-08-15，系列顺序20。

## 真正读过什么

- [LLM.int8() arXiv v2 正文](https://arxiv.org/html/2208.07339v2)：逐段读 §2.1、§3.1–3.4、§4.1–4.2、§6、附录D.1–D.2与表5/6相关文字；读了§1相关叙述。没有逐段复读全部相关工作和训练/微调附录E/F，也未复现论文模型实验。v2正文时间是2022-11-10，但首发仍以[abs历史](https://arxiv.org/abs/2208.07339)的2022-08-15为准。
- [bitsandbytes0.47.0 functional官方文档](https://huggingface.co/docs/bitsandbytes/v0.47.0/en/reference/functional)：实际读 int8_vectorwise_quant、int8_mm_dequant的参数、dtype、返回；threshold=0.0明确表示不保留outlier支路。
- [bnb0.47.0 autograd/_functions.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/autograd/_functions.py#L146)：实际读 MatMul8bitLt.forward 的activation quant、weight quant与threshold分支、反量化支路、state；本篇若谈HF模块的物理weight布局，要注明这里B是(N,K)，与教学W(K,N)转置。
- [bnb0.47.0 default/ops.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/default/ops.py#L30)：实际读int8_mixed_scaled_mm，以及int8_vectorwise_quant，必须保留下面“实现版本边界”。
- [bnb0.47.0 cuda/ops.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/cuda/ops.py#L23)：实际读int8 matmul、dequant、vectorwise quant；特别是inner dim不能被4整除时的FP32 fallback和outlier列整列置零。
- [Transformers4.55.4 bnb量化文档](https://huggingface.co/docs/transformers/v4.55.4/en/quantization/bitsandbytes)：打开了版本固定页；在作者决定保留API示例前还须针对该页参数段细读，不据打开动作声称所有依赖兼容性已核实。本稿最好以可运行stdlib算法示意为主，不强加需要下载模型的使用recipe。

## 精确数学契约

统一系列维度 `X∈R^(M×K),W∈R^(K×N)`。先定义本篇scale是重建步长，不是论文有时使用的其倒数normalizer：

- `sx_m=max_k |X_mk|/127`；`sw_n=max_k |W_kn|/127`；全零vector专设scale=1。
- `QX_mk=RNE(X_mk/sx_m)`，`QW_kn=RNE(W_kn/sw_n)`。
- `Yhat_mn=sx_m sw_n sum_k QX_mk QW_kn`。scale产品是outer product逐元素广播，不是一个维度不明的普通矩阵乘法。
- 论文vectorwise主路采用symmetric absmax，前一篇zero point拓展不应生硬塞入本篇核心分解。
- 异常维集合 `O={k∈{0,...,K-1}:存在m使|X_mk|>α}`，补集 `R`；论文常见α6.0是其模型实验设置，不是所有模型阈值定理。论文叙述有larger than阈值，bnb0.47源码实际使用 `>=`；文章要选一种并明言，测试必须覆盖等于阈值，不混写。
- 先无量化严格恒等 `XW=X[:,R]W[R,:]+X[:,O]W[O,:]`。
- 概念性LLM.int8式以重算的常规维尺度量化第一项、FP16计算第二项，最终相加；FP16计算也有舍入，不是实数意义精确。代码用Pythonfloat支路只是参考，不能自称FP16硬件复现。
- O是当前输入矩阵所有token/row的列索引并集。只把超阈值元素置零且不正确补回所有selected-column贡献可能重复/漏算。教学实现可明确整列拆分并重算R scale，避免逐元素混淆。

## 论文结论与实现细节的区别

| 结论 | 支持 | 限定 |
| --- | --- | --- |
| row X与column W缩放可逐点积恢复 | 论文§3.1式7 | K内scale保持固定；不是任意逐元素scale |
| outlier集中在少数hidden dimensions，破坏每row量化分辨率 | §3.2与§4 | 论文所测模型经验，不把6.7B阈值推广到2026所有模型 |
| 拆分少数列进入浮点计算可兼顾大部分INT8与质量 | §3.2式8 | 不能说这是完整Transformer每个算子都INT8 |
| attention projection与attention function不同 | §6 | 原论文未量化QK/softmax/PV attention function，也不等于KV cache INT8 |
| quant/dequant与decomposition可抵消GEMM收益 | 附录D | 不抄历史speedup当当前硬件保证；小shape可能慢 |
| bnb0.47.0 mixed支路权重从CB反量化 | default/ops.py L41–52 | 该版本路径里subB并非直接取原始FP16W；源码TODO明确区分 |
| bnb0.47.0 CUDA K%4!=0走FP32 fallback | cuda/ops.py L46–51 | 不能仅凭load-in-8bit推断MMA dtype；不推断其他版本仍如此 |

特别注意：bnb0.47的 `int8_mixed_scaled_mm` 只接收A、CA、CB与scale，没有原始B。它将 selected CB weights dequant到浮点后做addmm（源码还有TODO未来取原FP16 B）。因此本篇“论文式用高精度处理异常维”与“某版本系统的异常支路完全不含任何weight quant error”不能划等号。建议用一小段源码读法说明，不让这个细节挤占原理主线。

## 逐节提纲

1. 从一个三维点积开头：低位表示不是只让异常值自己不准，它还挤掉同row普通维的分辨率。
2. 把矩阵拆成许多点积，解释token-row activation与output-column weight尺度如何形成outer product；与per-tensor对照，示范两row范围不同。
3. 为什么vector-wise仍不够：同一token有普通值与巨大hidden feature，row尺度无法将该feature隔离。
4. 先做严格的列分解，再谈量化：定义O/R，画简小矩阵/索引示意；两条路相加，不删掉异常通道。
5. 把同一反例重新算一遍：naive与hybrid，两者都比较原点积，并保留hybrid残余误差，而非制造“完全无损”。
6. 一个清楚的标准库参考：finite rectangular矩阵校验，阈值/边界约定，空O、空R、全零vector；输出O、两支路、重建误差；不要写成不可运行省略号伪代码。
7. 从数学到一次runtime调用：量化activation、缓存量化weight、INT32累加/反量化、浮点支路、输出dtype；区分论文和pinned实现的subB来源。
8. 当batch或长度增加时，O的并集可能变大：不能默认每次|O|固定6/7，也不保证最大并发收益等于理论权重压缩倍数。
9. 论文里“无性能下降”的performance指任务/PPL，需要与latency区分；按论文具体实验限定，不证明logits逐bit相同。
10. 原论文边界：Linear projection不等于attention function、KV不自动量化、本文不是FP8；与后续SmoothQuant缩放迁移、GPTQweight重建各解不同位置的问题。
11. 不重复完整部署清单，以“精度预算的分配而非统一dtype转换”收束；链接基础篇与FP8篇。

## 已执行的数值构造与审核候选

2026-10-15已用Node标准库、自写RNE执行：

- `X=[0.11,0.2,100],W=[1,1,1]^T`，真实点积100.31。
- naive vectorwise activation codes `[0,0,127]`，W codes全127，结果100，误差约-0.31。
- α=6，仅O={2}，普通X重算step0.2/127，codes `[70,127]`；普通项约0.3102362205，高精度异常项100，总100.3102362205，保留约0.00023622误差。
- 多row `[[0.11,0.2,100],[0.2,-0.3,0.5]]`，第一row使O={2}，第二row该列0.5虽不超阈值也属于高精度分解项。用此检查没有重复/遗漏。
- 特意避开0.1/(0.2/127)这样的数学半整数：二进制浮点中直接除step与乘normalizer可能落在63.5两侧，实测Node直接除step得到code63而数学精确值RNE为64。不要在不注明运算路径时许诺所有kernel位级相同。
- threshold边界α=6，明确代码用>=还是>；α=0在bnb API是禁用隔离，而朴素定义>=0将全列选中，不能直接照搬。
- no outlier、all outlier、K=1、全零行与全零weight column、不同M/K/N、正负极值、非finite与不矩形输入。
- 对所有低精度路径关闭量化时，分解两项应在浮点容差内恢复XW；打开量化后只量化常规维，不能再次把outlier项算进低精度sum。
- 存储估算只能给抽象上界：低位weight主体MN? 这里weight参数为KN；若另留O的浮点weight，额外约`|O|N*2bytes`，但若实际实现由INT8反量化，不是长期另存这份原始权重。实现具体内存仍需实测。

## 没有完成/不能宣称

未运行 bitsandbytes、PyTorch、真实模型、GPU或端到端评测；没有复现论文质量和速度。Pythonfloat例子不模拟FP16积累/INT32overflow/cuda packing。最终文章代码必须由独立审核员直接提取执行，本笔记Node算数不能替代该审核。
