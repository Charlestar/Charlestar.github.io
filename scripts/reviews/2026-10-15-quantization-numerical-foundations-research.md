# 模型量化数值基础：调研与逐篇写作交接

调研日期：2026-10-15。范围：神经网络模型量化，不是金融量化。此笔记仅记录实际阅读、推导和候选验证，不替代独立审核。

## 文章定位

- 候选标题：`模型量化的数值基础：Scale、Zero Point 与误差怎样进入计算`。
- 建议文件：`_posts/2026-10-15-quantization-numerical-foundations.md`。
- 技术日期锚：Jacob et al. 首次 arXiv 公开为 2017-12-15；CVPR 会议年份是 2018。不得写“量化诞生于 2017 年”。系列用这项系统化 integer-only 方案作为统一记号起点。
- 本篇回答：一个浮点张量如何变成有限 code、如何重建、误差如何通过线性层、fake quant 为什么不等于加速。
- 不抢讲后续：GPTQ 的 Hessian 补偿、LLM.int8 的异常通道分离、NF4 codebook、FP8 指数编码、完整大模型 QAT recipe。
- 近邻现有链接：`/2026/08/01/fp8-formats-scaling-llm-inference/`、`/2026/08/02/smoothquant-w8a8-activation-outliers/`、`/2026/08/03/awq-activation-aware-weight-quantization/`、`/2026/08/04/w4a8-progressive-quantization-kernel/`。

## 实际读取的一手来源与边界

1. Jacob et al. [arXiv v1 正文](https://arxiv.org/html/1712.05877v1)：实际逐段读了 §2.1–2.4、§3–3.2，包括式 (1)–(14)、fake quant 图位置、整数 accumulator 和 bias 的 scale。也核验了封面日期。没有完整复读所有实验表、附录或复现移动端性能。CVF PDF 返回 403 后改读作者 arXiv HTML，不把失败页面当已阅读。
2. [ONNX QuantizeLinear 文档](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-13)：实际读了 opset 13 的 Summary、axis、Inputs；本篇仅引用其中 INT8/UINT8 affine、对 `x/scale` round-to-nearest-even 再加整数 zero point、saturation。当前文档页还含更新版本，本篇不据此宣称所有低位浮点也遵循整数范围。
3. [ONNX DequantizeLinear](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html#dequantizelinear-13)：实际读 opset 13 的 Summary、scale/zero-point/axis 输入；反量化为 `(q-z)*s`。
4. [Google LiteRT int8 规范](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec)：实际读 Specification summary、Signed vs unsigned、Per-axis vs per-tensor、Symmetric vs asymmetric、零点展开。其 weight `[-127,127], z=0` 与 activation `[-128,127]` 是该规范约定，不是所有 INT8 格式的通则。
5. [PyTorch 2.8 fake quant API](https://docs.pytorch.org/docs/2.8/generated/torch.fake_quantize_per_tensor_affine.html)：实际读公式、输入与返回 dtype、参数和示例。它输出 float32 模拟值，不是 packed INT8 artifact。
6. [PyTorch v2.8.0 FakeQuantPerTensorAffine.cpp](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp#L110)：实际读 cachemask backward，`return dY * mask`。
7. [PyTorch v2.8.0 CPU kernel](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp#L2515)：实际读 `_fake_quantize_tensor_helper`，先求 rounded code，再以未 clamp 的 rounded code 是否处于合法区间生成 backward mask。
8. Bengio et al. [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)：目前仅核验元数据和摘要中的 STE 定义，未读全文。因此正文如解释具体 PyTorch mask，只引用上述 pinned 源码；不要声称本次精读了 Bengio 全文。

## Claim/source mapping 与必须保留的限定

| 可写结论 | 证据 | 不可扩大为 |
| --- | --- | --- |
| 整数 affine 量化保存 code、正 scale、整数 zero point | Jacob §2.1；ONNX Q/DQ 13 | 所有量化都使用均匀整数格点 |
| `z` 是实数零对应的 code，`q=z` 重建为 0 | Jacob 式 (1)；LiteRT Summary | 非对称分布只能采用非零 z |
| INT8 常用对称量化可选 `[-127,127], z=0`，放弃 -128 | LiteRT weight 规范；Jacob §3.1 | INT8 的硬件范围只有 255 个 code |
| nearest-even 的舍入位置必须固定在 `x/s` 上，之后再加 z | ONNX Q13；PyTorch 2.8 API | 把 z 移入 round 总等价 |
| INT8 operand 可用 INT32 accumulator，bias scale 与输入、权重 scale 乘积匹配 | Jacob §2.4 | 任意 K 都不会发生 INT32 overflow |
| scale 随 reduction 轴变化时，不能一次整体移到点积外 | 逐组展开的代数推导 | 任意 per-channel 或 per-group 都有同样 kernel 代价 |
| fake quant 的前向模拟 Q/DQ，返回浮点，反向使用估计量 | PyTorch API 与 pinned CPU/backward 源码 | QAT 是用整数参数直接做通常梯度下降 |
| STE 不是离散阶梯函数的真实导数 | 阶梯函数的直接数学事实；源码明确自定义 backward | STE 无偏、必然收敛或恢复原精度 |

## 建议文章叙事和逐节内容

1. **两种完全不同的“变小”**：dtype cast 与有 scale/code 的量化；先说明 weight、activation、KV 是三个位置，不对所有位宽重复性能宣传。
2. **从一条离散数轴定义 Q/DQ**：令 `s>0`、整数 `z∈[qmin,qmax]`，定义 `q=clip(RNE(x/s)+z,qmin,qmax)` 和 `xhat=s(q-z)`；明确例子全部有限实数，实际工具另需规定 NaN/Inf。用一小张 code/reconstruction 表解释网格，而不从术语表开头。
3. **Zero point 不是给浮点值加的偏置**：unsigned z=128 例；padding 写 code z 而非 code0；整数 z 与浮点加性偏置 `b=-sz` 必须带单位区分。后续 QuantSpec 有此重要互操作边界。
4. **范围、步长与饱和只能一起选**：对称 `a=max|x|,s=a/127`，全零时专门约定 `s=1,z=0`；`a=0` 直接除会 NaN。非对称 min/max 公式先把 0 包进范围，再 nudge 整数 z；若没空间讲完整算法，仅给固定 s/z 范例，避免错误声称端点精确表示。
5. **舍入并不是无关细节**：正负 ties-to-even；`round(0.5)+1=1`，`round(0.5+1)=2`；奇数 z 时把整数零点移入 round 会改变 tie 的奇偶性。不可声称整数平移不影响 nearest-even。
6. **误差为什么要分 rounding 与 clipping**：对最邻近格点且 x 位于可重建端点区间，`|xhat-x|≤s/2`；越界时 clipping error 不受此界控制。缩范围可能增大极值误差却降低整体 MSE；不是总有利。
7. **Granularity 是哪一组数共享尺度**：固定 `W∈R^(K×N), X∈R^(M×K)`；per-output-channel 沿 N 保存 scale、per-group 沿 K 分组；不可把所有“per-channel”当同一轴。讲 metadata 的 bit 预算，主体 `bN` 加 `ceil(N/G)*scale_bits` 和 padding（符号避免混淆矩阵 N 与参数数，另用 P）。
8. **把恢复公式代入一次线性层**：先同一 reduction 内 scale 固定，展开零点修正；演示 INT8 是 operand，不代表 INT8 accumulator/output。随后显示 group scale 随 k 变化不能提到总和外面。说明这里不实现 fixed-point requantization，不宣称 Python 浮点模拟和整数 kernel bit-exact。
9. **Weight MSE 和实际输出不是同一个目标**：`DeltaY=X DeltaW`，小 weight MSE 也会因 activation 方向被放大；用两误差向量反例给后续 GPTQ/AWQ 留自然问题。
10. **Fake quant 怎样把这个误差放进训练**：Q/DQ 前向仍是浮点张量，阶梯函数真实导数几乎处处 0；STE 是约定的 surrogate。可给概念性 unclipped-region 梯度，但必须明确边界依实现；若要精确写 PyTorch，采用 rounded-code mask。QAT 更新高精度 master weights；QLoRA 冻结量化基座、更新 adapter，两者别混淆。
11. **一个不依赖深度学习框架的参考实现**：Python 标准库，round 是 ties-even，validate finite x/s、positive finite s、integer合法 z/range，返回 code/reconstruction/saturation 统计；强调 CPU 参考不是 GPU kernel benchmark。每段代码保持可提取执行。
12. **落回需要记录的数值契约**：code范围、rounding位置、axis/group、scale dtype、zero-point单位、accumulator/output、padding/packing；用短收束而非巨型部署清单重复后续验收篇。

## 已执行的独立算数与建议审核用例

2026-10-15 已运行 Node 标准库参考（自行实现 nearest-even，避免 JavaScript Math.round 与 Python round 的 tie 规则不同），结果如下；文章中的最终 Python 代码仍须 auditor 直接提取执行，而不能据此代替。

- ties `[-2.5,-1.5,-0.5,0.5,1.5,2.5] -> [-2,-2,0,0,2,2]`。
- zero point 移入 round 反例：`RNE(0.5)+1=1`，`RNE(0.5+1)=2`。
- fixed affine：s=0.1,z=128,UINT8，x `[-1,0,0.2,1]` 得 `[118,128,130,138]`，重建原值（浮点容差比较）。
- 零点点积：`qx=[3,4],zx=2,qw=[5,7],zw=4`，直接 centered dot=7；raw dot43，修正 `43-4*7-2*12+2*2*4=7`；sx=0.5,sw=0.25 后输出0.875。
- clipping tradeoff：100个1和1个10，三值对称码 `[-1,0,1]`；absmax s=10 的总平方误差100；裁到±1 后 s=1 的总平方误差81。只是构造分布的反例，不证明一般 clipping 最优。
- `x=100,s=1,z=0,q∈[-7,7]`，重建7，误差93，大于s/2；只有舍入区间内才有半步长界。
- 同 tensor 含100和小数0.1/0.2：s=100/7 时小数都变0；单独小数组s=0.2/7时重建约0.1142857与0.2。说明分组能改善这个例子，不保证任何输入都单调改善。
- output-vs-weight：`X=[1,100]`，`DeltaW_A=[0,0.1]^T` 的weight MSE0.005、output error10；`DeltaW_B=[0.2,0]^T` 的weight MSE0.02、output error0.2。
- metadata：P个4-bit weights、每G个16-bit scale，忽略zero/alignment时 effective bits=`4+16*ceil(P/G)/P`。P是分组整数倍时为`4+16/G`；尾组不能无条件用P/G。
- 建议 fake-quant finite difference：s=1,z=0，x=0.2，真实前向局部导数0，STE返回1；可比较x=7.2在码域[-7,7]的PyTorch rounded-code mask仍为1，而抽象`|x|≤7` mask会给0，不能把两者混写。

## 验证与不可声称的内容

本调研未运行 PyTorch、量化真实模型、GPU GEMM、整数 requantization、QAT 训练或线上模型 benchmark。数值构造证明概念区别；实际任务精度和加速都必须另做目标环境实测。本篇重点是让后续各篇共用精确定义，而不是保证任意模型压缩无损。
