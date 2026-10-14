# 模型量化数值基础 独立审核记录

审核日期：2026-10-15。

对象：`_posts/2026-10-15-quantization-numerical-foundations.md`。本篇在写作者停止编辑后独立审核，覆盖全文、公式、正文三个实际 Python 块、来源对应和系列位置；不是只检查研究笔记或元数据。

## 实际独立阅读的来源

- [Jacob 等人 arXiv v1](https://arxiv.org/html/1712.05877v1)：逐段核对 §2.1–2.4、§3–3.1 中 affine 定义、实数零、整数点积、zero-point 展开、bias scale、INT32 accumulator 和浮点模拟训练。封面与 submission history 证实 2017-12-15；没有将其描述为量化技术起源。未完整复读实验表及附录，未复现 CPU 性能。CVF 页面未成功返回，不把该请求算作已阅读正文。
- [ONNX QuantizeLinear opset 13](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#quantizelinear-13)：实际核对该版本 Summary、axis、输入输出与类型约束，尤其是对 `x / scale` nearest-even，再加 zero point 和饱和。没有把较新 float8/float4 算子的类型约束倒灌给 opset 13。
- [ONNX DequantizeLinear opset 13](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html#dequantizelinear-13)：核对该版本的 `(x - zero_point) * scale`、参数 shape 与 axis。
- [LiteRT int8 规范](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec)：核对 summary、per-axis、symmetric/asymmetric、zero-point 展开与 bias 规范。本文沿用自己的 `W[K,N]` 数学约定，没有把规范内转置后的权重 axis 编号照搬过来。
- [PyTorch 2.8 fake quant API](https://docs.pytorch.org/docs/2.8/generated/torch.fake_quantize_per_tensor_affine.html)：核对 Q/DQ 公式与浮点输入/返回值。
- [PyTorch v2.8.0 CPU helper](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp#L2515)：实际读取 `_fake_quantize_tensor_helper` 与其调用者。两条 dtype 分支都先以 float 计算 `inv_scale`，再执行 `z_point + nearbyint(input * inv_scale)`；clip 前的 code 生成 mask。这个 per-tensor helper 使用 `TensorIterator.for_each`，未将其他 per-channel vec 分支或 learnable 参数梯度实现当作它的规则。
- [PyTorch v2.8.0 backward](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp#L110)：实际核对固定参数 cachemask backward 返回 `dY * mask`；未据此宣称所有 fake quant 的梯度边界相同。
- [QLoRA 摘要及元数据](https://arxiv.org/abs/2305.14314)：仅用于核对冻结量化基座、梯度流入 LoRA adapters 的简短对比，本篇没有复现 NF4 或训练结果，也未声称精读了 QLoRA 全文。

## 实修问题与保留的正确边界

1. 初稿标量实现的范围 guard `-128 <= qmin < qmax <= 255` 会接受 `[-128,255]`、`[-1,255]` 等不能完整装入 INT8 或 UINT8 的范围。改为要求整个区间属于其中一种类型，并加入拒绝用例；保留更窄有效范围、zero point 必须是合法整数，以及 normalization/reconstruction overflow 拒绝策略。
2. 原值略微超出实数重建区间，不等于舍入后的 code 已经被 clip。修正文中容易混同两者的表述，以 7.2 与 7.6 明确区分，并连接 STE mask 的边界。
3. 在 PyTorch 源码结论旁补上 float 倒数乘法、dtype 和 tie 附近浮点误差的限定。标准库例子并非 PyTorch 任意 dtype、CPU/CUDA 路径的逐位兼容实现。
4. 首次展开 QAT、PTQ 名称与训练安排，避免在说明 fake quant 后直接引入未解释的训练缩写。为 QLoRA 冻结基座的对比加入一手引用。
5. 保留并独立核对：RNE 的零点不能任意移入 round；`s` 是反量化乘子；全零组正 scale；signed INT4 有效 15-code recipe 不等于类型只剩 15 个 code；矩阵轴、K-group scale、bias 的 scale 条件、INT32 overflow、完整双侧量化误差项和 metadata 尾组公式。

## 已执行的验证

命令：`node scripts/reviews/oct15-quantization-foundations-examples.mjs`。Windows 上通过 `BLOG_PYTHON` 指向当前可用的 Python；脚本本身在其他平台默认使用 `python3`，只依赖标准库。

结果：9 个具有独立语义的验证组通过，不把遍历次数、断言数或每个参数值另算一组。

| 验证组 | 实际实现与独立证据 |
| --- | --- |
| 三个真实正文块 | 从唯一 marker 后提取 affine、dot、STE Python，每块独立作用域执行自身断言，并解析真实打印结果；检查没有遗漏第四个 Python 块。 |
| 标量 nearest-grid oracle | 对实际 `quantize_scalar` 执行 1,395 个有符号/无符号、有零点、正负 tie 与端点用例。独立 oracle 使用精确 Fraction 枚举重建格点、比较距离与归一化整数奇偶，不复用 Python `round` 计算期望 code。二次幂 scale 和二进制精确输入使浮点误差不掩盖量化规则。 |
| 拒绝与零组 | 覆盖 NaN/Inf、非正 scale、非整数或布尔零点、非法区间/零点、混合 signed/unsigned 范围、有限输入触发归一化或恢复溢出；正 scale 下的零组编码仍有效。 |
| clipping 与 zero point | 实算总平方误差 100 对 81、7.2 未 clip 与 7.6 已 clip、100 的误差 93、零填充应写 zero-point code，以及奇数零点移入 round 的正负 tie 反例。 |
| 矩阵/分组点积 | 高精度精确有理数重建矩阵独立执行 rectangular GEMM，再核对 raw-dot 修正形式；覆盖不同输出列参数、K=5 的非整除分组及随组变化的 zero point/scale。 |
| 输出误差 | 以完整矩阵乘法差值验证 `Delta X W + X Delta W + Delta X Delta W`；精确复核较小 weight MSE 反而有更大输出误差的构造。 |
| 粒度与存储 | 实际标量实现复算 per-tensor/per-output-channel 例子；按每列列举不同 K/N/G 的实际尾组验证 `4KN + 16N ceil(K/G)`，并给出不能跨输出列合并尾组的反例。这里只计理论 payload 与 scale bits，不声称生成了 packed artifact。 |
| accumulator 范围 | 用 Python 标准库 signed-INT32 序列化检查合法 INT8 乘积在足够长 K、或再加 bias 后可以越界；没有模拟硬件溢出语义。 |
| STE | 执行实际 `fake_quant_and_mask/ste_dx`；用远离跳点的有限差分得到真实局部导数 0，与非零代理区分；检查两侧 7.5 tie 及 `nextafter` 相邻值、跳点差商、不合法非有限输入。未将该模拟当作真实 PyTorch autograd 实验。 |

另外已执行 `node scripts/audit-posts.mjs`，91 篇文章、10 个系列、24 个标签通过。`git diff --check` 未发现空白错误。此结果只说明当时整个 checkout 的静态元数据状态，不等于所有文章完成本轮事实审核。

## 验证边界与交接

本轮未安装或运行 PyTorch、ONNX runtime、真实模型、校准数据集、packed INT4/INT8 kernel、GPU、QAT 训练或质量/性能 benchmark。阅读 pinned 源码可支持对该实现规则的描述，不可替代执行证据；数值 oracle 证明教学算例，不证明真实模型质量。

内容独立审核及标准库专项检查完成。Jekyll 实际构建、生成页面链接/公式检查、浏览器视觉检查、逐篇提交推送、commit-specific CI 与线上发布由 root 继续验证并据实际结果补记；本记录没有预先宣称它们已经完成。

## Root 的构建与浏览器复核

静止稿移交后，root 直接运行专项检查，9 组通过。生产构建与生成页面校验通过：106 个 HTML 页面、131 个生成文件；84 篇数学文章中的 619 个行内公式和 912 个独立公式通过检查。

浏览器实查发现初稿行内绝对值的裸竖线被 GFM 解析为意外表格，正文残留双美元定界符。它不一定产生 MathJax error，也不能仅靠枚举已解析成功的数学节点发现。root 改为 `\lvert x_i\rvert`，并为 `scripts/validate-math.rb` 增加排除 code/script/style 后的双美元泄露检查：先用尚未重建的坏页面运行，确认明确拒绝；重建修正版后整站通过。README 同步记录写法，避免后续重复。

修复后重新加载真实页面：97 个 MathJax 容器，0 个数学渲染错误，正文无双美元残留，只有预期的手算表格；1265 像素视口下整页宽度不溢出。实际查看定义式、范围/绝对值与半步长误差界的页面截图，公式和段落正常。此记录仍不预先声称提交之后的线上部署已经成功。
