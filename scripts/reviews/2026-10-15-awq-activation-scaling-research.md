# AWQ 既有文章逐篇调研与改造交接

日期2026-10-15。对象 `_posts/2026-08-03-awq-activation-aware-weight-quantization.md`，保留URL，系列70/首发2023-06-01。已读完现有全文。

## 本轮实际读取

- [AWQ arXiv v5](https://arxiv.org/html/2306.00978v5)：实际读§3.1–3.2方法和误差推导，§4系统相关段落；实验比较仅用于范围，不复现。v5是2024-07-18，不把该修订当2023全部内容。
- 官方llm-awq main经API固定 **`d6e797a42b9ef7778de8ee2352116e0f48a78d61`**，commit日期2025-07-17：
  - [auto_scale.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py)：实际读get_act_scale、scale_ln_fcs/fc_fc/gelu_fc、_search_module_scale、apply_scale相关调用；input meanabs、20ratios、outputMSE、state恢复、共享consumer边界。
  - [auto_clip.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_clip.py)：实际读完整88行，按group partialoutput误差搜索；此实现跳过q/k相关模块clipping。
  - [quantizer.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/quantizer.py)：实际读pseudo与real入口，zero-point支持、groupreshape、Q/DQ和WQLinear转换。
- 现有vLLM stable/TensorRT latest引用未用于本次current支持结论；继续保留具体兼容性事实时需另查固定版本。没有执行Torch、TinyChat或GPUkernel。

## 核心claim及修改意见

| 现稿claim | 原始证据与限制 | 处理建议 |
| --- | --- | --- |
| activation决定误差重要性 | §3.1；代码meanabs X | 正确，但meanabs是proxy，相关性/误差抵消仍影响真实output，不是完整重要性度量 |
| 约1%权重保FP16能提高质量 | 论文表1具体INT3/g128/OPT实验 | 原文已写实验观察非固定阈值，应保留；不是AWQ最终格式永久混FP16 |
| 等价scale `(XS^-1)(SW)` | §3.2；auto_scale | 没quant时等价；量化后`X S^-1 DQ(Q(SW))`不是严格原输出 |
| 放大salient权重必提高分辨率 | 论文以group Δ'近似不变作为局部论证；表2同时显示其他权重受损 | 加“若同组step没有同比扩大”；不能无条件保证每个scale改善 |
| 误差平均0.25格点 | 论文均匀fractionalpart近似 | 不推荐当普适事实；构造性数值例更可靠 |
| s由activation统计给出 | pinned代码meanabs^ratio，ratio0..0.95、20grid并归一化 | 可说明具体实现，不冒称所有backend都固定20或含alpha1 |
| 不做reconstruction | 论文意思是不像GPTQ做weight二阶补偿/回归，不用backprop | 必须补“仍以前向output误差搜索scale与clip”，否则误导成完全不用outputobjective |
| 不易overfit | 论文实验观察对比，不是算法保证 | 现稿“降低风险”可保留为设计动机，不能写不会overfit、calibration无关 |
| clipping总启用所有层 | auto_clip跳过q_/k_/query/key/Wqkv | 不可笼统写所有投影都同样clip；典型流程与pinned排除分清 |
| 对称Q/DQ示例 | 教学合法但official real path要求zero_point=true | 附明确“数学示例非artifact schema”；现有四舍五入措辞改最近值/ties明言 |
| AWQ不是activation量化 | 典型W4A16 | 正确；Activation-aware描述离线统计，不自动量化A4 |
| TinyChat加速复制到vLLM | 论文明确特定系统/GPU/baseline | 现稿已拒绝套用，保留；可删未固定历史“stable曾提示慢”不必要动态话题 |

## 对结构的针对性改善

1. 以“同weight误差不同activation放大”两通道引入，随后立即给完整scale前后量化数值，而不是只有重要性比例。
2. 紧接group coupling展示s过大反而伤其他权重，同一组里不是所有通道可免费各自独立变精确。
3. 明确实际搜索的proxy：no_grad前向比较原module输出与pseudoquant后输出；选scale不是训练权重，也不是不看output。
4. AWQ和SmoothQuant同样代数式，但SmoothQuant减少A8风险、AWQ以A16换权重低bit保护；只靠“都activationaware”不够区分。
5. Folding尊重graph，norm+多consumers共同改；有些非线性后scale不能穿过GELU，只能明确保留/融合inverse scaling算子，不能说所有scale都零成本吸收。
6. 伪量化、真正packing、W4A16runtime三个节点可用短具体source读法替代大量通用部署清单；最终工程验收篇再统一报告质量/SLO。
7. 不删除原文已正确的全零组、zero-point格式、K/output轴差别与W4A16不是INT4MMA限制。

## 统一数学与已执行反例

系列X(M,K),W(K,N)。定义重建误差`E_W=W-DQ(Q(W))`，则输出差`X E_W`；若改变误差定义符号，同篇全部同步。

meanabs大只能说明典型绝对误差敏感方向，不完整刻画跨channel协方差。比如X=[1,1]与X=[1,-1]拥有相同per-channel abs统计，对误差[1,-1]^T的输出分别0与2。

2026-10-15用Node标准库、自写RNE执行：

- X=[10,1],W=[0.1,1]^T，原输出2；教学对称Q/DQ码[-7,7]，baseline s=[1,1]时quantW=[1/7,1]，输出2.4285714286。
- 只将通道0等价scale10，X'=[1,1],W'=[1,1]，quant后输出2。
- 同通道scale100，X'=[0.1,1],W'=[10,1]；groupstep10/7，quantW'=[10,10/7]，输出2.4285714286，再次变差。说明scale不能无限增大，不是4bitLLM真实性能实验。
- 所有同group权重同乘公共正因子，在理想absmax量化下常被step同比抵消，不会免费改变相对分辨率；真正保护来自组内相对通道变换。
- 建议auditor执行固定activation covariance与group变化，验证原浮点equivalence、Q/DQ差异、no/allzero、negativevalues、整组尾巴、nearesttiepolicy、scale正性。

## 尚未验证

没有真实AWQ校准、TinyChat/vLLM/TensorRT推理、kernel packing一致性、GPU性能或任务精度。本篇source只支持原理与pinned实施读法。独立审核仍需直接运行最终文章代码并核对公式/结论/链接。

## 独立预审补充的固定实现边界

审核者已另读 qmodule：`ScaledActivation` 在非线性输出之后除 S，不意味着 S 可以穿过 GELU。scale 候选先 clamp `meanabs**ratio` 再按 `sqrt(max*min)` 归一化，因此最终 S 不保证仍大于预归一化 clamp 值。默认 clip 候选为 10 个收缩系数 1 至 0.55，目标是各 group 部分输出误差，不是所有层共享的全局任务 loss。

Root 已实际读回固定 `quantizer.py`：`zero_point=False` 的 else 分支在赋值前有 `assert min_val is None`，正常 Python 断言开启时会引用未赋值局部变量；源码注释也说该分支未使用。这里只作代码阅读判断，没有运行 Torch 或声称收到运行错误。real_quant 入口明确只支持 `zero_point=True`；正文若提供对称教学参考，应声明为独立数值合同，不把旧版这个分支写成已验证可运行路径。source 中 min/max、epsilon、round 次序、packing 与教学公式的区别也不能省略。
