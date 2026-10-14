# QuaRot / SpinQuant 独立审核记录

日期：2026-10-15。对象：`_posts/2026-10-15-quarot-spinquant-rotation.md`。收到作者静止稿后独立阅读全文、三个 Python 块、公式与参考对应关系，并完成以下修正和执行。写作前预读不等于本次正文已经通过；本记录以最终实际提取执行为依据。

## 一、实际读取的来源

- [QuaRot v2](https://arxiv.org/html/2404.00456v2)：§3.1–3.4、§4 Stage 1a–1d / Stage 2a–2c、§5.2 性能语境。另查 [arXiv 提交历史](https://arxiv.org/abs/2404.00456)，核实首发 2024-03-30、v2 2024-10-29。
- [SpinQuant v4](https://arxiv.org/html/2405.16406v4)：§2.2、§3.1–3.2、§4.1、§4.3.2 与 Appendix B 关于旋转不变网络/梯度的讨论。另查 [arXiv 提交历史](https://arxiv.org/abs/2405.16406)，核实首发 2024-05-26、v4 2025-02-20。未把后续修订的所有结果归入初版。
- QuaRot 固定提交 `5008669b08c1f11f9b64d52d16fddd47ca754c5a`（2024-11-26）：完整读取 [`fake_quant/rotation_utils.py`](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/rotation_utils.py)、[`model_utils.py`](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/model_utils.py)、[`main.py`](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/main.py)。不是对全库所有内核的逐行审计。
- SpinQuant 固定提交 `8f47aa3f00e8662caf1a484153920a07e5281c3a`（2025-02-14）：完整读 [`eval_utils/rotation_utils.py`](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/eval_utils/rotation_utils.py)、[`optimize_rotation.py`](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/optimize_rotation.py)、[`train_utils/optimizer.py`](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/train_utils/optimizer.py)、[`quant_linear.py`](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/train_utils/quant_linear.py)；定点读 [`utils/quant_utils.py`](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/utils/quant_utils.py) 的 STE、activation quantizer、旋转权重调用。
- 两个固定提交日期已独立通过 GitHub commit API 核实，不以当前 main 分支代替固定源码。

## 二、概念与数学核对

1. 行式 XW 的旋转方向正确：X'=XR、W'=R^T W，输出基也变时为 R^T W S。源码 Linear 存储转置与文章数学布局有明确区分；正交泛指也包含反射。
2. Hadamard 的归一化为 1/√d，未归一化矩阵 Gram 为 dI。没有复制 QuaRot/SpinQuant 部分松散文字或错误 footnote 中的 ±1 / ±√d 正交说法。L2 保持不等于 absmax 一定降低。
3. 两组 Q/DQ 算例成立；[-7,7] 明确是 15 值教学子域，不是官方 16-code artifact。第二个例子两侧 absmax 都降低，但最终点积误差变差，未从降低 absmax 推出任务质量普遍改善。
4. RMSNorm 去 gamma 后、保持同一 epsilon 时满足旋转等变。gamma 必须融合到所有消费者；输出 bias 必须转入残差基。两块小网络与 embedding/head 闭合，并未称其为完整 Transformer。
5. LayerNorm 的去均值、逐元素 SiLU/GELU 和门控乘积一般不能穿越任意坐标混合。down-projection 前的局部 H 必须放在已经形成的 Z 后。
6. Q/K 同 head、同一个正交变换才能抵消；不同 head 的独立 softmax 不能任意混合。Value/head 变换还须补偿输出投影的对应分块。普通代数关系不证明真实 GQA wrapper 已实现所有路径。
7. post-RoPE QK 配对成立；pre-RoPE 一般不交换。二维位置反例的正负号与 Attention 1/√head_dim 正确；Hadamard 归一化与 score 缩放是两个独立步骤，不能删掉其中之一。
8. SpinQuant R1 全局共享，R2 每层一套 Dhead 方阵、层内共享；R3/R4 为固定在线 Hadamard，不是四类矩阵都独立学习。基础权重冻结不切断对旋转参数的梯度。
9. 自定义 STE 反向明确是代理导数；不能把真实量化阶梯当普通光滑层。GPTQ 配置的旋转学习先量化 activation、随后对旋转权重 GPTQ，KV 条件另按实验设置说明。
10. Cayley 的精确正交性推导正确，但示意 A 不冒充官方梯度下降方向。实际负梯度/动量、5 次固定点迭代、行范数 epsilon、步长限制和随机 QR 不能被说成每次显式精确求逆或浮点下 Gram 恒为 I。
11. QuaRot Stage 2c 是 KV 反量化后 FP16 attention，Query FP16；不能从摘要扩成全部矩阵乘原生 INT4。§5.2 延迟主要为 RTX3090 单 Transformer block microbenchmark，不是本机完整模型端到端性能。
12. 理想数学等价未被包装成整个固定预处理脚本无条件等价。源码 embedding 去均值/default norm epsilon 的额外限制留在 research，正文只精炼强调 affine、epsilon、全部消费者与残差基条件，没有扩成项目 bug 清单。

## 三、局部修正

- 第一块补有限向量/维度、Hadamard 中间溢出、步长下溢与重建范围检查。明确只针对下方有限小参考，不提供全动态范围数值保证，不重排极端输入来扩大计算域。
- 第二块用共享短检查防止不规则矩阵、gamma 长度不匹配、门控/up 不匹配及非有限乘积静默通过；RMS 保证正 epsilon 和有限分母。SiLU 改用正负两支，避免普通大负输入直接 exp 溢出。
- 融合后的单位 gamma 长度从输入 gamma 推出，仍只展示原有两块小网络；没有增加完整 Transformer 实现。
- 第三块补矩阵形状、非有限输入与结果检查，不引入新的优化器实现。
- 将 Hadamard/Attention 两个因子的说明由“形状相同”改为“同为 1/√head_dim、作用不同”。
- §6 增一句冻结 W 而沿旋转后权重/激活回传到 R，避免把冻结理解成切断计算图。
- 参考资料补明确两篇首发日期，保留修订日期并注明 v4 内容不都属于 2024 初版。

## 四、真实执行与独立核验

专项：`scripts/reviews/oct15-rotation-examples.mjs`，从最终正文三个唯一 marker 提取全部 Python，在不同 namespace 中分别执行，包含正文原有 assertions 与打印。没有用抄写的另一份示例替代正文代码。

```powershell
$env:BLOG_PYTHON = 'C:\Users\iStar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
node scripts/reviews/oct15-rotation-examples.mjs
node scripts/audit-posts.mjs
```

最终 7 组专项通过：

1. 三个实际正文块独立执行，末块含真实 100 次复合，非只检查 marker 存在。
2. Fraction 距离/偶码 oracle 检查 dyadic 最近值格点；二维旋转的公共 1/√2 在码选择中消去、两侧乘积精确为1/2，因此不借用正文浮点 Hadamard 计算独立得出 10.2 与 561/700；另 exact H4 验证归一化及能量集中/分散。
3. 70 位 Decimal 独立执行归一化、SiLU、两层残差和非方阵 head，覆盖正文全部 embedding 行；另外用非对称有理 4D 正交矩阵，检查 hidden4→gate/up3→hidden4→logits5 的两块图及5个输入行，避免对称 H 把转置方向错误掩盖。
4. 故意遗漏 up/gate 一侧 gamma、输出 bias 旋转、输出残差基、原 epsilon，以及 embedding/head 端点，均使结果偏离独立目标；另用 Decimal LayerNorm、SiLU/门控不可交换反例和精确 post/pre-RoPE 目标值检验限制。
5. 显式索引的 4 Query heads/2 KV heads 数学映射、共同 R2 的 Value/output-projection 配对、每 head 的概率和 mask；原始 logits 用 Fraction，softmax 用 Decimal。另有只转一边、混合独立 head、未归一化 H 的反例。这不是实际模型 GQA 支持测试。
6. Fraction Gauss-Jordan 独立求 2D/3D/4D Cayley，不重抄二维闭式解；精确 Gram、复数指数角度核100次复合；独立5次固定点算术产生非单位 Gram，明确不是运行官方 SGDG。
7. 形状、gamma、epsilon、空/非有限向量、underflow 步长、有限输入但溢出的 Hadamard/点积/RMS/矩阵乘，以及 ±1000 的稳定 SiLU 等受限范围检查。

`audit-posts.mjs` 通过：96 篇、10 个系列、24 个标签。这个元数据/Markdown 结果不替代最终 HTML/MathJax 和浏览器布局检查。

## 五、验证边界与交接

仅运行 Python 标准库 CPU 数值/图/索引参考及站点静态审核。未运行官方 Torch 旋转学习、完整 Transformer、真实 Hadamard/CUDA/INT4 内核、低位 packing artifact、模型任务质量、GPU 显存/延迟/吞吐。有限参考精度通过不证明官方预处理或所有硬件路径逐位相同。

正文已冻结并通知 root 可并行构建/浏览器检查；随后只补专项和本记录，未再改正文。date、author、系列顺序、文件名与 URL 保持约定；未改其他文章/配置/计划、未提交。现正式移交本篇 post、专项脚本、本记录并停止写入；root 最终构建、浏览器与发布 QA 另行补记。

## 六、Root 本地验收

三个真实代码块与 7 组专项复跑通过；全部 29 个 JavaScript 审核套件、既有 Python、96 篇元数据和历史 coverage 通过，`git diff --check` 无空白错误。实际生产构建生成 111 个 HTML 页面、136 个文件，89 篇文章中的 1196 个原生行内公式和 998 个独立公式通过结构检查。

浏览器实际打开本地文章：112 个 MathJax 容器、0 个渲染错误、无单双美元残留，1265 像素视口无整页横向溢出，残差基和 RMS 等变推导截图正常。实际正文包含两项首发日与修订边界。Root 另将专项注释的 FP4 更正为本篇 INT4；执行逻辑及正文未变。发布结果另记，未运行官方模型/优化器或 GPU。
