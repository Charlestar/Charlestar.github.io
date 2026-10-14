# QuaRot / SpinQuant：旋转篇研究交接

研究日期：2026-10-15。仅研究笔记，不是已完成模型实验。系列第 9 篇，锚定 QuaRot 首次公开 2024-03-30；SpinQuant 2024-05-26 为后续演进，因此同篇后半部分会晚于下一篇 QServe 的 2024-05-07，不能把它写成严格逐论文单调时间轴。

建议标题：**QuaRot 与 SpinQuant：为什么换一个坐标系能改善低比特量化**。

## 一、实际阅读与版本

- [QuaRot v2](https://arxiv.org/html/2404.00456v2)：实读 §3.1–3.4、§4 Stage 1a–1d 与 Stage 2。这里的方法阅读覆盖旋转等价、归一化、MLP、Attention、KV 与执行精度；没有重新复现结果表。
- [SpinQuant v4](https://arxiv.org/html/2405.16406v4)：实读 §2.2、§3.1–3.2、§4.3.2，以及 Appendix B 关于旋转不变网络的梯度解释。没有逐项复验论文所有结果。
- QuaRot 官方固定版本 [`5008669b08c1f11f9b64d52d16fddd47ca754c5a`](https://github.com/spcl/QuaRot/tree/5008669b08c1f11f9b64d52d16fddd47ca754c5a)，2024-11-26：完整读取 [`fake_quant/rotation_utils.py`](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/rotation_utils.py)。本文件提供权重融合与 QK wrapper，不等于已审阅/执行全部 CUDA 内核。
- SpinQuant 官方固定版本 [`8f47aa3f00e8662caf1a484153920a07e5281c3a`](https://github.com/facebookresearch/SpinQuant/tree/8f47aa3f00e8662caf1a484153920a07e5281c3a)，2025-02-14：完整读取 [`eval_utils/rotation_utils.py`](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/eval_utils/rotation_utils.py)、[`optimize_rotation.py`](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/optimize_rotation.py)、[`train_utils/optimizer.py`](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/train_utils/optimizer.py)。没有把论文公式视作源码逐位等价：实现用迭代近似 Cayley，并偶尔 QR 修正。

## 二、可直接采用的数学主线（统一行向量 XW）

令正交矩阵 R 满足 R^T R=I。若 X'=XR、W'=R^T W，则 X'W'=XW。若输出也转换到基 Q，则 W'=R^T WQ。所有公式都必须声明权重是数学布局 (K,N)；PyTorch Linear 的存储是 (N,K)，源码乘法方向因此不同。

旋转保留二范数但改变坐标最大值。归一化 Hadamard H2=1/sqrt(2)*[[1,1],[1,-1]]，具有 H^T H=I。普通 ±1 的未归一化 Hadamard 只有 H^T H=dI；不要照抄论文某些文字中松散的“±1 且正交”表述。正交矩阵也可以包含反射，本文沿用领域中的“旋转”统称，不要求 det=1。

对 x=(M,0,...,0)，Hadamard 能把能量均摊到 |M|/sqrt(d)；但反过来，均匀向量也可能被它集中成尖峰。因此不是任意旋转都降低 absmax，也不是对任意样本都改善量化。

去掉可学习 gamma 后，RMSNorm(x)=x/sqrt(mean(x^2)+eps) 满足 RMSNorm(xR)=RMSNorm(x)R，eps 不破坏等价，因为范数保持。gamma 是对角缩放，通常不能与 R 交换，须先融合到所有相邻消费者权重。一般 LayerNorm 的去均值与任意旋转不交换；官方对 OPT 还做均值融合、替换等处理，不能仅凭 RMSNorm 公式宣称任意 LayerNorm 架构直接支持。

MLP 的 SiLU/GELU 不与一般正交矩阵交换。应对激活/门控乘积 Z 插入 ZH，再把 H^T 融合到 down-projection：ZH(H^T W_down)=ZW_down。不可把 H 随意移过非线性。

Attention：同一 head 内，Q'=QH、K'=KH 保持 Q'K'^T=QK^T；V'=VH、W_out'=H^T W_out 保持 (PV')W_out'=PVW_out。不同 head 的 Q/K 不应任意混合；QuaRot 另有利用 Kronecker 结构跨 head 处理 out-proj 输入的方案，初篇不必展开所有索引。

RoPE 后再对 Q/K 同时施加 H，可直接抵消。若把 H 放到 RoPE 前，通常不与各位置旋转矩阵交换，不能自动融合进 Q/K projection。官方 QKRotationWrapper 包装 positional function 的输出，且显式除 sqrt(head_dim)。

量化破坏坐标不变性：Q(XR) Q(R^T W) 通常不等于 Q(X)Q(W)。这才是优化基的空间；全精度代数等价不代表量化后结果等价，也不代表浮点实现逐位相等。

## 三、论文/源码映射及必须保留的边界

| 文章主张 | 可核查位置 | 写作边界 |
| --- | --- | --- |
| 全局旋转与 norm 融合 | QuaRot §3.4、Stage1a；rotation_utils 的 fuse_layer_norms / rotate_* | gamma、bias、embedding、lm_head、残差的基必须一致，不能只改一个 Linear |
| MLP 需要在线 Hadamard | QuaRot Stage1b；rotate_mlp_output | 不能穿越激活；在线操作有开销，不声称完全免费 |
| QK 在位置编码后旋转 | QuaRot Stage1d Eq14–15；QKRotationWrapper | 同时旋转 Q 与 K，保留 mask 与 1/sqrt(d)；不复制论文 HTML 中不规范的乘法 mask 记法 |
| SpinQuant 的可学习对象 | SpinQuant §3.1–3.2；optimize_rotation.py | R1 全局与 R2 value/head 可学习；R3/QK、R4/MLP 在线 Hadamard 不学习；基础 W 冻结 |
| 使用 GPTQ 时的学习顺序 | SpinQuant §4.3.2 | 该配置先以 activation quantization 优化旋转，之后 GPTQ 量化 W；不能一概说训练时同时量化所有 W/A/KV |
| Cayley 约束 | SpinQuant §3.2；train_utils/optimizer.py | 论文精确变换保持正交，源码有限次迭代/浮点存在误差，需测 ||R^TR-I|| |
| 真实执行精度 | QuaRot Stage2 | Linear 的 INT4×INT4/INT32 路径，与量化 KV 读取后 FP16 attention 路径分开。不得将摘要泛称扩展成“所有 GEMM 都以 INT4 执行” |

SpinQuant 的学习可解释为 min_{R1,R2 正交} L(model_quantized(W_fixed,R), calibration)。因量化阶梯不可普通求导，需要训练用代理梯度；不可说“精确导数证明可学习”。不要把此方法写成更新全部模型参数的常规 QAT。

可以用一个一般的 Cayley 示意：若 A^T=-A，则 C=(I-tA)^(-1)(I+tA) 正交，再令 R_new=CR。**此处仅展示保正交原理，不将自选 A 的符号约定冒充论文梯度下降步骤**。论文 Eq3 与具体优化器方向应按各自定义阅读。

## 四、针对本篇的结构

1. 从“一个异常坐标拖大整个量化步长”进入，衔接 LLM.int8 的拆分、SmoothQuant/AWQ 的对角缩放；本篇改为坐标混合。
2. 二维坐标实验，先保二范数、再看 absmax，不先堆方法名。
3. 推导 XW=(XR)(R^TW)，解释权重布局与正交。
4. Hadamard 的归一化、快速变换与维数限制；不要把所有维度都叫 2 的幂。
5. 把等价嵌入 Transformer：残差、RMSNorm gamma、embedding/head；仅一个局部公式不足以证明全网络改写正确。
6. 两个不能随意移动的位置：非线性之后的 down-proj 输入、RoPE 之后的 Q/K。
7. QuaRot 的离线融合/在线变换/随后量化三类操作表，说明可与 GPTQ 配合，而非替代所有量化器。
8. 同一个旋转有时更好、有时更坏的可运行反例，引出校准而不是宣称万能。
9. SpinQuant：冻结模型、学习 R1/R2、正交约束、Cayley；保留 R3/R4 固定以控制在线开销。
10. 实现验收：先关量化测 FP 输出差，再开量化测重构/任务指标，另测 Hadamard 开销与实际 kernel，不把论文实验当本机结果。
11. 与前文 KIVI 的不同：KIVI 改分组轴，旋转改坐标；后文 W4A8 讨论恢复/计算契约，不继续扩大旋转算法名录。

## 五、已经执行的 CPU 数学检查

使用 Node 标准算术、自实现 nearest-even、对每个二维向量用 absmax/7 的对称 [-7,7] 量化，并对输入和权重都量化。没有运行 Torch、CUDA、模型或官方 repo。

| x | w | 真实点积 | 不旋转 Q/DQ 点积 | 同时 Hadamard 后 Q/DQ 点积 |
| --- | --- | ---: | ---: | ---: |
| [10,0.2] | [1,1] | 10.2 | 10 | 10.199999999999998 |
| [10,0.2] | [0.1,1] | 1.2 | 1.4285714285714284 | 0.8014285714285716 |

第二行是改善 absmax 也未必改善最终点积的反例。两组并列优于只展示有利样本。不是 LLM 精度基准。

Cayley 取 tA=[[0,-0.1],[0.1,0]]，C≈[[0.9801980198,-0.1980198020],[0.1980198020,0.9801980198]]；行 Gram 对角 0.9999999999999999，非对角 0。

审核代理应重新运行 Python 标准库版本，并额外断言：线性恒等、RMSNorm 去 gamma 后等变、带未融合 gamma 不等变、ReLU/Hadamard 不交换、Q/K 同时旋转保持 scores、只旋转一边不保持、归一化遗漏时多出 d 因子。真实模型的 norm 融合、GQA、bias、RoPE wrapper、低位打包、GPU 性能均尚未运行，必须单独注明。

## 六、独立预读补注（审核代理，2026-10-15）

本轮仍是写作前的来源核对，不是对尚未完成正文的审核通过。另完整读取 QuaRot 同一固定提交的 [`fake_quant/model_utils.py`](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/model_utils.py)、[`fake_quant/main.py`](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/main.py)，以及 SpinQuant 同一提交的 [`train_utils/quant_linear.py`](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/train_utils/quant_linear.py)；额外核对 [`utils/quant_utils.py`](https://github.com/facebookresearch/SpinQuant/blob/8f47aa3f00e8662caf1a484153920a07e5281c3a/utils/quant_utils.py) 的 STEQuantize / AsymSTEQuantize 与 ActQuantizer.forward。GitHub commit API 独立核实两个提交日期分别为 2024-11-26、2025-02-14。

1. **理想旋转恒等式不等于整个固定预处理脚本严格等价。** QuaRot 的 `fuse_layer_norms` 会对包含 LLaMA 在内的 embedding 行无条件去均值；替换 norm 时又直接构造默认 `eps=1e-5` 的 `RMSN`，没有继承原 norm epsilon。任意去均值不是 RMSNorm 的对称性，改变 epsilon 也不是：标准库算术下，`RMS([1,3],1e-5)` 约为 `[0.447213,1.341639]`，去均值后约为 `[-0.999995,0.999995]`。这里只验证了该局部反例，没有执行官方模型，不能据此报告某模型任务指标。正文应保持数学等价条件，精炼注明 affine、epsilon 与全部消费者须正确保留，不把整个脚本每一步都包装成该恒等式的推论。
2. **R2 的共享粒度要说清。** SpinQuant §3.1、§4.1 和 `optimize_rotation.py` 对应的是每层一个 `D_head × D_head` 的 R2，层内各头共享、各层分别学习；不是每个 head 各自独立学习一套矩阵。R1 则是模型全局共用。
3. **Cayley 示例只证明正交性。** 固定优化器先使用 `V = momentum * V - g.T` 构造负梯度方向，再建立反对称矩阵；`Cayley_loop` 仅执行 5 次固定点迭代，并有行范数 epsilon、步长限制与随机触发的 QR 修正。因此不能把自选反对称 A 的正号公式直接称作论文/代码逐步等价的下降更新，也不能宣称实际矩阵 Gram 在浮点下永远恰为单位阵。论文 §2.2 footnote 的 Hadamard 元素 `±sqrt(n)` 同样不能照抄，归一化元素应为 `±1/sqrt(n)`。
4. **代理梯度不是量化阶梯的精确导数。** SpinQuant `STEQuantize.backward` / `AsymSTEQuantize.backward` 明确把输入梯度原样返回，而不是对 `round` 计算普通导数；scale/zero 对该自定义反向返回 None。这可支持“使用 STE 代理反向”的有限源码主张，不代表所有量化配置或框架反向都相同。
5. **性能语境不要扩展。** QuaRot §5.2 的主要 GPU 延迟比较是 RTX 3090 上单个 Transformer block 的 microbenchmark，不应改写成已在本机复现的完整模型端到端加速。Stage 2c 明确把 KV 解包到 FP16 后用于 attention；“所有矩阵乘都以 INT4 执行”不符合该节计算路径。
