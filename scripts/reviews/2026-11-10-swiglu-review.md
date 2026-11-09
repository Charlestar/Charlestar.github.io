# SwiGLU 与 Transformer FFN 独立全文审核

- 审核日期：2026-11-10。
- 正文：`_posts/2026-11-10-swiglu-transformer-ffn.md`。
- 流程：研究 → 作者完成三文件并冻结 → 独立审读全文、固定来源及专项，补充修订 → 冻结交主审。

## 全文覆盖

完整检查了逐 token FFN 的作用、三投影形状与 Linear 转置约定、Swish 参数为 1、门控非概率/非路由、参数与主要 GEMM 预算、宽度取整、完整一阶反向、残差/norm 边界、零输出反例、两段真实 Python、三种融合边界、激活保存/覆写/重算、Tensor Parallel 通道配对与合并投影布局，以及架构变化和等价执行的区分。不是仅审查改动行。

## 独立阅读的原始来源

- [GLU Variants v1 §1–3.2](https://arxiv.org/html/2002.05202v1)：无 bias 约定、Swish/GLU 变体、三矩阵 FFN、2/3 宽度预算与 T5 训练比较。原实验不是任意 checkpoint 替换后的质量保证。
- [LLaMA v1 §2.2、§2.4](https://arxiv.org/html/2302.13971v1)：SwiGLU 宽度选择、架构和 activation 重算相关设计。没有复现训练或性能实验。
- [Transformers v4.46.3 LlamaMLP](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py)：gate/up/down Linear、普通前向与 pretraining_tp 分片分支。该本地 Python 分支不是多进程 collective 的实测证据。
- [同版 LlamaConfig](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/configuration_llama.py)：hidden_size=4096、intermediate_size=11008、SiLU 和 mlp_bias 默认值；配置类直接存值，不自动对任意 d 计算 8d/3 并对齐。
- [Liger v0.5.2 ops/swiglu.py](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/swiglu.py)：全部前后向 kernel 与 autograd Function。读取 A 后转 FP32、直接形成乘积、重算 sigmoid/SiLU、将 A/B 存储写成梯度及 saved_tensors 生命周期。
- [同版 transformers/swiglu.py](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/transformers/swiglu.py)：完整两种 MLP wrapper；三 Linear 的 LigerSwiGLUMLP 和合并 gate/up 的 Phi3 形式，bias=False。
- [同版 utils.py](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/utils.py)：ensure_contiguous 和 calculate_settings；复制与受支持 width 是实现条件，不是矩阵恒等式的保证。

仅声明这些实际读过的章节/函数，没有把论文全文、所有模型配置或全部后端宣称为已验证。

## 发现、补充与保留

1. 前后向推导、上游梯度两路相加、零点 1.5、小矩阵输出和梯度均正确，保留。
2. 预算成立的无 bias/主导 GEMM 前提、对齐后多 0.78125%、172/516/64 MiB 账本均一致，没有把这些大小当成实际峰值。
3. 补充“融合保持实数公式，不保证与低精度分步执行逐位相同”：独立存储 SiLU 会多一次低精度舍入。用 CPU BF16-RNE 账本给出可复算反例，但不声称它复现 Triton sigmoid 或 GPU 运算。
4. 补充专项执行命令及 CPU/后端/通信的证据边界；润色两处空格硬切标题。
5. 加强专项，使正确通道分片和共同置换不仅核对输出、输入梯度，也核对三份权重梯度。
6. 增加完整单通道网络的两个零输出例：A=0，B 分别为 3、6，U 均为零而 gate 权重梯度分别为 1.5、3。仅保存 U 不能恢复全部局部导数，仍需其他状态或重算。
7. 保留适中有限非空教学输入合同，不扩张成生产级输入验证器；保存状态/权重必须来自同一次 forward，未声明能检测任意错误复用或中间溢出。

## 实际运行与结果

`node scripts/reviews/swiglu-examples.mjs` 提取正文两个唯一 marker 内的真实 Python 并执行，最终通过。

- 独立标量循环 oracle 不使用正文矩阵乘 helper；四组形状覆盖非方形和 h=1。
- 对 X、Wg、Wu、Wd 共 109 个元素分别执行中心差分，核对全部手推梯度；包含负上游梯度，输入和参数不被改写。
- 具体 Y、dX、标量目标，负门控、门控大于 1、SiLU 零点及 sigmoid 正负极端分支。
- gate/up 合并投影的输出等价；含尾部的教学通道分片，合并输出、dX 与三份权重梯度。
- gate/up/down 一致通道置换的输出与全部梯度等价；单独错排 up、错误地平分 `[all_gate, all_up]`、试图折叠成线性函数的反例。
- 残差路径的额外输入梯度；同零输出、不同 gate 梯度的保存状态反例。
- 适中正有限数的 BF16 round-to-nearest-even 模型：a=1、b=9，先舍入 SiLU 再乘得到 6.5625，保留较高精度乘完再舍入得到 6.59375。仅用于展示舍入位置。
- 非空与 shape 约束、134217728/135266304 参数元素、256/258 MiB、0.78125% 取整差、172/516/64 MiB，以及 6/12/18 Ndh 主要工作量和理想等预算。

`git diff --check` 通过；未暂存新文件并不由此命令完整覆盖。站点构建与浏览器检查由主审另行记录。

## 结论与边界

全文审核及必要补充已完成，可交主审做最终站点验证与单篇发布。未运行 PyTorch/Triton/CUDA SwiGLU、真实多进程 TP、BF16 模型训练、GPU 性能或峰值显存测试。CPU oracle、有限差分、舍入账本和固定源码是不同证据；有限算例不证明所有输入、所有低精度实现或模型质量结论。

## 主审发布前验证

41 套 JavaScript 示例回归与独立 Python Agent 示例通过，最终独审修订后重跑 SwiGLU 专项。107 篇元数据、10 个系列、24 个既有标签检查通过。Jekyll 生产构建生成 123 个 HTML 页面和 148 个文件；100 篇数学文章的 1873 个行内、1116 个块级公式通过标记检查。独审冻结正文与构建前后 blob 一致，本地无界面浏览器渲染本篇 96 个 MathJax 公式，无错误。

同批前两篇分别随 `582b1fd`（Actions `42665203931`）和 `82d6312`（Actions `42667092318`）独立推送，对应 build、deploy 和线上最终正文及公式已验证。第三篇部署须在此次提交推送后另行确认。
