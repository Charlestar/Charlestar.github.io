# SwiGLU 与 FFN 研究和作者交接

- 日期2026-11-10；正文`_posts/2026-11-10-swiglu-transformer-ffn.md`。
- 第二篇FLCE推送82d6312后才写入；year2020、gpu-runtime-precision order30。
- 范围：无bias dense FFN分支，Swish beta=1即SiLU；原文数学用行向量，Linear存储权重为其转置。

## 实际核对的一手来源

1. [GLU Variants v1](https://arxiv.org/html/2002.05202v1)：研究读全文重点§1–3.2，作者核对§1–3.2；三权重与2/3宽度是匹配主导预算，质量结论限定T5设置。
2. [LLaMA v1](https://arxiv.org/html/2302.13971v1)：研究读§2.2与2.4，说明使用(2/3)*4d，不泛化所有SwiGLU模型。
3. [HF Llama v4.46.3](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py)：研究核对LlamaMLP投影、pretraining_tp路径；不是多进程TP执行证据。
4. [HF LlamaConfig v4.46.3](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/configuration_llama.py)：研究核对默认d4096、h11008、hidden_act及mlp_bias；对齐256是正文显式示例，不声称配置类自动套公式。
5. [Liger SwiGLU ops v0.5.2](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/swiglu.py)：研究和作者核对全部114行；FP32 sigmoid、重算激活、A/B覆写为梯度。
6. [Liger SwiGLU wrappers v0.5.2](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/transformers/swiglu.py)：研究和作者核对全部58行；三Linear与Phi3 gate_up合并形式，bias=False限定。
7. 研究另核对[PyTorch v2.8.0 activation.py](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/activation.py)SiLU/GLU，未将GLU split半边约定混入本篇gate/up顺序。

## 文章结构与核心核算

- 逐token通道变换→两路乘法→参数预算→完整反向→小矩阵代码→融合/保存→同通道TP→架构与等价优化。
- X[N,d]，Wg/Wu[d,h]，Wd[h,d]；Y=(SiLU(XWg)*XWu)Wd。
- h=8d/3只匹配三份权重与普通4d两份权重的预算。d4096/h11008参数135266304、BF16 258MiB，比134217728/256MiB多0.78125%。
- 全部权重和X均需grad时前向6Ndh、反向12Ndh，合18Ndh，只计主导GEMM。
- N8192,h11008,BF16每份Nh为172MiB，A+B+U三份516MiB，X64MiB。节省S不是总峰值保证。
- 单步小例Y=[.73105857863,-.53788284274]，G=[1,2]，dX=[1.85190482639,1.55825937388]。
- A0/B3/D1时U0但dA1.5，否定零输出等于所有分支零梯度。

## 从正文提取的专项

`swiglu-examples.mjs`提取`swiglu-reference`和`swiglu-small-example`。代码限定非空、有限、数值适中的教学矩阵，不构造工业输入验证器。专项对X和三权重全部元素执行有限差分；覆盖非方形、merged gate_up、通道一致置换/错配、正确分片和部分和、零点/负值、字节与FLOP账本。

仅CPU数学与教学代码验证；不执行Torch/Triton/多进程TP，不代表实际峰值、吞吐或训练质量。独立审核记录另建，主审负责全栈构建和发布。
