# 融合 LM head 与 CE 研究与作者交接

- 正文：`_posts/2026-11-10-fused-linear-cross-entropy.md`。
- 日期：2026-11-10；技术年份2024对应Liger/CCE代表工作，不是交叉熵或融合思想起源。
- 作者在AdamW已单独推送582b1fd后开始第二篇；三角色调研、作者、独审保持分离。

## 经主源核对的范围

1. [Liger论文v1](https://arxiv.org/html/2410.10989v1)：研究实际读§3.2 CE/FLCE/Remark及§3.3.1，作者再浏览同版本。
2. [FLCE v0.5.2](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/fused_linear_cross_entropy.py)：研究与作者实际读完整227行，固定token chunk、预计算梯度、有效计数加权与detach saved gradients语义。
3. [CE v0.5.2](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/cross_entropy.py)：研究读前180行；[utils](https://github.com/linkedin/Liger-Kernel/blob/v0.5.2/src/liger_kernel/ops/utils.py)核对element_mul与AMP decorators。
4. [Online Normalizer v2](https://arxiv.org/html/1805.02867v2)：研究读§2–3算法3，作者核对稳定表达与在线合并。
5. [PyTorch v2.8.0 loss.py](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/loss.py)：研究核对hard-label mean/sum与ignore。
6. [Transformers v4.46.3 loss_utils.py](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/loss/loss_utils.py)：研究核对内部shift/float，正文以已对齐行作为输入，不复制不完整集成片段。
7. [CCE v2](https://arxiv.org/html/2411.09009v2)：独审补充§4机制与§5.3预训练过滤/累加边界。研究最初读v1，但正文改为v2引用，不泛化早期微调结论。

## 正文合同与风险控制

- X[R,H]、W[V,H]，Z=XW^T；最小化CE为LSE-target，梯度softmax-onehot，未照搬最大化log-likelihood的相反符号。
- dense hard labels、mean/sum、ignore_index、无bias/classweights/smoothing/softcap/z-loss。
- 只有当前行块产生C×V logits，梯度先按全局有效计数归约，保存局部导数再乘实际标量upstream。
- mean无有效target显式拒绝；sum全忽略为0/0梯度。忽略行不等于删除词表类别。
- CPU用(m,s)而非已舍入LSE做大数相消，loss=(m-target)+log(s)，p=exp(z-m)/s；1e16全同logits仍得到log(V)。
- Liger v0.5.2有额外grad_logits_chunk乘法分配，64MiB只是一份块大小；zeros_like梯度不保证FP32。
- 只说明一阶、单次、scalar loss，不承诺reduction none、高阶或重复backward。Tied embedding仍须累加其他路径梯度。
- CCE对照仅讲vocabulary tile、统计与重算，梯度过滤明确是额外数值策略。

## 可复算例与专项

- `fused-linear-ce-examples.mjs`提取正文`flce-reference`、`flce-small-example`，对照不分块dense oracle和对X/W全部元素的中心差分。
- 固定小例loss=0.5626709415555414，忽略行梯度0；覆盖chunk1/2/尾块/大于行数、sum/mean、不同有效计数、upstream0.3与负值、空输入和全忽略、ignore_index在词表内仍参与其他行分母。
- Online normalizer对照独立max后exp和，验证共同偏移与1e16；非法label、shape、reduction、chunk与非有限输入必须拒绝。
- 字节模型：8192×131072 BF16 logits 2GiB，FP32副本4GiB；C256单块64MiB，X/dX各64MiB，W/BF16dW各1GiB。

## 验证边界

专项为标准库CPU算法、梯度和字节模型证据，不运行Torch/Triton/CUDA，不代表实际显存、吞吐或长程收敛。独立审核、全栈构建与发布由后续角色执行。
