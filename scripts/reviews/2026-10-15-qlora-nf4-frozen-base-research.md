# QLoRA / NF4：调研与写作交接

调研日期2026-10-15。新文建议 `_posts/2026-10-15-qlora-nf4-frozen-base.md`；候选标题 `QLoRA：冻结的 4-bit 权重，怎样参与一次微调`。技术首发2023-05-23，系列60，在AWQ六月首发之前。

## 实读来源和边界

- [QLoRA arXiv v1](https://arxiv.org/html/2305.14314v1)：实际读§2–3、§4相关比较、§8、附录E精确NF4表、F正态观察、G内存边界。没有逐条复读全部chatbot评测和人工/GPT4评分协议；不引其99.3%历史ChatGPT比较作为一般质量保证。
- bitsandbytes **0.47.0** pinned tag：
  - [backends/utils.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/utils.py#L23)，读NF4 exacttable与FP4 table，确认NF4不是后续NVFP4 E2M1格式。
  - [backends/default/ops.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/backends/default/ops.py#L175)，读quantize/dequant4bit、nearestcode与高nibble-first packing；仅用于版本路径示例，不承诺所有设备打包相同。
  - [functional.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/functional.py#L765)，读quantize4bit参数/压缩statistics分支、dequantnested scales、GEMV路径的nested恢复。
  - [nn/modules.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/nn/modules.py#L388)，读Linear4bit/LinearNF4构造、weight requires_grad=False、compute_dtype与forward；默认Linear4bit quant_type是fp4，不可声称类名本身默认NF4。
  - [autograd/_functions.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/bitsandbytes/autograd/_functions.py#L328)，实际读MatMul4Bit.forward/backward：dequant至A.dtype后F.linear，gradA用反量化B，gradB=None。不要误写对4-bit codes做STE更新。
- 没有运行这些库、训练模型或GPU算子。以下stdlib算数仅验证讲解逻辑。

## 符号与关键推导

系列维度 X(M,K),W(K,N)，LoRA A(K,r),B(r,N)，常数c可为alpha/r：

`Y = X W_hat + c X A B`, `W_hat=DQ(codes, scales, codebook)`。

W_hat来自冻结量化基座；A/B是可训练较高精度参数。给定输出梯度G(M,N)：

- `dA=c X^T G B^T`；
- `dB=c A^T X^T G`；
- `dX=G W_hat^T + c G B^T A^T`。

冻结的是base参数更新和baseoptimizerstates，不是切断dX。前面若有adapter仍需要这个dX。因此不能用torch.no_grad罩住整个基座forward，否则早期adapter梯度可能断掉。也不需要对被冻结NF4查表/round求真导数或STE；在计算图中base代表固定线性算子。

这些导数是对理想高精度线性表达的数学推导；实际BF16/FP16运算有数值误差。不要照抄论文正文一处奇怪的`partialX/partialW`记号当标准反向公式。

## NF4的确切性质

NF4是16个非均匀重建值的codebook，不是均匀signedINT4，不是sign/exponent/mantissa拆成E2M1。它依据正态分布的quantile构造并归一化到[-1,1]，留精确0；实际表有7个负值、0和8个正值，不要硬说正负完全对称。

对一block，absmax `a=max|w|`，finite a>0时 `u_i=w_i/a`，选nearestcode `j_i=argmin_j |u_i-C_j|`，重建 `w_hat_i=a C_ji`。全零block应专门返回0对应code7并合法scale/metadata；这项教学处理须单独说明，不假定所有kernel同样处理。

正常假设是设计动机，不保证真实block都正态，也不意味着NF4在任意数据、误差指标、group/scale下全局最优。不要把“信息论最优”口号等同于最小MSE定理；block absmax归一化后分布也不是独立无限支持的标准正态。

## Double quant内存与误差契约

第一层64weight一FP32 absmax，metadata 32/64=0.5bit/weight。再用8-bit表示这些scale、每256个scale保留FP32二级scale，忽略全局offset、tail/headers/alignment：

`metadata = 8/64 + 32/(64*256)=0.126953125`；
`total=4.126953125 bits/weight`，相对4.5节约0.373046875。

是scale被再量化，不是weight连续降成4bit再2bit。bnb0.47具体实现先减absmax均值offset，再quantize_blockwise(blocksize256)，重建时加回offset；二级uint8/codebook与FP8 E4M3硬件数值格式不应混为一谈。论文用8-bit Floats术语，代码路径要说明version/codebook，不擅自称“把scale强制cast成E4M3”。

若第一层code固定而scale误差δa，weight误差项 `δw_i=C_ji δa`；scale误差对同block相关传播，不是完全免费。

## 逐节提纲

1. 从问题开头：只训练少量adapter已省优化器状态，为何基座权重仍占显存；4-bit存储解决哪一项，不承诺整个训练只占0.5Pbytes。
2. LoRA一层算式和冻结边界：XW_hat+cXAB，用一个小数值例看两支路；不要复刻完整LoRA历史。
3. 4-bit里到底存什么：NF4非均匀codebook+blockabsmax；nearestlookup、zero、packing演示，解释NF4和INT4/FP4不同。
4. 为什么normal-derived levels在零附近更密：用非均匀表的局部间隔而不是抽象“高信息”口号；承认真实block与假设差异。
5. Double quant详细比特预算：0.5降0.127，offset与二级scale也要存；小模型/tail/header不能用渐近式精确估total。
6. 前向解码到compute dtype，反向为何仍能穿过冻结W_hat：给dX/dA/dB，解释no_grad和requires_grad不是同一事；与第一篇STE-QAT清楚区分。
7. 可跑的stdlib NF4block+LoRA梯度有限差分例，重点验证码表、packing、冻结base和梯度；不是低位训练kernel复现。
8. Paged optimizer管什么：统一内存迁移optimizerstate，处理某些峰值；不能解决无限activation显存或保证paging无代价，hostram/互连和pagefault有成本。
9. 训练内存分账：frozenbase payload/scales、adapter+grad+optimizer、activation/gradientcheckpointing、dequantworkspace、allocator；具体batch/length和目标modules决定峰值。
10. 训练后的交付：adapter与确切base量化artifact匹配；`W_hat+cAB`高精度merge与`Q(W_hat+cAB)`重新量化不等价，拿原始未量化W merge也不是原训练函数。
11. 质量限制：rank/targetmodules/data任务比只看NF4dtype更关键；论文65B/48GB是指定条件，不保证任何模型/长度；§8还明确说没有建立33B/65B规模与16-bit full finetuning等效的直接证据，不能只复述摘要写“65B已全面等效全参微调”。本地没有训练则明确未验证任务效果。

## 已执行算数与审核建议

Node标准库结果（最终文中Python需独立提取执行）：

- NF4表与原附录E逐项对应；block `[0,0.2,-0.4,1]` absmax1，indices `[7,9,3,15]`，重建 `[0,0.16093020141124725,-0.39491748809814453,1]`。
- 教学高nibble-first打包成bytes `[0x79,0x3f]`；不是证明所有bnb GPU格式都如此；需要奇数尾部padding与原length。
- x=[1,2],W_hat=[0.5,-0.25]^T，A=[0.1,0.2]^T,B=[0.3],c1,target1：y0.15、loss0.36125、dB-0.425、dA[-0.255,-0.51]、dX[-0.4505,0.1615]。
- 建议auditor对x/A/B每一坐标中央有限差分；basecodes/scales前后hash不变；演示B初始0导致首步dA0而dB非0，避免把合法初始化当“梯度断了”。
- NF4code7精确zero，16code有序但非正负对称；testnearest/tie policy、allzero、nonfinite、oddlength、group64tail；不能只hardcode打印预期结果。
- merge反例可以固定NF4codebook和absmax：对某个code值加小delta，不一定仍落codebook；requant后丢失delta。独立base量化误差与adaptermerge误差不能混账。

## 不可声称的结果

没有实际训练/微调质量、GPU加速、所有bnb backends bitwise一致、65B装载或48GB峰值测试。代码算例只证明固定W_hat下的反向与存储逻辑；完整质量仍需相同base revision/tokenizer/template与独立eval数据上的真实实验。
