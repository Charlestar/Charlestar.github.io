# 序列打包与变长 Attention：research 交接

日期：2026-09-13。状态：论文相关章节、固定接口和源码已核对，可交 writer。未运行GPU/FlashAttention，不声称复现论文吞吐或验证完整训练收敛。

## 覆盖与编排

- 已检查现有正文。FA2文章只用短段给出packed tokens和cu_seqlens，Ring/Context Parallel提到跨文档隔离，RoPE文给相对位置公式；都没有完整串起sample隔离、position、shifted labels、loss权重、DDP分母。
- root确认 `series: distributed-training`、`series_order: 75`、`technology_year: 2021`、`tags: [分布式训练, 注意力机制]`、`mathjax: true`。最终正文仅保留这两个既有标签。2021是代表论文预印本公开年，绝不说packing在2021首次发明；实际阅读v2发布日期是2022-10-05，v1提交2021-06-29（arXiv编号2107）。
- 建议标题：`序列打包与变长 Attention：少算 Padding 之后，怎样保持训练目标不变`。
- 前置链接RoPE/FA2，后续链接Context Parallel与本批混合精度累积窗口；承接ZeRO的DDP平均语义，但重新严格推不等token权重。

## 实际阅读范围与一手来源

1. [Efficient Sequence Packing without Cross-contamination，v2](https://arxiv.org/html/2107.02027v2)：实际读取摘要、§1、§3.1概念、§3.2.1–3.2.3、§3.3、Appendix A、T.2–T.6，核对[position reset/attention block mask/per-sequence loss]三层分离和batch改变。没有完整读packing solver附录或复现实验。论文主体BERT/MLM/NSP与IPU，不是本文causal decoder/FA2代码的直接来源；本文对causal labels与DDP是另据固定源码做的推导。论文T.6建议expected-token分母是特定近似方案，本文采用当前窗口实际全局有效target数，不能混写成论文通用结论。
2. [论文元数据与提交历史](https://arxiv.org/abs/2107.02027)：实际核到v1为2021-06-29、v2为2022-10-05。可用于year说明。
3. [FlashAttention v2.7.4.post1 interface](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/flash_attn/flash_attn_interface.py)：实际读取 `flash_attn_varlen_func` 参数、docstring与调用。q=[total_q,Hq,D]，k/v=[total_k,Hkv,D]，cu_q/cu_k=[batch_size+1] int32，max_seqlen为各逻辑序列最大长度。本文只做self-attention且每段q/k等长，cu_q=cu_k，因此causal是段内下三角。源码的非方形causal bottom-right规则已核但不拓展成跨框架通则。
4. [同tag flash_api.cpp](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/flash_api.cpp)：实际读取`mha_varlen_fwd`与set_params_fprop有关参数。核到CUDA路径FP16/BF16同dtype、cu int32/CUDA/contiguous、最后维stride1与head dim限制，并把cu指针交下游。包装层并不替调用者保证cu内容每项有序正确；host侧构造数据时必须自行保证。未深入检查每个GPU内核路径。本文可只给D=64同头数的保守API示例，避免列跨版本硬件兼容百科。
5. [Transformers v4.46.3 loss_utils.py](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/loss/loss_utils.py)：实际读取 `ForCausalLMLoss`与`fixed_cross_entropy`。logits[..., :-1,:]对应labels[...,1:]，ignore=-100；提供num_items时sum/num_items，未提供时mean。本文只借接口合同，不说所有模型/所有版本/Trainer都完全相同。参数名的“items”不自动保证调用者计数正确。
6. [Transformers v4.46.3 Llama源码](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py)：实际读取LlamaRotaryEmbedding的`_dynamic_frequency_update`/forward、Q/K应用RoPE的位置与CausalLM调用loss_function处。动态RoPE会用max(position_ids)+1重新选频率；仅reset位置也不保证不同batch分别触发相同动态recipe，比较时必须固定频率/一致有效长度策略。varlen接口本身没有自动替你完成RoPE/labels。
7. [RoFormer v5 §3.2](https://arxiv.org/html/2104.09864v5#S3.SS2)：实际核对二维旋转和相对位置eq11–16。由此自行推导同一segment整体position偏移常数c会在固定频率RoPE内积中抵消；论文原式有排版简写，正文写完整q^T R(n-m)k。本文不是重写RoPE推导。
8. [PyTorch v2.5.1 distributed.py](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/parallel/distributed.py)：实际读取DDP docstring的跨rank平均梯度说明与`no_sync`。forward也须位于no_sync上下文，首次退出后的forward-backward同步已累积梯度。DDP默认平均、不使用自定义comm hook/uneven-input join是下面公式的显式前提。
9. [PyTorch v2.5.1 CrossEntropyLoss](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/loss.py)：实际读取class-index target的ignore_index、sum/mean定义。本文无class weights、无label smoothing，分母是shift之后未ignore的targets数，不是所有input tokens。若加class weights分母需另核。
10. [PyTorch v2.5.1 SDPA文档源码](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/functional.py)：实际读取scaled_dot_product_attention的参考式、mask、dropout和is_causal合同。SDPA bool True表示可见，而MultiheadAttention bool True常表示屏蔽；不要无说明通用复制bool mask。若给SDPA参考代码，显式完整block-causal bool mask且is_causal=False、dropout_p=0；不要同时传is_causal=True和attn_mask。
11. [Hugging Face官方Fixing Gradient Accumulation，2024-10-16](https://huggingface.co/blog/gradient_accumulation)：实际读取问题描述、sum/total targets而非mean of microbatch means的修正动机。只作为工程问题旁证，不拿该页旧“tomorrow”安装main指导2026用户。

## 正文必须先建立的范围

**两个不同任务**：有意把多个文档当连续预训练流，是训练目标/数据方案；保持样本独立的packing，是改变物理布局但保留原任务。后者必须隔离，否则改变目标；不能一概说任何跨文档attention都错。EOS是内容token不是硬件边界，模型看见EOS不代表attention自动断开。

本文等价性先限定：同一批逻辑样本、同样token顺序与上下文、dense decoder、token-local norm/MLP、无跨sample统计/对比损失/MoE容量竞争等耦合；dropout关闭或正确映射同一随机路径；固定或一致position recipe；相同loss权重及optimizer-update窗口。这样block attention的等价可逐层推出到logits。浮点kernel/reduction顺序仍可能不同，只期望容差相近，不承诺bitwise。后文再指出MoE/batch统计/dropout改变此推导前提。

## 一个贯穿例：三条逻辑序列，九个物理token

长度L=[3,2,4]，内容为：

```text
A = [a0, a1, EOS]
B = [b0, EOS]
C = [c0, c1, c2, EOS]
flat = [a0, a1, EOS, b0, EOS, c0, c1, c2, EOS]
segment = [0,0,0,1,1,2,2,2,2]
position = [0,1,2,0,1,0,1,2,3]
cu_seqlens = [0,3,5,9]
max_seqlen = 4
```

物理下标i并非逻辑position。cu_r≤i<cu_(r+1)定义属于r，局部位置p_i=i-cu_r；cu最后一项是total有效token数T=9，长度是样本数B+1=4。本文要求每段正长度；cu[0]=0、严格递增、cu[-1]=T、max_seqlen=max(diff(cu))，dtype与device按版本契约。空segment/超int32容量属非目标，不臆测自动支持。

### 可见性公式

对self-attention，定义allow(i,j)=[segment_i=segment_j]且[p_j≤p_i]，只在有效token域内。

M_ij=0当allow，否则-∞；Attention=softmax(QK^T/√d + M)V。

这是三块下三角组成的块对角结构，不是整条flat的单一下三角。后者允许B/C看之前段。padding query若全mask，普通手写softmax可能NaN；教学例去掉padding且每个有效query至少能看自己，避免全mask行；若有padding参考实现需明确安全处理并不参与loss。

反例：令QK所有score为0，A的V都为1，B的V都为10。B首token在隔离mask下输出10；整条flat的causal下会均匀看3个A token与自身，输出(1+1+1+10)/4=3.25。即使A末token是EOS也不会自动从softmax删去。这个CPU可复算例比只断言“会污染”更有解释力。

### varlen改变执行范围，而不只是改变一个mask的表示

显式9×9 block mask可定义正确语义，但dense QK可能仍算全部跨段块。`flash_attn_varlen_func(q,k,v,cu,cu,4,4,dropout_p=0.0,causal=True)`是在同一调用里按cu的逻辑序列范围计算；示意q/k/v形状[9,H,64]、同GPU FP16/BF16、q/k已经按本文逻辑position完成RoPE。API不接受“document文本”也不生成labels。

如果每个physical pack里还有多个独立segments，cu必须描述每个逻辑segment，不是只记录physical pack边缘；传[0,9]会把三条当一条。只把position_ids重置但未确认attention backend识别边界，仍可能跨段；有的框架有pos→varlen适配但不能当通用约定。

## RoPE reset的准确边界

保持独立样本通常用局部position重置，特别是绝对position embedding会实质改变输入。但**纯固定频率RoPE不应写成“不reset必错”**：

(R(m+c)q)^T(R(n+c)k)=q^T R(n-m)k。

前提同段所有Q/K整体平移同一c、频率/缩放一致、无其他absolute位置项、段内mask一致、精确数学。reset是清楚且稳妥的数据合同，不是针对固定RoPE的必要性定理。真实实现三角函数舍入、特别大的position、动态根据max(position_ids)+1改变频率、其他position-dependent机制都可能破坏简单等价。已核Llama动态频率源码可就近给一个明确反例边界。比较独立调用和packed batch时动态recipe若依据不同最大序列长度，也须显式统一，不能保证reset一个动作自动完成全部等价。

## mask隔离了上下文，却没有修复跨段监督

模型采用内部shift标签约定时，`labels`与输入token位置对齐，logit_i监督labels_(i+1)。本例应设：

```text
labels = [-100, a1, EOS, -100, EOS, -100, c1, c2, EOS]
```

每段首token label忽略，避免前一段EOS位置的logit被用于预测下一段首token。整条首label本来会被shift切掉，显式置ignore可统一规则。每段EOS target保持：a1→EOS、b0→EOS、c2→EOS是合法监督。EOS位置logit→下一段首token无监督。

实际有效预测对共6个：a0→a1、a1→EOS；b0→EOS；c0→c1、c1→c2、c2→EOS。总输入token为9，目标计数为Σ(Lr-1)=6，二者不可混用。若给每段加BOS，则BOS可预测首内容token，计数必须包含该数据合同；不要在公式里暗加BOS。

另一种自己预先生成`next_labels[i]=目标token`且不再内部shift的约定，应屏蔽各段末logit位置。两种约定等价，但mask偏移不同；正文只实现一种并用小注提示另一种。不要先shift一次再传自动shift的模型。

SFT中prompt labels也可ignore，但同段response通常仍应看prompt；**loss mask不等于attention mask**。有效目标集合同时扣除段边界、padding、prompt等；不能用position==0替代所有task mask。若pad_token_id==eos_token_id，不要按token ID把全部EOS labels过滤掉，使用真实padding位置/结构metadata。

## loss权重先决定，再决定分母

设样本r有n_r>0个有效监督target，loss和S_r=Σ_t ell_rt。

- token平均：L_tok=(Σ_r S_r)/(Σ_r n_r)。每个有效token同权，因此长响应可有更多总权重。
- sample平均：L_sample=(1/B)Σ_r S_r/n_r。每样本同权，因此短样本每个token权重更大。

二者都是可定义目标，但packing不能擅自从一者变另一者。n_r=0时sample mean未定义，须制定过滤/零权重政策且统一报告；若整个update有效target数N=0，不执行除0与无意义更新，所有ranks需一致处理。

小例：A有2个有效target、每target loss=1；B有6个、每target loss=3。token平均=(2+18)/8=2.5，sample平均=(1+3)/2=2。若二者各在不同pack或microbatch而直接平均各自mean，就得到2，packing方案/微批划分反过来决定目标。

写作重点不是哪个平均“更正确”，而是训练前定好目标，改变padding/pack/microbatch边界时保持它。本文后续选token平均；分类/对比/RL序列级目标另核，不能照搬。

## DDP分母属于整个optimizer更新窗口

固定world size W、默认DDP平均梯度、无自定义comm hook或uneven-input join、所有rank相同步数；每rank本窗口loss和S_r、有效target计数N_r，全局N=Σ_rN_r>0。目标L=Σ_rS_r/N。

若各rank backprop本地mean S_r/N_r，DDP得到(1/W)Σ_r ∇S_r/N_r，N_r不等时不等于全局token平均。正确局部反传量可取：

L_r^(backward) = W*S_r/N。

DDP再平均得(1/W)Σ_r ∇(W S_r/N)=∇Σ_rS_r/N。N为不依赖θ的数据计数。W前因子来自默认DDP平均，不是任意框架必需；若通信只sum不平均、框架已自动按token校正或额外除G，需要重核，避免二次归一化。

数值同上一例W=2：rank0本地sum2/count2、rank1 sum18/count6；直接local means平均2；反传缩放值分别2*2/8=.5与2*18/8=4.5，再平均2.5。严格说这个标量例展示权重差异，梯度公式才是训练等价依据。

有G个microbatch时，S_r=Σ_gS_rg、N_r=Σ_gN_rg，N覆盖**所有rank的整个窗口**。已知N后每microbatch反传W*S_rg/N，窗口累加并在末次同步；不要再除G。可在forward前由已准备好的target masks先统计本地窗口count，再AllReduce一个int64 count；不在每token上做通信。若利用no_sync，前G-1个forward和backward都放其上下文、最后正常forward/backward同步。

如果分母只能窗口结束才知道，也可先累加sum梯度，随后在正确同步/AMP unscale与clip边界统一乘W/N；此方案更易与现成Trainer已有缩放冲突，正文优先使用预先得到分母的推导，不强塞一份多GPU完整脚本。

与AMP串联：窗口S_scale恒定、loss归一化分母与gradient scaler是独立量；sum/count的线性归一化放在backward loss最清楚；clip在所有累加、同步和unscale/归一化完成后。不同rank非有限跳步/空target判断须保持一致。本文只推导默认DDP，不扩展FSDP/PP/CP全部实现。

## 计算工作量不能只看padding比例

对长度L_r、T=ΣL_r、B样本、Lmax=maxL_r：token-local层在dense padded布局处理B Lmax个位置，紧凑布局处理T个有效位置，理想载荷比例T/(B Lmax)。注意实际projection/MLP是否也去padding取决于上游布局；只替换attention kernel不一定减少其他层padding工作。

注意力score的非因果稠密元素计数：padded B Lmax²，逐段varlen ΣL_r²；若把所有token误当一个dense长序列则T²。Causal有效pair计数：逐段Σ L_r(L_r+1)/2，对比整条T(T+1)/2。不要把这两个口径混着算speedup；kernel tile边缘、softmax、reduction、显存流量另计，pair数不等于FLOP实测。

本例L=[3,2,4]：padded token12→有效9；score槽位48→逐段29；误作整条9长dense有81槽位。段内有效causal pairs=6+3+10=19；单条causal有45，其中26个为额外跨段pairs。padded固定4 causal名义30只是规则三角槽位，不是所有库必实际执行30单位。数值只说明不同账本，不预测1.66倍/2.37倍实测。

varlen避免跨段矩阵块不改变某一条长度L本身O(L²)的dense attention；它不是稀疏attention也不延长上下文能力。packing+varlen常有收益但短段/长度差大可产生tile浪费、launch/metadata/调度成本；长度bucketing可能更简单合适；单长样本主导ΣL²与跨rank负载时仅token数均衡不保证时间均衡。

持续数据pipeline还包括tokenize、sort/bin-pack、metadata生成、H2D、重新排列、graph/compile形状稳定性；packing效率高不自动代表端到端吞吐高。不得摘原论文BERT/IPU 2x作为此decoder/GPU的加速成绩。监控有效input tokens/s、监督targets/s、samples/update与窗口loss分母，不只看填满的physical tokens/s。

## 建议递进正文大纲

1. **少了padding，训练的还是同一件事吗？** 明确独立packing vs 连续预训练，范围前提，三样本例。
2. **用四份metadata恢复样本身份。** flat/segment/position/cu表，物理下标与逻辑坐标，cu边界不变量。
3. **先证明每个query只看到原本该看到的key。** block-causal公式、3.25 vs10反例、EOS不隔离，mask要在softmax之前。
4. **从正确的mask走向只执行必要块。** dense mask与varlen不同层，q/k/v+cu接口示意、max=4不是9、position和labels仍上游责任。
5. **位置编号该重置，但理由需要准确。** absolute embedding与固定RoPE平移相消区别；动态调频实装边界；链接RoPE详细推导。
6. **上下文隔离之后，还有一个跨边界预测对。** 内部shift对齐表、屏蔽每段首labels、保留EOS target、6个有效pairs、prompt loss mask≠attention mask。
7. **每token同权，还是每sample同权？** 两目标与2.5/2反例，不能mean of means，n_r=0政策。
8. **跨microbatch与rank之后，分母归谁？** 默认DDP推W*S_r/N、窗口累计、no_sync与AMP边界，避免无条件 /G /W。
9. **省掉多少工作，为什么不等于快了多少？** T、ΣL²、T²与causal计数三组对比；短段、长尾、pipeline成本与effective-target吞吐。
10. **验证从一个样本不受另一段影响开始。** 修改A但B/C logits不变；packed和independent forward/loss/grad比较；labels pairs、total count、token/sample、DDP权重、ragged shapes与生产性能分开验证。

## 反例与验证状态

本轮实际用JavaScript复算cu/pair计数、48/29/81及19/45、2.5/2/DDP权重、3.25/10；未执行GPU。writer/auditor可加标准库CPU测试：随机有限小Q/K/V的packed block mask vs每段独立attention、改A检查B不变、shift labels list与EOS保留、固定RoPE平移、global weighted gradient代数反例。CPU正确不代表FlashAttention GPU backend已跑通。

## writer/auditor特别检查

- causal本例每段Q/K等长，勿套到decode不等长对齐；SDPA与FA非方形causal规则不同，非目标。
- 不声称只传2D padding mask或只reset positions即可所有模型自动隔离；必须实际确认backend适配。direct varlen仅Attention示意，不是假装全模型训练框架。
- 不按input总token数9归一化此例6个监督target；不mask掉合法EOS；不把ignore labels当作禁止attention。
- 不将grad accumulation等价说成任意BatchNorm/MoE/dropout都成立；先固定样本/顺序/随机与跨sample耦合前提，再讲纯dense情形。
- 不把数学等价升级为低精度bitwise相同/训练收敛保证；不同kernel/reduction/动态RoPE状态可能影响数值。
- 不给“packing factor翻倍所以学习率直接翻倍”的建议；effective update样本/target变化必须作为训练超参变化单独评估。
- 不把cu[0]=0、末项=T、有序等内容检查说成库必为你实现；这些是调用者必须维持的合同。
