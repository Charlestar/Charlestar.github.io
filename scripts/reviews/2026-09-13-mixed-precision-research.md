# 混合精度训练：research 交接

日期：2026-09-13。状态：一手资料与固定版本源码核对完成，可交 writer；未运行 GPU 训练，不声称论文全文精读或训练效果复现。

## 覆盖与编排

- 已读 README、`_data/series.yml`、`_data/tags.yml`，核对全部正文关键词。现有 FP8 文解释格式/缩放但面向推理；ZeRO/FSDP 文介绍经典 16 bytes/parameter 账本；activation checkpointing 文讨论重计算。没有系统覆盖 AMP 数值链与 optimizer step 边界的独立文章。
- 建议标题：`混合精度训练：从 FP16/BF16 的数值边界到一次正确的参数更新`。
- root确认 `series: distributed-training`、`series_order: 47`、`technology_year: 2017`、`tags: [分布式训练, GPU优化]`、`mathjax: true`。2017 对应 Mixed Precision Training 预印本首次公开，不说该年首次发明低精度训练；本文阅读论文 v3 为 2018-02-15。order47置于2016 activation checkpoint45与2018 pipeline50之间。
- 正文链接已有 FP8、ZeRO/FSDP、activation checkpointing、Roofline；不重复大段 state sharding 或显存重计算百科。

## 实际阅读范围与固定来源

1. [Mixed Precision Training，arXiv v3](https://arxiv.org/html/1710.03740v3)：实际读取摘要、§1、§3.1–3.3、§4 开头的实验精度定义和 §5。核心是 FP32 权重更新、loss scaling、FP32 累加三个问题，不能把论文特定网络“约减半显存”推广为现代 Adam LLM 全状态规律。未逐个精读 §4 的各任务实验，不引用速度/准确率为本文成绩。论文 §3.1 的“低于最小 subnormal 变零”措辞不够细，正文按明确 round-to-nearest-even 和半步边界准确解释。
2. [PyTorch v2.5.1 AMP 文档源码](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/amp.rst)：实际核对 Gradient Scaling、Op Eligibility、CUDA Op-Specific Behavior。matmul/linear 和 loss/reduction 等不必须采用同一 dtype；原地调用、显式 out、显式 dtype 不遵循同一自动转换路径。无需抄整张 op 名单。版本化 HTML 本次打开失败，固定 tag 的文档源码成功读取，不伪称核过最新版。
3. [v2.5.1 autocast_mode.py](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/amp/autocast_mode.py)：实际读取 autocast 类文档与 CUDA/局部禁用示例，特别是不要先对普通 autocast 模型做 `.half()` / `.bfloat16()`、forward/loss 在上下文内而 backward 在外、禁用子区域需显式转输入。
4. [v2.5.1 amp_examples.rst](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/amp_examples.rst)：实际读取 Typical Mixed Precision Training、Gradient clipping、Gradient accumulation。有效累积窗口中 scale 恒定，全部梯度累积完才 unscale；每 optimizer 每次更新间只能 unscale 一次；clip 在 unscale 后；update 在 step 后。
5. [v2.5.1 grad_scaler.py](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/amp/grad_scaler.py)：实际读取类文档、`_unscale_grads_`、`unscale_`、`_maybe_opt_step`、`step`、`update`、`get_scale`。核到 found_inf 记录、READY/UNSCALED/STEPPED 状态、重复 unscale 报错；普通 optimizer 路径出现非有限梯度会跳过 optimizer.step；`step` 返回 optimizer 的返回值而不是成功布尔值；scale 不保证 >=1；get_scale 可能触发 CPU/GPU 同步。自定义/fused optimizer 可在 fused 内部消费 grad_scale/found_inf，正文不泛化其内部流程。
6. [v2.5.1 adamw.py](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/adamw.py)：实际读取 `_init_group`、`step` 与算法文档。m/v 用 `zeros_like(p)` 分配，因此常规 FP32 parameter+autocast 路线获得 FP32 moments，并非 autocast 给所有 optimizer 自动创建 FP32 master/moments。手工把 parameters 转 BF16 不保证 AdamW 自动保留 FP32 moments。
7. [Google 官方 BF16 说明，2019](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)：实际读取 Bfloat16 semantics、Choosing bfloat16、Mixed-precision training、Choosing values to represent。只借格式和精度分工；TPU flush subnormal 行为限定为该硬件说明，不扩展为所有 GPU；不复制2019旧 API。BF16 与 FP32 指数范围相近，不写最大有限值完全相等。

## 必须解释清楚的因果链

### 1. “低精度”至少四个位置，不能只贴一个 dtype 标签

存储参数、算子输入、乘加/归约累积、optimizer state 的 dtype 分开。激活在 forward 保存供 backward 使用，减小其体积不意味着 persistent states 同比例缩小。autocast 选的是算子精度，不等于 model 参数永久转换。FP32 累积不找回在输入 cast 时已经丢失的信息，也不保证相同 reduction 顺序与 FP32 baseline bitwise 一致。

### 2. 数值范围与分辨率是两条独立轴

格式表建议只列 sign/exponent/fraction、最小正常值、最大有限值、[1,2) 间距，避免小数表把重点淹没。

| 格式 | exponent/fraction | 最小正常值 | 最大有限值 | [1,2) ULP |
| --- | --- | --- | --- | --- |
| FP16 | 5/10 | 2^-14 | 65504 | 2^-10 |
| BF16 | 8/7 | 2^-126 | (2-2^-7)2^127 | 2^-7 |
| FP32 | 8/23 | 2^-126 | (2-2^-23)2^127 | 2^-23 |

均另有 sign 1bit；正常数有隐含 leading 1，不把 fraction 位数误写有效数字总位数。FP16 最小正 subnormal 为 2^-24；在支持 subnormal 的最近偶数舍入模型，|x|<2^-25 会舍入零，恰好半步按 ties-to-even 也为零。不要写“所有小于 2^-24 都直接为0”：0.75*2^-24 可以舍入为最小 subnormal。硬件具体算子若 flush-to-zero 需另核。

例A（更新吸收）：w=1.5，δ=2^-12=0.000244140625，FP16 在 [1,2) 相邻点间距 2^-10。一次 w-δ 舍入仍为1.5；多次直接在 FP16 w 上更新可以每次都丢掉δ。FP32 master 可累计小更新，再生成计算副本；这解决权重更新的时间累积，不等于所有训练总能收敛。选1.5避免1.0左右 ULP 不对称的陷阱。

例B（反向下溢）：g=2^-26，最近偶数 FP16 转换为0；令 S=2^12=4096，Sg=2^-14，已在正常值范围。若反向路径以放大的梯度传播，并最后在 FP32 gradient/optimizer 路径中除S，可以保留该量级。不是先把已下溢为零的g乘S后复原；也不是无条件说所有梯度都会被保护。S过大可使其他反向中间值溢出。

### 3. 为什么 loss scaling 在理想数学里不改优化目标

对本次 backward 固定、无梯度的正标量S，有 ∇θ(SL)=S∇θL；在 update 前除S得到原梯度。有限精度并非严格等价，但目标是减少中间表示损失，不是增大学习率。它不能扩大 forward 激活的数值范围、不能给 BF16 添加 fraction bits、不能修复错误mask或非有限输入。BF16 常无需 FP16 式 scaling，但并非永不溢出/数值问题；看具体模型/算子。

### 4. 参数与 optimizer 两条具体布局

- 论文式显式低精度计算参数+FP32 master：以 Adam 示意，计算权重2P、低精度gradient2P、master4P、m4P、v4P，合计16P；如果gradient累积为FP32则18P，明确只是主体且激活/临时/通信/allocator另计。
- 本文 PyTorch 2.5.1 标准 autocast 示范：FP32 params4P、FP32 grads4P、AdamW moments8P，主体也是16P；算子转换/缓存等不是独立持久master承诺。相同16P却不同组成，正适合一张精简对照表。optimizer scalar step、lazy state 初始化等小项略去应注明。不要把第一条“计算参数2P+master4P”再额外叠加到第二条参数4P形成双重记账。
- FP32 m/v 不只是大范围：m/v长期累积、平方与epsilon路径也可能敏感。只需解释持久状态作用，不展开 Adam 整篇推导。PyTorch zeros_like 核对是重要反例：不能说所有mixed precision永远FP32 optimizer。

### 5. 累积窗口才是一次更新的边界

等权、相同有效分母的G个microbatch若每个loss为均值，目标L=(1/G)ΣL_j，scaled累积为 S/G Σg_j。所有microbatch保持S；结束才 unscale一次、clip一次、step一次、update一次、zero_grad一次。最后不足G个microbatch不能机械除G；若目标按token平均且有效token数不同，也不能平均各microbatch mean，交给packing文章详述。

反例C：g1=2，g2=4，S1=8，S2=16，累加得到80；用最后S2解码得到5而非6。提前unscale后又加scaled梯度亦不能用一个scale恢复。

反例D：g=(3,4)，S=128，clip阈值τ=1。正确 unscale→clip 得(0.6,0.8)；先clip(Sg,1)再除S只得(0.0046875,0.00625)，范数1/128。理想clip定义 g*min(1,τ/||g||)，g=0单独定义返回0。PyTorch为数值稳定使用epsilon，本文是忽略epsilon的理想模型，不宣称最后浮点bit精确一致。

以上数字已在本轮用 JavaScript 算术复算；还没运行PyTorch/GPU。writer/auditor可加CPU精确浮点转换/状态反例测试，但须标示仅验证局部性质。

## 建议正文递进大纲

1. **同样是16bit，为什么训练行为不同？** 用“forward能跑、参数却更新错”的问题引入四类精度，说明本文只讲FP16/BF16训练主线；不说入门或新手。
2. **指数决定能走多远，fraction决定步子有多细。** 格式表→正常/subnormal→更新吸收，读者先建立数值失败的两种机制。
3. **把高精度留在会积累误差的位置。** FP32乘加累积 vs FP32 master更新，接persistent state对照，解释不承诺总显存减半。
4. **用链式法则移动反向梯度的范围。** 推S梯度恒等式、例B、为什么不能先下溢再放大、S过大/forward溢出边界。区分BF16更大范围与更粗分辨率。
5. **autocast与GradScaler分工。** 以已锁定v2.5.1讲fp32 model+autocast、不手工half、局部敏感算子fp32、backward在外、scaler不替代optimizer。
6. **把一个更新拆成microbatch后，哪些量必须保持一致？** 推梯度归一化，例C说明scale窗口；尾窗口及不等token分母只点出并链接packing。
7. **unscale、clip、step不能交换位置。** 例D→状态顺序图→紧凑版本锁定训练骨架，示例显式处理窗口G/等权前提。不要贴无法独立执行的长训练项目。
8. **一次step被跳过之后，时间究竟前进了多少？** 普通optimizer非有限梯度跳过意味着参数/moments/optimizer step不前进，数据仍消费、scaler可调低；epoch/data步和成功更新步不同。scheduler若按成功更新数定义，不能每microbatch/每尝试无条件推进；若按token数定义则另有合理语义。别以scaler.step返回None判断失败，因为AdamW成功也可能返回None。checkpoint需含scaler状态及训练进度语义。
9. **怎样验证省下来的时间没有换走训练目标。** 固定FP32参考、loss/grad范数/finite/跳步频率/验证集指标、实际state dtype与显存、同样global objective/effective tokens、预热和实际吞吐。短跑局部数值一致不是长程收敛证明；无GPU不得写实测。

## writer 的代码约束与审核关卡

- 优先给单optimizer、普通AdamW、FP32参数、FP16 autocast骨架，`torch.amp.GradScaler("cuda")`，不用已旧的torch.cuda.amp命名。示例标为按v2.5.1 API核对、未GPU执行。
- 可以让调用方显式提供一个完整 `microbatches` 窗口，`G=len(microbatches)>0`，每个microbatch loss定义为等权单元的均值且单元数相同，再loss/G；不写loader末尾漏step bug。BF16替代路径必须解释scaler关闭后不再自动finite跳步，不把禁用scaler称等价保护。
- 若展示clip代码并且希望让GradScaler常规跳步生效，可使用默认 `error_if_nonfinite=False`；若设置True则非有限norm可能先抛异常，不能又承诺本示例总由scaler平滑跳过。unscale记录的是当时的非有限状态；后续人为gradient变换若引入NaN不能假设必被再次完整检查。只给普通clip流程，不拓展任意梯度改写保证。
- scheduler最稳妥正文说明语义而不强塞跨框架通用检测代码。单optimizer无手工new_scale的常规scaler中，比较old/new scale可作为该约束下的线索，但有同步且不宜包装为通用成功信号。多optimizer各自可独立skip，不纳入此文代码。
- DDP/FSDP要求所有rank更新/scale一致，不能让各rank独立决定更新而破坏同步；具体分片scaler/collective规范不在此文实现范围。
- 不说BF16不会underflow、FP32累积=完整FP32 GEMM、scaling提高有效位数、AMP自动建FP32 master、梯度finite保证optimizer更新finite、2bytes权重=全训练显存2P。
