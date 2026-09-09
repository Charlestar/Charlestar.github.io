# RoPE 与上下文扩展：调研与写作交接

状态：调研完成；2026-09-09。仅调研记录，不是最终文章。不包含模型实验复现。

## 选题与仓库定位

- 当前 `_posts` 为 77 篇；搜索标题、系列元数据及 RoPE/重计算/Roofline 正文命中后，确认本主题没有独立文章。
- 建议标题：`RoPE：相对位置怎样进入 Attention，长上下文又改变了什么`。
- 固定交接文件：`_posts/2026-09-09-rope-relative-position-context-extension.md`。
- 建议 `series: attention-long-context`、`series_order: 5`、`technology_year: 2021`、`mathjax: true`、`tags: [注意力机制, KV Cache, LLM推理]`。order 5 在现有 FlashAttention=10 之前，不改旧文。
- 与 GQA 的 KV-head 共享、MLA 的 decoupled RoPE、CP 的全局位置规则形成衔接；不重复讲 FlashAttention 分块算法。
- 未发现仓库及其 D:\、D:\Codes\ 祖先目录的 AGENTS.md。

## 实际查阅的一手资料

1. [RoFormer, arXiv v5](https://arxiv.org/html/2104.09864v5)：实际读取 §2.1、§3.1–3.3、§3.4.1 的开头；核心是 §3.2.1 二维旋转及 §3.2.2 多维推广。2021 是首次提出年份，v5 是 2023 修订。支持旋转 Q/K、正交性质、显式相对位置。不要照抄 HTML Eq.16 的疑似转置排版错误，按本文下方矩阵恒等式自行推导。
2. [Position Interpolation, arXiv v2](https://arxiv.org/html/2306.15595v2)：实际读取 §2.2 后半、§2.3（Eq.4–8）、§3 开头。PI 的确定定义是将位置 m 改为 mL/L'；文章无需转述“600 倍界”或复述原实验数字，避免把理论上界当质量保证。
3. [YaRN, arXiv v3](https://arxiv.org/html/2309.00071v3)：实际读取 §2.1–3.4、Appendix A.2–A.3、B.1–B.5 中返回的表格/说明。重点 §3.2 的按频率分段、§3.3 的 attention temperature、A.2 的 base-change 推导。α=1、β=32、幅度 1+0.1 ln s 是作者针对 LLaMA 系列的经验选择，不是普适常量。**HTML §2.3 Eq.9 写 g(m)=s·m，与该文 s=L'/L 定义及 PI 原论文矛盾；必须采用 PI 原论文的 m/s，不照抄该式。**
4. [Transformers v4.57.1 modeling_rope_utils.py](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_rope_utils.py)：实际读取 `_compute_default_rope_parameters`、`_compute_linear_scaling_rope_parameters`、`_compute_dynamic_ntk_parameters`，核实 linear 为 inv_freq/factor，dynamic 是依赖 seq_len 的另一公式。固定版本，不称其为最新。
5. [Transformers v4.57.1 modeling_llama.py](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py)：实际读取 `LlamaRotaryEmbedding.forward`、`rotate_half`、`apply_rotary_pos_emb`、`LlamaAttention.forward`。核实 FP32 生成相位/三角函数、split-half 配对，以及先旋转 K 再写 cache 的此实现顺序。
6. [Transformers RoPE utilities 文档](https://huggingface.co/docs/transformers/main/en/internal/rope_utils)：实际读取 Overview、Configuration、Per-Layer-Type Configuration。main 文档存在 `rope_parameters`，固定 v4.57.1 源码则使用 `rope_scaling` 等；正文不要提供混用版本的万能配置片段。

以上为机制查阅；没有完整复现论文实验。最终正文宜在机制段落就近引用，并用自己的推导与例子展开，勿大段复述来源。

## 公式与约定（写作时逐个引入）

以下是基于矩阵定义的独立推导。用列向量；q_m、k_n 指已投影但未旋转的 head 向量。旋转维 d_r 为正偶数；attention 缩放维 d_h 与 d_r 可不同。

- 二维：R(φ)=[[cosφ,-sinφ],[sinφ,cosφ]]，R(a)^T R(b)=R(b-a)。于是 `(R(mω)q)^T(R(nω)k)=q^T R((n-m)ω)k`。正负号必须和列向量、旋转矩阵统一。
- 多维：i=0,...,d_r/2-1，ω_i=b^(-2i/d_r)；R_m 是 R(mω_i) 的 block diagonal。`q̃_m^T k̃_n=q_m^T R_(n-m) k_n`。若只旋转部分维度，未旋转部分直接保留，其内积加回 score；softmax 常用 `/sqrt(d_h)` 而非擅自改成 `/sqrt(d_r)`。
- 正交只保证无额外幅度缩放的 RoPE 保范数，不保证每一项 attention 随距离单调减小，也不保证训练外位置泛化。YaRN 的幅度缩放不再保范数。
- 波长 λ_i=2π/ω_i，训练区间旋转圈数 r_i=L/λ_i。说明高频/低频是各维相位变化速度，不是某一 token 的内容频率。
- PI：s=L'/L>1，m'=m/s，等价 ω_i'=ω_i/s。局部相邻 token 相位差也缩小，不是只有窗口外的位置变化。
- 静态 NTK-aware base change（d_r>2）：要求最高频不变、最低频变成原来 1/s，可推得 b'=b*s^(d_r/(d_r-2))，继而 ω_i'=ω_i*s^(-2i/(d_r-2))。它并非每维 PI，不等同于所有库的 dynamic NTK。
- YaRN 概念公式：γ_i=clip((r_i-α)/(β-α),0,1)，ω_i'=(1-γ_i)ω_i/s+γ_iω_i。这是论文按 r 的 ramp 描述；库代码可能按维度区间实现 ramp，不能宣称逐点相同。完整 YaRN 再调整 attention logits；若 Q、K 都乘 a，则 score 乘 a²，不是 a。论文常用 a=1+0.1 ln s，等价 temperature t=1/a²，正文明确经验性和全维假设。

## 可复算例（代入计算，不是实测）

1. 二维正确性：q=k=(1,0)，ω=1 rad/token，m=2、n=5，则旋转后内积 cos(3)≈-0.989992；位置同时加 10 仍相同。n-m=6 时 cos(6)≈0.960170，直接反驳逐对内积随距离单调衰减。
2. d_r=8、b=10000：ω=[1,0.1,0.01,0.001]。L=4096，r≈[651.899,65.190,6.519,0.652]。s=8 时 PI 为 [0.125,0.0125,0.00125,0.000125]；静态 base change b'=160000，频率为 [1,0.05,0.0025,0.000125]。前两维为何保留更细相位、最后一维为何相同，可以逐项解释。
3. PI 位置：原 L=4096，扩展 L'=32768，最后一个下标 32767 映射 4095.875，不要写成 4096；范围采用半开区间。
4. YaRN 幅度：s=8，a≈1.207944，a²≈1.45913；因此缩放 cos/sin 的幅度会影响 attention temperature，不只是位置。
5. 可建议写一个只依赖 Python math 的小型代数断言：保范数、相对位置、共同平移、PI 相位等价。若写作 agent 未执行，必须标为可运行检查，不能写“已验证”。

## 针对性递进大纲

1. **同样的 token，顺序为什么会改变计算**：只说明无位置输入、无 mask 的 attention 具有置换等变性；causal mask 已带来顺序约束，不能说 decoder 完全没有顺序信息。先定义 Q/K logits，再提出要让内容匹配携带相对位移。
2. **把一个二维向量随位置旋转**：矩阵与具体向量例；说明先投影、后旋转；随后逐步推出相对位置式，不一上来列完整频谱。
3. **一个 head 为什么需要一组转速**：多维配对、ω 与 λ、partial RoPE、d_h 与 d_r；放 d_r=8 可核算表。仅用小表即可，不需生成图片。
4. **相对位置性质究竟保证什么**：固定内容的相对位移恒等式、范数、非单调反例；区分数学可计算与模型分布外泛化。
5. **窗口扩大八倍时，频率在改什么**：PI 压位置，再讲保高频的 base change，再讲 YaRN 按频段和温度；每种回应前一种留下的问题，而不是堆名词。表中仅对比同一 d_r=8 假设。
6. **K 一旦写进 Cache，位置规则就是状态的一部分**：读取到的 Llama 固定版本先旋转后存储；prefix cache 的模型配置、全局位置、scaling 配置必须匹配；paged block 的物理槽不等于 position ID；CP/packed sequence 内链。
7. **改频率为什么可能破坏增量解码一致性**：用一组二维相位显示旧 K 与新 Q 混用频谱并不等于固定新频谱的 full prefill。动态方法必须核对实现的重建/转换策略；仅重新旋转缓存 K 并不必然恢复全网络一致性，因为历史 hidden states 已受旧规则影响。
8. **验证应分成代数、执行与任务质量三层**：代数 checks；full prefill 对逐 token/chunked logits、left padding、packed boundaries、长短 batch、cache reuse；长文本 perplexity 与检索位置/距离矩阵、短上下文回归、真实任务。不存在跑通 128K 请求就证明有效 128K 的捷径。
9. **收束**：RoPE 是几何规则，扩展是修改频谱和学习分布，serving 是维护状态一致性。自然串起 GQA、MLA、CP，而非泛化的总结清单。

## 审核必须守住的边界

- q^T R(n-m) k 的转置与正负号；频率索引从 0 起；d_r 为偶数，base-change d_r>2。
- 不把 token 位置、KV 写入槽位、padding 后下标、packed 样本位置混为一谈。
- 不把改变 rope_theta 当作无损推理优化；不把 YaRN 写成只有插值；不把 amplitude a 写成 logits a。
- 不把 RoPE 数学相对位移性质上升为整个多层、带 mask、分布外 LLM 的严格平移不变性。
- 不声称高频一圈后全部位置编码碰撞：那仅是某一频率的周期，不等于整组多频状态同时重复。
- 不强求等价 kernel bitwise equality；应按 dtype、后端制定数值容差。
- 引用代码必须固定版本；文中若没有真实 GPU 或质量实验，只交付可复算例与验证方法。
