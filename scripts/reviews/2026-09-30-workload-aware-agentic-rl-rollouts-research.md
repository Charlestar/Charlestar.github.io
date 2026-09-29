# WAR 调研与写作交接

调研日：2026-09-30。此记录是研究角色的一手资料、大纲与验证建议，不是独立审核或 GPU 复现声明。

## 一手资料、版本与实际阅读

1. [WAR: Workload-Aware Rollouts for Synchronous Agentic Reinforcement Learning, 2607.17299v1](https://arxiv.org/html/2607.17299v1)，[abs 元数据](https://arxiv.org/abs/2607.17299v1) 核实 v1 首次公开为 2026-07-19，作者 Ryan Xu、Atlas Zhao、David Bao、Frank Du。已阅读主文 Introduction、Background、WAR System Design 全部、Implementation、Evaluation 全部、Related Work 与 Conclusion。本文是预印本，不宣称已经同行评审。未取得足以审计 WAR 修改细节的作者实现，不把论文的实现描述当作已核对的源码行为。
2. [SuffixDecoding, 2411.04975v3](https://arxiv.org/html/2411.04975v3)，[abs 历史](https://arxiv.org/abs/2411.04975) 显示 v1=2024-11-07、v2=2025-06-02、v3=2025-10-07，NeurIPS 2025 Spotlight。已读 §2、§3（树构造、扩展、动态长度、评分、混合方案）、§4.1 实验设置及 Appendix C 的 batch 控制；也查看 Appendix B 的 scalability。不能把 WAR 2026 年采用 SuffixDecoding 写成该算法首次提出。
3. [ArcticInference 官方仓库](https://github.com/snowflakedb/ArcticInference)：只查看项目页与目录，未逐一审计其 verifier 后端。因此不能凭“官方开源”声称 WAR 所用 SGLang 适配已被我们证明无损。
4. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)：读取到的 PDF 为 v2（2023-05-18），实际阅读 §2.1–§2.3、Algorithm 1、§3.1–§3.4 及 Appendix A.1。用于核对分布保持所需的接受概率与拒绝修正，而非声称 WAR 恰好逐行使用这套线性草稿算法。

## 本篇的切口与现有覆盖差异

建议题目：`WAR：Agent RL 的 Rollout 为什么需要两层负载感知`。

既有文章已经解释 GRPO 数据流、训练/推理切换、异步 RL 与 speculative decoding 基础。本篇应围绕一个多轮编程 agent 的完整轨迹讲清楚：模型调用、工具执行、带着更长上下文返回，以及最后几条轨迹如何拖住同步训练。重点是同一 rollout 阶段中，瓶颈从并发请求之间的资源争抢转到尾部单条请求延迟，调度层与解码层要分别响应。

这不是异步 RL 论文。每轮仍等待本轮 rollout 后才更新 actor；异步 HTTP 更新 suffix cache，不等于 rollout 与新策略训练异步重叠。

## 必須保持准确的系统关系

### 两层控制，不是一个全局二选一开关

- 全局调度器：开始时把 rollout prompts 入队；全局总请求数 `N_req,tot < Thr_sched` 时退回 veRL 的 least-request + sticky-session，否则使用 cache-aware priority。
- 每个推理副本：按本副本的 runtime effective batch size 独立开关 SuffixDecoding。它不是训练配置里的 prompt batch size，也不是全局队列长度。
- 因此全局仍做 cache-aware scheduling 时，某个低负载副本也可开启投机解码；尾部尤其可能如此。不要画成“全局高负载→所有副本都禁用 spec”的单开关。
- 每个 RL step 结束后，广播完整 prompt-response token 序列；通过 HTTP 异步更新各副本 CPU suffix tree。共享的是 token 模式，不是 GPU KV tensor，也不是直接复用旧 policy 的训练轨迹。

### suffix history 与 KV cache 是两类不同的复用

SuffixDecoding 从当前 token 序列的后缀匹配历史 token 模式，取历史后继作为草稿。它不是语义检索，也不直接跳过目标模型验证。

可用自己的离散例子：历史 `[10,20,30,40,50]`，当前 token 后缀为 `[20,30]`，可提出 `[40,50]`。两个请求只要后缀相同就可能提出同样草稿，但它们完整左侧上下文不同，所以不能据此复用对应 KV。Suffix tree 存于 CPU，KV 取决于模型权重、完整合法前缀及相应执行条件。

原 SuffixDecoding 中有全局历史输出树与请求本地 prompt/output 树；节点的历史条件频率 `C(N)` 与沿路径乘积 `D(N)` 是候选扩展/评分启发式，不是当前 actor 的精确条件概率。不要把这些统计直接当作 target logits。

模型更新之后，旧历史文本仍可以当“提议”；是否有价值由新 target 的接受结果决定。在正确验证、tokenizer ID 空间一致等前提下，旧提议不会自动引入 policy staleness。相反，旧权重产生的 KV 不能因为 token 没变就跨更新沿用。

“Model-free”只排除了额外 draft network 的 GPU 参数和前向计算，不是零开销：CPU 查询、草稿树、验证计算、mask、临时 KV 与模式切换都有成本。未命中或成本过高时回到普通 decode。

## 从吞吐关系理解为什么接受更多仍可能更慢

建议自行构造计时例子，不把它写成论文硬件结果。以同样测量边界下的一条逻辑请求为例，普通一次 decode 提交 1 token，耗时 `T_decode(B)`；一次投机迭代（含必要草稿与验证）平均最终提交 `A(B)` 个 token，耗时 `T_spec(B)`。有利的条件是：

`T_spec(B) / A(B) < T_decode(B)`。

这里 `A` 必须定义为所有最终提交 token，包括算法可能生成的 correction/bonus token，而不是含糊混用论文 acceptance length。批次吞吐应用 `sum(committed tokens)/iteration time`，并保持相同并发/工作量，不能跨不同 B 只比单请求商。

纯算术例子：低负载普通 4 ms/token，投机 12 ms 提交 4 token，得到 3 ms/token，约 1.333 倍；高负载普通 1 ms/token，投机 7 ms 提交 6 token，得到约 1.167 ms/token，反而慢。接受长度从 4 到 6，并不能推导速度提高。

同理 rollout 阶段加速不是整轮训练加速。假设原 rollout 60 秒、reward/update 合计 40 秒，rollout 加速 1.5 倍后总耗时 80 秒，整轮只有 1.25 倍。这只是边界说明，不是 WAR 的测量。

## cache-aware placement 与请求优先级

论文把四种信号相加：等待时间、同组估计 assistant turns、目标副本 cache hit rate、副本 inflight requests。等待超阈值时给高正分；估计 turns 和 inflight 带负系数，cache hit 带正系数。系数实验调节，量级设计为 waiting > turns > cache > inflight。

同组 turns 的更新为 `T_new = alpha*T_observed + (1-alpha)*T_old`，其中观测是已经结束的同组轨迹。它是总 assistant turns 的轻量代理，不是精确剩余 token 数，更不是已知未来完成时间。一次工具调用的时长和各轮 token 数都可能不同，不能说 WAR 实现了 oracle shortest-job-first 或最优 makespan。

sticky session 只承诺路由偏好，不等于 pin 住 KV。工具执行期间，原副本的 cache 仍可能被其他请求驱逐；同组的另一个副本也可能已有共同前缀。因此必须区分“我曾在这里跑过”和“现在真正命中”。

sandbox TTL 会改变正确性边界：等待太久被回收会让轨迹提前终止。优先级中的等待项是工程保护，不是资源不足时依然保证所有 deadline 的形式化定理。论文公式的 waiting 项在阈值未触发时是 0，不应原样沿用其后面无条件的 `S_waiting>0` 简写。

### 不应照搬成生产算法的 AIMD 细节

论文修改后的窗口规则写的是：平均 hit rate 低于阈值时 additive increase，高于阈值时 multiplicative decrease；这与常见“内存压力高则减少窗口”的直觉不是同一个信号。不要擅自反转分支后仍称论文公式。没有作者实现核对，不能解决其信号含义/实验调参的不足。

论文还写：

`W_t = min(max(1, W_hat), inflight_rep + queue_total / N_rep)`。

但文字称窗口至少 1。如果 inflight=0、queue=0，右侧上界为 0，公式输出 0；若 queue 少于副本数，则可能出现小于 1 的小数。整数舍入与上下界冲突时的策略未明确。建议正文不把该式作为可直接运行的实现，而明确说并发窗口是需工作负载校准的反馈控制，并指出论文没有给足边界处理。若正文展示公式，必须同时指出反例，不能默默“修好”后归因作者。

## 输出分布是 rollout 优化的前置契约

WAR 的实验用 temperature=1、top-p=1、top-k=200，不是贪心。仅比较 draft 是否等于 target argmax 并不足以证明采样分布保持。

经典单步 speculative sampling：令 `p` 为 target 分布、`q` 为草稿实际采样分布；都必须在温度和截断等处理之后重新归一化。草稿 `x~q` 的接受概率为 `min(1,p(x)/q(x))`；拒绝后从归一化的正残差 `max(p-q,0)` 采样。若 q(x)=0，该 token 本来不会从 q 被抽到；不能直接运行除零表达式。树验证还要正确处理树 mask、祖先关系和提交状态，不能把兄弟草稿互相作为上下文。

可 CPU 复算的反例：`p=[0.6,0.3,0.1]`、`q=[0.2,0.5,0.3]`。接受质量为 `min(p,q)=[0.2,0.3,0.1]`，总接受率 0.6。拒绝概率 0.4，残差条件分布为 `[1,0,0]`，合并恰好还原 p。若拒绝后错误地再从原始 p 抽一次，最终为 `[0.44,0.42,0.14]`，已改变策略分布。

这只是经典分布保持原理的可复算例子，不是在声明 WAR 的 suffix-tree verifier 使用了上述 q 表，更不是已经证明其实现通过。原始 suffix tree 的历史频率不一定就是草稿实际抽样分布；应按实际构造机制验证。输出同分布也不等于同随机种子得到同一轨迹或 bitwise 相同 loss。

实现检查点可以在文章末落成清单：当前 policy version、tokenizer/token IDs、采样参数、正确 attention/tree mask、未接受分支的临时 KV 回收、EOS/tool-call 边界、模式切换后逻辑位置与 RNG 状态。工具调用不能在未验证的 draft 上执行。

## 实验数字与限制必须成对出现

- 软件：veRL 0.8.0、SGLang 0.5.5、Megatron-LM 0.13.1；作者称 reward/train 组件不变，不等于我们已审计完整实现。
- 模型/数据：Qwen3-32B，YaRN 扩展到 96K；500 条 SWE prompts。
- 硬件：16 H100，两个节点、每节点 8 卡，InfiniBand。
- 训练 TP/PP/CP=4/2/2；rollout TP=4；这些参数在比较中保持不变。
- training batch size=4、8、16、32、64；每个 prompt 8 条 rollout；每种配置 5 个 training steps。不要把 5 steps 写成 5 个独立随机种子。
- temperature/top-p/top-k=1/1/200，shuffle 关闭，最大 assistant turns 120。
- 主要测量是 end-to-end rollout time、prefill throughput、decode throughput。摘要报告低负载约 1.4 倍、高负载最高约 1.6 倍 rollout throughput；不能推广为整体训练都快这么多。
- 作者只保留 prefill tokens、decode tokens、assistant turns 相对 baseline 差异都在 5% 以内的 runs。它降低工作量差异，但不足以证明采样分布不变，也不是完整无筛选性能分布。没有所有被筛掉 runs/独立重复，不能据此声称稳健无偏平均收益或长程收敛等价。
- isolated decode benchmark 的固定 input/output=8192/512，batch>16 时投机可能劣于普通 decode。阈值 16 只是该设置的经验结果，不是所有硬件与模型的常数。低/高负载中观测 acceptance length 约 2.8/6，后者更高仍不保证更快。
- 最值得讲清楚的消融：training batch64 若取消 TTL 等待保护，相比启用等待项的配置（不是 veRL_base）prefill tokens -44.5%、decode tokens -41.3%、assistant turns -38.9%、rollout time -46.9%。这里“更快”包含提前截断、少做工作，不能称为有效性能提升。

## 建议逐层文章结构

1. 从同一个修复代码的 agent 轨迹讲起：LLM→工具→新上下文→再次 LLM。用时序图说明同步 barrier，不把模型每一轮调用与整条 trajectory 混用。
2. 同一轮内部也会从拥挤走到尾部：为什么全局 queue、每副本 effective batch、训练 prompt batch 是三种不同量。
3. 历史后缀能预测下一段，但不是旧答案直抄：token ID 小例子、suffix tree 与 KV 的区别、当前 target 的验证契约。
4. 接受长度不是速度：展开一次投机迭代成本，并用高低负载两组纯算术例子说明控制目的。
5. 请求放在哪里、先服务谁：cache hit、siblings turns、inflight 与 TTL，让一次具体排队选择串起这四个信号；讲经验启发式的边界。
6. 合起来才是 WAR：两层控制图；共享 token 历史与局部 KV 生命周期；不要引入不存在的跨新旧策略 KV 复用。
7. 看懂作者测量：局部与整轮指标、固定比较工作量、5 步与筛选限制、TTL 消融“假加速”反例。
8. 实现/阅读这类系统的检查契约：正确提交、采样分布、工具 side effects、deadline、反馈边界；AIMD 未解释处明确留白，不把论文预印本当可无条件搬用的生产配方。

## CPU 复算建议（writer/root 自行落地测试）

1. 接受分布：验证 `minimum(p,q) + (1-sum(minimum(p,q)))*residual == p`，同时验证错误 rejection-to-p 的反例。
2. 投机盈亏：核对 4→3 ms/token 的 4/3 倍与 1→7/6 ms/token 的退化，强调是给定参数的算术。
3. Amdahl 边界：60+40→60/1.5+40=80，总速比 1.25。
4. 尾部示意：4 个抽象独立 worker 的任务时长 `[2,2,2,10]`，barrier=10、忙碌时间总和16，占用比16/(4*10)=0.4。不是 GPU utilization 实测，也未模拟 continuous batching。
5. AIMD 反例：任意正 `W_hat`，inflight=0/queue=0 得窗口0；queue=1/N_rep=4 时上界0.25。用于暴露论文公式边界，不作为建议采用的控制器。

所有数字验证仅覆盖构造的数学关系。它们不能复现论文 GPU 吞吐、tree verifier 正确性、RL 收敛或真实 sandbox 完成率。
