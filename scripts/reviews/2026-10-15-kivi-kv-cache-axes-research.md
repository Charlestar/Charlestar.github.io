# KIVI：调研与逐篇写作交接

调研日期2026-10-15。建议 `_posts/2026-10-15-kivi-kv-cache-axes.md`；候选标题 `KIVI：Key 和 Value 为什么不能沿同一条轴量化`。系列80，首发2024-02-05；链接KV内存基础、NVFP4 KV与QuantSpec作为后续，而不将已提过KIVI的引用当已有完整文章。

## 本轮实际读取

- [KIVI arXiv v2](https://arxiv.org/html/2402.02750v2)：实际读§2、§3.1–3.3、相关实验设置/ablation边界、附录A算法1关键步骤；不复现任务/吞吐结果。v2修订2024-07-25，首发仍二月。
- 官方KIVI main经API固定commit **`876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`**，提交2025-11-20：
  - [models/llama_kivi.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/models/llama_kivi.py)：实际读decode quant/full logits拼接、Kflush顺序、VFIFO、prefill分割、repeat_kv与GQA相关段落；未逐行审完整模型所有类。
  - [quant/new_pack.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/new_pack.py)：实际读quant_and_pack_kcache/vcache的维度分组、minmax和triton_quantize_and_pack_along_last_dim相关路径；未执行Triton/CUDA。
- 论文Algorithm A与源码的flush相对当前attention时点有差异，文章需固定明确演示哪条路径，不能把两者糊成“永远保留R个全精度KV”。

## 维度与zero-point单位

单head先省略batch/head：`K,V∈R^(T×D)`。

- **Key per-channel**：固定d，沿T上的连续G个tokens为组，`K[t:t+G,d]`共享range。不是把同一token的所有D通道绑一scale。
- **Value per-token**：固定t，沿D上的连续G个channels为组，`V[t,d:d+G]`共享range。不是同一channel跨T共享scale。
- 一张小表/矩阵轴示意有价值，避免“along channel dimension”自然语言容易误读。
- 原实现保存float `mn=min(group)`与step `(max-min)/(2^b-1)`，`q=round((x-mn)/step)`, `xhat=mn+step*q`。这里论文把mn叫zero-point，但它是浮点原点/offset，不是前篇affine整数z；可换名m避免混用。
- 这套minmax网格一般不保证0恰落integercode，不能硬套“零点一定整数且精确零”的ONNX affine约定。若要转换成浮点bias型式，明确单位，不声称mn/step必整数。
- constantgroup mx=mn会使朴素step0；教学实现必须单独表示constant或设置positive step+code0+offset，且说它是参考约定，不假装固定源有完整相同处理。

## Key/Value误差怎样进入attention

单query `q∈R^(1×D)`，logits `s=q K^T/sqrt(D)`，`p=softmax(s)`，`o=pV`。

- `Khat=K+E_K`令logits改变 `delta_s=q E_K^T/sqrt(D)`；softmax非线性使attention权重改变。
- `Vhat=V+E_V`；若用phat表示量化K后的softmax，则完整输出差精确分解 `ohat-o=(phat-p)V+phat E_V`。不是只看KV本身平均MSE就能推导任务误差。
- 论文观察Key固定channel outliers，per-channel隔离它们；Value未呈同样规律，attention加权聚合使per-token局部误差控制重要。它是所测模型的动机与经验，不是per-channel V在任意矩阵永远更差的数学定理。
- “asymmetric”同时涉及K/V不同轴策略和每组minmax非对称数值，不要只解释其一。

## Residual buffer的真实不变量

分开定义group_size G和residual_length R，**R%G=0**；二者不必须相等。

Key：新K追加到full residual，长度达到R时，将整段按G沿T分组quant，追加历史，residual清空。故persisted Keyfull长度在0..R-1周期变化，而非永远至少R；论文说平均窗口约R/2仅均匀周期意义，不是每step最低保护。

Value：按token可立即量化，但为质量维持recentFIFO；长度超过R时量化最旧token并保留最近R，故成熟后长度R（当前compute前可暂时R+1）。

具体pinned代码顺序：decode先appendK并用full residual计算当前step logits，之后若达到R就flush；再全局softmax；appendV，混合oldquant+full计算当前output，最后若Vlen>R将最旧V量化持久化。因此刚好flush的最新K/V对**当次attention**仍可按full支路用，持久化状态供下一步不同。不得只画cache状态就推断当次compute用了什么。

注意旧quant/full **logits拼接后统一softmax**；不是两段各softmax然后直接加output。Value quant/full分界按V的token长度切p，不能把K的分界复制给V，因K/Vresidual长度不同。

Prefill源码先用原始K/V完成prefill attention，再准备量化cache；不能写prefill在第一层就只用低位cache自注意力。

## 逐节大纲

1. 从长context decode必须反复读cache开头，解释权重量化没覆盖这个动态状态。
2. 把K/V放回两次不同attention计算，建立误差路径，不先讲压缩倍数。
3. 用明确T×D矩阵画group axis；Key固定channel大值为何污染token-wise scale。
4. Value反例展示无关token决定sharedscale会损害被attention选中的token；这不是只“V没outlier所以随便量化”。
5. 2bit code+floatmin+step怎样重建，和常见integerzero_point单位区别；constant/tail处理。
6. Streaming难题：K沿时间分组需要等未来同组数据，buffer收齐再封存；Value按token可增量但仍保留recent。
7. 分开Kflush与VFIFO，给R4/G2小时间线/状态机，并标compute与persist时点。
8. Mixed-cache attention如何算：拼接logits统一softmax，再按V正确split概率；不保证softmax概率没变化。
9. 标准库参考：smallmatrix grouping+2bitpack+buffertrace、toyattention，明确非真实Triton内核。
10. 实际bytes计算：2bitpayload+每组m/s+残留FP16+padding，不能16/2=8直接当服务显存压缩；短context元数据/残留占比大。
11. 质量与系统边界：不丢token、不缩KVheads、不是KV eviction；GQA缓存按Hkv而非Hq；pack/layout/append/rollback需另外集成；论文吞吐含更大batch，不是单请求固定speedup。
12. 与NVFP4 KV、QuantSpec的联系：前者改表示/硬件，后者加入草稿/目标不同视图与事务回滚，不能直接复用KIVIcache状态机当推测解码rollback证明。

## 已执行的构造

Node标准库 minmax+RNE（教学constantgroup单独精确表示）执行：

- Key4×4：`[[100,0,1,-1],[101,0.5,1.5,-0.5],[102,1,2,0],[103,1.5,2.5,0.5]]`。
- 2bit per-token重建普通三channel都塌到本行min：行1[100,-1,-1,-1]、行2[101,-0.5,-0.5,-0.5]、行3[102,0,0,0]、行4[103,0.5,0.5,0.5]；per-channel的4点恰落各自grid而完全恢复。是构造例，不证明真实Key必exact。
- Value3×2：`[[-100,-100],[1,0],[100,100]]`，p=[0,1,0]选择中间token。token-wise重建中行[1,0]，channel-wise两列取[-100,100]scale=200/3后中行均33.3333；重要token被无关token污染。p是人为给定固定概率，用来隔离V误差，不冒称由某个Q/K softmax恰好产生onehot。
- R4持久状态t1..9：Kres `[1,2,3,0,1,2,3,0,1]`；Vres `[1,2,3,4,4,4,4,4,4]`。注意这是单步decode后persisted状态，当前compute前暂存长度不同。
- 建议auditor检查每token计数不漏不重、oldquant+residual=logicalT、Kres%G关系、R/G和shape边界、K/Vsplit不同、globalsoftmax与错误splitsoftmax反例；使用实际bodycode而不复制表里预期值。

## 尚未完成/不可声称

未运行原KIVI模型、GPUpack/BMM、质量/吞吐或长期cache演化实测。不是所有backends都相同constant/rounding策略，也没有证明2bit在所有长上下文任务无损。最终稿需独立审核模型逻辑、数值例和公式渲染。

## Root 额外来源交叉检查

Root 实读 KIVI v2 §2–3.3、§4.1 时发现，§4.1 将 Mistral 笼统归入 multi-head attention，不能据此在本篇写成 Mistral-7B 使用普通 MHA。已另查 [Mistral 官方 2023 发布说明](https://mistral.ai/news/announcing-mistral-7b/) 与 [Mistral 7B 原论文摘要](https://arxiv.org/abs/2310.06825)，其明确采用 GQA。本文最好直接按实际模型的 Hkv 计算，不复制这句架构分类。原论文和源码也需逐条判断，不能因是一手来源就把旁支表述全部当作正确结论。

独立预审者额外读了 `quant/matmul.py` 与模型各分支，提醒：R4/G2 仅是状态教学，不代表原生 2-bit/INT32 packing 能直接接受该尺寸（每 INT32 装 16 项）。固定普通 Attention 限制 Hq/Hkv=1；Flash 的部分短 cache `value_quant=None` 分支没有对 `value_full` 作 repeat，不能因其他分支出现 `repeat_kv` 就称所有 GQA 状态已认证。此为代码阅读风险，不是实际 Torch/GPU 失败复现。正文可解释 GQA 的 Hkv 容量公式，但不据此承诺该 commit 的全路径兼容；min/scale 的 dtype、constant-group 处理也要与独立标准库参考分开。
