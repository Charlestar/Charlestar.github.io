# Mamba-2 与 SSD 调研交接

日期：2026-10-21。论文固定 arXiv:2405.21060v1（2024-05-31），官方源码固定 v2.2.2 对应 commit 8ffd905c91d207f5c0cc84fc2a2fb748655094f0。依调研 → 写作 → 独审顺序完成，未以摘要中的性能倍数代替核心机制解释。

## 实际来源与阅读范围

调研代理实际读取论文 §§2、4–8 对应定义、结构对偶、分块、架构和系统部分；作者 Part I / III 解释；官方 ssd_minimal.py、mamba2.py、mamba2_simple.py。写作代理再次阅读论文 §§5–7 对应方法/复杂度片段、作者 Part III 的四步算法/最小代码/非线性复杂度警告/数值稳定部分，并读取固定源码 minimal 全文及 step、cache 分配与 block 相关路径。此记录不声称精读全部实验、证明或附录。

- https://arxiv.org/abs/2405.21060
- https://arxiv.org/html/2405.21060v1
- https://goombalab.github.io/blog/2024/mamba2-part1-model/
- https://tridao.me/blog/2024/mamba2-part3-algorithm/
- https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/mamba_ssm/modules/ssd_minimal.py
- https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/mamba_ssm/modules/mamba2.py
- https://github.com/state-spaces/mamba/blob/8ffd905c91d207f5c0cc84fc2a2fb748655094f0/mamba_ssm/modules/mamba2_simple.py

## 本篇结构及关键边界

- 单 head 的离散状态方向自定为 N×P：H_t=a_t H_prev+b_t x_t^T，y_t=H_t^T c_t。当前写入不额外乘 a。Mamba-2 同 head 标量转移是比一般对角 SSM 更强的结构限制。
- 从 H_initial=0 展开 M_ts=(c_t^T b_s)prod(a_{s+1:t})，非零 initial 另加项；当前时刻空乘积为 1。
- L ⊙ C B^T 对偶是未归一化 structured masked linear attention，不是任意 softmax attention 的等价替换；参数由输入生成不代表整个网络是线性算子。
- M 整体可满秩。真正的低秩是跨时间分割的 off-diagonal 子块，rank≤N；mask 本身对应 1-semiseparable。
- 推导 local outputs、zero-state chunk summary、boundary affine scan、incoming-state correction 四部分；边界复合 associative、不 commutative；首个 token 的 carry 含本步 a_l。
- 理想块内 O(TQ(N+P))、状态 O(TNP)、边界 O(KNP)，K=ceil(T/Q)；1<=Q<=T 时再简写为 O((T/Q)NP)，固定 Q/N/P 对 T 线性。官方最小例第 3 步显式构造 K×K 矩阵，不是长 T 的真正线性实现；作者明确优化路径使用实际 scan。
- 实际 step dA=exp(dt*A_cont)，写入 dt*B*x。教学 b 可吸收 dt；官方 minimal X 吸收 dt、A 参数为 log decay，不能两处重复乘 dt。作者博客末尾记号不作为覆盖固定 step 源码的依据。
- 完整 block 另含 projections、短卷积/SiLU、D*x、gate/norm/outproj，不能把核心 recurrence 冒充完整官方层。
- 固定 inference cache 含 SSM 和 convolution states；示例容量独立计算 16MiB+0.5625MiB，不含其他内存、不作质量等价对比。恢复、前缀和回滚需保存相关状态；hybrid attention 层仍有 KV。

## 代码验证范围

正文三种核心算法及输入检查/示例由 mamba2-ssd-examples.mjs 提取执行。递推、完整矩阵和 chunk 不互相调用来假装独立；另用逐写入 tagged-contribution oracle 检验全部输出与最终状态。Fraction 有理数输入保持精确比较；浮点输入用明确容差。覆盖非零初态、尾块、chunk1、chunk>T、零/单位 gate、负系数、空序列、小衰减长序列、shape/NaN/非法 chunk 参数和状态容量算术。

逆向累乘区间 decay 不使用 prefix product 相除，也不在每一 pair 重算整段乘积。Python 程序是教学模型，不等于官方 backend 的数值/梯度/性能测试。没有导入或执行 Torch、CUDA/Triton、预训练模型、GPU 速度或任务质量评估。独审和主代理构建/页面/发布证据另行记录。
