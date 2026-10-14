# GPTQ：调研与逐篇写作交接

日期：2026-10-15。建议 `_posts/2026-10-15-gptq-second-order-reconstruction.md`；候选标题 `GPTQ：量化一个权重之后，怎样让其余权重补偿误差`。核心技术首发2022-10-31，系列顺序40；ICLR2023是会议年份，不应改变顺序。

## 实际阅读的一手来源

- [GPTQ arXiv v2](https://arxiv.org/html/2210.17323v2)：逐段读§3、§4全部（含算法1）、§5 Setup、Practical Speedups、Additional Tricks、Extreme Quantization及§6；查看相关表用于边界，不照搬为当前速度保证。未完整复读所有附录实验和相关工作。
- [官方实现gptq.py](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py)：实际读完整160行，包括add_batch、dead、damp、Cholesky/inverse/upper factor、actorder/groups、block update和写回weight。
- [官方quant.py](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py)：实际读完整197行，包含grid参数、Q/DQ、3-bit pack与单tokenkernel入口。
- [官方README](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/README.md)：已读取说明；act-order与true-sequential是后续代码增强，不能把所有现代可选项倒叙成初版固定算法。
- 以上commit **`2d65066eeb06a5c9ff5184d8cebdf33662c67faf`** 通过GitHub公开commit API核验，提交时间2023-07-11；不是猜测hash或可变main引用。

## 统一系列矩阵方向，不照抄论文符号

系列使用 `X∈R^(M×K), W∈R^(K×N), Y=XW`。论文使用 `Wpaper∈R^(N×K), Xpaper∈R^(K×M)`，把series两者转置即可。官方nn.Linear.weight也按(N,K)。

对series的一个output column `w∈R^K`，令量化后的向量为w_hat，目标为：

`L(w_hat)=||X(w_hat-w)||_2^2=(w_hat-w)^T X^T X(w_hat-w)`。

这是线性层重建代理的**精确二次型**，其Hessian `H=2X^T X`，不是完整LLM任务loss Hessian。称GPTQ approximate second order，不等于这里的H是随意近似的任务二阶导；近似来自逐层proxy、有限校准与逐步离散决策。

### 一次约束最优补偿

在当前未量化坐标集合F上，假设H_F正定，当前q权重w_q量化到grid值Q(w_q)，定义残差 `r=w_q-Q(w_q)`。固定该维变化 `delta_q=-r`，其余仍允许连续调整。最小化 `1/2 delta^T H_F delta` 得

`delta_F= -r / (H_F^-1)_(qq) * (H_F^-1)_(:,q)`。

可以用Lagrange multiplier 3行推导：Hδ+νe_q=0，δ=-νH^-1e_q，约束δ_q=-r推出ν=r/Hinv_qq。补偿会随相关性改变其他权重，符号不能凭“减去误差”口述。注意这个更新包含q分量本身；若只实现未来坐标，q已经被显式写为Q(wq)，不可再重复更新一次。

限制：这是固定一个量化决定后的连续最优；后续坐标还要量化，不能推出整层离散global optimum，也不保证对任意输入最终优于RTN。

### Raw inverse 与 Cholesky factor 必须分开

教学先用P=H_F^-1；真正GPTQ预计算upper triangular U，使 `H^-1=U^T U`，算法使用 `err=(w-q)/U_qq`，再 `W_remaining-=err*U_q,remaining`。源码把U变量命名Hinv并不代表它此时仍是raw inverse矩阵。

不可在同一个公式中分母用U_qq却分子乘raw Hinv row；也不可把U_qq写成完整Hinv_qq而忘了factor变化。文章建议严格用P/U不同符号，清楚说明两个形式的关系和未量化集合语义，避免把完整逆矩阵全行一次套用到所有后续步骤。

## 逐节提纲

1. 开头不是“4bit可以省75%”老话：两个量化误差相同的权重向量，输入方向不同可使层输出误差截然不同；从前篇`DeltaY=XDeltaW`自然进入。
2. Calibration究竟提供什么：把前向样本收集成X，保存Gram而非所有activation；不需task labels、不对完整模型反传，但仍数据依赖，不是data-free。
3. 固定grid下，weight MSE为什么是特殊情形：XTX若是常数倍I，方向等权；一般有diagonal敏感性与off-diagonal关联。解释H的维度K×K以及与output数N共享。
4. 用两维例子先看“先量化、后挪动”过程，保留中间连续weight不在grid上的事实。
5. 约束补偿式的推导，区分r定义与delta符号；重点是用未量化自由度补偿已经付出的误差。
6. OBQ到GPTQ的三点工程改造：相同列顺序共享H信息，lazy block updates变成更适合GPU的矩阵工作，Cholesky避免反复inverse downdate误差；不是“GPTQ简单对H求逆然后round”。
7. 解释damping/dead channels：有限样本常导致rank deficient；H+λI稳定而改变代理，zero diagonal是校准未激活方向，不等于真实部署永远无用。官方会把dead weights置0，这是实现选择/校准风险。
8. Group size和act-order是不同轴：group决定共享grid，blocksize决定更新批；act-order按Hdiag启发式处理列，最终必须恢复排列并保存packing所需group/perm契约。不要把group128与blocksize128混为一个参数。
9. 一小段标准库2×2参考（或通用小矩阵reference）：验证直接inverse约束一步、loss对比、独立brute-force可行网格；必须明确不是高性能GPTQ实现。
10. 从浮点模拟Q/DQ到packed weights是另一步：gptq.py最后Q写回仍是原dtype浮点tensor（落在格点），quant.py的pack才构造真实低位artifact；存储、decode、MMA dtype分别看。
11. 论文速度边界：其batch1 matrix-vector路径动态dequant，主要节省weight traffic；未量化activation、不改变计算操作数量；3-bit OPT175B塞A100与4-bit完全不同，不能将“4bit175B单A100”写错。
12. 质量收束：calibration domain/length影响X，layer proxy低≠PPL/任务/长上下文无损；与AWQ都activation-aware但目标/求法不同（补偿vs缩放）。不在此提前教AWQ全文。

## Claim/source mapping

| Claim | 来源 | 边界 |
| --- | --- | --- |
| layer reconstruction objective与Hessian | §3式1/2；gptq.py add_batch | 不是完整任务loss的真实Hessian |
| 所有output rows共享input Hessian | §4 Step1 | 共享来自相同未量化列顺序，不是无需重算任何数据 |
| lazy block updates | §4 Step2、algorithm1、gptq.py L99–132 | blocksize不是quant group size |
| damp与Cholesky稳定 | §4 Step3；源码L91–97 | 不是所有矩阵天然正定，也非global optimum证明 |
| actorder=sort(Hdiag desc) | pinned gptq.py L83–87/144–145 | heuristic，含后续实践增强，不当作初版论文不可少条件 |
| group grids可按补偿后的weight更新 | §5 AdditionalTricks；源码groupsize/static_groups | 不同exporter/group策略可有差异 |
| 本体方法weight-only、speed来自traffic | §5 PracticalSpeedups、§6 | 不自动W4A4、不保证nativeINT4MMA |
| 浮点Q/DQ与真正packing不同 | gptq.py L148；quant.py Quant3Linear.pack | 同一个文件dtype无法证明runtime算术dtype |

## 已执行构造与建议审核

2026-10-15用Node标准库跑过以下2D构造（没有PyTorch或模型）：

- 取 `X=[[1,0.5],[0,sqrt(3)/2]]`，则 `H=2X^TX=[[2,1],[1,2]]`，`P=H^-1=[[2/3,-1/3],[-1/3,2/3]]`。
- 原始 `w=[0.49,0.49]`，演示grid `{-1,0,1}`（三值是教学方便，算法不依赖它，不冒称实际4bitGPTQ格式）。
- 先量化q=0到0，r=0.49，最优delta `[-0.49,+0.245]`，中间w `[0,0.735]`，layer SSE0.180075。
- 后一维再round到1，最终w_hat `[0,1]`，layer SSE0.2503。RTN `[0,0]` SSE0.7203。
- RTN weight MSE0.2401，补偿方案weight MSE0.2501（更差），但layer output误差更好，说明优化目标的区别，而非“每个weight都量化得更准”。
- 若只一个样本X=[1,1]，XTX rank1，直接inverse失败；加λI后可逆，但目标增加λ||delta||²项，必须说明不是悄悄改变理论假设。
- 建议auditor：finite difference确认XTX Hessian；枚举固定deltaq时连续最优；检查与2D Cholesky实现一致；随机SPD；zero/dead coordinate；negative残差；组尾/perm恢复；不保证全离散最优。

## 尚未执行

没有实际校准LLM、运行官方PyTorch GPTQ、打包并加载GPU artifact、执行真实kernel或复现论文PPL/延迟。文稿的玩具最优步骤只能支持该proxy推导，不能冒充模型质量验证或全部GPTQ实现等价证明。

## 定向补充：完整 Cholesky 怎样编码当前自由集的逆（2026-10-15）

### 1. 先区分两个不相等的对象

固定处理顺序0→…→K-1（若有permutation，先对H和w同步排列）。定义原始SPD矩阵H、完整逆 `P=H^-1`、upper Cholesky `P=U^T U`。已固定坐标I={0,…,q-1}，当前自由集F={q,…,K-1}。

当前自由集需要的是 `P_F=(H[F,F])^-1`，**不是完整逆的子矩阵 `P[F,F]`**。前者是其他坐标固定后，剩余变量的二次曲率之逆；后者包含那些已固定变量还可联动时的信息。

### 2. 从分块逆/Schur complement 推导

将P与U按I/F分块：

```
U = [ A  B ]       P = U^T U = [ A^T A       A^T B       ]
    [ 0  C ]                     [ B^T A       B^T B+C^T C ]
```

由分块逆公式，P关于I块的Schur complement满足：

```
P_F = (H[F,F])^-1
    = P[F,F] - P[F,I] (P[I,I])^-1 P[I,F]
    = B^T B + C^T C - B^T A (A^T A)^-1 A^T B
    = C^T C.
```

最后一步用A可逆，`A(A^TA)^-1A^T=I`。I为空时，直接有C=U、P_F=P。这里的Schur complement取在**P**上；不要混成H关于I的Schur complement（其逆对应完整P的F块，是另一对象）。

q是F的首个坐标，C的第一列只有首项Uqq非零，因此：

```
(P_F)[q,q] = U[q,q]^2
(P_F)[q,j] = U[q,q] * U[q,j],  j∈F.
```

将当前残差 `r=w_current[q]-Q(w_current[q])` 代入约束补偿：

```
delta[j] = -r * (P_F)[j,q] / (P_F)[q,q]
         = -(r/U[q,q]) * U[q,j],  j∈F.
```

所以一次性求完整P的upper Cholesky，随后逐行读U，等价于每一步重新求 `(H[F,F])^-1` 的当前行；并非把初始P的一整行不加处理地重复使用。尤其full `P[q,q]=sum_{i<=q} U[i,q]^2`，通常大于 `U[q,q]^2`，后者才是此时 `(P_F)[q,q]`。

此证明限定固定H、固定顺序、精确算术；damping时把H统一替换H+λI即可，但目标相应改变。数值误差、动态grid重估、groups/packing等不由这个等式自动证明。

### 3. 为什么当前增量目标没有一次项

初始w是未约束二次代理的最小点。每次约束一个坐标后，用上述补偿在剩余连续变量上重新达到最小点；归纳可得当前自由方向的梯度为0。因此下次只在F上改变δ时，目标增长恰是 `1/2 δ^T H[F,F] δ`，没有被遗漏的一次项。这是连续补偿的逐步最优，不是剩余变量还必须落在grid的离散最优。

## 定向补充：最终可以严格差于RTN的3维构造

取下面可逆X（列为输入特征），无需damping；用原始layer SSE作为统一比较目标：

```
X = [[1,1,1],
     [0,1,1],
     [0,0,1]]
H = 2 X^T X = [[2,2,2],[2,4,4],[2,4,6]]
w = [-0.4,-0.4,0.2]
grid = {-8,-7,...,7}   # 固定signed INT4格点，step1；全程不重估scale
Q(x) = clamp(roundTiesToEven(x), -8, 7)
order = 0,1,2
```

这是固定INT4网格下的教学反例，不运行实际GPTQ的自动min-max校准、默认damping或act-order，因此不声称复现其模型结果。本例关键决策处没有tie，但完整量化器仍明确采用nearest-even。

完整逆与upper factor为：

```
P = [[1,-1/2,0],[-1/2,1,-1/2],[0,-1/2,1/2]]
U = [[1,-1/2,0],
     [0,sqrt(3)/2,-1/sqrt(3)],
     [0,0,1/sqrt(6)]]
```

q=1时，当前 `P_F=[[3/4,-1/2],[-1/2,1/2]]`，首对角3/4=U11²，**不是**完整P11=1；q=2时P_F=[1/6]=U22²，而fullP22=1/2。

| 步骤 | 量化前当前坐标 | 选定code | 连续补偿后的完整w | 当前SSE |
| --- | ---: | ---: | --- | ---: |
| q0 | -0.4 | 0 | [0,-0.6,0.2] | 0.08 |
| q1 | -0.6 | -1 | [0,-1,7/15] | 0.1866666667 |
| q2 | 7/15 | 0 | [0,-1,0] | 0.84 |

最后一步前仍有连续自由度，较低中间SSE不是最终离散SSE的上界。逐步约束使当前最小SSE单调增加也不等于最终优于另一条约束路径。

- sequential compensation最终 `[0,-1,0]`，SSE0.84；
- RTN `[0,0,0]`，SSE0.44；
- 穷举16³=4096个网格向量，唯一全局最优 `[-1,0,0]`，SSE0.24。

因此“每一步在剩余连续坐标上最优”既不能推出整层离散最优，也不能推出任意样本最终SSE一定优于RTN。这个构造不是论文中的实测结论，是独立生成并执行验证的反例。

### 可直接运行的 Node/标准库复核

以下代码不用数值库：Gauss-Jordan独立求每个自由集的逆，与完整逆Cholesky的trailing block及逐步更新交叉检查；再穷举4096个离散候选。直接保存为`.js`或运行本文代码块即可；仅此research中的代码，不新增测试/生产文件。

```javascript
// GPTQ_SCHUR_COUNTEREXAMPLE_CHECK
const assert = (ok, msg) => { if (!ok) throw new Error(msg); };
const close = (a, b) => Math.abs(a - b) < 1e-11;
const transpose = A => A[0].map((_, j) => A.map(row => row[j]));
const matmul = (A, B) => A.map(row => B[0].map((_, j) =>
  row.reduce((s, v, k) => s + v * B[k][j], 0)));
const trailing = (A, q) => A.slice(q).map(row => row.slice(q));
function inverse(A) {
  const n = A.length;
  const B = A.map((row, i) => row.concat(
    Array.from({length: n}, (_, j) => Number(i === j))));
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++)
      if (Math.abs(B[i][k]) > Math.abs(B[p][k])) p = i;
    [B[k], B[p]] = [B[p], B[k]];
    assert(Math.abs(B[k][k]) > 1e-12, "singular matrix");
    const d = B[k][k];
    B[k] = B[k].map(v => v / d);
    for (let i = 0; i < n; i++) if (i !== k) {
      const c = B[i][k];
      B[i] = B[i].map((v, j) => v - c * B[k][j]);
    }
  }
  return B.map(row => row.slice(n));
}
function upperCholesky(A) {
  const n = A.length;
  const L = Array.from({length: n}, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j];
    for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    if (i === j) assert(s > 0, "not SPD");
    L[i][j] = i === j ? Math.sqrt(s) : s / L[j][j];
  }
  return transpose(L);
}
function nearestEven(x) {
  const f = Math.floor(x), d = x - f;
  return d < 0.5 ? f : d > 0.5 ? f + 1 : f % 2 === 0 ? f : f + 1;
}
const quantize = x => Math.max(-8, Math.min(7, nearestEven(x)));
const X = [[1,1,1],[0,1,1],[0,0,1]], w = [-0.4,-0.4,0.2];
const H = matmul(transpose(X), X).map(row => row.map(v => 2 * v));
const P = inverse(H), U = upperCholesky(P);
const sse = candidate => X.reduce((s, row) => s +
  row.reduce((t, x, j) => t + x * (candidate[j] - w[j]), 0) ** 2, 0);
const currentRaw = w.slice(), currentChol = w.slice(), trace = [];
for (let q = 0; q < w.length; q++) {
  const PF = inverse(trailing(H, q)), C = trailing(U, q);
  const gramC = matmul(transpose(C), C);
  for (let i = 0; i < PF.length; i++) for (let j = 0; j < PF.length; j++)
    assert(close(PF[i][j], gramC[i][j]), "Schur/trailing factor mismatch");
  assert(close(PF[0][0], U[q][q] ** 2), "current inverse diagonal mismatch");
  const codeRaw = quantize(currentRaw[q]), codeChol = quantize(currentChol[q]);
  assert(codeRaw === codeChol, "quantized decisions differ");
  const r = currentRaw[q] - codeRaw;
  const err = (currentChol[q] - codeChol) / U[q][q];
  for (let j = q; j < w.length; j++) {
    currentRaw[j] -= r * PF[j - q][0] / PF[0][0];
    currentChol[j] -= err * U[q][j];
    assert(close(currentRaw[j], currentChol[j]), "compensation mismatch");
  }
  trace.push({q, code: codeRaw, w: currentRaw.slice(), sse: sse(currentRaw),
    currentInverseDiagonal: PF[0][0], fullInverseDiagonal: P[q][q]});
}
const rtn = w.map(quantize);
let best = Infinity, bestVectors = [], count = 0;
for (let a = -8; a <= 7; a++) for (let b = -8; b <= 7; b++)
  for (let c = -8; c <= 7; c++) {
    const candidate = [a,b,c], loss = sse(candidate);
    count++;
    if (loss < best - 1e-11) { best = loss; bestVectors = [candidate]; }
    else if (close(loss, best)) bestVectors.push(candidate);
  }
assert(count === 4096, "enumeration incomplete");
assert(close(sse(currentRaw), 0.84) && close(sse(rtn), 0.44), "wrong SSE");
assert(close(best, 0.24) && bestVectors.length === 1 &&
  bestVectors[0].join(",") === "-1,0,0", "wrong discrete optimum");
assert([-1.5,-0.5,0.5,1.5,2.5].map(nearestEven).join(",") === "-2,0,0,2,2",
  "tie rule mismatch");
console.log(JSON.stringify({H, P, U, trace, sequential: currentRaw,
  sequentialSSE: sse(currentRaw), rtn, rtnSSE: sse(rtn), count, best, bestVectors}, null, 2));
```

### 补充来源映射与实际执行范围

- GPTQ v2 §3 Eq2支撑固定自由集的连续约束补偿，Eq3/§4 Eq5是inverse downdate/Schur结构；§4 Step3及Algorithm1支撑upper Cholesky行、`err=r/Uqq`与行更新。
- 官方固定commit `2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py` 本次重读实际L98–104（damping→Cholesky→inverse→upper Cholesky）、L114–137（d、d²损失、err、补偿）。此前research表中的简写行号并非本次逐行位置，应以本次读到的固定源码为准。
- `P_F=C^TC`的分块证明与该3D反例为独立推导/构造，不冒称论文提供此例或证明实际模型效果。
- 执行范围：已从本Markdown直接提取上面的完整JS代码并用Node成功运行（exit0），所有Schur/行更新/4096点/nearest-even断言通过；输出SSE依次0.08000000000000002、0.18666666666666665、0.8400000000000001，RTN0.44000000000000017，best0.24、唯一[-1,0,0]。无PyTorch、GPU、LLM、官方exporter；未开启act-order、grid重估、分组、damping。浮点尾差不解释成理论不等价。
