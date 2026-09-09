# RoPE 新文独立全文审核

- 审核日期：2026-09-09。
- 对象：`_posts/2026-09-09-rope-relative-position-context-extension.md`。
- 角色：research → writer 交接后由独立 auditor 全文审读并补充；不是只查看差异。

## 实际覆盖

已完整阅读正文所有章节：二维旋转、多频率与 partial RoPE、布局置换、外推边界、PI、base change、YaRN、配置契约、动态 KV 状态、三层验证和结语。逐式检查列向量约定、转置、相对位移符号、维度、量纲、表格数值、就近引用与文章内部链接。

独立打开并核对的关键原始材料：

- [RoFormer v5 §3.2、§3.3、§3.4.1](https://arxiv.org/html/2104.09864v5)：二维/分块旋转、相对位置式、范数与距离性质的适用范围。
- [PI v2 §2.3 与 §3.1–3.2](https://arxiv.org/html/2306.15595v2#S2.SS3)：`m L/L'` 映射、继续训练与质量证据的区别。
- [YaRN v3 §2.2–2.3、§3.1–3.4、附录 A.1–A.2](https://arxiv.org/html/2309.00071v3)：ramp 方向、温度与幅度关系、静态 base change、动态缓存说明。
- [Transformers v4.57.1 Llama 源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py)：`LlamaRotaryEmbedding.forward`、`rotate_half`、`apply_rotary_pos_emb`、attention projection → RoPE → cache 更新顺序、head 维度缩放。
- [同版本 RoPE 工具源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_rope_utils.py)：linear 频率除 factor、dynamic 更新、YaRN 维度区间 ramp。

以上是实际核对的关键章节/函数，不代表精读了各篇论文的全部实验与附录，也没有把研究记录当成独立验证的替代品。

## 发现、修正与保留的边界

1. 独立确认 YaRN v3 正文 Eq. 9 的 `g(m)=s*m` 与同文 `s=L'/L`、附录 Eq. 16 及 PI 原论文不一致。新文已采用 PI 原文的 `m/s`，未沿用该不一致记号；不需要为此强行修改原已正确的推导。
2. 检查二维交叉项为 `(q2*k1-q1*k2) sin((n-m)ω)`，文中算例为 2；partial RoPE 的标准分母保持 `sqrt(d_h)`，而非 `sqrt(d_r)`。
3. 复算 `b'=160000`、两张频谱表、YaRN `a≈1.207944` 与 logits 倍率 `a²≈1.45913`；保留只旋转部分维度时不能将完整 score 一概乘 `a²` 的限定。
4. 动态缓存反例明确固定内容向量的角度不一致；保留“仅修正历史 K 旋转不能一般性复原深层 hidden states”的限定，未把论文对 pre-RoPE 缓存的建议误写成完整网络等价性证明。
5. 实际补充正文中的配套复算运行命令和验证层级边界，并将虚构的 `09:00` 发文时间改为当日日期。
6. 初次脚本测试中的缓存反例向量恰好产生较小差异，未满足选定阈值；改用单个有效平面的明确反例后重跑通过。此为测试样本修正，不是隐去失败或宣称数值容差普适。

## 已运行验证

执行 `node scripts/reviews/new-rope-examples.mjs`，9 组标准库检查通过：

- 旋转保范数、相对位移符号、共同平移。
- 正文二维算例与投影/旋转不交换。
- 两张频谱表、波长、圈数与 base 数值。
- interleaved/split-half 一致置换与错误替换反例。
- PI 连续位置映射、最高频也被缩放。
- base change 非统一缩放。
- 任意 score 不保证单调距离衰减。
- YaRN 幅度平方、partial scaling 和 ramp 端点。
- 新旧频率混用缓存反例及固定内容的重旋转恒等式。

## 尚未实测

未加载真实模型，未执行 PyTorch/Transformers、GPU kernel、prefill/decode/分块路径对照、动态缓存实现或长上下文质量评测。当前结论是已覆盖的推导与小例未发现未修正问题，不是绝对正确性证明。

## 主审发布前验证

主审另行通读正文，并在 auditor 停止编辑后执行以下检查：

- `audit-posts.mjs`、历史范围 `check-coverage.mjs`、全部 10 个 JavaScript 示例套件及 `full-agent-examples.py` 通过。历史审查仍只声明原有 77 篇的范围，新文由本记录单独覆盖。
- 本地生产构建成功；当时工作区含 79 篇（包括另一篇待发布新文），页面检查通过 92 个 HTML / 117 个生成文件，公式保留检查通过 269 个原生行内式与 804 个展示式。
- 在真实浏览器打开 RoPE 生成页面：90 个 MathJax 公式容器，无 `merror` 或 `data-mjx-error`；目视核对二维旋转矩阵、频率对比表及相邻正文，未见排版异常。
- 这些检查不扩展为真实模型质量、PyTorch API 执行或 GPU 性能的验证结论。
