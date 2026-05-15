# ZZC's Blog

> AI Infra 工程师的技术博客 — 专注大模型推理系统优化

🌐 [https://charlestar.github.io](https://charlestar.github.io)

## 📝 关于

这里是我的个人技术博客，主要分享以下方向：

- **LLM 推理引擎**：SGLang、vLLM、TensorRT-LLM 架构解析与性能优化
- **Kernel 优化**：FlashAttention、FlashInfer、CUDA Kernel 编写
- **推理加速**：推测解码（Speculative Decoding）、MoE 推理、PagedAttention
- **AI Agent**：LangGraph + MCP 多 Agent 系统实战
- **推荐算法**：召回、排序、特征交叉、行为序列建模

## 📚 近期文章

| 日期 | 文章 |
|------|------|
| 2026-05-15 | [推理模型推测解码完全指南：Thinking Budget 机制与 EAGLE-3 实战](https://charlestar.github.io/2026/05/15/speculative-decoding-thinking-budget-eagle3/) |
| 2026-05-14 | [LangGraph + MCP 实战：构建生产级多 Agent 系统的完整指南](https://charlestar.github.io/2026/05/14/langgraph-mcp-practical-guide/) |
| 2026-05-13 | [MoE 推理优化全景指南：从架构原理到 GPU 部署实践](https://charlestar.github.io/2026/05/13/MoE推理优化全景指南/) |
| 2026-05-13 | [FlashInfer 深度解析：从 JIT 编译到 AI 生成 Kernel 的 LLM 推理加速革命](https://charlestar.github.io/2026/05/13/flashinfer-deep-dive/) |
| 2026-05-12 | [SGLang 与 RadixAttention 详解](https://charlestar.github.io/2026/05/12/SGLang与RadixAttention详解/) |
| 2026-05-12 | [推测解码（Speculative Decoding）原理与实战](https://charlestar.github.io/2026/05/12/推测解码SpeculativeDecoding原理与实践/) |
| 2026-05-11 | [2026 大模型推理引擎全景对比](https://charlestar.github.io/2026/05/11/2026大模型推理引擎全景对比/) |

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 框架 | [Jekyll](https://jekyllrb.com/) + [GitHub Pages](https://pages.github.com/) |
| 主题 | 基于 [Hux Blog](https://github.com/Huxpro/huxpro.github.io) 深度定制 |
| 评论 | [Giscus](https://giscus.app/)（GitHub Discussions） |
| 数学公式 | MathJax 3.2+ |
| 代码高亮 | Rouge |

## ✨ 主题特性

- 🌙 **深色模式** — 自动跟随系统 + 手动切换
- 📑 **目录导航** — 桌面端侧边栏 + 移动端折叠 TOC
- 🔍 **站内搜索** — Lunr.js 全文搜索
- 📊 **阅读进度条** — 顶部固定进度指示
- 🏷️ **标签系统** — Featured Tags + 标签云
- 🔗 **社交分享** — Twitter / 微信 / GitHub
- 💬 **Giscus 评论** — Markdown 支持，GitHub 账号登录
- 📱 **响应式设计** — 完美适配手机 / 平板 / 桌面

## 🚀 本地开发

```bash
# 安装依赖
bundle install

# 本地预览
jekyll serve

# 访问 http://127.0.0.1:4000
```

## ✍️ 撰写新文章

在 `_posts/` 目录下创建 Markdown 文件，命名格式：`YYYY-MM-DD-title.md`

```yaml
---
layout: post
title: 文章标题
subtitle: 副标题（可选）
date: 2026-05-15
author: iStar
header-img: /img/post-bg-ai-infra.jpg
catalog: true
mathjax: true
tags:
  - AI Infra
  - SGLang
  - vLLM
---
```

> ⚠️ **注意**：`header-img` 必须使用以 `/` 开头的**绝对路径**（如 `/img/xxx.jpg`），否则分页页面图片会加载失败。

## 📄 License

MIT License
