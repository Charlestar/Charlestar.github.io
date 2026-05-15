# ZZC's Blog

> AI Infra 工程师的技术博客

🔗 [https://charlestar.github.io](https://charlestar.github.io)

## 关于

专注于大模型推理系统优化的技术博客，主要分享以下内容：

- **AI Infra**：SGLang、vLLM、FlashInfer、TensorRT-LLM 等推理引擎深度解析
- **LLM 推理优化**：PagedAttention、推测解码、MoE 推理、KV Cache 优化
- **AI Agent**：LangGraph、MCP 等多 Agent 系统实战
- **计算机视觉**：视频生成、光流法、图像处理
- **推荐算法**：召回、排序、特征交叉、序列建模

## 近期文章

| 日期 | 标题 |
|------|------|
| 2026-05-15 | 推理模型推测解码完全指南：Thinking Budget 机制与 EAGLE-3 实战 |
| 2026-05-14 | LangGraph + MCP 实战：构建生产级多 Agent 系统的完整指南 |
| 2026-05-13 | MoE 推理优化全景指南：从架构原理到 GPU 部署实践 |
| 2026-05-13 | FlashInfer 深度解析：从 JIT 编译到 AI 生成 Kernel 的 LLM 推理加速革命 |
| 2026-05-12 | SGLang 与 RadixAttention 详解——大模型推理服务的 KV Cache 复用革命 |
| 2026-05-12 | 推测解码（Speculative Decoding）原理与实战 |
| 2026-05-11 | 2026 大模型推理引擎全景对比 |

## 技术栈

- **框架**：[Jekyll](https://jekyllrb.com/) + [GitHub Pages](https://pages.github.com/)
- **主题**：基于 [Hux Blog](https://github.com/Huxpro/huxpro.github.io) 深度定制
- **评论**：[Giscus](https://giscus.app/)（基于 GitHub Discussions）
- **数学公式**：MathJax 3.2+
- **代码高亮**：Rouge

## 主题特性

- 🌙 **深色模式**：自动跟随系统，支持手动切换
- 📑 **目录导航**：桌面端侧边栏 + 移动端折叠 TOC
- 🔍 **站内搜索**：Lunr.js 全文搜索
- 📱 **响应式设计**：完美适配手机/平板/桌面
- 🔗 **社交分享**：Twitter / 微信 / GitHub
- 📊 **阅读进度条**：顶部固定进度指示
- 💬 **Giscus 评论**：Markdown 支持，GitHub 账号登录
- 🏷️ **标签系统**：Featured Tags + 标签云

## 本地开发

```bash
# 安装依赖
bundle install

# 本地预览
jekyll serve

# 访问 http://127.0.0.1:4000
```

## 撰写新文章

在 `_posts/` 目录下创建 Markdown 文件，文件名格式为 `YYYY-MM-DD-title.md`：

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

> ⚠️ `header-img` 必须使用以 `/` 开头的绝对路径，否则分页页面会出现图片加载失败。

## License

遵循 MIT 许可证。

---

Built with ❤️ by [iStar](https://github.com/Charlestar)
