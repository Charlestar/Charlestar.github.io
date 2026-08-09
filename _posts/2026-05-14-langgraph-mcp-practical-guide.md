---
layout: post
title: "LangGraph + MCP：构建可恢复的 Agent 工作流"
subtitle: "状态机负责流程，协议负责工具边界"
date: 2026-05-14
last_modified_at: 2026-08-09
author: iStar
catalog: true
tags: [AI Agent, LangGraph, MCP]
---

> **校订说明**：原文基于早期 SDK 写法，混合了伪 API、未实现的 MCP 调用和自制基准。本文改为稳定的架构原则；具体代码应以当前 LangGraph 与 MCP SDK 文档为准。

## 1. 两者职责不同

- **LangGraph**：描述有状态工作流，包括节点、边、条件路由、持久化与恢复。
- **MCP**：标准化模型应用如何发现并调用 tools、读取 resources 和获取 prompts。

MCP 不负责多 Agent 编排，LangGraph 也不会自动让任意外部系统成为安全工具。常见组合是：工作流节点通过 MCP client 调用一个或多个 server，并把结构化结果写回 graph state。

```text
request -> LangGraph state
              |
              +-> planner node
              +-> MCP client -> server tools/resources
              +-> review node
              +-> final node
```

## 2. 状态设计

状态应保存可恢复的业务事实，例如消息、任务计划、工具结果引用和审批状态；不要把不可序列化的网络连接或完整大文件直接塞入 state。并行分支写同一字段时，需要显式 reducer 或合并策略。

## 3. MCP 调用边界

- 启动时协商 protocol version 和 capabilities；
- 只调用 server 实际声明的 tool，不猜测名称或参数；
- 对 tool 输入做 schema 校验，对输出设置大小和超时限制；
- 把认证信息留在 server/transport 边界，不写入 prompt 或持久化 state；
- 对有副作用的 tool 加审批、幂等键和审计日志。

## 4. 可靠性

重试要区分临时错误与业务错误。对写操作盲目重试可能产生重复副作用。checkpoint 能恢复 graph 状态，但不能自动回滚外部系统；需要幂等设计或补偿动作。

## 5. 最小验证清单

1. 单元测试每个节点的纯状态转换。
2. 用假的 MCP transport 测 schema、超时和错误映射。
3. 测进程中断后的 checkpoint 恢复。
4. 测并行分支合并与重复 tool response。
5. 在部署前固定 SDK 版本并按官方迁移指南更新示例。

## 6. 一个最小状态机

下面是概念结构，不绑定某个 SDK 小版本：

```python
class State(TypedDict):
    messages: list
    plan: list[str]
    tool_results: list[dict]
    status: str

def plan_node(state): ...
def tool_node(state): ...      # 通过 MCP client 调用已发现的 tool
def review_node(state): ...

# START -> plan -> tool -> review
# review -> tool   (需要补充证据)
# review -> END    (结果已满足要求)
```

节点尽量返回状态增量而非原地修改全局对象，这更容易测试、持久化和重放。

## 7. MCP 初始化流程

```text
open transport
 -> initialize(version, capabilities, client info)
 <- server capabilities
 -> initialized notification
 -> tools/list or resources/list
 -> tools/call / resources/read
```

Client 应以协商结果为准。例如 server 没声明 resources，就不应猜测 `resources/read` 可用。工具 schema 也应在调用前缓存并校验。

## 8. Human-in-the-loop

对发送邮件、修改工单、执行部署等有副作用动作，可以在 graph 中插入 interrupt/approval 节点。checkpoint 保存待审批状态，恢复时携带批准/拒绝结果。审批应绑定动作摘要和幂等键，防止恢复后执行了不同参数。

## 9. 安全模型

MCP server 返回的数据也可能包含不可信文本。不要把 tool output 视为系统指令；限制可访问资源、执行目录、网络目标和响应大小。Agent 的“会调用工具”不是授权模型，真正权限应由 server 和运行环境强制执行。

## 10. 可观测性

为每个 graph run、node 和 MCP call 记录 trace ID、耗时、重试、输入/输出大小与错误分类。敏感参数应脱敏。性能测试最好区分模型时间、graph orchestration 时间和外部 tool 时间。

## 参考资料

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Model Context Protocol specification](https://modelcontextprotocol.io/specification/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
