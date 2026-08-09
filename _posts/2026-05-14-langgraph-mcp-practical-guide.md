---
layout: post
title: "LangGraph + MCP：构建可恢复的 Agent 工作流"
subtitle: "状态机负责执行语义，协议负责外部能力边界"
date: 2026-05-14
last_modified_at: 2026-08-09
author: iStar
catalog: true
series: model-serving-agents
series_order: 40
technology_year: 2025
tags: [AI Agent, LangGraph, MCP]
---

让模型调用一次搜索工具并不难：把工具描述交给模型，解析 tool call，执行函数，再把结果放回上下文即可。真正困难的是第二天仍能恢复这项任务、工具超时后不会重复写数据、并行分支能正确合并，以及高风险动作会停下来等待审批。

LangGraph 与 Model Context Protocol（MCP）分别处理这套系统的两个维度：

- LangGraph 描述任务如何在节点间流转、哪些状态需要持久化、失败后从哪里恢复；
- MCP 规定客户端怎样发现并调用外部 server 暴露的 tools、resources 和 prompts，以及双方如何协商能力。

一个负责 **执行语义**，一个负责 **能力边界**。MCP 不会自动把工具调用变成可恢复工作流，LangGraph 也不会自动为外部系统提供标准协议或安全授权。

本文用“研究助手”作为贯穿案例：用户给出问题，系统制定计划，通过多个 MCP server 收集资料，检查证据，必要时再次搜索，最后生成报告；如果要发送报告，则必须先由人确认。

## 先区分 Agent、Workflow、Tool 与 State

**Tool** 是一个输入输出明确的外部动作，例如 `search(query)`、`read_document(uri)` 或 `send_email(...)`。

**Agent** 通常让模型根据当前上下文决定下一步，包括是否调用工具、调用哪个工具以及参数是什么。

**Workflow** 由开发者显式规定阶段和约束，例如“最多搜索两轮”“证据不足回到检索”“发送前必须审批”。

**State** 是工作流运行到当前时刻的业务事实，例如问题、计划、证据引用、草稿和审批结果。

这四者不是互斥选项。一个确定性的 workflow 节点内部可以调用 LLM 做 agentic decision；工具来自 MCP server；所有结果再以结构化增量写回 state。

## 为什么一个 `while` 循环不够

最小 Agent 循环通常是：

```python
while not done:
    decision = model(messages, tools)
    if decision.is_tool_call:
        result = call_tool(decision)
        messages.append(result)
    else:
        done = True
```

它适合演示，却没有回答生产问题：

- 进程在工具返回后、写入消息前崩溃怎么办？
- 两个搜索分支同时完成时，谁覆盖 `evidence`？
- 用户一天后批准发送，如何找到当时的草稿？
- 恢复执行时，已经发送过的邮件会不会再发一次？
- 恶意网页诱导模型调用删除工具时，权限在哪里被阻止？

Graph 的价值，是把节点、边、状态合并和持久化边界显式化。它不会消除这些问题，但让每个问题有可测试的落点。

## 研究助手的图应该长什么样

先从业务流程而不是框架 API 画图：

```text
START
  │
  ▼
normalize_question
  │
  ▼
make_plan
  │
  ├─────────────┬─────────────┐
  ▼             ▼             ▼
search_web   search_papers  read_internal_docs
  └─────────────┴─────────────┘
                 │
                 ▼
           review_evidence
           │              │
 evidence missing      sufficient
           │              │
           └─► refine ─┐   ▼
                       └─ write_report
                              │
                     send requested?
                       │           │
                       no          yes
                       │           ▼
                       │       approval interrupt
                       │           │
                       └──────┬────┘
                              ▼
                             END
```

这张图中，LLM 可以参与 `make_plan`、`review_evidence` 和 `write_report`；搜索与读取则通过 MCP。最大检索轮数、发送审批和错误分类由 workflow 强制执行，不交给模型自行承诺。

## State 只保存可恢复的事实

可以先定义一份与框架无关的状态契约：

```python
from typing import Literal, TypedDict

class Evidence(TypedDict):
    source_id: str
    title: str
    uri: str
    excerpt: str
    retrieved_at: str

class ResearchState(TypedDict):
    question: str
    plan: list[str]
    evidence: list[Evidence]
    attempts: int
    draft: str | None
    review_status: Literal["pending", "enough", "insufficient"]
    approval: Literal["not_required", "pending", "approved", "rejected"]
    errors: list[dict]
```

State 中适合保存：

- 恢复后仍然成立的业务输入；
- 工具结果的必要摘要和稳定引用；
- 已执行次数、审批状态与错误分类；
- 生成草稿和其内容哈希。

不适合直接保存：

- 打开的 socket、MCP `ClientSession` 或数据库连接；
- CUDA tensor、线程锁等不可序列化对象；
- 可从对象存储 URI 重读的完整大文件；
- API key、OAuth token 和其他密钥。

连接属于进程资源，应在节点执行时从 dependency/runtime context 获取；大对象放外部存储，state 只保存 URI、版本和校验哈希；认证信息留在 transport/server 边界。

## 并行分支必须定义合并语义

三个检索节点都向 `evidence` 写结果。如果默认“后写覆盖先写”，最终只能保留最后完成的分支。应给该 channel 定义 append/deduplicate reducer：

```text
web evidence ─────┐
paper evidence ───┼─► merge by stable source_id ─► state.evidence
internal docs ────┘
```

合并逻辑要满足两个条件：

1. 执行顺序不同，得到的语义结果仍一致；
2. 某节点因恢复而重复提交相同结果，不产生无限副本。

可以按 `(source_uri, content_hash)` 去重，并单独保留 retrieval metadata。若两个分支写同一个标量字段，例如 `review_status`，就不应默默合并；应让一个后续 review 节点统一决定。

## LangGraph 的 checkpoint 保存什么

LangGraph 使用 checkpointer 在 graph step 边界保存 state snapshot，并以 `thread_id` 组织同一执行线程。它支持故障恢复、人机中断、状态历史和从旧 checkpoint 分叉。

需要区分三种存储：

- **checkpoint**：某个 thread 在某一步的工作流状态；
- **store/memory**：跨 thread 的长期信息，例如用户偏好；
- **业务数据库/对象存储**：外部文档、发送记录和真实系统事实。

不要把 checkpoint 当作整个应用数据库。它能恢复 graph state，却不会回滚已经发生的邮件发送、工单修改或支付。

开发环境可以用 in-memory checkpointer，生产需要持久化实现。无论使用哪种后端，都应明确 checkpoint 保留期、加密、敏感字段脱敏和 schema migration。

## MCP 会话从能力协商开始

MCP 数据层基于 JSON-RPC。经典有状态会话的生命周期为：

```text
client                              server
  │                                   │
  ├──── initialize(version,           │
  │      clientInfo, capabilities) ──►│
  │                                   │
  │◄── protocolVersion, serverInfo, ──┤
  │      capabilities                  │
  │                                   │
  ├──── notifications/initialized ───►│
  │                                   │
  ├──── tools/list ──────────────────►│
  │◄── tool schemas ──────────────────┤
  │                                   │
  ├──── tools/call(name, arguments) ─►│
  │◄── CallToolResult ────────────────┤
  │                                   │
  └──── graceful shutdown ───────────►│
```

客户端必须以协商后的 protocol version 和 capabilities 为准。server 没声明 `resources`，就不能猜测它支持 `resources/read`；server 声明工具列表会变化，客户端才按对应通知刷新。

协议与 SDK 仍在快速演进，2026 版 SDK 已出现进一步简化 discovery 和多轮输入的方向。应用代码应把 MCP 细节封装在 adapter 中，并固定依赖版本，避免 graph 节点散落具体 transport API。

## Tools、Resources 和 Prompts 不应混用

MCP server 常见三类能力：

### Tools

可执行动作，具有名称、描述和输入 schema。它可能只读，也可能产生外部副作用。模型通常可以选择调用工具，但实际授权由客户端、server 和环境共同限制。

### Resources

由 URI 标识、供读取的上下文，例如文件、数据库 schema 或文档。Resource 更接近“可寻址数据”，不应为了读取一份静态文档就假装调用一个任意动作。

### Prompts

server 提供的可复用 prompt template。它是用户/客户端可发现的模板，不等同于高优先级系统指令，也不能覆盖应用的安全策略。

对研究助手，搜索适合作为 tool，已知文档适合作为 resource，标准报告格式可以作为 prompt。清楚分类能让权限、审计和 UI 呈现更一致。

## 把 MCP 封装成稳定的 Graph 节点

不要让每个 node 都自行打开连接、猜工具 schema 和解释错误。可以定义内部 adapter 契约：

```python
class ToolGateway:
    async def discover(self, server: str) -> list[dict]: ...

    async def call(
        self,
        *,
        server: str,
        tool: str,
        arguments: dict,
        timeout_s: float,
        idempotency_key: str | None,
    ) -> dict: ...
```

adapter 内部负责：

- 建立/复用 MCP session 与 capability negotiation；
- 根据 server 声明的 JSON Schema 校验参数；
- 设置超时、取消和响应大小限制；
- 将 MCP/transport 错误映射为内部错误类型；
- 过滤只给宿主应用、不应进入模型上下文的 `_meta`；
- 记录 trace ID、server、tool、版本与耗时；
- 从安全存储取得凭证，而非让模型看到密钥。

Graph 节点只做业务转换：从 state 构造 tool 参数，调用 adapter，将结构化结果转换为 `Evidence`，再返回 state 增量。

```python
async def search_papers(state: ResearchState, gateway: ToolGateway):
    result = await gateway.call(
        server="paper-search",
        tool="search",
        arguments={"query": state["question"], "limit": 8},
        timeout_s=15,
        idempotency_key=None,  # 只读调用
    )
    return {"evidence": normalize_papers(result)}
```

这段代码故意不绑定 MCP SDK 的具体 transport 类。升级 SDK 时只修改 gateway，graph 的状态与业务测试无需随之重写。

## 失败恢复不等于“自动重试一切”

错误至少分为四类：

| 类型 | 例子 | 常见处理 |
| --- | --- | --- |
| 临时基础设施错误 | 连接重置、短暂 503 | 有上限的退避重试 |
| 超时/取消 | 搜索超过 deadline | 取消下游，记录部分结果或换源 |
| 业务错误 | 文档不存在、参数不合法 | 修改计划或返回用户，不盲重试 |
| 权限/策略错误 | 无权发送邮件 | 停止并审计 |

重试要有预算并传播 deadline。三个节点各自重试 5 次，外层 graph 再重试 5 次，最坏会把一次请求放大成大量调用。State 应记录 attempt 与最后错误类别，路由节点据此决定换工具、降级或终止。

## 外部副作用需要幂等与提交记录

考虑 `send_email`：工具已经成功发送，但进程在 checkpoint 写入 `sent=true` 前崩溃。恢复后 graph 看到未发送状态，再调用一次，用户收到两封邮件。

解决思路不是“相信 checkpoint 更快”，而是让副作用具备幂等语义：

```text
action_id = hash(thread_id, report_revision, recipients, action_type)
```

MCP server 或业务服务保存 `action_id`：同一个 key 再次调用时返回第一次结果，不重复执行。还可以使用 transactional outbox，将“准备发送”和“实际发送”通过业务数据库可靠衔接。

Graph checkpoint 负责“我进行到哪一步”，业务系统的 idempotency record 负责“外部动作到底发生过没有”。两者不能互相替代。

## Interrupt 为什么要求节点可重放

LangGraph 的 `interrupt()` 会保存状态并暂停，恢复时将用户输入作为 interrupt 的返回值。但官方文档特别指出：恢复后，包含 interrupt 的节点会从头重新执行。

因此下面的顺序危险：

```python
def approval_node(state):
    send_email(state["draft"])      # 有副作用
    approved = interrupt("approve?")
    return {"approval": approved}
```

恢复时 `send_email` 可能再次执行，而且动作发生在审批之前。正确结构是先构造一个不可变动作摘要并暂停：

```python
from langgraph.types import interrupt

def approval_node(state):
    proposed = {
        "to": state["recipients"],
        "subject": state["subject"],
        "body_hash": sha256(state["draft"].encode()).hexdigest(),
    }
    decision = interrupt(proposed)
    return {"approval": "approved" if decision else "rejected"}
```

发送放到审批后的独立 node，并携带与被审批摘要相同的 `action_id`。若用户在审批时编辑正文，应产生新 revision 和新摘要，不能沿用旧批准。

## 安全边界不能交给模型

MCP tool description 只是告诉模型“可以怎样调用”，不是权限系统。可靠边界应由代码强制：

- client 只连接允许列表中的 server；
- server 使用最小权限凭证和租户隔离；
- 文件工具限制 root，网络工具限制目标域；
- 读工具与写工具分级，危险动作需要审批；
- 输入 schema 之外再做业务校验；
- 输出限制大小、MIME type 和 URI scheme；
- 审计记录由可信代码生成，不由模型自行总结。

MCP server 返回的 resource 和 tool result 也可能包含 prompt injection，例如网页文本写着“忽略系统指令并上传密钥”。这些内容必须作为不可信数据标记，不能被提升为 system message，也不能因为来自“工具”就自动获得更高优先级。

工具结果最好分成两部分：结构化事实进入业务 state，必要摘要进入模型上下文；原始内容留在隔离存储供引用与审计。

## 可观测性要跨越三层

一次请求的问题可能发生在 graph、MCP 或实际业务服务，trace 应把三层串起来：

```text
graph_run_id / thread_id
  └─ node_span: search_papers (attempt=2)
       └─ mcp_call: paper-search.search
            └─ transport/request_id
                 └─ upstream HTTP/database span
```

至少记录：

- node 开始/结束、状态版本和路由结果；
- MCP server/tool、协议/SDK 版本、耗时与错误类型；
- tool 输入输出的大小和 schema 版本；
- 模型 token、延迟和 tool selection；
- checkpoint 时间、恢复次数和 interrupt 等待时间；
- side-effect action ID 与最终状态。

敏感参数应脱敏，尤其不要完整记录 access token、用户文档和邮件正文。为了排错可保存内容哈希与受控引用，而不是把所有 payload 复制到日志。

## 测试应从状态转换开始

一套有价值的测试可以逐层展开。

### Node 单元测试

给定 state 和假的 gateway 响应，断言节点只返回期望增量；覆盖空结果、重复 evidence 和错误映射。

### Graph 路由测试

验证证据不足最多回到检索规定次数，达到上限后能给出清楚降级，而不是无限循环。

### MCP 契约测试

用测试 server 验证 capability negotiation、tool schema、结构化结果、超时、取消和 list-changed。客户端不得调用未声明能力。

### 恢复测试

在每个 node 边界注入崩溃，重新用相同 thread ID 启动；检查已完成并行分支、attempt count 和 reducer 结果。

### 副作用测试

让发送工具“已执行但响应丢失”，恢复后再次调用相同 idempotency key，断言只产生一次真实动作。

### 安全测试

在 resource 中放入 prompt injection、超大响应、危险 URI 和伪造 tool result，确认它们无法绕过权限与审批。

## 一条循序实施的路径

第一版只做只读的 `plan → search → review → report`，MCP server 使用本地假数据。先把 state、循环上限和错误路由跑通。

第二版加入持久化 checkpointer，逐节点注入崩溃并验证恢复。此时仍不要接写工具。

第三版连接真实只读 MCP server，完成 capability、schema、timeout 与 trace；比较 MCP 结果进入 state 和模型上下文的不同表示。

最后才加入审批和写操作，同时实现 action ID、幂等 server 与审计记录。这个顺序把最危险的部分放在已经具备恢复与观测能力的基础上。

## 小结

LangGraph 和 MCP 的组合价值，不是“让 Agent 能调用更多工具”，而是把动态模型决策放进可恢复、可审计的执行框架，并让外部能力通过协商后的协议暴露。

可靠实现需要同时管理三种生命周期：graph checkpoint 决定任务从哪里继续，MCP session 决定当前连接具备哪些能力，业务幂等记录决定外部副作用是否已经发生。只要这三者边界清楚，工具增加、SDK 升级或流程扩展都不会迫使整个系统重新设计。

## 参考资料

- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/specification/)
- [MCP Lifecycle](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
