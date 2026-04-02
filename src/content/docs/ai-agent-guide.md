---
title: "AI Agent 开发入门"
description: "从 Claude Code 源码中提炼的 AI Agent 工程方法论"
date: "2026-04-02"
category: ai-agent
tags:
  - AI
  - Agent
  - Architecture
order: 1
---

## 什么是一个 AI Agent Harness？

很多人以为 AI Agent 就是一个能"对话+调用工具"的聊天机器人。但如果你深入研究过 Claude Code、Codex 这些产品的架构，你会发现一个根本性的区别：

**Chatbot 是"问答机器"，Agent 是"执行引擎"。**

一个 Chatbot 接收用户输入，生成文本回复，结束。一个 Agent 接收用户输入后，会进入一个**循环**——它思考下一步该做什么，调用工具，获取结果，再思考，再调用，直到任务完成。

这个循环就是 **Agentic Loop**，它是所有 Agent 系统的核心。

Claude Code 之所以强大，不是因为它用的模型有多聪明，而是因为它围绕这个循环构建了一整套基础设施：工具系统、权限控制、上下文管理、会话持久化、沙箱隔离。这套基础设施才是一个 "Agent Harness" 的真正价值所在。

## 架构设计的核心原则

从 Claude Code 的架构中，我们可以提炼出几个关键的设计原则：

### 分层架构，各司其职

Claude Code 的架构分为 5 层：

1. **CLI 层**：用户交互入口，处理命令行参数、REPL 循环、Slash 命令
2. **Runtime 层**：对话循环核心，编排 API 调用和工具执行
3. **Core 层**：会话管理、权限策略、上下文压缩、配置加载
4. **Tools 层**：具体的工具实现（文件操作、Bash 执行、搜索等）
5. **Infrastructure 层**：HTTP 客户端、OAuth 认证、JSON 解析

每一层只解决一类问题，层与层之间通过清晰的接口通信。这种设计让每一层都可以独立测试和替换。

### 依赖注入让一切可替换

Claude Code 的对话循环使用了三个泛型参数：

```rust
pub struct ConversationRuntime<C: ApiClient, T: ToolExecutor> {
    api_client: C,
    tool_executor: T,
    permission_policy: PermissionPolicy,
    // ...
}
```

这意味着你可以用不同的 API 客户端（OpenAI、Anthropic、本地模型）、不同的工具执行器（本地工具、MCP 远程工具）、不同的权限策略来组合出一个完全不同的 Agent 运行时，而核心循环代码一行都不用改。

在测试中，你可以注入一个 `ScriptedApiClient`，它按调用次数返回预设的响应，从而完整测试整个对话循环而不需要真实的网络连接。

## Agentic Loop 的设计哲学

Claude Code 的对话循环非常简洁，但每个细节都有深思熟虑的设计理由：

### 为什么设置迭代上限？

```rust
max_iterations: 16  // 默认最多 16 次迭代
```

没有上限的循环是危险的。如果 Agent 陷入"调用工具 → 失败 → 重试 → 再失败"的死循环，不仅浪费 Token 和钱，还可能执行大量无用甚至危险的操作。16 次迭代足够完成绝大多数任务，同时防止失控。

### 为什么错误不中断循环？

当工具执行失败或权限被拒绝时，Claude Code **不会中断循环**，而是将错误信息作为 `ToolResult(is_error: true)` 返回给模型，让模型自行决定下一步。

这是一个非常关键的设计决策。如果直接中断，Agent 就失去了自我修正的机会。通过把错误当作正常输入，模型可以：
- 换一种方式重试
- 尝试替代方案
- 向用户报告问题并请求帮助

### 流式事件处理

API 返回的是 SSE（Server-Sent Events）流，Claude Code 将其解析为一个类型安全的事件枚举：

```rust
pub enum AssistantEvent {
    TextDelta(String),       // 文本增量
    ToolUse { id, name, input },  // 请求调用工具
    Usage(TokenUsage),       // Token 用量
    MessageStop,             // 消息结束
}
```

ToolUse 事件使用"pending tool"状态机处理——开始时创建占位，每个 Delta 追加 JSON 片段，Stop 时完成组装。这确保了即使工具输入很大，也能通过流式增量接收。

## 权限系统：安全与体验的平衡

权限系统可能是 Claude Code 最精妙的设计之一。

### 五级权限模型

| 模式 | 说明 |
|------|------|
| ReadOnly | 只允许读操作 |
| WorkspaceWrite | 允许读写文件，Bash 等危险操作需审批 |
| DangerFullAccess | 完全访问，无需审批 |
| Prompt | 每次工具调用都需确认 |
| Allow | 跳过所有权限检查 |

### "智能权限提升"

最精妙的设计在于：当用户处于 `WorkspaceWrite` 模式，但 Agent 想执行 Bash 命令（需要 `DangerFullAccess`）时，系统**不是直接拒绝**，而是自动弹出交互式审批。

这意味着：日常的文件读写操作不需要每次确认，但 Bash 执行等危险操作会被拦截。既保证了安全性，又不会过度打扰用户。

这个设计背后是一个深刻的产品洞察：**用户讨厌的不是安全检查，而是不必要的麻烦。**

## 上下文管理的关键技术

### DYNAMIC BOUNDARY

Claude Code 的系统提示分为两部分，中间用一个特殊标记分隔：

```
[Intro + Style + Rules]
__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__
[Environment + Project Context + Instructions + Config]
```

上半部分是**静态**的（不随环境变化），下半部分是**动态**的（每次运行都不同）。

为什么要这样分？因为 Anthropic API 支持 **Prompt Caching**——相同的 prompt 前缀可以被缓存，后续调用只收取缓存读取费用（是正常费用的 1/10）。静态部分命中缓存，动态部分实时计算，这是实打实的成本优化。

### 指令文件链式发现

从当前目录向上遍历所有祖先目录，查找 `CLAUDE.md`、`CLAUDE.local.md` 等指令文件。找到后对内容做哈希去重（相同内容只保留一份），每个文件限制 4000 字符，总计限制 12000 字符。

这确保了项目根目录的全局指令和子目录的局部指令都能被加载，同时避免上下文爆炸。

## 上下文压缩：让 Agent 拥有"无限记忆"

长对话会消耗大量 Token。Claude Code 的解决方案是**上下文压缩**：

当会话 Token 数超过阈值时，保留最近 N 条消息，将旧消息替换为一条系统消息，内容是自动生成的摘要：

```
<summary>
Conversation summary:
- Scope: 20 earlier messages compacted (user=8, assistant=6, tool=6).
- Tools mentioned: bash, read_file, write_file.
- Recent user requests: Refactor auth module...
- Pending work: Next: update tests...
- Key files: src/auth/oauth.rs, src/middleware/rate_limit.rs
- Key timeline:
  - user: Refactor the auth module...
  - assistant: tool_use bash(cargo test --lib auth)
  - tool_result bash: running 12 tests... ok
</summary>
```

关键是，这个摘要**不依赖 API 调用**，完全在本地生成。它从消息中提取：用户请求、工具使用记录、关键文件引用、待办事项，并按时间线排列。

压缩后，Agent 被告知："不要问用户任何问题，直接从上次中断的地方继续。"这让 Agent 能无缝恢复工作状态。

## 如何自己构建一个 AI Agent 系统

基于以上分析，如果你要构建自己的 Agent 系统，核心要做的事情：

1. **实现 Agentic Loop**：用户输入 → API 调用 → 解析工具调用 → 执行工具 → 返回结果 → 继续调用
2. **设计工具接口**：定义统一的工具描述格式（名称、参数 schema、执行函数）
3. **加入权限控制**：至少区分"安全操作"和"危险操作"
4. **管理上下文**：系统提示 + 项目指令 + 对话历史
5. **实现会话持久化**：保存/恢复对话状态
6. **考虑上下文压缩**：长对话的 Token 管理策略

记住：**Agent 的核心竞争力不在模型，而在围绕模型构建的基础设施。**
