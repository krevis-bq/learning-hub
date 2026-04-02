---
title: "07 - LangChain 1.0 入门"
description: "LangChain 核心概念、模块架构、与 TypeScript 版本的关键用法"
date: "2026-04-02"
category: rag
tags:
  - LangChain
  - TypeScript
  - 框架实战
order: 7
---

## 什么是 LangChain？

LangChain 是目前最流行的 LLM 应用开发框架，提供了一套标准化的组件抽象和组合方式：

```
LangChain 核心价值：
1. 统一的模型接口 → 换模型只改一行配置
2. 丰富的组件库 → Embedding、向量库、检索器、工具，开箱即用
3. Chain/Agent 抽象 → 复杂流程用声明式代码表达
4. TypeScript + Python → 前后端都能用
```

本文聚焦 **LangChain TypeScript 版本**，基于 1.0 架构。

## 安装

```bash
# 核心包
npm install @langchain/core @langchain/openai @langchain/community

# 按需安装
npm install @langchain/pinecone    # Pinecone向量库
npm install @langchain/cohere      # Cohere模型
npm install @langchain/textsplitters # 文本分割
```

## 核心模块架构

```
@langchain/core
├── language_models     # 模型抽象
│   ├── chat_models     # Chat Model（对话式）
│   └── llms            # LLM（文本补全式，已较少用）
├── embeddings          # Embedding 抽象
├── vectorstores        # 向量数据库抽象
├── retrievers          # 检索器抽象
├── documents           # Document 数据结构
├── messages            # 消息格式（Human/AI/System）
├── output_parsers      # 输出解析器
├── prompts             # Prompt 模板
├── chains              # Chain 组合
└── tools               # 工具定义（Agent用）
```

## 基础用法

### Chat Model

```typescript
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, SystemMessage, AIMessage } from '@langchain/core/messages';

const model = new ChatOpenAI({
  model: 'gpt-4o',
  temperature: 0.7,
  maxTokens: 2000,
});

// 基础对话
const response = await model.invoke([
  new SystemMessage('你是一个RAG技术专家'),
  new HumanMessage('什么是检索增强生成？'),
]);

console.log(response.content);

// 多轮对话
const conversation = [
  new SystemMessage('你是技术顾问'),
  new HumanMessage('RAG和微调的区别是什么？'),
  new AIMessage('RAG是检索增强生成，微调是修改模型参数...'),
  new HumanMessage('那什么时候用RAG更好？'),  // 带上下文
];

const response2 = await model.invoke(conversation);
```

### 流式输出

```typescript
const stream = await model.stream([
  new HumanMessage('用500字解释RAG的工作原理'),
]);

for await (const chunk of stream) {
  process.stdout.write(chunk.content as string);
}
```

### Prompt Template

```typescript
import { ChatPromptTemplate } from '@langchain/core/prompts';

const prompt = ChatPromptTemplate.fromMessages([
  ['system', '你是一个{role}，用{style}的风格回答问题'],
  ['human', '{question}'],
]);

// 格式化 prompt
const formatted = await prompt.invoke({
  role: '资深工程师',
  style: '简洁直接',
  question: '如何设计一个RAG系统？',
});

const response = await model.invoke(formatted);
```

### Output Parser

```typescript
import { StringOutputParser } from '@langchain/core/output_parsers';
import { StructuredOutputParser } from 'langchain/output_parsers';
import { z } from 'zod';

// 字符串解析器（提取AI回复的纯文本）
const stringParser = new StringOutputParser();

// 结构化输出解析器
const structuredParser = StructuredOutputParser.fromZodSchema(
  z.object({
    answer: z.string().describe('回答内容'),
    sources: z.array(z.string()).describe('参考来源'),
    confidence: z.number().min(0).max(1).describe('置信度'),
  })
);

const formatInstructions = structuredParser.getFormatInstructions();

const response = await model.invoke([
  new HumanMessage(`根据以下信息回答问题。
  
{context}

问题：{question}

${formatInstructions}`),
]);

const parsed = await structuredParser.parse(response.content as string);
console.log(parsed);
// { answer: "...", sources: ["doc1.md"], confidence: 0.85 }
```

## Chain（链）

LangChain 1.0 的核心组合方式——用管道操作符 `pipe` 串联组件：

```typescript
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnablePassthrough } from '@langchain/core/runnables';

// 构建一个RAG Chain
const ragPrompt = ChatPromptTemplate.fromMessages([
  ['system', `基于以下上下文回答用户的问题。如果上下文中没有相关信息，请说"我没有找到相关信息"。

上下文：
{context}`],
  ['human', '{question}'],
]);

// 方式1：用 pipe 串联
const chain = ragPrompt.pipe(model).pipe(new StringOutputParser());

const answer = await chain.invoke({
  context: 'RAG是检索增强生成技术...',
  question: '什么是RAG？',
});

// 方式2：用 RunnableSequence.from
import { RunnableSequence } from '@langchain/core/runnables';

const chain2 = RunnableSequence.from([
  ragPrompt,
  model,
  new StringOutputParser(),
]);
```

### 带 Retriever 的 Chain

```typescript
// 从向量库创建检索器
const retriever = vectorStore.asRetriever({ k: 5 });

// 构建完整 RAG chain
const ragChain = RunnableSequence.from([
  {
    context: (input: { question: string }) =>
      retriever.invoke(input.question).then(docs =>
        docs.map(d => d.pageContent).join('\n\n')
      ),
    question: new RunnablePassthrough(),
  },
  ragPrompt,
  model,
  new StringOutputParser(),
]);

const answer = await ragChain.invoke({ question: 'RAG和微调的区别？' });
```

## Tool Calling（函数调用）

LangChain 1.0 的 Agent 基础——让模型调用外部工具：

```typescript
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

// 定义工具
const searchTool = tool(async ({ query }: { query: string }) => {
  const results = await vectorStore.similaritySearch(query, 3);
  return results.map(r => r.pageContent).join('\n\n');
}, {
  name: 'knowledge_search',
  description: '在知识库中搜索相关信息',
  schema: z.object({
    query: z.string().describe('搜索查询'),
  }),
});

// 绑定工具到模型
const modelWithTools = model.bindTools([searchTool]);

// 模型会自动决定是否调用工具
const response = await modelWithTools.invoke([
  new HumanMessage('RAG系统中向量数据库怎么选？'),
]);

if (response.tool_calls?.length) {
  for (const tc of response.tool_calls) {
    console.log(`调用工具: ${tc.name}`);
    console.log(`参数: ${JSON.stringify(tc.args)}`);
  }
}
```

## Document 数据结构

LangChain 中所有文档相关操作的统一数据格式：

```typescript
import { Document } from '@langchain/core/documents';

// 创建文档
const doc = new Document({
  pageContent: 'RAG是检索增强生成技术，通过外部知识库增强LLM的回答质量。',
  metadata: {
    source: 'rag-guide.md',
    page: 1,
    author: 'Krevis',
    tags: ['RAG', 'LLM'],
  },
});

// 文档经过 splitDocuments 后元数据会自动继承
const chunks = await splitter.splitDocuments([doc]);
chunks.forEach(chunk => {
  console.log(chunk.metadata.source); // 'rag-guide.md' ← 自动继承
});
```

## 常用模式速查

### 模型切换

```typescript
// 只改一行就能换模型
import { ChatAnthropic } from '@langchain/anthropic';
// import { ChatOpenAI } from '@langchain/openai';

const model = new ChatAnthropic({
  model: 'claude-sonnet-4-20250514',
});
```

### 错误处理

```typescript
import { RetryOutputParser } from 'langchain/output_parsers';

const chain = ragPrompt.pipe(model).pipe(parser);

try {
  const result = await chain.invoke(input);
} catch (e) {
  if (e instanceof OutputParsingException) {
    // 自动重试一次
    const retryResult = await new RetryOutputParser()
      .parseWithPrompt(e.output, e.prompt);
  }
}
```

### 回调与可观测性

```typescript
import { CallbackHandler } from 'langchain/langsmith';

// LangSmith 追踪（开发调试神器）
const handler = new CallbackHandler({
  projectName: 'rag-system',
});

const response = await chain.invoke(input, {
  callbacks: [handler],
});
```

## 下一步

下一篇我们将用 LangChain 1.0 从零搭建一个**完整的 RAG Chain**，把前面学的所有组件串联起来。
