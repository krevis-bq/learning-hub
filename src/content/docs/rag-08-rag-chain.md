---
title: "08 - 构建完整 RAG Chain"
description: "用 LangChain 1.0 从零搭建端到端的 RAG 系统，包含索引、检索、生成的完整代码"
date: "2026-04-02"
category: rag
tags:
  - RAG Chain
  - LangChain
  - 端到端
  - 框架实战
order: 8
---

## 概览：完整 RAG 系统架构

```
┌─────────────────────────────────────────────┐
│              离线索引 Pipeline                │
│                                             │
│  文档加载 → 分块 → Embedding → 存入向量库    │
│                                             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│              在线查询 Pipeline                │
│                                             │
│  用户查询                                    │
│    ↓                                        │
│  查询改写/扩展                               │
│    ↓                                        │
│  混合检索（向量 + BM25）                     │
│    ↓                                        │
│  Re-ranking 精排                            │
│    ↓                                        │
│  Prompt 组装                                │
│    ↓                                        │
│  LLM 生成回答                               │
│    ↓                                        │
│  返回结果 + 来源引用                         │
│                                             │
└─────────────────────────────────────────────┘
```

## Step 1: 文档加载与索引

```typescript
// src/indexing.ts
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { OpenAIEmbeddings } from '@langchain/openai';
import { FAISS } from '@langchain/community/vectorstores/faiss';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { MarkdownLoader } from '@langchain/community/document_loaders/fs/markdown';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { Document } from '@langchain/core/documents';
import * as fs from 'fs';
import * as path from 'path';

const embeddings = new OpenAIEmbeddings({
  model: 'text-embedding-3-small',
});

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
  separators: ['\n## ', '\n### ', '\n\n', '\n', '。', ' ', ''],
});

// 加载不同格式的文档
async function loadDocuments(dir: string): Promise<Document[]> {
  const docs: Document[] = [];
  
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    
    if (entry.isFile()) {
      let loader;
      if (entry.name.endsWith('.md')) {
        loader = new MarkdownLoader(fullPath);
      } else if (entry.name.endsWith('.pdf')) {
        loader = new PDFLoader(fullPath);
      } else if (entry.name.endsWith('.txt')) {
        loader = new TextLoader(fullPath);
      }
      
      if (loader) {
        const loaded = await loader.load();
        // 给每个文档加上来源元数据
        loaded.forEach(doc => {
          doc.metadata.source = entry.name;
          doc.metadata.loadedAt = new Date().toISOString();
        });
        docs.push(...loaded);
      }
    }
  }
  
  return docs;
}

// 完整索引流程
export async function buildIndex(docsDir: string, indexDir: string) {
  console.log(`加载文档 from ${docsDir}...`);
  const rawDocs = await loadDocuments(docsDir);
  console.log(`加载了 ${rawDocs.length} 个文档`);
  
  console.log('分块中...');
  const chunks = await splitter.splitDocuments(rawDocs);
  console.log(`分成 ${chunks.length} 个块`);
  
  console.log('创建向量索引...');
  const vectorStore = await FAISS.fromDocuments(chunks, embeddings);
  
  console.log(`保存索引到 ${indexDir}...`);
  await vectorStore.save(indexDir);
  
  console.log('索引完成！');
  return vectorStore;
}
```

## Step 2: 检索模块

```typescript
// src/retriever.ts
import { FAISS } from '@langchain/community/vectorstores/faiss';
import { OpenAIEmbeddings } from '@langchain/openai';
import { BM25Retriever } from '@langchain/community/retrievers/bm25';
import { EnsembleRetriever } from '@langchain/core/retrievers/ensemble';
import { CohereRerank } from '@langchain/cohere';
import { ContextualCompressionRetriever } from 'langchain/retrievers/contextual_compression';
import { VectorStoreRetriever } from '@langchain/core/vectorstores';
import { Document } from '@langchain/core/documents';

let vectorStore: FAISS;
let hybridRetriever: EnsembleRetriever;

// 加载已建好的索引
export async function loadRetriever(indexDir: string, allDocs: Document[]) {
  const embeddings = new OpenAIEmbeddings();
  vectorStore = await FAISS.load(indexDir, embeddings);
  
  // 向量检索器
  const vectorRetriever = vectorStore.asRetriever({ k: 15 });
  
  // BM25 检索器
  const bm25Retriever = BM25Retriever.fromDocuments(allDocs, { k: 15 });
  
  // 混合检索器
  hybridRetriever = new EnsembleRetriever({
    retrievers: [bm25Retriever, vectorRetriever],
    weights: [0.4, 0.6],
  });
  
  return hybridRetriever;
}

// 检索 + Re-ranking
export async function retrieve(
  query: string,
  options: {
    topK?: number;
    useReranking?: boolean;
  } = {}
): Promise<Document[]> {
  const { topK = 5, useReranking = true } = options;
  
  // Step 1: 混合检索
  const candidates = await hybridRetriever.invoke(query);
  
  if (!useReranking) {
    return candidates.slice(0, topK);
  }
  
  // Step 2: Re-ranking
  const reranker = new CohereRerank({
    model: 'rerank-v3.5',
    topN: topK,
  });
  
  const reranked = await reranker.compressDocuments(candidates, query);
  return reranked;
}
```

## Step 3: 生成模块

```typescript
// src/generator.ts
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Document } from '@langchain/core/documents';
import { RunnableSequence } from '@langchain/core/runnables';

const model = new ChatOpenAI({
  model: 'gpt-4o',
  temperature: 0.3,  // RAG场景用较低温度，减少幻觉
});

const ragPrompt = ChatPromptTemplate.fromMessages([
  ['system', `你是一个专业的技术助手。基于以下参考资料回答用户的问题。

要求：
1. 回答必须基于参考资料，不要编造信息
2. 如果参考资料中没有相关信息，明确告知用户
3. 在回答末尾标注引用来源
4. 使用清晰的结构（必要时用列表或表格）

参考资料：
{context}`],
  ['human', '{question}'],
]);

// 格式化文档为上下文
function formatDocs(docs: Document[]): string {
  return docs.map((doc, i) => {
    const source = doc.metadata.source || '未知来源';
    return `[来源${i + 1}: ${source}]\n${doc.pageContent}`;
  }).join('\n\n---\n\n');
}

export async function generate(
  question: string,
  docs: Document[]
): Promise<{ answer: string; sources: string[] }> {
  const context = formatDocs(docs);
  
  const chain = RunnableSequence.from([
    ragPrompt,
    model,
    new StringOutputParser(),
  ]);
  
  const answer = await chain.invoke({
    context,
    question,
  });
  
  const sources = [...new Set(docs.map(d => d.metadata.source).filter(Boolean))];
  
  return { answer, sources };
}
```

## Step 4: 完整 Pipeline

```typescript
// src/rag.ts
import { buildIndex, loadRetriever, retrieve, generate } from './index';
import { Document } from '@langchain/core/documents';

export interface RAGConfig {
  docsDir: string;
  indexDir: string;
  topK?: number;
  useReranking?: boolean;
}

export class RAGPipeline {
  private config: RAGConfig;
  private allDocs: Document[] = [];
  private initialized = false;
  
  constructor(config: RAGConfig) {
    this.config = {
      topK: 5,
      useReranking: true,
      ...config,
    };
  }
  
  // 初始化：建索引或加载索引
  async init(forceReindex = false) {
    if (forceReindex) {
      await buildIndex(this.config.docsDir, this.config.indexDir);
    }
    
    this.allDocs = await loadDocuments(this.config.docsDir);
    await loadRetriever(this.config.indexDir, this.allDocs);
    this.initialized = true;
    console.log('RAG Pipeline 初始化完成');
  }
  
  // 查询
  async query(question: string) {
    if (!this.initialized) {
      throw new Error('Pipeline 未初始化，请先调用 init()');
    }
    
    console.log(`\n查询: ${question}`);
    
    // 检索
    const docs = await retrieve(question, {
      topK: this.config.topK,
      useReranking: this.config.useReranking,
    });
    
    console.log(`检索到 ${docs.length} 条相关文档`);
    
    // 生成
    const { answer, sources } = await generate(question, docs);
    
    return {
      question,
      answer,
      sources,
      retrievedDocs: docs.length,
      context: docs.map(d => ({
        content: d.pageContent.slice(0, 100) + '...',
        source: d.metadata.source,
        score: d.metadata.relevanceScore,
      })),
    };
  }
}

// 使用示例
async function main() {
  const rag = new RAGPipeline({
    docsDir: './knowledge-base',
    indexDir: './faiss-index',
    topK: 5,
    useReranking: true,
  });
  
  await rag.init();
  
  const result = await rag.query('RAG系统中向量数据库怎么选？');
  console.log('\n=== 回答 ===');
  console.log(result.answer);
  console.log('\n=== 来源 ===');
  result.sources.forEach(s => console.log(`- ${s}`));
}

main().catch(console.error);
```

## Step 5: 进阶——对话式 RAG

支持多轮对话，带历史记忆：

```typescript
// src/conversational-rag.ts
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { RunnablePassthrough, RunnableSequence } from '@langchain/core/runnables';
import {AIMessage, HumanMessage, BaseMessage } from '@langchain/core/messages';
import { StringOutputParser } from '@langchain/core/output_parsers';

const conversationalPrompt = ChatPromptTemplate.fromMessages([
  ['system', `你是一个专业的技术助手。基于以下参考资料和对话历史回答用户的最新问题。

参考资料：
{context}

要求：
1. 结合对话历史理解用户意图
2. 回答基于参考资料，不编造
3. 引用来源`],
  new MessagesPlaceholder('chatHistory'),
  ['human', '{question}'],
]);

export async function conversationalQuery(
  question: string,
  chatHistory: BaseMessage[] = [],
  retriever: any,
  model: ChatOpenAI,
) {
  // 检索相关文档
  const docs = await retriever.invoke(question);
  const context = docs.map(d => d.pageContent).join('\n\n');
  
  const chain = RunnableSequence.from([
    {
      context: () => context,
      chatHistory: () => chatHistory,
      question: new RunnablePassthrough(),
    },
    conversationalPrompt,
    model,
    new StringOutputParser(),
  ]);
  
  return chain.invoke(question);
}

// 使用
const history: BaseMessage[] = [];

// 第一轮
const answer1 = await conversationalQuery(
  '什么是RAG？',
  history,
  retriever,
  model,
);
history.push(new HumanMessage('什么是RAG？'));
history.push(new AIMessage(answer1));

// 第二轮（带上下文）
const answer2 = await conversationalQuery(
  '它的缺点是什么？',  // "它"指的是上一轮的RAG
  history,
  retriever,
  model,
);
```

## 常见问题与优化

### 回答中出现幻觉

```typescript
// 在 prompt 中加强约束
const strictPrompt = `严格基于以下参考资料回答问题。
如果参考资料中没有足够的信息来回答，请回复：
"根据现有资料，我无法回答这个问题。"
绝对不要编造任何信息。`;
```

### 检索不到相关内容

```typescript
// 1. 检查查询改写
const rewrittenQuery = await model.invoke([
  new HumanMessage(`将以下问题改写为更适合检索的查询：
原始问题：${userQuery}
改写后：`),
]);

// 2. 多查询检索
// 3. 降低相似度阈值
```

### 回答太长/太短

```typescript
// 通过 prompt 控制长度
const lengthControlledPrompt = `请用200-300字回答以下问题。不要超出这个范围。`;

// 或通过 maxTokens
const model = new ChatOpenAI({ maxTokens: 500 });
```

## 下一步

下一篇我们探讨**多模态 RAG**——如何处理图片、表格、代码等非文本内容。
