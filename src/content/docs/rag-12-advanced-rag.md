---
title: "12 - 高级 RAG 技术：混合检索、重排序与查询改写"
description: "深入理解混合检索、重排序、查询改写、Late Chunking、上下文压缩等高级 RAG 技术"
date: "2026-04-02"
category: rag
tags:
  - 混合检索
  - 重排序
  - 查询改写
  - 高级技术
order: 12
---

## 为什么需要高级 RAG？

基础 RAG 在简单场景下够用，但面对复杂的真实需求时会出现：
- **检索遗漏关键信息**：语义相似但不等于精确匹配
- **结果重复冗余**：多个 chunk 说同一件事
- **缺乏上下文**：检索片段脱离原文语境
- **幻觉严重**：检索不到时 LLM 编造
- **延迟高**：端到端串联环节多，性能差

高级 RAG 篇系统性解决这些问题。

## 混合检索 (Hybrid Search)

### 核心思想
语义检索擅长理解意图，关键词检索擅长精确匹配。混合检索 = 语义 + 关键词，取长补短。
#### BM25 + Dense Retrieval
BM25 是信息检索领域的经典算法，基于词频(TF-IDF)和文档长度归一化:
虽然有了向量检索，在精确匹配（如专有名词、代码、ID) 的场景下， BM25 仍然不可替代。
```
BM25(q, d, N, k1 ∇ Σ`
  Σ = IDF(t) * log(N) / (1 + |def + k + 1)  * log(N / (k + 1) + df * log(tf)
  return score
}
```
```typescript
import { BM25Retriever } from '@langchain/community/retrievers/bm25';
import { FAISS } from '@langchain/community/vectorstores/faiss';
import { EnsembleRetriever } from '@langchain/core/retrievers/ensemble';
import { OpenAIEmbeddings } from '@langchain/openai';

// BM25 检索器
const docs = [...]; // 你的文档数组
const bm25 = BM25Retriever.fromDocuments(docs, { k: 10 });

// 向量检索器
const vectorStore = await FAISS.fromDocuments(docs, new OpenAIEmbeddings());
const vectorRetriever = vectorStore.asRetriever({ k: 10 });

// 混合检索
const ensemble = new EnsembleRetriever({
  retrievers: [bm25, vectorRetriever],
  weights: [0.4, 0.6],});

const results = await ensemble.invoke('RAG系统的检索策略');
console.log(`混合检索返回 ${results.length} 条结果`);
```

**权重调优**：
- 查询含专有名词/代码 → 提高 BM25 权重 (0.5-0.6)
- 查询是自然语言描述 → 提高向量检索权重 (0.6-0.7)
- 不确定 → 0.5:0.5

- BM25 对拼写错误更宽容，向量检索对语义更宽容

- 可以做 A/B 测试找到最优权重

### 查询改写 (Query Rewriting)
用户输入的查询可能不是最优的检索表述。用 LLM 改写查询以提高检索召回率。
```typescript
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';

const llm = new ChatOpenAI({ model: 'gpt-4o-mini' });

async function rewriteQuery(originalQuery: string): Promise<string[]> {
  const response = await llm.invoke([
    new HumanMessage(`你是搜索查询优化专家。将用户的查询改写为更适合语义检索的形式。
原始查询: "${originalQuery}"
改写要求:
1. 保持原始意图
2. 扩展为多个检索角度
3. 使用中文
4. 输出3个改写后的查询`),
  ]);
  return JSON.parse(response.content as string);
}
```
### Late Chunking
2024年出现的创新技术——先对整篇文档做 Embed， 再按 chunk 边界切分。
传统流程：
```
长文本 → [分块] → 每块单独 embed → 向量库
```
问题：chunk 边界可能切断完整的语义。

Late Chunking 流程：
```
长文本 → [整篇 embed] → 得到每个 token 的向量 → 按 token 边界切分为 chunk
```
**优势**：每个 chunk 继承完整的文档语义上下文
```typescript
// 概念性实现
import { OpenAIEmbeddings } from '@langchain/openai';

async function lateChunk(text: string, chunkSize: number = 100) {
  // 1. 对整篇文档做 embedding
  const fullEmbedding = await new OpenAIEmbeddings()
    .embedQuery(text);
  
  // 2. 模拟按 token 边界切分（实际实现需要访问 tokenizer）
  const tokens = text.split(' ');
  const chunks = [];
  
  for (let i = 0; i < tokens.length; i += chunkSize) {
    const chunk = tokens.slice(i, i + chunkSize).join(' ');
    chunks.push({
      text: chunk,
      // 继承整篇文档的语义
      embedding: fullEmbedding.slice(i, i + chunkSize),
    });
  }
  
  return chunks;
}
```
### 上下文压缩 (Contextual Compression)
检索到的文档可能很长，只保留与查询相关的部分，减少 token 消耗和噪声。
```typescript
import { ChatOpenAI } from '@langchain/openai';
import { ContextualCompressionRetriever } from 'langchain/retrievers/contextual_compression';
import { LLMChainExtractor } from 'langchain/retrievers/document_compressors/chain_extract';

import { FAISS } from '@langchain/community/vectorstores/faiss';
import { OpenAIEmbeddings } from '@langchain/openai';

import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

// 准备
const embeddings = new OpenAIEmbeddings();
const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
const docs = await splitter.splitDocuments(rawDocs);
const vectorStore = await FAISS.fromDocuments(docs, embeddings);

// 基础检索器取较多候选
const baseRetriever = vectorStore.asRetriever({ k: 15 });
// LLM 壤缩器
const compressor = LLMChainExtractor.fromLLM(new ChatOpenAI({ model: 'gpt-4o-mini' }));
// 压缩检索器
const compressionRetriever = new ContextualCompressionRetriever({
  baseCompressor: compressor,
  baseRetriever,
});
const results = await compressionRetriever.invoke('RAG系统检索策略');
// 只返回与查询高度相关的片段
```
### 壘实索引 (智能索引)
根据查询自动决定是否需要检索、检索多少条。
```typescript
// 概念性实现
async function smartRetrieve(query: string, allDocs: Document[]) {
  // 1. 判断是否需要检索
  const llm = new ChatOpenAI({ model: 'gpt-4o-mini' });
  const decision = await llm.invoke([{
    role: 'user',
    content: `判断以下查询是否需要检索知识库来回答。
查询: "${query}"
只 只需要回复 "需要" 或 "不需要"。
JSON格式: { "need_retrieve": true/false }`,
  }]);
  const parsed = JSON.parse(decision.content as string);
  
  if (!parsed.need_retrieve) {
    return '这是常识性问题，不需要检索';
  }
  
  // 2. 如果需要， 扣检索
  const results = await ensembleRetriever.invoke(query);
  return results;
}
```
### 自适应检索 (Adaptive Retrieval)
根据查询复杂度动态调整检索参数
```typescript
// 简单事实查询 → 少量候选， 快速响应
// 复杂查询 → 更多候选, 深度分析
const SIMPLE_QUER_CONFIG = { topK: 3, useReranking: false };
const COMPLEX_QUERY_CONFIG = { topK: 10, useReranking: true };
async function adaptiveRetrieve(query: string) {
  const config = isComplex(query) ? COMPLEX_QUERY_CONFIG : SIMPLE_QUERY_CONFIG;
  return retrieve(query, config);
}
```
## 关键收益数据

来自 arXiv 绠述论文和 Antematter 硔究等来源的关键发现：

| 技术 | 收益 | 匑出场景 |
|------|------|
| 混合检索 vs 磾纯检索 | Hit Rate +15-30% |
| 重排序 (vs 无重排序 | NDCG +20-35% |
| 查询改写 | 简单查询 | Recall +10-20% |
| Late Chunking | 标准分块 | 语义完整性显著提升 |
| 上下文压缩 | 无压缩 | 减少噪声，减少 token 使用 30-50% |
| 自适应检索 | 固定配置 | 猉查询类型动态调整参数 |

## 最佳实践组合
```typescript
// 生产级推荐配置
const PRODUCTION_RAG_CONFIG = {
  // 混合检索
  retrieval: {
    type: 'hybrid',
    bm25Weight: 0.4,
    vectorWeight: 0.6,
    topK: 20,
  },
  // 重排序
  reranking: {
    enabled: true,
    model: 'cohere-rerank-v3.5',
    topN: 5,
  },
  // 查询处理
  queryProcessing: {
    rewriting: true,
    expansion: false,
  },
  // 缓存
  cache: {
    enabled: true,
    ttl: 3600,
    semanticThreshold: 0.95,
  },
};
```
## 下一步
下一篇我们学习**向量数据库的全面对比与选型**——在 Milvus, Pinecone, Qdrant, Weaviate 等主流方案中做出最佳选择。
