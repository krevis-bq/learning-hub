---
title: "05 - 检索策略：从基础到高级"
description: "掌握相似度检索、混合检索、MMR、查询转换等核心检索策略"
date: "2026-04-02"
category: rag
tags:
  - 检索
  - 混合检索
  - MMR
  - 核心组件
order: 5
---

## 检索在 RAG 中的地位

检索是 RAG 的核心环节——LLM 的回答质量上限由检索质量决定。**垃圾进，垃圾出**（Garbage In, Garbage Out）。

```
用户查询 → [检索策略] → 相关文档 → [LLM] → 回答
               ↑
          这一步决定了上限
```

## 基础检索

### 相似度检索（Similarity Search）

最基本的策略：计算查询向量和所有文档向量的余弦相似度，取 Top-K。

```typescript
import { FAISS } from '@langchain/community/vectorstores/faiss';
import { OpenAIEmbeddings } from '@langchain/openai';

const vectorStore = await FAISS.fromDocuments(docs, new OpenAIEmbeddings());

// 基础相似度检索
const results = await vectorStore.similaritySearch(
  '如何优化RAG系统的检索效果？',  // 查询文本
  5,                               // 返回Top-5
);

// 带分数的检索（分数越小越相似，FAISS用的是L2距离）
const resultsWithScore = await vectorStore.similaritySearchWithScore(
  '如何优化RAG系统的检索效果？',
  5,
);
```

### K值选择

```typescript
// K太小：遗漏重要信息
const tooFew = await vectorStore.similaritySearch(query, 1); // ❌

// K太大：噪声多，浪费token，LLM困惑
const tooMany = await vectorStore.similaritySearch(query, 50); // ❌

// 合理范围：3-10
const justRight = await vectorStore.similaritySearch(query, 5); // ✅
```

| 场景 | 推荐K值 |
|------|---------|
| 简单问答 | 3-5 |
| 复杂推理 | 5-10 |
| 总结/报告 | 10-20 |
| 代码搜索 | 3-5 |

## 高级检索策略

### 1. MMR（最大边际相关性）

解决**检索结果重复**的问题。MMR 在保证相关性的同时，尽量让结果多样化：

```typescript
// 标准检索可能返回5个非常相似的结果
const standardResults = await vectorStore.similaritySearch('RAG技术', 5);
// 可能5个都在讲同一件事

// MMR检索：平衡相关性和多样性
const mmrResults = await vectorStore.maxMarginalRelevanceSearch(
  'RAG技术',
  {
    k: 5,          // 最终返回5个
    fetchK: 20,    // 先检索20个候选
    lambda: 0.5,   // 0=最大多样性，1=最大相关性
  }
);
```

**lambda 参数调节**：
- `lambda = 1.0`：退化为标准相似度检索
- `lambda = 0.7`：偏向相关性，适度去重（推荐）
- `lambda = 0.5`：平衡模式
- `lambda = 0.3`：偏向多样性，适合探索性查询

### 2. 混合检索（Hybrid Search）

结合向量检索和关键词检索的优势：

```
向量检索：擅长语义匹配
  "怎么提高性能" ≈ "如何优化速度"

关键词检索（BM25）：擅长精确匹配
  "Python 3.12" ≠ "Python 编程"（必须包含3.12）

混合检索 = 两者融合 → 最强检索效果
```

```typescript
import { BM25Retriever } from '@langchain/community/retrievers/bm25';
import { FAISS } from '@langchain/community/vectorstores/faiss';
import { OpenAIEmbeddings } from '@langchain/openai';
import { EnsembleRetriever } from '@langchain/core/retrievers/ensemble';

// 关键词检索器
const bm25Retriever = BM25Retriever.fromDocuments(docs, {
  k: 5,
});

// 向量检索器
const vectorStore = await FAISS.fromDocuments(docs, new OpenAIEmbeddings());
const vectorRetriever = vectorStore.asRetriever({ k: 5 });

// 混合检索器（ Ensemble）
const ensembleRetriever = new EnsembleRetriever({
  retrievers: [bm25Retriever, vectorRetriever],
  weights: [0.4, 0.6],  // BM25占40%，向量检索占60%
  c: 0,                  // RRF常量
});

const results = await ensembleRetriever.invoke('RAG中的向量检索策略');
console.log(`混合检索返回 ${results.length} 条结果`);
```

**权重调节建议**：
- 查询含专有名词、代码、编号 → 提高 BM25 权重（0.5-0.6）
- 查询是自然语言描述 → 提高向量检索权重（0.6-0.7）
- 不确定 → 各0.5

### 3. 多查询检索（Multi-Query Retriever）

用户的一次查询可能表述不精确，用 LLM 生成多个变体查询，扩大检索覆盖面：

```typescript
import { MultiQueryRetriever } from 'langchain/retrievers/multi_query';
import { ChatOpenAI } from '@langchain/openai';

const llm = new ChatOpenAI({ model: 'gpt-4o-mini' });

const multiQueryRetriever = MultiQueryRetriever.fromLLM({
  llm,
  retriever: vectorStore.asRetriever({ k: 3 }),
  queryCount: 3,  // 生成3个变体查询
});

// 内部过程：
// 原始查询: "RAG优化"
// 变体1: "如何提升检索增强生成系统的性能"
// 变体2: "RAG系统效果不佳的优化方法"  
// 变体3: "RAG pipeline调优策略"
// → 分别检索，合并去重

const results = await multiQueryRetriever.invoke('RAG优化');
```

### 4. 上下文压缩检索（Contextual Compression）

检索到的文档可能很长，只保留和查询相关的部分：

```typescript
import { ContextualCompressionRetriever } from 'langchain/retrievers/contextual_compression';
import { LLMChainExtractor } from 'langchain/retrievers/document_compressors/chain_extract';
import { ChatOpenAI } from '@langchain/openai';

const llm = new ChatOpenAI({ model: 'gpt-4o-mini' });

// 基础检索器
const baseRetriever = vectorStore.asRetriever({ k: 10 });

// LLM压缩器：从每个文档中提取与查询相关的内容
const compressor = LLMChainExtractor.fromLLM(llm);

// 组合
const compressionRetriever = new ContextualCompressionRetriever({
  baseCompressor: compressor,
  baseRetriever,
});

// 检索10条 → LLM压缩 → 只返回相关片段
const results = await compressionRetriever.invoke('RAG中的re-ranking策略');
```

### 5. 自查询检索（Self-Querying）

让 LLM 自动把自然语言查询拆分为"语义查询"和"元数据过滤"：

```typescript
import { SelfQueryRetriever } from 'langchain/retrievers/self_query';
import { ChatOpenAI } from '@langchain/openai';

// 用户说："2024年发布的技术文档中关于RAG的内容"
// LLM自动拆分为：
//   语义查询: "RAG"
//   过滤条件: year=2024 AND type="technical-doc"

const selfQueryRetriever = SelfQueryRetriever.fromLLM({
  llm: new ChatOpenAI(),
  vectorStore,
  documentContents: '技术文档和教程',
  attributeInfo: [
    { name: 'year', description: '发布年份', type: 'number' },
    { name: 'source', description: '文档来源', type: 'string' },
    { name: 'docType', description: '文档类型', type: 'string' },
  ],
});

const results = await selfQueryRetriever.invoke(
  '2024年发布的关于向量数据库的技术文档'
);
```

## 检索策略组合

实际生产系统中，往往组合多种策略：

```typescript
// 组合方案：混合检索 + MMR去重 + Re-ranking
const pipeline = async (query: string) => {
  // Step 1: 混合检索，取较多候选
  const hybridResults = await ensembleRetriever.invoke(query);
  
  // Step 2: MMR去重（在候选中去选多样化的结果）
  const diverseResults = await mmrFilter(hybridResults, { k: 10 });
  
  // Step 3: Re-ranking（精确排序）
  const rerankedResults = await reranker.rerank(query, diverseResults);
  
  // Step 4: 取Top-K
  return rerankedResults.slice(0, 5);
};
```

## 评估检索质量

### 命中率（Hit Rate）

```typescript
// 查询的正确答案是否在Top-K结果中
function hitRate(queries: TestQuery[], k: number): number {
  let hits = 0;
  for (const q of queries) {
    const results = retriever.invoke(q.query);
    const topKIds = results.slice(0, k).map(r => r.metadata.id);
    if (q.relevantDocIds.some(id => topKIds.includes(id))) {
      hits++;
    }
  }
  return hits / queries.length;
}
```

### MRR（Mean Reciprocal Rank）

```typescript
// 正确答案在结果中的排名越高，分数越高
function mrr(queries: TestQuery[]): number {
  let totalReciprocal = 0;
  for (const q of queries) {
    const results = retriever.invoke(q.query);
    for (let i = 0; i < results.length; i++) {
      if (q.relevantDocIds.includes(results[i].metadata.id)) {
        totalReciprocal += 1 / (i + 1);
        break;
      }
    }
  }
  return totalReciprocal / queries.length;
}
```

### NDCG（Normalized Discounted Cumulative Gain）

考虑了排序的质量——相关结果排在前面比排在后面贡献更大。

## 下一步

下一篇学习 **Re-ranking**——对检索结果做二次精排，进一步提升精度。
