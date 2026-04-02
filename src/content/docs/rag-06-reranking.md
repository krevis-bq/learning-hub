---
title: "06 - Re-ranking：检索结果的二次精排"
description: "理解 Re-ranking 的原理、主流模型、以及在 RAG pipeline 中的集成方式"
date: "2026-04-02"
category: rag
tags:
  - Re-ranking
  - 重排序
  - 核心组件
order: 6
---

## 为什么需要 Re-ranking？

初次检索（向量检索/BM25）追求**速度**，会牺牲一定精度。Re-ranking 用更精确但更慢的模型对候选结果做二次排序：

```
初次检索（快但粗）：1000条 → Top-20候选
Re-ranking（慢但精）：20条 → Top-5精确结果
```

**比喻**：初次检索像"海选"，快速从万人中筛出20人；Re-ranking 像"决赛"，认真评估这20人，选出最终的5个。

### 精度提升有多大？

在 BEIR 和 MTEB 基准测试中，加入 Re-ranking 后：
- **Hit Rate@10** 提升 10-30%
- **NDCG@10** 提升 15-35%
- **MRR** 提升 10-25%

这是 RAG 系统中**投入产出比最高的优化手段**之一。

## Re-ranking 原理

### Cross-Encoder 架构

Re-ranking 模型通常使用 Cross-Encoder（交叉编码器），和 Bi-Encoder（双编码器，即 Embedding 模型）不同：

```
Bi-Encoder（Embedding）:
  文档 → [Encoder] → 向量A    ← 分别编码
  查询 → [Encoder] → 向量B    ← 分别编码
  相似度 = cosine(A, B)        ← 后计算

Cross-Encoder（Re-ranking）:
  [查询 + 文档] → [Encoder] → 相关性分数  ← 一起编码
```

**Cross-Encoder 优势**：能捕获查询和文档之间的细粒度交互（比如"不"字改变了整个句意），精度远高于 Bi-Encoder。

**Cross-Encoder 劣势**：必须对每个查询-文档对都过一遍模型，O(N) 复杂度，不能预计算。

## 主流 Re-ranking 模型

### Cohere Rerank

最成熟的商业化方案，多语言支持好：

```typescript
import { CohereRerank } from '@langchain/cohere';

const reranker = new CohereRerank({
  model: 'rerank-v3.5',
  apiKey: 'your-cohere-api-key',
  topN: 5,  // 返回Top-5
});

// 对检索结果重新排序
const rerankedDocs = await reranker.compressDocuments(
  retrievedDocs,    // 初次检索的结果
  'RAG系统如何优化检索效果', // 查询
);

rerankedDocs.forEach((doc, i) => {
  console.log(`#${i+1} score=${doc.metadata.relevanceScore?.toFixed(4)}`);
  console.log(`  ${doc.pageContent.slice(0, 80)}...`);
});
```

### BGE Re-ranker（开源）

智源开源的中文最佳 Re-ranking 模型：

```typescript
// 通过 HuggingFace Inference API
import { HuggingFaceInference } from '@langchain/community/llms/hf';

// 或本地部署（需要GPU）
// BAAI/bge-reranker-v2-m3：多语言，最新版本
// BAAI/bge-reranker-large：英文为主
// BAAI/bge-reranker-base：轻量版
```

### Jina Reranker

```typescript
import { JinaReranker } from '@langchain/community/utilities/jina';

const reranker = new JinaReranker({
  model: 'jina-reranker-v2-base-multilingual',
  apiKey: 'your-jina-api-key',
  topN: 5,
});
```

### 模型对比

| 模型 | 语言 | 延迟 | 成本 | 精度 | 部署 |
|------|------|------|------|------|------|
| Cohere rerank-v3.5 | 多语言 | ~100ms/20docs | $0.002/1000次 | ⭐⭐⭐⭐⭐ | API |
| bge-reranker-v2-m3 | 多语言 | ~200ms/20docs | 免费（需GPU） | ⭐⭐⭐⭐ | 本地 |
| jina-reranker-v2 | 多语言 | ~150ms/20docs | 有免费额度 | ⭐⭐⭐⭐ | API |
| bce-reranker-base | 中英 | ~300ms/20docs | 免费 | ⭐⭐⭐⭐ | 本地 |

## LangChain 中集成 Re-ranking

### 方式1：ContextualCompressionRetriever

```typescript
import { ContextualCompressionRetriever } from 'langchain/retrievers/contextual_compression';
import { CohereRerank } from '@langchain/cohere';
import { FAISS } from '@langchain/community/vectorstores/faiss';
import { OpenAIEmbeddings } from '@langchain/openai';

// 基础检索器：多取一些候选
const vectorStore = await FAISS.fromDocuments(docs, new OpenAIEmbeddings());
const baseRetriever = vectorStore.asRetriever({ k: 20 }); // 取20个候选

// Re-ranker
const reranker = new CohereRerank({
  model: 'rerank-v3.5',
  topN: 5, // 精排后只留5个
});

// 组合成压缩检索器
const compressionRetriever = new ContextualCompressionRetriever({
  baseCompressor: reranker,
  baseRetriever,
});

const results = await compressionRetriever.invoke('RAG优化策略');
// 直接拿到精排后的Top-5
```

### 方式2：手动 Pipeline

```typescript
import { Document } from '@langchain/core/documents';
import { CohereRerank } from '@langchain/cohere';

const reranker = new CohereRerank({ model: 'rerank-v3.5', topN: 5 });

async function retrieveAndRerank(query: string) {
  // Step 1: 初始检索（多取）
  const candidates = await vectorStore.similaritySearch(query, 20);
  
  // Step 2: Re-ranking
  const reranked = await reranker.compressDocuments(candidates, query);
  
  // Step 3: 结果处理
  return reranked.map(doc => ({
    content: doc.pageContent,
    score: doc.metadata.relevanceScore ?? 0,
    source: doc.metadata.source,
  }));
}
```

### 方式3：Ensemble + Re-ranking（最强组合）

```typescript
async function superRetrieve(query: string) {
  // 混合检索取候选
  const hybridResults = await ensembleRetriever.invoke(query);
  
  // Re-ranking精排
  const reranked = await reranker.compressDocuments(hybridResults, query);
  
  return reranked;
}
```

## Re-ranking 的局限

1. **延迟**：每个查询-文档对都要过模型，候选太多时延迟明显
2. **成本**：商业API按调用次数收费，高频场景成本不低
3. **长度限制**：大多数模型只支持 512-8192 tokens，超长文档需要截断
4. **不改变召回**：Re-ranking 只能从已有候选中排序，不能召回之前遗漏的文档

## 实践建议

### 候选数量选择

```
候选太少（<10）：
  → Re-ranking 提升有限（本身就是高精度）
  
候选合适（15-30）：
  → Re-ranking 效果最佳（推荐）
  
候选太多（>50）：
  → 延迟和成本增加，但边际收益递减
```

### 何时使用 Re-ranking

**推荐使用**：
- 生产环境，对回答质量有要求
- 文档库较大（>1万条）
- 初次检索效果不理想

**可以不用**：
- 原型验证阶段
- 文档库很小（<1000条）
- 对延迟极度敏感（<100ms）
- 混合检索已经够用

## 下一步

下一篇进入框架实战——**LangChain 1.0 入门**，学习如何用框架把前面的组件串联起来。
