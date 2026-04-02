---
title: "04 - 向量数据库：存储与检索引擎"
description: "全面掌握向量数据库的选型、原理、操作与性能优化"
date: "2026-04-02"
category: rag
tags:
  - 向量数据库
  - Milvus
  - Pinecone
  - 核心组件
order: 4
---

## 什么是向量数据库？

向量数据库是专门为高维向量设计的存储和检索系统。在 RAG 中，它负责：

1. **存储**：持久化保存文档的 embedding 向量 + 元数据
2. **检索**：给定查询向量，快速找到最相似的 Top-K 个向量
3. **管理**：支持增删改查、过滤、分片等数据库操作

```
查询: "RAG是什么？"
    ↓ Embedding
查询向量: [0.12, -0.34, ...]
    ↓ 向量数据库检索
Top-5 结果:
  1. sim=0.92 "RAG是检索增强生成技术..." (source: doc1.md)
  2. sim=0.87 "RAG结合了检索和生成..."   (source: doc2.md)
  3. sim=0.83 "RAG vs 微调的区别..."     (source: doc1.md)
  4. sim=0.79 "RAG的工作流程包括..."     (source: doc3.md)
  5. sim=0.75 "RAG系统的评估指标..."     (source: doc4.md)
```

## 向量检索原理

### 暴力搜索（Flat）

直接计算查询向量与数据库中所有向量的相似度，取 Top-K。

- **时间复杂度**：O(N)，N = 向量总数
- **精确度**：100%
- **适用**：小规模（<10万条）

### 近似最近邻（ANN）

牺牲少量精度换取数量级的速度提升：

#### IVF（倒排索引）

```
1. 用 K-Means 把向量空间分成 n 个聚类中心
2. 每个向量归属到最近的聚类
3. 查询时只搜索最近的 nprobe 个聚类

速度提升：10-100x
精度损失：<5%
```

#### HNSW（分层可导航小世界图）

```
1. 构建多层图结构
2. 上层是稀疏的"高速公路"，下层是密集的精确图
3. 查询时从顶层开始，逐层向下逼近

速度提升：50-500x
精度损失：<3%
最主流的 ANN 算法
```

#### PQ（乘积量化）

```
1. 把高维向量切成 M 个子向量
2. 每个子向量量化为聚类中心 ID（1字节）
3. 距离计算变成查表操作

内存节省：8-32x（float32 → uint8）
速度提升：10-50x
```

## 主流向量数据库对比

| 数据库 | 类型 | ANN算法 | 开源 | 托管服务 | 最大规模 | 过滤 | 中文生态 |
|--------|------|---------|------|---------|---------|------|---------|
| **Milvus** | 专用 | IVF/HNSW/PQ | ✅ | Zilliz Cloud | 10亿+ | ✅ | ✅✅ |
| **Pinecone** | 专用 | 自研 | ❌ | ✅ | 10亿+ | ✅ | ✅ |
| **Qdrant** | 专用 | HNSW | ✅ | ✅ | 亿级 | ✅✅ | ✅ |
| **Weaviate** | 专用 | HNSW | ✅ | ✅ | 亿级 | ✅ | ✅ |
| **Chroma** | 轻量 | HNSW | ✅ | ❌ | 百万级 | ✅ | ✅ |
| **pgvector** | 扩展 | IVF/HNSW | ✅ | ✅ | 千万级 | ✅✅ | ✅ |
| **FAISS** | 库 | IVF/HNSW/PQ | ✅ | ❌ | 10亿+ | ❌ | ✅ |

### 选型建议

```
需要生产级、大规模？
├─ 是 → 自建运维团队？
│       ├─ 是 → Milvus（功能最全）或 Qdrant（Rust，性能好）
│       └─ 否 → Zilliz Cloud（Milvus托管）或 Pinecone
└─ 否 → 原型/小项目？
         ├─ 已有 PostgreSQL → pgvector（最省事）
         ├─ Python 快速原型 → Chroma
         └─ 纯内存/LangChain → FAISS / MemoryVectorStore
```

## LangChain 集成实战

### FAISS（本地开发首选）

```typescript
import { FAISS } from '@langchain/community/vectorstores/faiss';
import { OpenAIEmbeddings } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

const embeddings = new OpenAIEmbeddings();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});

// 从文档创建向量库
const docs = await splitter.splitDocuments(rawDocs);
const vectorStore = await FAISS.fromDocuments(docs, embeddings);

// 相似度检索
const results = await vectorStore.similaritySearch('什么是RAG？', 5);
results.forEach(doc => {
  console.log(`[${doc.metadata.source}] ${doc.pageContent.slice(0, 80)}...`);
});

// 带分数的检索
const resultsWithScore = await vectorStore.similaritySearchWithScore('RAG workflow', 3);
resultsWithScore.forEach(([doc, score]) => {
  console.log(`score=${score.toFixed(4)}: ${doc.pageContent.slice(0, 60)}`);
});

// 保存和加载
await vectorStore.save('./faiss-index');
const loaded = await FAISS.load('./faiss-index', embeddings);
```

### Pinecone（云端生产级）

```typescript
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { OpenAIEmbeddings } from '@langchain/openai';

// 初始化 Pinecone 客户端
const pinecone = new Pinecone({
  apiKey: 'your-pinecone-api-key',
});

// 创建索引（首次）
await pinecone.createIndex({
  name: 'rag-knowledge',
  dimension: 1536,        // 必须匹配 embedding 模型维度
  metric: 'cosine',       // 相似度度量
  spec: {
    serverless: {
      cloud: 'aws',
      region: 'us-east-1',
    },
  },
});

const index = pinecone.Index('rag-knowledge');

// 从文档创建
const vectorStore = await PineconeStore.fromDocuments(docs, embeddings, {
  pineconeIndex: index,
  maxConcurrency: 5,    // 并发写入
  batchSize: 100,        // 每批大小
});

// 检索（支持元数据过滤）
const results = await vectorStore.similaritySearch('RAG', 5, {
  source: 'technical-doc', // 只搜索特定来源
});

// MMR 检索（增加多样性）
const diverseResults = await vectorStore.maxMarginalRelevanceSearch(
  'RAG技术',
  { k: 5, fetchK: 20, lambda: 0.7 }
);
```

### Qdrant（自建推荐）

```typescript
import { QdrantClient } from '@qdrant/js-client-rest';
import { QdrantVectorStore } from '@langchain/qdrant';
import { OpenAIEmbeddings } from '@langchain/openai';

const client = new QdrantClient({ url: 'http://localhost:6333' });

const vectorStore = await QdrantVectorStore.fromDocuments(docs, embeddings, {
  client,
  collectionName: 'rag-docs',
  // 支持payload过滤
});

// 带过滤条件的检索
const results = await vectorStore.similaritySearch('检索策略', 5);
```

## 性能优化

### 索引参数调优

```typescript
// Qdrant HNSW 参数示例
const collectionConfig = {
  vectors: {
    size: 1536,
    distance: 'Cosine',
    hnsw_config: {
      m: 16,              // 每个节点的连接数（越大越精确，越占内存）
      ef_construct: 200,  // 构建时的搜索宽度（越大越精确，构建越慢）
    },
  },
  optimizers_config: {
    indexing_threshold: 20000, // 超过此数量才建索引
  },
};

// 查询时的 ef_search 参数
const searchParams = {
  hnsw_ef: 128,  // 查询时的搜索宽度（越大越精确，越慢）
  exact: false,  // false=ANN, true=暴力搜索
};
```

### 批量写入优化

```typescript
// 批量upsert，而不是逐条插入
const batchSize = 500;
for (let i = 0; i < allDocs.length; i += batchSize) {
  const batch = allDocs.slice(i, i + batchSize);
  await vectorStore.addDocuments(batch);
  console.log(`Indexed ${Math.min(i + batchSize, allDocs.length)} / ${allDocs.length}`);
}
```

### 冷热分离

```
热数据（频繁查询）：
  → 内存存储 + HNSW 索引
  → SSD 磁盘

冷数据（历史归档）：
  → 量化压缩（PQ/SQ）
  → HDD 磁盘
  → 访问时动态加载
```

## 常见问题

### 向量维度不匹配

```
错误：Collection dimension (768) != embedding dimension (1536)
原因：Embedding 模型换了，或者用错模型
修复：删除旧索引，用新模型重新 embedding
```

### 检索结果全是无关内容

可能原因：
1. **embedding 模型和查询不匹配**（比如文档用中文模型，查询用英文模型）
2. **分块太碎**，检索到的是无意义的片段
3. **没有做 re-ranking**，原始检索精度不够
4. **向量数据库距离度量设置错误**（cosine vs l2）

### 内存不足

```
估算公式：
  内存(GB) = 向量数 × 维度 × 4字节 / (1024³)
  
  100万 × 1536维 = ~5.7GB（纯向量，不含索引开销）
  
优化方案：
  1. 使用量化（uint8 → 内存减少4倍）
  2. 使用磁盘存储（Qdrant的memmap模式）
  3. 分片到多个节点
```

## 下一步

下一篇我们学习**检索策略**——如何从向量数据库中获取最相关的内容，包括混合检索、MMR 等高级技巧。
