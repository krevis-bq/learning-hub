---
title: "13 - 向量数据库全面对比与选型指南"
description: "深入对比 Milvus, Pinecone, Qdrant, Weaviate, Chroma, FAISS 等主流向量数据库的性能、功能和适用场景"
date: "2026-04-02"
category: rag
tags:
  - 向量数据库
  - Milvus
  - Pinecone
  - Qdrant
  - 选型
order: 13
---

## 选型决策树

选择向量数据库前，明确你的场景和约束：

```
你需要什么？
├─ 托管服务 vs 自建？
│   ├─ 托管 → Pinecone / Zilliz Cloud / Qdrant Cloud
│   └─ 自建 → Milvus / Qdrant / Weaviate
├─ 数据规模？
│   ├─ <100万 → Chroma / FAISS 够用
│   ├─ 100万-1亿 → Milvus / Qdrant / Weaviate
│   └─ >1亿 → Milvus 集群
├─ 是否已有 PostgreSQL?
│   ├─ 是 → pgvector（最省事）
│   └─ 否 → 专用向量数据库
├─ 延迟要求?
│   ├─ <50ms → 内存索引 (FAISS / Qdrant)
│   └─ <200ms → 任何方案都可以
└─ 查询复杂度?
    ├─ 纯相似度 → 任何方案
    ├─ 带过滤 → Qdrant / Pinecone / Milvus（支持最好）
    └─ 混合检索 → Milvus / Qdrant / Weaviate
```

## Milvus

开源生态最成熟的向量数据库，由 Zilliz 维护。

### 优势
- **功能最全**：标量过滤、向量检索、混合检索、全文检索一体化
- **高性能**：GPU 加速索引，Faiss IVF PQ 等
- **大规模**：支持 10 亿+ 向量，分布式架构
- **多索引类型**：FLAT, IVF_FLAT, IVF_PQ, IVF_SQ8, HNSW, SCANN, BIN_FLAT
- **云原生**：支持 K8s Operator 和 Helm

### 适用场景
- 企业级生产系统
- 大规模数据（>1亿向量）
- 需要复杂过滤（标量 + 向量混合）
- 已有 DevOps 团队

### 快速开始

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v milvus-data:/var/lib/milvus \
  milvusdb/milvus:v2.4-latest
```

```typescript
import { MilvusClient } from '@zilliz/milvus2-sdk-node';

const client = new MilvusClient({ address: 'localhost:19530' });

await client.createCollection({
  collection_name: 'documents',
  fields: [
    { name: 'id', data_type: 'VarChar', is_primary_key: true },
    { name: 'vector', data_type: 'FloatVector', dim: 1536 },
    { name: 'text', data_type: 'VarChar', max_length: 65535 },
    { name: 'source', data_type: 'VarChar', max_length: 512 },
    { name: 'created_at', data_type: 'Int64' },
  ],
});
```

## Pinecone

完全托管的服务，零运维。

### 优势
- **零运维**：完全托管，无需管理基础设施
- **简单易用**：API 设计简洁，上手极快
- **内置功能**：命名空间、元数据过滤、稀疏向量一体化
- **自动扩展**：根据负载自动扩缩
- **Pod-based 架构**：支持实时更新（非批量）

### 适用场景
- 快速原型
- 小团队无运维能力
- 对延迟不敏感（100-300ms）
- 需要命名空间隔离

### 注意事项
- 数据存在美国区域（合规性问题）
- 大规模数据成本较高
- 不支持 HNSW（用自己的索引算法）

```typescript
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

const pinecone = new Pinecone({ apiKey: 'your-key' });

await pinecone.createIndex({
  name: 'rag-docs',
  dimension: 1536,
  metric: 'cosine',
  spec: { serverless: { cloud: 'aws', region: 'us-east-1' } },
});
```

## Qdrant

Rust 编写的高性能向量数据库，推荐自建。

### 优势
- **性能极佳**：Rust 实现，内存效率高，延迟低
- **过滤强大**：最灵活的 payload 过滤条件
- **混合检索**：原生支持稀疏+密集向量混合检索
- **轻量级**：单二进制文件部署，资源占用小
- **量化支持**：uint8/float16/int8 量化减少内存

### 适用场景
- 自建部署（推荐首选）
- 需要复杂过滤
- 延迟敏感场景（<50ms 可达）
- 中大规模（千万级）

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

```typescript
import { QdrantClient } from '@qdrant/js-client-rest';

const client = new QdrantClient({ url: 'http://localhost:6333' });

await client.createCollection({
  collection_name: 'docs',
  vectors: { size: 1536, distance: 'Cosine' },
  hnsw_config: { m: 16, ef_construct: 200 },
});
```

## Weaviate

内置向量化的全功能数据库。

### 优势
- **内置向量化**：自动调用 embedding 模型，无需额外调用
- **GraphQL API**：灵活的查询方式
- **多模态**：支持文本、图片等多种数据类型
- **模块化**：可插拔的向量化和生成模块

### 适用场景
- 需要内置向量化（不想管理 embedding 流程）
- 全文检索 + 向量检索一体化
- GraphQL 查询偏好

### 注意
- Go 语言实现，资源占用比 Qdrant 大
- 内置向量化限制了模型选择

## Chroma

轻量级开发友好型向量库。

### 优势
- **极简 API**：几行代码就能用
- **内置 embedding**：自动调用 OpenAI 等模型
- **内存模式**：开发时不需要启动任何服务

### 适用场景
- 快速原型
- Jupyter Notebook 实验
- 小规模数据（<10万）

```typescript
import { Chroma } from '@langchain/community/vectorstores/chroma';
const chroma = await Chroma.fromDocuments(docs, new OpenAIEmbeddings());
```

### 不适用
- 生产环境
- 大规模数据
- 高并发

## FAISS

Meta 的向量检索库（不是数据库）。

### 特点
- **极致性能**：CPU/GPU 加速，亿级向量毫秒级检索
- **灵活索引**：Flat/IVF/HNSW/PQ 随意组合
- **内存操作**：无需磁盘 IO，延迟极低

### 限制
- **不是数据库**：无 CRUD、持久化需自己实现
- **无过滤**：不支持元数据过滤
- **单机**：不支持分布式

### 适用场景
- 研究实验
- 百万级以内的生产环境
- 需要极致性能

```typescript
import { FAISS } from '@langchain/community/vectorstores/faiss';

const vectorStore = await FAISS.fromDocuments(docs, new OpenAIEmbeddings());
await vectorStore.save('./index');
const loaded = await FAISS.load('./index', new OpenAIEmbeddings());
```

## pgvector

PostgreSQL 扩展，已有 PG 的团队最省事。

### 优势
- **零学习成本**：SQL 语法，开发者都会用
- **事务支持**：ACID 保证数据一致性
- **生态丰富**：直接集成现有 PG 工具链

### 限制
- **性能上限**：千万级以上性能不如专用向量库
- **索引类型少**：只有 IVF 和 HNSW
- **无混合检索**：不支持稀疏+密集混合

### 适用场景
- 已有大量 PostgreSQL 经验的团队
- 数据量 <1000万
- 向量检索是辅助功能而非核心

## 生产环境推荐

| 场景 | 推荐 | 理由 |
|------|------|------|
| 快速验证 | FAISS / Chroma | 零成本上手 |
| 中小规模生产 | Qdrant (自建) | 性能好，运维简单 |
| 大规模企业 | Milvus 集群 | 功能全、成熟稳定 |
| 无运维团队 | Pinecone / Zilliz Cloud | 托管、省心 |
| 已有 PG | pgvector | 最小迁移成本 |
