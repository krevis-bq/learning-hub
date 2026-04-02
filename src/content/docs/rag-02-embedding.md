---
title: "02 - Embedding 详解：从词向量到语义表示"
description: "深入理解 Embedding 技术的原理、主流模型、选型策略与代码实践"
date: "2026-04-02"
category: rag
tags:
  - Embedding
  - 向量
  - 基础组件
order: 2
---

## 什么是 Embedding？

Embedding（嵌入）是将离散的文本符号映射到连续的向量空间中的技术。简单来说，就是把一段文字变成一串数字（向量），让语义相近的文本在向量空间中的距离也相近。

```
"猫坐在垫子上" → [0.12, -0.34, 0.78, ..., 0.56]  // 768维
"小猫趴在地毯上" → [0.11, -0.33, 0.79, ..., 0.55]  // 语义相近，向量也相近
"股票市场暴跌"   → [-0.45, 0.89, -0.12, ..., 0.23]  // 语义不同，向量差异大
```

### 为什么需要 Embedding？

在 RAG 系统中，Embedding 是连接"文本世界"和"数学世界"的桥梁：

1. **语义检索**：通过向量相似度找到语义相关的文档，而不是关键词匹配
2. **聚类分析**：把相似的文档自动归类
3. **分类判断**：根据向量判断文本属于哪个类别

### Embedding 的演进

| 阶段 | 技术 | 特点 |
|------|------|------|
| 2013 | Word2Vec | 静态词向量，一词一向量 |
| 2014 | GloVe | 全局共现统计 |
| 2018 | ELMo | 上下文相关，LSTM架构 |
| 2018 | BERT | 双向Transformer，里程碑 |
| 2019 | Sentence-BERT | 专门优化句子级语义相似度 |
| 2023+ | OpenAI text-embedding-3 | 大规模预训练，多语言 |
| 2024+ | BGE / GTE / Jina | 开源SOTA，中文表现优秀 |

## 主流 Embedding 模型

### OpenAI 系列

```typescript
import OpenAI from 'openai';

const client = new OpenAI();

// text-embedding-3-small：性价比之选
const embedding1 = await client.embeddings.create({
  model: 'text-embedding-3-small',
  input: '什么是检索增强生成？',
});

// text-embedding-3-large：更高维度，更精确
const embedding2 = await client.embeddings.create({
  model: 'text-embedding-3-large',
  input: '什么是检索增强生成？',
  dimensions: 1024, // 可自定义维度（降维）
});

console.log(embedding1.data[0].embedding.length); // 1536
console.log(embedding2.data[0].embedding.length); // 1024
```

**特点**：
- `text-embedding-3-small`：1536维，便宜，速度快
- `text-embedding-3-large`：3072维（可降维），精度更高
- 支持自定义维度，灵活度好
- 多语言支持优秀

### 开源模型推荐

#### BGE 系列（智源）

```typescript
// 通过 HuggingFace Inference API
import { HfInference } from '@huggingface/inference';

const hf = new HfInference('your-hf-token');

const result = await hf.featureExtraction({
  model: 'BAAI/bge-large-zh-v1.5',
  inputs: '什么是检索增强生成？',
});
```

**模型选择**：
- `bge-large-zh-v1.5`：中文最佳，1024维
- `bge-base-zh-v1.5`：平衡选择，768维
- `bge-small-zh-v1.5`：轻量级，512维

#### GTE 系列（阿里巴巴）

- `gte-large-zh`：中文表现优秀
- `gte-Qwen2`：最新一代，多语言

#### Jina 系列

- `jina-embeddings-v3`：多任务，支持 LoRA 微调
- 1024维，支持8192 token长度

### 模型对比

| 模型 | 维度 | 最大长度 | 中文 | 开源 | MTEB排名 |
|------|------|---------|------|------|---------|
| text-embedding-3-large | 3072 | 8191 | ✅ | ❌ | Top 3 |
| bge-large-zh-v1.5 | 1024 | 512 | ✅✅ | ✅ | 中文Top 1 |
| gte-Qwen2 | 1536 | 32768 | ✅✅ | ✅ | Top 5 |
| jina-embeddings-v3 | 1024 | 8192 | ✅ | ✅ | Top 10 |
| Cohere embed-v3 | 1024 | 512 | ✅ | ❌ | Top 5 |

## 相似度计算

### 余弦相似度（最常用）

衡量两个向量方向的相似程度，范围 [-1, 1]：

```typescript
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

const sim = cosineSimilarity(
  [0.12, -0.34, 0.78],
  [0.11, -0.33, 0.79]
);
console.log(sim); // ≈ 0.999
```

### 其他距离度量

```typescript
// 欧氏距离：向量间的直线距离
function euclideanDistance(a: number[], b: number[]): number {
  return Math.sqrt(
    a.reduce((sum, val, i) => sum + (val - b[i]) ** 2, 0)
  );
}

// 点积（内积）：某些模型直接用点积作为相似度
function dotProduct(a: number[], b: number[]): number {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}
```

**选择建议**：
- 大多数向量数据库默认用**余弦相似度**
- 归一化向量（单位向量）的点积 = 余弦相似度
- 欧氏距离对向量长度敏感，适合非归一化向量

## LangChain 中的 Embedding

### 使用 OpenAI Embedding

```typescript
import { OpenAIEmbeddings } from '@langchain/openai';

const embeddings = new OpenAIEmbeddings({
  model: 'text-embedding-3-small',
  // 可选：自定义维度
  dimensions: 512,
});

// 单条文本嵌入
const docEmbedding = await embeddings.embedQuery('什么是RAG？');

// 批量嵌入
const docEmbeddings = await embeddings.embedDocuments([
  'RAG是检索增强生成技术',
  'Embedding将文本转为向量',
  '向量数据库存储和检索嵌入向量',
]);

console.log(docEmbedding.length);      // 512
console.log(docEmbeddings.length);     // 3
console.log(docEmbeddings[0].length);  // 512
```

### 使用 HuggingFace 本地模型

```typescript
import { HuggingFaceTransformersEmbeddings } from '@langchain/community/embeddings/huggingface_transformers';

const embeddings = new HuggingFaceTransformersEmbeddings({
  model: 'BAAI/bge-small-zh-v1.5',
});

const result = await embeddings.embedQuery('检索增强生成的核心是什么？');
```

### 使用 Ollama 本地部署

```typescript
import { OllamaEmbeddings } from '@langchain/ollama';

const embeddings = new OllamaEmbeddings({
  model: 'nomic-embed-text', // 或 mxbai-embed-large
  baseUrl: 'http://localhost:11434',
});

const result = await embeddings.embedQuery('本地部署的嵌入模型');
```

## Embedding 质量评估

### 内在评估

```typescript
// 语义相似度对测试
const testPairs = [
  { a: '猫在沙发上睡觉', b: '小猫趴在沙发上', expected: 'high' },
  { a: '今天天气很好', b: '股市大跌', expected: 'low' },
  { a: '如何学习编程', b: '编程入门教程', expected: 'high' },
];

for (const pair of testPairs) {
  const [embA, embB] = await embeddings.embedDocuments([pair.a, pair.b]);
  const sim = cosineSimilarity(embA, embB);
  const pass = pair.expected === 'high' ? sim > 0.7 : sim < 0.5;
  console.log(`${pass ? '✅' : '❌'} "${pair.a}" vs "${pair.b}": ${sim.toFixed(4)}`);
}
```

### MTEB 基准

[Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard) 是评估 Embedding 模型的标准基准，涵盖：

- **分类**（Classification）
- **聚类**（Clustering）
- ** pairwise 分类**（Pair Classification）
- **重排**（Reranking）
- **检索**（Retrieval）← RAG 最关心的指标
- **语义文本相似度**（STS）

## 实践建议

### 模型选择决策树

```
需要最高精度？
├─ 是 → OpenAI text-embedding-3-large
└─ 否 → 需要本地部署？
         ├─ 是 → 中文为主？
         │       ├─ 是 → BGE-large-zh-v1.5
         │       └─ 否 → GTE-large
         └─ 否 → 预算有限？
                 ├─ 是 → text-embedding-3-small
                 └─ 否 → Cohere embed-v3
```

### 性能优化

1. **批量处理**：一次 embed 多条文本，减少 API 调用次数
2. **缓存**：相同文本的 embedding 结果缓存起来，避免重复计算
3. **降维**：高维向量可以降维存储，节省空间，牺牲少量精度
4. **量化**：float32 → float16 甚至 int8，减少内存占用

### 常见坑

1. **模型混用**：不同模型生成的向量维度和语义空间不同，**绝对不能混用**
2. **长度截断**：超过模型最大长度的文本会被截断，丢失信息
3. **语言不匹配**：用英文模型处理中文，效果会大打折扣
4. **版本锁定**：模型更新后向量可能变化，生产环境要锁定版本

## 下一步

下一篇我们将学习 **Chunking（分块）策略**——如何把长文档切成合适大小的片段，让 Embedding 和检索效果最佳。
