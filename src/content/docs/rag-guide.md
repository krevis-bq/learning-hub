---
title: "RAG 检索增强生成：从原理到实战"
description: "全面理解 RAG 的架构设计、核心技术与工程实践"
date: "2026-04-02"
tags:
  - RAG
  - LLM
  - Vector Database
  - LangChain
  - TypeScript
order: 2
---

## RAG 是什么？

RAG（Retrieval-Augmented Generation，检索增强生成）不是一个大模型，也不是一个产品，而是一种**架构模式**——让 LLM 在生成回答之前，先从外部知识库中检索相关内容，再结合检索结果与用户问题一起输入模型。

**为什么需要 RAG？** 因为 LLM 有三个根本性的局限：

1. **知识截止**：模型的训练数据有截止日期，它不知道昨天发生的事
2. **幻觉问题**：模型会自信地编造看起来正确但实际上错误的答案
3. **领域缺失**：模型不了解你公司的内部文档、代码库、业务逻辑

RAG 通过"检索外部知识 → 注入上下文 → 生成回答"这个流程，在不对模型进行重新训练的情况下，解决了这三个问题。

**一个类比：** 你让一个律师帮你打官司。传统 LLM 就像让律师凭记忆回答——他可能记错法条。RAG 就像给了律师一个法律数据库——他可以先查法条，再基于准确信息给你建议。

## RAG 的核心架构

一个完整的 RAG 系统由 5 个核心阶段组成：

```
用户提问
   ↓
[1] 文档处理（离线）
   ├─ 加载文档
   ├─ 文本分块（Chunking）
   ├─ 向量化（Embedding）
   └─ 存入向量数据库
   ↓
[2] 查询处理（在线）
   ├─ 用户问题向量化
   ├─ 向量相似度检索
   └─ 召回 Top-K 文档块
   ↓
[3] 上下文构建
   ├─ 检索结果重排序（Re-ranking）
   ├─ 提示词组装（Prompt Assembly）
   └─ 注入系统指令
   ↓
[4] LLM 生成
   ├─ 模型基于上下文生成回答
   └─ 引用来源标注
   ↓
[5] 答案输出
```

## 文档分块（Chunking）策略

分块是 RAG 中最被低估的环节。分块质量直接决定了检索质量。

### 固定大小分块
最简单的方式，按字符数切割：

```typescript
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,        // 每块 1000 字符
  chunkOverlap: 200,      // 块间重叠 200 字符
  separators: ["\n\n", "\n", "。", "！", "？", " "],  // 按优先级尝试分割
});

const chunks = await splitter.splitText(documentContent);
```

`RecursiveCharacterTextSplitter` 的工作原理：按分隔符优先级列表依次尝试分割——先尝试双换行（段落级），不够再用单换行（行级），再不够用句号，最后用空格。这确保了分块边界尽量落在语义自然断点。

### 语义分块
比固定大小更智能，基于语义边界分割：

```typescript
import { SemanticChunker } from "@langchain/experimental/text_splitters";
import { ChatOpenAI } from "@langchain/openai";

const chunker = new SemanticChunker({
  embeddings: new OpenAIEmbeddings(),
  breakpointThresholdType: "percentile",  // 使用百分位阈值
});

const chunks = await chunker.splitText(documentContent);
```

语义分块通过计算相邻句子的 embedding 相似度，在相似度骤降处断开。成本更高，但分块质量明显更好。

### 分块的最佳实践
- **重叠（Overlap）**：块间保留 10-20% 的重叠，避免关键信息被切断
- **块大小**：通常 500-1000 个 token，太小丢失上下文，太大引入噪声
- **元数据附加**：每个块附加来源、页码、章节等元数据，检索时可以过滤
- **递归分块**：对长文档先按大块分割，再对大块做细分割，构建层级索引

## 向量化（Embedding）

向量化是把文本转换为高维数值向量的过程。向量之间的距离代表语义相似度。

### 选择 Embedding 模型

| 模型 | 维度 | 特点 |
|------|------|------|
| OpenAI text-embedding-3-small | 1536 | 性价比最高，适合大多数场景 |
| OpenAI text-embedding-3-large | 3072 | 精度更高，大规模知识库推荐 |
| Cohere embed-multilingual-v3 | 1024 | 多语言支持好，中文场景推荐 |

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
  dimensions: 512,  // 可降维，减少存储成本
});

const vector = await embeddings.embedQuery("什么是 RAG？");
// 输出：[0.023, -0.015, 0.078, ...] 一个 512 维的浮点数组
```

**降维技巧：** text-embedding-3-small 默认 1536 维，但支持通过 `dimensions` 参数降到 512 维。存储成本降低 3 倍，精度损失通常不到 5%。对于大规模知识库（百万级以上），这是性价比最高的选择。

## 向量数据库选型

| 数据库 | 类型 | 适合场景 | 是否需要自建 |
|------|------|----------|-------------|
| Pinecone | 云服务 | 生产环境，零运维 | 否 |
| Supabase pgvector | 云服务 | 已有 Supabase 项目 | 否 |
| Chroma | 嵌入式 | 开发/原型 | 否 |
| FAISS | 内存库 | 高性能检索，数据量不大 | 否 |
| Qdrant | 自建 | 生产环境，需要完全控制 | 是 |
| Milvus | 自建 | 超大规模（亿级） | 是 |

### 使用 Chroma（本地开发推荐）

```typescript
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OpenAIEmbeddings } from "@langchain/openai";

// 创建向量存储
const vectorStore = await Chroma.fromDocuments(
  chunks.map(chunk => new Document({
    pageContent: chunk.content,
    metadata: chunk.metadata,
  })),
  new OpenAIEmbeddings(),
  {
    collectionName: "my-knowledge-base",
    url: "http://localhost:8000",
  }
);
```

### 使用 FAISS（纯内存，零依赖）

```typescript
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OpenAIEmbeddings } from "@langchain/openai";

const vectorStore = await FaissStore.fromDocuments(docs, new OpenAIEmbeddings());

// 保存到本地文件
await vectorStore.save("./faiss-index");

// 从文件加载
const loadedStore = await FaissStore.load("./faiss-index", new OpenAIEmbeddings());
```

## 检索策略

### 基础向量检索（Semantic Search）

```typescript
const retriever = vectorStore.asRetriever({
  k: 5,                // 召回 5 个最相似的文档块
  searchType: "mmr",   // 使用 MMR（最大边际相关性）算法
});

const docs = await retriever.invoke("用户的问题");
```

**MMR vs 癮通相似度：** 普通相似度检索会返回高度重复的结果。MMR 算法在保证相关性的同时，尽量让返回结果多样化——覆盖更多不同方面的信息。

### 混合检索（Hybrid Search）

生产环境中，纯向量检索是不够的。用户可能用完全不同的措辞问同样的问题，向量检索能处理；但用户也可能用精确的关键词搜索（如错误代码、API 名称），向量检索反而可能遗漏。

混合检索 = 向量检索 + 关键词检索（BM25），结果合并后去重排序：

```typescript
import { EnsembleRetriever } from "@langchain/core/retrievers/ensemble";
import { BM25Retriever } from "@langchain/community/retrievers/bm25";

const vectorRetriever = vectorStore.asRetriever({ k: 5 });
const bm25Retriever = BM25Retriever.fromDocuments(docs, { k: 5 });

const ensembleRetriever = new EnsembleRetriever({
  retrievers: [vectorRetriever, bm25Retriever],
  weights: [0.6, 0.4],  // 向量检索权重 60%，关键词权重 40%
});

const results = await ensembleRetriever.invoke("TypeError: Cannot read properties of undefined");
```

## 完整的 RAG 链

把上面所有部分串联起来：

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

// 1. 初始化模型
const llm = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0,  // RAG 场景建议 temperature=0，更精确
});

// 2. 构建提示词模板
const prompt = ChatPromptTemplate.fromTemplate(`
你是一个专业的问答助手。基于以下参考文档回答用户的问题。
如果你在参考文档中找不到答案，就说"我无法从提供的文档中找到答案"。
不要编造信息。每个回答都要标注信息来源。

参考文档：
{context}

用户问题： {input}
`);

// 3. 构建文档合并链
const combineDocsChain = await createStuffDocumentsChain({
  llm,
  prompt,
  documentSeparator: "\n\n---\n\n",
});

// 4. 构建完整 RAG 链
const ragChain = await createRetrievalChain({
  combineDocsChain,
  retriever: ensembleRetriever,
});

// 5. 执行查询
const response = await ragChain.invoke({
  input: "RAG 中的分块策略有哪些？",
});

console.log(response.answer);          // 模型的回答
console.log(response.context.length);   // 检索到的文档块数量
```

## Re-ranking：检索结果的二次排序

向量检索召回的文档块不一定是最相关的。生产环境中，通常会在向量检索之后加一个 Re-ranking 步骤，用更强的模型对召回结果重新打分排序。

### 使用 Cohere Reranker

```typescript
import { CohereRerank } from "@langchain/cohere";

const reranker = new CohereRerank({
  model: "rerank-v3",
  topN: 3,  // 从召回结果中选出最相关的 3 个
});

// 先用向量检索召回 20 个候选
const candidates = await vectorStore.similaritySearch(query, 20);

// 再用 Reranker 精排到 3 个
const rerankedResults = await reranker.compressDocuments(candidates, query);
```

**为什么需要 Re-ranking？** 向量检索用 embedding 相似度做粗排，速度快但不够精确。Reranker 用更强的模型（如 Cohere Rerank、Cross-Encoder）做精排，慢但更准。两步结合，兼顾速度和精度——这是"漏斗式检索"的核心思想。

## RAG 的评估

不知道 RAG 效果好不好，就没法改进。三个核心指标：

### 1. 检索质量（Retrieval Quality）
检索到的文档块中有多少是真正相关的？用 **Hit Rate** 和 **MRR（Mean Reciprocal Rank）** 衡量。

### 2. 生成忠实度（Faithfulness）
模型的回答是否忠实于检索到的文档？有没有"无中生有"？用 **Faithfulness Score** 衡量。

### 3. 答案相关性（Answer Relevance）
模型的回答是否真正回答了用户的问题？用 **Answer Relevance** 衡量。

```typescript
// 使用 RAGAS 评估框架
// ragas 是 Python 库，这里展示思路
/*
评估数据集格式：
{
  question: "什么是 RAG？",
  answer: "RAG 是检索增强生成...",
  contexts: ["RAG（Retrieval-Augmented Generation）是一种..."],
  ground_truth: "RAG 是一种结合检索和生成的技术框架"
}

指标：
- faithfulness: 0.85     （回答是否忠实于上下文）
- answer_relevancy: 0.92  （回答是否切题）
- context_precision: 0.78 （检索结果中相关文档的排名）
- context_recall: 0.88   （检索结果覆盖了多少真实答案需要的信息）
*/
```

## 生产环境的 RAG 优化清单

从原型到生产，这些是必须考虑的问题：

| 问题 | 解决方案 |
|------|----------|
| 文档更新后索引过期 | 增量索引：只重新向量化变更的文档 |
| 检索太慢 | 缓存热门查询的 embedding 结果 |
| 回答包含敏感信息 | 检索后做内容过滤（PII 脱敏） |
| 多语言混合 | 用多语言 Embedding 模型 + 查询翻译 |
| 用户问的问题太模糊 | 查询改写（Query Rewriting）：用 LLM 先把模糊问题改写成精确查询 |
| 检索结果不够全面 | 多跳检索（Multi-hop Retrieval）：先检索，基于结果再检索 |
| 成本太高 | 缓存 + 降维 embedding + 异步索引更新 |

## 从这里开始

1. **动手实践**：用 Chroma + OpenAI Embeddings 搭一个最小 RAG 原型
2. **优化检索**：实现混合检索（向量 + BM25）
3. **加入 Re-ranking**：用 Cohere Rerank 做二次排序
4. **评估和迭代**：用 RAGAS 量化评估效果，针对性优化
5. **生产部署**：增量索引、查询缓存、内容安全过滤

RAG 的核心竞争力不在于用了多好的模型，而在于**检索质量**——Garbage in, garbage out。花 80% 的精力在文档处理和检索策略上，20% 在生成侧，这个比例关系不能反。
