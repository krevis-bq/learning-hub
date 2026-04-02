---
title: "14 - RAG 緺与与反模式"
description: "了解生产中最容易犯的错误、如何避免， 以及 Self-RAG、 GraphRAG 癷新架构解析"
date: "2026-04-02"
category: rag
tags:
  - 反模式
  - Self-RAG
  - GraphRAG
  - 陷阱
order: 14
---

## 10 个最常见的 RAG 锻弯与反模式
\ 综合多篇论文和实战经验总结

### 1. 只用 Embedding 潲傻用
用大模型做 embedding， 但维度可能不匹配, 换了模型后向量维度变了。
 生产环境锁定 embedding 模型版本, 不要随便升级。
```typescript
// 锠品： 同时使用 text-embedding-3-small
// 2023年版本: text-embedding-3-small → 1536维
// 2024年新版本: text-embedding-3-large → 3072维
// 你的文档是 1536维
// 维度不匹配！ 需要重建索引
```
**修复**： 在模型配置中锁定版本号，不要在生产环境自动升级 embedding 模型。

### 2. 忽略元数据质量
分块时不丢失了文档来源、 作者信息:
```typescript
// 文档分块时保留完整元数据
const chunks = splitter.splitDocuments(rawDocs);

chunks.forEach(chunk => {
  // 确保每个 chunk 都有 source 元数据
  if (!chunk.metadata.source) {
    console.error(`文档丢失来源信息: ${chunk.pageContent.slice(0, 50)});
  }
});
```
**修复**: 在分块逻辑中强制要求每个 chunk 携带 source、 metadata。

### 3. 检索结果不重新排序
初次检索结果可能有高度重复的文档,没有经过去重直接用。
```typescript
// 磀索后对结果去重
const uniqueResults = new Map<string, Document>();
const seen = new Set<string>();
for (const doc of initialResults) {
  const key = `${doc.metadata.source}|${doc.pageContent.slice(0, 30)}`;
  if (seen.has(key)) continue;
  seen.add(key);
  uniqueResults.set(key, doc);
}
return Array.from(uniqueResults.values());
```
### 4. 忽视评估
没有评估体系 = 无法量化优化效果， 不知道改了之后是变好还是变差。

```typescript
// 自动评估 pipeline
interface TestSample {
  question: string;
  expectedRelevantSources: string[];
  expectedKeywords: string[];
}

async function autoEvaluate(result: RAGResult, sample: TestSample) {
  const hasRelevantSource = result.sources.some(s =>
    sample.expectedRelevantSources.includes(s)
  );
  const hasKeywords = sample.expectedKeywords.some(kw =>
    result.answer.includes(kw)
  );
  
  return {
    relevantSourceFound: hasRelevantSource,
    keywordsFound: hasKeywords,
    overallPass: hasRelevantSource && hasKeywords,
  };
}
```
### 5. 只用 LLM 壚检索到内容
让 LLM 对检索结果做过滤或但不是喂给 LLM 壀直接返回,浪费 token 和增加延迟。
```typescript
// ❌ 反模式： 把检索到的所有文档都塞给 LLM
const allContext = retrievedDocs.map(d => d.pageContent).join('\n\n');
// ✅ 正确做法: 只传相关片段
const relevantContext = rerankedDocs
  .slice(0, 3)
  .map(d => d.pageContent)
  .join('\n\n');
```
### 6. 不处理边界情况
用户问"数据库里有哪些关于RAG的文章？"检索到 0 条结果，但没有一条相关。
```typescript
// ❌ 没有处理: 直接返回"我没有找到相关信息"
// ✅ 正确做法: 回退到对话模式， 引导用户换个问法
const response = `我检索到的信息中没有直接回答您您的问题。不过我可以提供以下相关主题的信息：...您是否感兴趣？`;
```
### 7. 过度依赖 Re-ranking
每次查询都做 re-ranking，即使候选结果很少。
```typescript
// ❌ 反模式: 所有查询都做 re-ranking
// ✅ 正确做法: 只在候选 > 10 时才启用
if (retrievalResults.length > 10) {
  const reranked = await reranker.compressDocuments(retrievalResults, query);
  return reranked;
}
```
### 8. 不缓存查询结果
相似查询重复执行,浪费 embedding 计算费用。
```typescript
// ✅ 正确做法: 语义缓存
const queryEmbedding = await embeddings.embedQuery(query);
const cached = await semanticCache.get(queryEmbedding);
if (cached) return cached;

const results = await retrieve(query);
await semanticCache.set(queryEmbedding, results);
return results;
```
### 9. 粗粒度固定
所有文档用相同的 chunkSize, 不考虑内容类型。
```typescript
// ❌ 反模式: 代码和文档用同样的分块参数
const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500 });
// 对所有文档一视同仁
// ✅ 正确做法: 按内容类型调整
const splitters = {
  code: RecursiveCharacterTextSplitter.fromLanguage('typescript', { chunkSize: 1000 }),
  markdown: new MarkdownTextSplitter(),
  pdf: new RecursiveCharacterTextSplitter({ chunkSize: 800 }),
};
```
### 10. 不做增量更新
文档更新后重建整个索引,时间长成本高。
```typescript
// ❌ 反模式: 每次更新都全量重建索引
// ✅ 正确做法: 增量更新
await vectorStore.addDocuments(newDocs);
// 只添加新文档，不影响已有索引
```
## Self-RAG: 自我反思的 RAG

2024 年提出的架构，让 RAG 稡型自己判断检索到的内容是否有用。
```
用户查询 → 检索 Top-K 文档 → LLM 生成回答
                                      ↓
                              LLM 自我评估: 回答是否充分？
                                      ↓
                              不充分 → 重新检索/改写查询 → 再次生成
```
**核心思想**: 在生成回答的同时让模型评估自己的回答质量。
**优势**: 减少幻觉,自动决定是否需要更多信息。
**代价**: 鯌需额外的 LLM 调用。

## GraphRAG: 知识图谱增强
微软提出的架构,利用知识图谱捕获实体之间的关系。
```
传统 RAG: 检索文本片段 → 丢失实体关系
GraphRAG: 检索知识图谱子节点和关系 → 保留完整上下文
```
**适用场景**:
- 文档之间有复杂引用关系（如法律条文、技术标准)
- 需要多跳推理（问题涉及多个实体)
- 知识密集型领域（医疗、金融)

## 实践建议
1. **先用 Naive RAG**， 然后逐步优化, 不要一上来就用所有高级技术
2. **混合检索 + 重排序 是性价比最高的组合**， 先加这个
3. **建立评估体系**， 才知道优化是否有效
4. **缓存常见查询**， 减少 API 调用
5. **监控关键指标**: 检索延迟、回答质量、用户反馈
