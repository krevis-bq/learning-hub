---
title: "15 - RAG 前沿趋势: Agentic RAG 与 GraphRAG"
description: "探索 RAG 的最新发展方向: Agentic RAG、 GraphRAG、 Self-RAG、 Modular RAG 等前沿架构"
date: "2026-04-02"
category: rag
tags:
  - Agentic RAG
  - GraphRAG
  - Self-RAG
  - 前沿趋势
order: 15
---

## RAG 的演进路线
RAG 技术正在快速演进。从 2023 年的 Naive RAG 到 2025 年的 Agentic RAG, 每一代都在解决上一代的核心问题:
```
Naive RAG (2023)     → 磅索慢, 幻觉多
Advanced RAG (2024)  → 磊更准, 但流程固定
Modular RAG (2024)  → 棒灵活, 但缺乏自主决策
Agentic RAG (2025) → 模型自主决定检索策略, 多轮迭代
```
## Agentic RAG
Agentic RAG 赋予 LLM Agent 自主决定"何时检索、检索什么、如何检索"的能力。
### 核心特征
- **自主决策**: 模型自己判断是否需要检索
- **工具调用**: 模型可以调用多种检索工具
- **多轮迭代**: 可以反复检索直到找到满意答案
- **动态规划**: 根据问题类型选择不同的检索策略
### 架构
```
用户查询
    ↓
LLM Agent (带工具)
    ├─ 判断: 需要检索？
    │   ├─ 是 → 选择检索工具
    │   │   ├─ 向量检索
    │   │   ├─ 关键词检索  
    │   │   ├─ 知识图谱查询
    │   │   └─ 网络搜索
    │   └─ 否 → 直接回答
    ├─ 分析检索结果
    │   ├─ 信息充足 → 生成回答
    │   └─ 信息不足 → 换个策略重新检索
    └─ 生成最终回答
```
### 代码示例
```typescript
import { ChatOpenAI } from '@langchain/openai';
import { tool } from '@langchain/core/tools';
import { HumanMessage } from '@langchain/core/messages';
import { z } from 'zod';

// 定义检索工具
const vectorSearch = tool(async ({ query }: { query: string }) => {
  const results = await vectorStore.similaritySearch(query, 5);
  return JSON.stringify(results.map(r => ({
    content: r.pageContent.slice(0, 200),
    source: r.metadata.source,
  })));
}, {
  name: 'vector_search',
  description: '在知识库中做语义检索',
  schema: z.object({ query: z.string() }),
});

const keywordSearch = tool(async ({ keywords }: { keywords: string }) => {
  // BM25 或 Elasticsearch 检索
  return '关键词检索结果...';
}, {
  name: 'keyword_search',  
  description: '用关键词精确搜索',
  schema: z.object({ keywords: z.string() }),
});

// 创建 Agent
const llm = new ChatOpenAI({ model: 'gpt-4o' }).bindTools([vectorSearch, keywordSearch]);

// Agent 自主决定检索策略
async function agenticRAG(query: string) {
  const messages = [
    new HumanMessage(`你是一个智能检索助手。根据用户的问题决定使用哪些工具来回答。
    
用户问题: ${query}

你可以使用以下工具:
- vector_search: 语义检索
- keyword_search: 关键词检索

请先用工具检索, 然后基于检索结果回答。如果检索结果不够, 可以再次检索。`
  ),
  ];

  const response = await llm.invoke(messages);
  return response.content;
}
```
## GraphRAG
微软 Research 提出的架构,利用知识图谱增强 RAG。
### 为什么需要 GraphRAG?
传统 RAG 的向量检索是"扁平的"——只能找到语义相似的片段,丢失了实体之间的关系。
```
问题: "OpenAI 的创始人和他的其他公司有什么关系？"
传统 RAG: 可能检索到"OpenAI由Sam Altman创立" 和 "Altman投资了..."
        → 但不知道这两条信息之间的"创立"和"投资"关系
GraphRAG: 检索到 Sam Altman → OpenAI → 创立
                 Sam Altman → Helion → 投资
                 OpenAI → 微软 → 合作
        → 完整的关系网络
```
### 构建知识图谱
```typescript
// Step 1: 从文档中提取实体和关系
async function extractEntities(text: string, llm: ChatOpenAI) {
  const response = await llm.invoke([{
    role: 'user',
    content: `从以下文本中提取实体和关系。
    
文本: ${text}
    
JSON格式输出:
{
  "entities": [{"name": "实体名", "type": "类型"}],
  "relations": [{"source": "实体A", "relation": "关系", "target": "实体B"}]
}`
  }]);
  
  return JSON.parse(response.content as string);
}

// Step 2: 构建图数据库
import { Neo4jGraph } from '@langchain/community/graphs/neo4j';

const graph = await Neo4jGraph.initialize({
  url: 'bolt://localhost:7687',
  username: 'neo4j',
  password: 'password',
});

// 写入实体和关系
await graph.query(`
  MERGE (a:Entity {name: $source})
  MERGE (b:Entity {name: $target})
  MERGE (a)-[:${relation}]->(b)
`, { source, target, relation });

// Step 3: 图检索 + 向量检索融合
const graphContext = await graph.query(
  'MATCH (n)-[r]->(m) WHERE n.name = $entity RETURN n, r, m',
  { entity: extractedEntity }
);
```
### 适用场景
- **多跳推理**: 问题需要综合多个文档的信息
- **关系密集型**: 法律、医疗、金融等领域
- **知识库大**: 百万级文档,图结构帮助导航
- **复杂问答**: 鶶要对比分析、趋势总结等

### 局限
- **构建成本高**: 鯌要额外的实体提取和图谱构建步骤
- **图数据库运维**: Neo4j 等图数据库的维护成本
- **实体提取质量**: 依赖 LLM 提取实体,可能有遗漏或错误
## Self-RAG
模型自己评估检索质量和回答质量。
### 流程
```
查询 → 检索 → 生成初步回答 → 自我评估
                                    ↓
                        评估通过 → 返回回答
                        评估不通过 → 补充检索 → 重新生成
```
### 评估维度
1. **检索充分性**: 检索到的信息是否足够回答问题？
2. **回答忠实度**: 回答是否基于检索到的内容？
3. **回答相关性**: 回答是否真正回答了用户的问题？

### 实现
```typescript
async function selfRAG(query: string, maxIterations: number = 3) {
  for (let i = 0; i < maxIterations; i++) {
    // 检索
    const docs = await retrieve(query);
    
    // 生成
    const answer = await generate(query, docs);
    
    // 自我评估
    const evaluation = await evaluate({
      question: query,
      context: docs.map(d => d.pageContent).join('\n'),
      answer,
    });
    
    if (evaluation.isSufficient) {
      return { answer, sources: docs, iterations: i + 1 };
    }
    
    // 改写查询
    query = await rewriteQuery(query, evaluation.missingAspects);
  }
  
  return { answer: '无法找到完整答案', iterations: maxIterations };
}
```
## CRAG (Corrective RAG)
在 Self-RAG 卉础上增加动作评估: 刯检索、不检索、还是需要纠正？
```
查询 → LLM 判断 → 检索/不检索/纠正
                    ↓
               检索结果 → 生成 → 评估 → 返回/纠正
```
## Modular RAG
模块化 RAG 把每个环节都变成可插拔的模块:
```
索引模块: 文档加载器 → 分块器 → Embedding → 向量库
检索模块: 查询改写 → 混合检索 → 重排序
生成模块: Prompt → LLM → 后处理
评估模块: 上下文相关性 → 忠实度 → 答案相关性
编排模块: 意图识别 → 模块选择 → 流程编排
```
每个模块都可以独立升级替换,不影响其他模块。
## 抣术选型建议
| 场景 | 推荐架构 |
|------|---------|
| 简单问答 | Naive RAG 或 Advanced RAG |
| 复杂推理 | Agentic RAG |
| 关系密集 | GraphRAG |
| 高质量要求 | Self-RAG + 人工审核 |
| 企业级 | Modular RAG + 评估体系 |
## 未来展望
RAG 的趋势是 **越来越智能**:
1. **更少的检索**: 模型自己判断什么值得检索
2. **更精确的检索**: 语义理解更深入,不只是关键词匹配
3. **更灵活的流程**: 动态调整检索策略
4. **更强的评估**: 自动检测幻觉和质量问题
5. **更广的数据源**: 不只是文本,图像/代码/结构化数据都能检索
