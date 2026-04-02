---
title: "10 - RAG 系统评估"
description: "掌握 RAG 系统的评估框架、关键指标、自动化测试方法与持续优化策略"
date: "2026-04-02"
category: rag
tags:
  - 评估
  - RAGAS
  - 质量保障
  - 生产部署
order: 10
---

## 为什么评估很重要？

RAG 系统不是"能跑就行"的——它有太多环节可能出问题：

```
文档质量差 → 分块不合理 → Embedding不准 → 检索不到
→ Re-ranking失效 → Prompt不好 → LLM幻觉 → 垃圾回答
```

没有评估，你只能凭感觉调参，永远不知道改了之后是变好还是变差。

## RAG 评估框架：RAG Triad

RAG 系统的三个核心评估维度：

```
1. 上下文相关性（Context Relevance）
   检索到的文档和查询有多相关？
   → 评估检索质量

2. 答案忠实度（Faithfulness / Groundedness）
   回答是否严格基于检索到的上下文？
   → 评估是否产生幻觉

3. 答案相关性（Answer Relevance）
   回答是否真正回答了用户的问题？
   → 评估生成质量
```

### RAG Triad 可视化

```
           用户查询
          /       \
    上下文相关?    答案相关?
    (检索质量)      (回答质量)
         |          |
    检索到的文档 → LLM → 回答
              忠实度?
           (幻觉检测)
```

## 评估指标详解

### 1. 上下文相关性（Context Relevance）

检索回来的文档是否和问题相关？

```typescript
// 人工标注
interface ContextRelevanceSample {
  query: string;
  retrievedDocs: string[];
  relevanceLabels: number[]; // 每个文档的相关性：0=无关，1=部分相关，2=高度相关
}

// 自动评估（用LLM做Judge）
async function evaluateContextRelevance(
  query: string,
  docs: string[],
  judgeModel: ChatOpenAI,
): Promise<number[]> {
  const scores: number[] = [];
  
  for (const doc of docs) {
    const response = await judgeModel.invoke([{
      role: 'user',
      content: `评估以下检索到的文档和查询的相关性。

查询：${query}
文档：${doc}

评分标准：
1 - 无关
2 - 部分相关
3 - 高度相关

只返回分数数字。`,
    }]);
    
    scores.push(parseInt(response.content as string));
  }
  
  return scores;
}
```

**计算指标**：
- **Precision@K**：Top-K 中相关文档的比例
- **Recall@K**：所有相关文档中被检索到的比例
- **MRR**：第一个相关文档的排名倒数
- **nDCG@K**：考虑排序位置的加权指标

### 2. 忠实度（Faithfulness）

回答是否基于上下文，没有编造？

```typescript
interface FaithfulnessSample {
  question: string;
  context: string;
  answer: string;
}

async function evaluateFaithfulness(
  sample: FaithfulnessSample,
  judgeModel: ChatOpenAI,
): Promise<{ score: number; hallucinations: string[] }> {
  const response = await judgeModel.invoke([{
    role: 'user',
    content: `判断以下回答是否完全基于给定的上下文。

上下文：${sample.context}
回答：${sample.answer}

任务：
1. 找出回答中所有不在上下文中的声明（幻觉）
2. 给出忠实度评分（0-1）

JSON格式输出：
{
  "score": 0.0到1.0,
  "hallucinations": ["幻觉声明1", "幻觉声明2"]
}`,
  }]);
  
  return JSON.parse(response.content as string);
}
```

### 3. 答案相关性（Answer Relevance）

回答是否真正解决了用户的问题？

```typescript
async function evaluateAnswerRelevance(
  question: string,
  answer: string,
  judgeModel: ChatOpenAI,
): Promise<number> {
  const response = await judgeModel.invoke([{
    role: 'user',
    content: `评估以下回答与问题的相关性。

问题：${question}
回答：${answer}

评分标准：
1 - 完全不相关
2 - 部分相关
3 - 相关但不够完整
4 - 完全相关且完整
5 - 超出预期，额外提供了有价值的信息

只返回分数。`,
  }]);
  
  return parseInt(response.content as string);
}
```

## RAGAS 框架

RAGAS（Retrieval Augmented Generation Assessment）是最流行的 RAG 评估框架：

```typescript
// RAGAS 的核心指标
interface RAGASMetrics {
  // 上下文精确度：检索到的相关文档在Top-K中的比例
  contextPrecision: number;
  
  // 上下文召回率：所有应该检索到的文档中，实际被检索到的比例
  contextRecall: number;
  
  // 忠实度：回答中声明的忠实程度
  faithfulness: number;
  
  // 答案相关性：回答与问题的相关程度
  answerRelevance: number;
  
  // 答案语义相似度：生成答案与参考答案的语义相似度
  answerSimilarity: number;
}

// 模拟 RAGAS 评估流程
async function ragasEvaluate(
  testSet: TestSample[],
  ragPipeline: RAGPipeline,
  judgeModel: ChatOpenAI,
): Promise<RAGASMetrics> {
  let totalContextPrecision = 0;
  let totalContextRecall = 0;
  let totalFaithfulness = 0;
  let totalAnswerRelevance = 0;
  
  for (const sample of testSet) {
    // 运行 RAG pipeline
    const result = await ragPipeline.query(sample.question);
    
    // 评估每个维度
    const ctxRel = await evaluateContextRelevance(
      sample.question, result.context, judgeModel
    );
    const faith = await evaluateFaithfulness({
      question: sample.question,
      context: result.context,
      answer: result.answer,
    }, judgeModel);
    const ansRel = await evaluateAnswerRelevance(
      sample.question, result.answer, judgeModel
    );
    
    totalContextPrecision += avg(ctxRel);
    totalFaithfulness += faith.score;
    totalAnswerRelevance += ansRel;
  }
  
  const n = testSet.length;
  return {
    contextPrecision: totalContextPrecision / n,
    contextRecall: totalContextRecall / n,
    faithfulness: totalFaithfulness / n,
    answerRelevance: totalAnswerRelevance / n,
    answerSimilarity: 0, // 需要参考答案
  };
}
```

## 测试数据集构建

### 方式1：人工标注

```typescript
interface TestSample {
  question: string;       // 用户问题
  referenceAnswer: string; // 标准答案（人工编写）
  relevantDocIds: string[]; // 相关文档ID（人工标注）
  difficulty: 'easy' | 'medium' | 'hard';
  category: string;        // 问题类别
}

const testSet: TestSample[] = [
  {
    question: 'RAG和微调的区别是什么？',
    referenceAnswer: 'RAG通过外部检索增强回答质量，不需要修改模型参数...',
    relevantDocIds: ['doc-rag-vs-finetune'],
    difficulty: 'easy',
    category: '基础概念',
  },
  {
    question: '在生产环境中，如何平衡检索延迟和精度？',
    referenceAnswer: '可以通过分层检索策略：先用ANN快速召回Top-50...',
    relevantDocIds: ['doc-production-design', 'doc-retrieval-strategy'],
    difficulty: 'hard',
    category: '生产部署',
  },
];
```

### 方式2：LLM 生成测试集

```typescript
async function generateTestSet(
  documents: Document[],
  model: ChatOpenAI,
  numQuestions = 20,
): Promise<TestSample[]> {
  const testSamples: TestSample[] = [];
  
  for (let i = 0; i < numQuestions; i++) {
    const doc = documents[Math.floor(Math.random() * documents.length)];
    
    const response = await model.invoke([{
      role: 'user',
      content: `基于以下文档内容，生成一个测试问题和标准答案。

文档内容：
${doc.pageContent}

JSON格式输出：
{
  "question": "具体的问题",
  "referenceAnswer": "基于文档内容的标准答案",
  "difficulty": "easy/medium/hard"
}`,
    }]);
    
    const parsed = JSON.parse(response.content as string);
    testSamples.push({
      ...parsed,
      relevantDocIds: [doc.metadata.id],
      category: doc.metadata.source,
    });
  }
  
  return testSamples;
}
```

## 端到端评估 Pipeline

```typescript
// src/evaluate.ts
export class RAGEvaluator {
  constructor(
    private pipeline: RAGPipeline,
    private judgeModel: ChatOpenAI,
  ) {}
  
  async evaluate(testSet: TestSample[]) {
    const results = [];
    
    for (const sample of testSet) {
      const start = Date.now();
      const ragResult = await this.pipeline.query(sample.question);
      const latency = Date.now() - start;
      
      // 各维度评估
      const [contextRel, faithfulness, answerRel] = await Promise.all([
        this.evalContextRelevance(sample, ragResult),
        this.evalFaithfulness(sample, ragResult),
        this.evalAnswerRelevance(sample, ragResult),
      ]);
      
      results.push({
        question: sample.question,
        latency,
        contextRelevance: contextRel,
        faithfulness,
        answerRelevance: answerRel,
        overallScore: (contextRel + faithfulness + answerRel) / 3,
      });
    }
    
    return this.aggregateResults(results);
  }
  
  private aggregateResults(results: any[]) {
    return {
      totalTests: results.length,
      avgLatency: avg(results.map(r => r.latency)),
      avgContextRelevance: avg(results.map(r => r.contextRelevance)),
      avgFaithfulness: avg(results.map(r => r.faithfulness)),
      avgAnswerRelevance: avg(results.map(r => r.answerRelevance)),
      avgOverallScore: avg(results.map(r => r.overallScore)),
      worstCases: results.sort((a, b) => a.overallScore - b.overallScore).slice(0, 5),
    };
  }
}
```

## 持续优化循环

```
评估 → 发现问题 → 定位瓶颈 → 优化 → 再评估 → ...

评估示例报告：
┌─────────────────────────────┐
│ RAG 系统评估报告 v1.2       │
│                             │
│ 上下文相关性: 0.78 ↑ (+0.05)│
│ 忠实度:      0.91 → (稳定)  │
│ 答案相关性:  0.82 ↑ (+0.03)│
│ 平均延迟:    1.2s  ↓ (优化) │
│                             │
│ 瓶颈分析：                  │
│ - 上下文相关性最低的5个问题  │
│ - 延迟最高的3个查询         │
│ - 忠实度<0.8的4个案例       │
│                             │
│ 优化建议：                  │
│ 1. 增加混合检索的BM25权重   │
│ 2. 调整chunkSize从500→800  │
│ 3. 添加Re-ranking层        │
└─────────────────────────────┘
```

## 下一步

最后一篇我们学习**生产级 RAG 系统设计**——把评估、监控、扩展性都考虑进去，构建可以在真实环境中运行的系统。
