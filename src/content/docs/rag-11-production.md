---
title: "11 - 生产级 RAG 系统设计"
description: "从架构设计到部署运维，构建可在真实环境运行的 RAG 系统"
date: "2026-04-02"
category: rag
tags:
  - 生产部署
  - 架构设计
  - 运维
  - 生产部署
order: 11
---

## 生产 RAG vs 原型 RAG

```
原型 RAG：
  - 单机，FAISS 内存索引
  - 手动触发索引更新
  - 无监控，无日志
  - 没有评估体系
  - 一个 Prompt 打天下

生产 RAG：
  - 分布式，Milvus/Qdrant 集群
  - 自动增量索引 + 版本管理
  - 全链路监控 + 告警
  - 自动化评估 + A/B测试
  - 查询路由 + 多策略组合
```

## 系统架构

### 整体架构图

```
                    ┌──────────────┐
                    │   API 网关   │
                    │  (Nginx/CDN) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   RAG 服务   │
                    │  (Node.js)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──┐  ┌─────▼─────┐  ┌──▼──────┐
       │ 查询分析 │  │  检索引擎  │  │ 生成引擎 │
       │         │  │           │  │         │
       │ 意图识别 │  │ 混合检索  │  │ Prompt  │
       │ 查询改写 │  │ Re-ranking│  │ LLM     │
       │ 路由分发 │  │ 缓存层    │  │ 后处理  │
       └─────────┘  └─────┬─────┘  └─────────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
       ┌──────▼──┐  ┌────▼────┐  ┌──▼──────┐
       │ 向量库   │  │ 关键词库 │  │ 对象存储 │
       │ Milvus  │  │ Elastic │  │ 文档原文 │
       └─────────┘  └─────────┘  └─────────┘
```

### 索引 Pipeline

```
文档源                    处理                      索引
┌──────┐           ┌───────────┐           ┌──────────────┐
│ 文档库 │──→ 消费队列 → 文档解析 → 分块 → Embedding → 写入向量库
│ S3   │           └───────────┘           └──────────────┘
│ 数据库│                  │
└──────┘                   ↓
                    ┌───────────┐
                    │ 索引版本管理│
                    │ 元数据同步  │
                    └───────────┘
```

## 关键设计决策

### 1. 索引管理

```typescript
// 索引版本管理
interface IndexVersion {
  version: string;       // v1.2.3
  createdAt: Date;
  documentCount: number;
  chunkCount: number;
  embeddingModel: string;
  status: 'building' | 'active' | 'deprecated';
}

export class IndexManager {
  private activeVersion: IndexVersion | null = null;
  
  // 蓝绿部署：新索引建好后原子切换
  async rebuildIndex(newDocs: Document[]): Promise<void> {
    const newVersion = `v${Date.now()}`;
    
    // 1. 构建新索引（不影响线上）
    const newIndex = await this.buildIndex(newDocs, newVersion);
    
    // 2. 验证新索引
    const validationResult = await this.validateIndex(newIndex);
    if (!validationResult.pass) {
      throw new Error(`索引验证失败: ${validationResult.reason}`);
    }
    
    // 3. 原子切换
    await this.atomicSwap(newVersion);
    
    // 4. 清理旧索引
    await this.cleanupOldVersions(keepLatest = 2);
  }
  
  // 增量更新
  async incrementalUpdate(
    addedDocs: Document[],
    deletedDocIds: string[],
  ): Promise<void> {
    // 只更新变化的部分，不全量重建
    const vectorStore = await this.getActiveVectorStore();
    
    // 删除
    if (deletedDocIds.length > 0) {
      await vectorStore.delete({ ids: deletedDocIds });
    }
    
    // 新增
    if (addedDocs.length > 0) {
      const chunks = await this.splitAndEmbed(addedDocs);
      await vectorStore.addDocuments(chunks);
    }
  }
}
```

### 2. 缓存层

```typescript
import { Redis } from 'ioredis';

const redis = new Redis(process.env.REDIS_URL);

export class RAGCache {
  // 语义缓存：相似查询复用结果
  async get(query: string, queryEmbedding: number[]): Promise<CachedResult | null> {
    // 方式1：精确缓存（相同查询）
    const exactKey = `rag:exact:${this.hashQuery(query)}`;
    const exact = await redis.get(exactKey);
    if (exact) return JSON.parse(exact);
    
    // 方式2：语义缓存（相似查询）
    // 在缓存向量库中查找相似查询
    const similarQuery = await this.findSimilarCachedQuery(queryEmbedding);
    if (similarQuery && similarQuery.similarity > 0.95) {
      return similarQuery.result;
    }
    
    return null;
  }
  
  async set(query: string, result: RAGResult, ttl = 3600): Promise<void> {
    const key = `rag:exact:${this.hashQuery(query)}`;
    await redis.setex(key, ttl, JSON.stringify(result));
  }
  
  private hashQuery(query: string): string {
    // 标准化后hash
    const normalized = query.trim().toLowerCase();
    const { createHash } = await import('crypto');
    return createHash('md5').update(normalized).digest('hex');
  }
}
```

### 3. 查询路由

```typescript
export class QueryRouter {
  private classifiers: Map<string, QueryClassifier>;
  
  async route(query: string): Promise<RoutedQuery> {
    // 意图识别
    const intent = await this.classifyIntent(query);
    
    // 根据意图选择检索策略
    switch (intent) {
      case 'factual':
        return {
          strategy: 'hybrid',
          topK: 5,
          useReranking: true,
          promptTemplate: 'fact-based',
        };
      case 'analytical':
        return {
          strategy: 'multi-query',
          topK: 10,
          useReranking: true,
          promptTemplate: 'analytical',
        };
      case 'conversational':
        return {
          strategy: 'hybrid',
          topK: 5,
          useReranking: false,
          promptTemplate: 'conversational',
          includeHistory: true,
        };
      case 'code':
        return {
          strategy: 'keyword-heavy', // 代码搜索偏关键词
          topK: 3,
          useReranking: false,
          promptTemplate: 'code-assist',
        };
      default:
        return {
          strategy: 'vector',
          topK: 5,
          useReranking: true,
          promptTemplate: 'general',
        };
    }
  }
}
```

### 4. 监控与可观测性

```typescript
export class RAGMonitor {
  // 记录每次查询的全链路数据
  async recordQuery(trace: RAGTrace) {
    await this.store.insert({
      queryId: trace.queryId,
      timestamp: new Date(),
      
      // 查询阶段
      queryText: trace.query,
      queryIntent: trace.intent,
      queryLatency: trace.intentLatency,
      
      // 检索阶段
      retrievalStrategy: trace.strategy,
      retrievalLatency: trace.retrievalLatency,
      candidateCount: trace.candidateCount,
      topKResults: trace.topKResults,
      rerankingLatency: trace.rerankingLatency,
      
      // 生成阶段
      generationLatency: trace.generationLatency,
      totalLatency: trace.totalLatency,
      tokenUsage: trace.tokenUsage,
      
      // 质量指标
      contextRelevance: trace.contextRelevance,
      faithfulness: trace.faithfulness,
      
      // 用户反馈
      userFeedback: trace.userFeedback, // 👍/👎
    });
  }
  
  // 告警规则
  checkAlerts() {
    // 检索召回率下降
    // 延迟突增
    // LLM幻觉率上升
    // 用户负反馈率上升
  }
}
```

### 5. 成本控制

```typescript
export class CostController {
  private budget: DailyBudget;
  
  // Token 用量控制
  estimateTokenUsage(query: RoutedQuery): number {
    const retrievalTokens = query.topK * 500; // 每个chunk约500 tokens
    const promptTokens = 200;  // prompt模板
    const outputTokens = 800;  // 预估输出
    
    return retrievalTokens + promptTokens + outputTokens;
  }
  
  // 降级策略
  getDegradedStrategy(currentLoad: number): RoutedQuery {
    if (currentLoad > this.budget.threshold.high) {
      return {
        strategy: 'vector',    // 去掉BM25，减少延迟
        topK: 3,               // 减少检索量
        useReranking: false,   // 关闭re-ranking
        promptTemplate: 'concise', // 用更短的prompt
      };
    }
    // ... 更多降级级别
  }
}
```

## 部署方案

### Docker Compose（中小规模）

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "3000:3000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MILVUS_URL=milvus:19530
      - REDIS_URL=redis:6379
    depends_on:
      - milvus
      - redis

  milvus:
    image: milvusdb/milvus:v2.4-latest
    ports:
      - "19530:19530"
    volumes:
      - milvus-data:/var/lib/milvus

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    # Milvus依赖etcd

volumes:
  milvus-data:
```

### Kubernetes（大规模）

```yaml
# rag-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-service
  template:
    spec:
      containers:
        - name: rag-api
          image: your-registry/rag-service:latest
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
          env:
            - name: MILVUS_URL
              value: "milvus-service:19530"
          readinessProbe:
            httpGet:
              path: /health
              port: 3000
          livenessProbe:
            httpGet:
              path: /health
              port: 3000
---
apiVersion: v1
kind: Service
metadata:
  name: rag-service
spec:
  selector:
    app: rag-service
  ports:
    - port: 80
      targetPort: 3000
```

## 安全考虑

### 访问控制

```typescript
// 文档级权限过滤
async function queryWithPermission(
  query: string,
  userId: string,
  userGroups: string[],
) {
  // 在检索时加入权限过滤
  const results = await vectorStore.similaritySearch(query, 20, {
    // 只检索用户有权访问的文档
    $or: [
      { 'access.public': true },
      { 'access.groups': { $in: userGroups } },
      { 'access.users': userId },
    ],
  });
  
  return results;
}
```

### 数据安全

```
敏感信息处理：
1. PII 脱敏 → 文档入库前脱敏处理
2. 加密存储 → 向量和文档加密存储
3. 传输加密 → TLS/HTTPS
4. 日志脱敏 → 查询和回答中的敏感信息不记日志
5. 审计追踪 → 记录所有查询操作
```

## 性能基准参考

| 规模 | 文档数 | 检索延迟 | 端到端延迟 | 硬件 |
|------|-------|---------|-----------|------|
| 小型 | 1万 | <50ms | <2s | 单机 8C16G |
| 中型 | 100万 | <100ms | <3s | 3节点集群 |
| 大型 | 1000万 | <200ms | <5s | 10+节点 |
| 超大 | 1亿 | <500ms | <8s | 分布式集群 |

## 扩展阅读与下一步

恭喜你完成了 RAG 知识库的全部学习！你现在应该能够：

1. 理解 RAG 的核心原理和演进路线
2. 选择合适的 Embedding 模型和分块策略
3. 使用向量数据库存储和检索
4. 实现混合检索 + Re-ranking
5. 用 LangChain 构建完整的 RAG Pipeline
6. 处理多模态内容
7. 评估和持续优化系统
8. 设计生产级架构

**持续学习资源**：
- [LangChain 官方文档](https://js.langchain.com/)
- [RAGAS 评估框架](https://docs.ragas.io/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Milvus 文档](https://milvus.io/docs)
- [LlamaIndex](https://docs.llamaindex.ai/) — 另一个流行的 RAG 框架
