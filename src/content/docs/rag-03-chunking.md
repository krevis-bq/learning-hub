---
title: "03 - Chunking 策略：如何切分文档"
description: "掌握文档分块的核心策略、参数调优、以及不同场景的最佳实践"
date: "2026-04-02"
category: rag
tags:
  - Chunking
  - 文档分块
  - 基础组件
order: 3
---

## 为什么需要分块？

LLM 有上下文窗口限制，不可能把整个文档库塞进去。Embedding 模型也有最大输入长度（通常512-8192 tokens）。所以必须把长文档切成小块：

```
原始文档（50000 tokens）
    ↓ 分块
[chunk1, chunk2, chunk3, ..., chunkN]（每块 200-1000 tokens）
    ↓ Embedding
[vec1, vec2, vec3, ..., vecN]
    ↓ 存入向量数据库
```

分块质量直接决定 RAG 系统的效果——**切太碎，丢失上下文；切太大，噪声太多**。

## 分块策略一览

### 1. 固定大小分块（Fixed Size Chunking）

最简单粗暴，按字符数或 token 数切割，有重叠：

```typescript
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,      // 每块最大字符数
  chunkOverlap: 50,    // 相邻块重叠字符数
  separators: ['\n\n', '\n', '。', '！', '？', '；', ' ', ''],
});

const chunks = await splitter.splitText(longDocument);
console.log(`分成 ${chunks.length} 块`);
console.log(`第一块长度: ${chunks[0].length} 字符`);
```

**优点**：简单、可预测、实现成本低
**缺点**：可能切断句子或段落，破坏语义完整性
**适用场景**：通用文档、日志、结构不规则的文本

### 2. 递归字符分块（Recursive Character Splitting）

LangChain 默认推荐的策略。按分隔符层级递归切割：

```typescript
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
  separators: [
    '\n## ',     // 先尝试按二级标题切
    '\n### ',    // 再按三级标题
    '\n\n',      // 按段落
    '\n',        // 按行
    '。',        // 按句号
    ' ',         // 按空格
    '',          // 最后按字符
  ],
});

// 处理文档并保留元数据
const docs = await splitter.splitDocuments([
  {
    pageContent: markdownContent,
    metadata: { source: 'rag-guide.md', author: 'Krevis' },
  },
]);

// 元数据会自动继承到每个 chunk
console.log(docs[0].metadata); // { source: 'rag-guide.md', author: 'Krevis' }
```

**工作原理**：
1. 优先用第一个分隔符切分
2. 如果某块还是太长，用下一个分隔符继续切
3. 递归直到所有块都 ≤ chunkSize

**优点**：尽量保持语义边界，通用性好
**缺点**：不是所有文本都有清晰的结构分隔符

### 3. 按标题/结构分块（Markdown / HTML Header Splitting）

利用文档本身的标题结构来分块，语义最完整：

```typescript
import { MarkdownTextSplitter } from '@langchain/textsplitters';

// 按Markdown标题层级分块
import { MarkdownHeaderTextSplitter } from '@langchain/textsplitters';

const markdownSplitter = new MarkdownHeaderTextSplitter({
  headersToSplitOn: [
    ['##', 'section'],    // 二级标题 → section 元数据
    ['###', 'subsection'], // 三级标题 → subsection 元数据
  ],
});

const mdChunks = await markdownSplitter.splitText(markdownDoc);
// 每个 chunk 的 metadata 会包含它所属的标题路径
console.log(mdChunks[0].metadata);
// { section: 'RAG 的工作流程', subsection: '离线索引阶段' }
```

```typescript
// HTML 按标签分块
import { HTMLSectionSplitter } from '@langchain/textsplitters';

const htmlSplitter = new HTMLSectionSplitter({
  tagsToSplitOn: [
    ['h1', 'h1'],
    ['h2', 'h2'],
    ['h3', 'h3'],
    ['p', 'paragraph'],
  ],
});
```

**优点**：完全保持语义完整性，元数据丰富
**缺点**：只适用于结构化文档（Markdown/HTML）
**适用场景**：技术文档、知识库、博客

### 4. 语义分块（Semantic Chunking）

根据语义相似度来决定切分点，而不是靠固定大小或分隔符：

```typescript
import { SemanticDoubleMergingSplitter } from '@langchain/textsplitters';
import { OpenAIEmbeddings } from '@langchain/openai';

const embeddings = new OpenAIEmbeddings();

const semanticSplitter = new SemanticDoubleMergingSplitter({
  embeddings,
  bufferSize: 1,           // 检测断点时的上下文窗口
  breakpointThresholdType: 'percentile', // 断点阈值类型
  breakpointThresholdAmount: 0.3,        // 阈值：相似度变化超过30%就切
});

const chunks = await semanticSplitter.splitText(text);
```

**工作原理**：
1. 把文本切成小句
2. 计算相邻句子的 embedding 相似度
3. 相似度骤降的地方就是语义断点
4. 在断点处合并相邻句子成 chunk

**优点**：语义边界最精准
**缺点**：需要额外的 embedding 计算，成本高、速度慢
**适用场景**：高质量知识库、对检索精度要求极高

### 5. 句子级分块（Sentence Splitting）

```typescript
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

const sentenceSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 300,
  chunkOverlap: 0,
  separators: ['。', '！', '？', '\n'],
});

const sentences = await sentenceSplitter.splitText(text);
```

**适用场景**：FAQ、短文本检索、需要精确定位到句子

### 6. Agentic 分块（Agentic Chunking）

用 LLM 自己决定怎么分块，最灵活但最贵：

```typescript
// 概念性代码：让 LLM 根据命题（proposition）分块
import { ChatOpenAI } from '@langchain/openai';

const llm = new ChatOpenAI({ model: 'gpt-4o-mini' });

// 步骤1：让LLM提取文本中的独立命题
const propositions = await llm.invoke([{
  role: 'user',
  content: `将以下文本拆分为独立的语义命题，每个命题一行：
  
  ${paragraph}`
}]);

// 步骤2：根据命题相关性重新组合成chunk
// 这部分逻辑需要自定义实现
```

**适用场景**：极度追求质量、不在乎成本

## 参数调优指南

### chunkSize 选择

| 场景 | 推荐 chunkSize | 理由 |
|------|---------------|------|
| FAQ/短回答 | 200-400 | 答案通常很短 |
| 技术文档 | 500-1000 | 保持完整段落 |
| 长文章/报告 | 1000-2000 | 需要足够的上下文 |
| 代码文件 | 1000-1500 | 保持函数/类完整 |

### chunkOverlap 选择

一般设为 chunkSize 的 10%-20%：

```typescript
const chunkSize = 500;
const chunkOverlap = Math.floor(chunkSize * 0.15); // 75
```

**重叠的作用**：避免关键信息正好在切割边界被截断。

### 分块 Granularity vs 检索效果

```
太细（50-100 tokens）:
  ✅ 检索精度高
  ❌ 缺乏上下文，LLM 答不好
  
合适（300-1000 tokens）:
  ✅ 精度和上下文的平衡
  ✅ 大多数场景的最佳选择

太粗（2000+ tokens）:
  ✅ 上下文丰富
  ❌ 噪声多，检索精度下降
  ❌ 浪费 token 预算
```

## 高级技巧

### 父子文档策略（Parent-Child Chunking）

小块检索，大块喂给 LLM：

```typescript
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { InMemoryStore } from 'langchain/storage/in_memory';
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

// 父文档：大块（用于LLM）
const parentSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 2000,
  chunkOverlap: 200,
});

// 子文档：小块（用于检索）
const childSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 400,
  chunkOverlap: 50,
});

// 存储父文档
const parentStore = new InMemoryStore();
// 存储子文档的向量索引
const embeddings = new OpenAIEmbeddings();

async function indexDocument(doc: string, docId: string) {
  // 切成大块
  const parentChunks = await parentSplitter.splitText(doc);
  
  const childDocs = [];
  const parentIdMap = new Map();
  
  for (let i = 0; i < parentChunks.length; i++) {
    const parentId = `${docId}-parent-${i}`;
    
    // 存储父文档
    await parentStore.mset([[parentId, parentChunks[i]]]);
    
    // 切成小块
    const children = await childSplitter.splitText(parentChunks[i]);
    for (let j = 0; j < children.length; j++) {
      const childId = `${parentId}-child-${j}`;
      childDocs.push({
        pageContent: children[j],
        metadata: { parentId, childId },
      });
      parentIdMap.set(childId, parentId);
    }
  }
  
  return { childDocs, parentIdMap };
}

// 检索时：先搜子文档，再取父文档
async function retrieveWithParent(childResults: any[]) {
  const parentIds = [...new Set(
    childResults.map(r => r.metadata.parentId)
  )];
  
  const parentTexts = await Promise.all(
    parentIds.map(id => parentStore.mget([id]))
  );
  
  return parentTexts.flat().map(t => t);
}
```

### 元数据增强（Metadata Enrichment）

给每个 chunk 加上丰富的元数据，支持过滤检索：

```typescript
const enrichedChunks = chunks.map((chunk, index) => ({
  pageContent: chunk,
  metadata: {
    source: 'knowledge-base.pdf',
    page: Math.floor(index / 5) + 1,
    section: extractSection(chunk),  // 自定义：提取所属章节
    keywords: extractKeywords(chunk), // 自定义：提取关键词
    docType: 'technical-doc',
    createdAt: new Date().toISOString(),
  },
}));
```

## 常见错误

1. **一刀切**：所有文档用同一个 chunkSize，不根据内容类型调整
2. **忽略重叠**：chunkOverlap=0，导致边界信息丢失
3. **忘记元数据**：分块后丢失文档来源、页码等信息
4. **过度追求语义分块**：语义分块成本高10倍，效果不一定好10%
5. **不分块**：把整篇文章作为一个 chunk 检索，效率极低

## 下一步

下一篇我们将进入**向量数据库**——学会如何存储和高效检索这些 embedding 向量。
