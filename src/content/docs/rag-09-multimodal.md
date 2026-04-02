---
title: "09 - 多模态 RAG：处理图片、表格与代码"
description: "扩展 RAG 系统处理非文本内容的能力，涵盖图片理解、表格提取、代码检索等场景"
date: "2026-04-02"
category: rag
tags:
  - 多模态
  - 图片
  - 表格
  - 框架实战
order: 9
---

## 为什么需要多模态 RAG？

真实世界的知识不只是纯文本。技术文档里有图表、架构图，论文里有公式和表格，代码仓库里有源码。传统 RAG 只处理文本，丢失了大量信息。

```
多模态 RAG 要处理的内容类型：

📄 文本：段落、标题、列表
🖼️ 图片：流程图、架构图、截图
📊 表格：对比表、数据表
💻 代码：源码、配置文件
📐 公式：数学公式、LaTeX
```

## 多模态 RAG 三种架构

### 架构1：统一嵌入（Unified Embedding）

把所有模态转成文本描述，再用统一的 embedding 模型：

```
图片 → [VLM] → 文本描述 → [Embedding] → 向量
表格 → [格式化] → Markdown → [Embedding] → 向量
代码 → [AST解析] → 摘要文本 → [Embedding] → 向量
```

**优点**：架构简单，复用现有向量库
**缺点**：信息有损（图片细节、代码结构会丢失）

### 架构2：多模态嵌入（Multimodal Embedding）

使用原生支持多模态的 embedding 模型：

```
图片 → [CLIP/ImageBind] → 向量（和文本同一空间）
文本 → [CLIP/ImageBind] → 向量
```

**优点**：跨模态语义对齐好
**缺点**：模型选择少，精度可能不如纯文本模型

### 架构3：分模态处理（Modality-Specific）

每种模态用独立的 pipeline：

```
文本 → 文本Embedding → 文本向量库
图片 → 图片Embedding → 图片向量库
代码 → 代码Embedding → 代码向量库

查询时：并行检索各向量库 → 合并结果
```

**优点**：每个模态用最优方案
**缺点**：架构复杂

## 图片处理

### 方案1：VLM 图片描述

用视觉语言模型（VLM）把图片转成文本：

```typescript
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';
import * as fs from 'fs';

const vlm = new ChatOpenAI({ model: 'gpt-4o' });

async function describeImage(imagePath: string): Promise<string> {
  const imageData = fs.readFileSync(imagePath);
  const base64 = imageData.toString('base64');
  const mimeType = imagePath.endsWith('.png') ? 'image/png' : 'image/jpeg';
  
  const response = await vlm.invoke([
    new HumanMessage({
      content: [
        {
          type: 'image_url',
          image_url: {
            url: `data:${mimeType};base64,${base64}`,
          },
        },
        {
          type: 'text',
          text: '详细描述这张图片的内容。如果包含流程图，描述每一步；如果包含架构图，描述每个组件和连接关系。用中文描述。',
        },
      ],
    }),
  ]);
  
  return response.content as string;
}

// 处理文档中的所有图片
async function processDocumentImages(
  imageDir: string,
  outputDir: string
) {
  const images = fs.readdirSync(imageDir)
    .filter(f => f.endsWith('.png') || f.endsWith('.jpg'));
  
  for (const img of images) {
    const description = await describeImage(path.join(imageDir, img));
    
    // 保存描述为markdown文件
    const outputPath = path.join(outputDir, `${img}.md`);
    fs.writeFileSync(outputPath, `# ${img}\n\n${description}`);
    
    console.log(`处理完成: ${img} → ${outputPath}`);
  }
}
```

### 方案2：多模态 Embedding（CLIP）

```typescript
import { OpenAIEmbeddings } from '@langchain/openai';

// OpenAI的clip兼容模型
const embeddings = new OpenAIEmbeddings({
  model: 'text-embedding-3-small',
});

// 图片先用VLM转文本，再embedding
// 或直接使用支持图片输入的模型
```

## 表格处理

### 方案1：Markdown 格式化

```typescript
// 从PDF/HTML提取表格并转为Markdown
function tableToMarkdown(table: string[][]): string {
  if (table.length === 0) return '';
  
  const header = table[0];
  const separator = header.map(() => '---');
  const rows = table.slice(1);
  
  const lines = [
    '| ' + header.join(' | ') + ' |',
    '| ' + separator.join(' | ') + ' |',
    ...rows.map(row => '| ' + row.join(' | ') + ' |'),
  ];
  
  return lines.join('\n');
}

// 示例
const tableData = [
  ['模型', '维度', '中文支持'],
  ['bge-large-zh', '1024', '✅'],
  ['text-embedding-3', '1536', '✅'],
];

console.log(tableToMarkdown(tableData));
// | 模型 | 维度 | 中文支持 |
// | --- | --- | --- |
// | bge-large-zh | 1024 | ✅ |
// | text-embedding-3 | 1536 | ✅ |
```

### 方案2：自然语言摘要

```typescript
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';

const llm = new ChatOpenAI({ model: 'gpt-4o-mini' });

async function summarizeTable(tableMarkdown: string): Promise<string> {
  const response = await llm.invoke([
    new HumanMessage(`将以下表格内容用自然语言摘要，保留关键数据和对比关系：

${tableMarkdown}

摘要：`),
  ]);
  
  return response.content as string;
}
```

### 方案3：表格 + 摘要组合

```typescript
// 最佳实践：同时保留结构化表格和自然语言摘要
function createTableDocument(
  tableMarkdown: string,
  summary: string,
  metadata: Record<string, any>
): Document {
  return new Document({
    pageContent: `表格摘要：${summary}\n\n原始表格：\n${tableMarkdown}`,
    metadata: {
      ...metadata,
      docType: 'table',
    },
  });
}
```

## 代码处理

### 代码分块策略

代码不能用普通的文本分块——必须尊重代码结构：

```typescript
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

const codeSplitter = RecursiveCharacterTextSplitter.fromLanguage('typescript', {
  chunkSize: 1000,
  chunkOverlap: 100,
});

const codeChunks = await codeSplitter.splitText(`
export class RAGPipeline {
  private config: RAGConfig;
  
  constructor(config: RAGConfig) {
    this.config = config;
  }
  
  async query(question: string) {
    const docs = await this.retrieve(question);
    return this.generate(question, docs);
  }
  
  private async retrieve(query: string) {
    // ...
  }
}
`);
```

### 代码语义增强

```typescript
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';

const llm = new ChatOpenAI({ model: 'gpt-4o-mini' });

async function enrichCode(code: string, filePath: string): Promise<string> {
  const response = await llm.invoke([
    new HumanMessage(`分析以下代码文件（${filePath}），生成：
1. 一句话功能描述
2. 导出的主要函数/类列表
3. 依赖关系

代码：
${code}`),
  ]);
  
  // 组合：摘要 + 原始代码
  return `${response.content}\n\n--- 原始代码 ---\n${code}`;
}
```

## 完整多模态 RAG Pipeline

```typescript
// src/multimodal-rag.ts
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { OpenAIEmbeddings } from '@langchain/openai';
import { FAISS } from '@langchain/community/vectorstores/faiss';
import { Document } from '@langchain/core/documents';

interface MultimodalDoc {
  type: 'text' | 'image' | 'table' | 'code';
  content: string;      // 处理后的文本
  rawContent?: string;   // 原始内容
  metadata: Record<string, any>;
}

export class MultimodalRAG {
  private embeddings = new OpenAIEmbeddings();
  private vectorStore?: FAISS;
  
  async index(documents: MultimodalDoc[]) {
    const allChunks: Document[] = [];
    
    for (const doc of documents) {
      let chunks: Document[];
      
      switch (doc.type) {
        case 'text':
          chunks = await this.splitText(doc);
          break;
        case 'code':
          chunks = await this.splitCode(doc);
          break;
        case 'image':
        case 'table':
          // 图片和表格已经处理成文本，直接用
          chunks = [new Document({
            pageContent: doc.content,
            metadata: { ...doc.metadata, type: doc.type },
          })];
          break;
      }
      
      allChunks.push(...chunks);
    }
    
    this.vectorStore = await FAISS.fromDocuments(allChunks, this.embeddings);
  }
  
  private async splitText(doc: MultimodalDoc): Promise<Document[]> {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });
    return splitter.splitDocuments([
      new Document({ pageContent: doc.content, metadata: doc.metadata }),
    ]);
  }
  
  private async splitCode(doc: MultimodalDoc): Promise<Document[]> {
    const ext = doc.metadata.filePath?.split('.').pop() || 'typescript';
    const splitter = RecursiveCharacterTextSplitter.fromLanguage(
      ext as any,
      { chunkSize: 1000, chunkOverlap: 100 }
    );
    return splitter.splitDocuments([
      new Document({ pageContent: doc.content, metadata: doc.metadata }),
    ]);
  }
  
  async query(question: string, topK = 5) {
    if (!this.vectorStore) throw new Error('请先调用 index()');
    
    const docs = await this.vectorStore.similaritySearch(question, topK);
    
    // 按类型分组展示
    const grouped = {
      text: docs.filter(d => d.metadata.type === 'text'),
      image: docs.filter(d => d.metadata.type === 'image'),
      table: docs.filter(d => d.metadata.type === 'table'),
      code: docs.filter(d => d.metadata.type === 'code'),
    };
    
    return { docs, grouped };
  }
}
```

## 多模态检索的特殊考虑

### 查询路由

```typescript
// 根据查询内容判断应该检索哪些模态
function routeQuery(query: string): string[] {
  const modalities: string[] = ['text']; // 总是包含文本
  
  if (query.includes('图') || query.includes('架构') || query.includes('流程')) {
    modalities.push('image');
  }
  if (query.includes('代码') || query.includes('函数') || query.includes('API')) {
    modalities.push('code');
  }
  if (query.includes('对比') || query.includes('数据') || query.includes('表格')) {
    modalities.push('table');
  }
  
  return modalities;
}
```

### 跨模态对齐

确保图片描述和正文内容的引用关系不丢失：

```typescript
// 给图片chunk加上它在原文中的位置信息
const imageDoc = new Document({
  pageContent: imageDescription,
  metadata: {
    type: 'image',
    sourcePage: 5,              // 在原文的第5页
    surroundingText: paragraph, // 图片周围的段落文本
    caption: '图3.1 RAG架构图',
  },
});
```

## 下一步

下一篇学习**RAG 系统评估**——如何量化评估你的 RAG 系统效果，建立持续优化的闭环。
