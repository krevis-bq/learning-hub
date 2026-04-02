import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const docs = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/docs' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.string(),
    category: z.string().default('uncategorized'),
    tags: z.array(z.string()).default([]),
    order: z.number().default(0),
  }),
});

export const collections = { docs };

// 文档库定义
export const categories: Record<string, { name: string; description: string; icon: string; order: number }> = {
  'rag': {
    name: 'RAG 知识库',
    description: '从零开始系统学习检索增强生成技术，涵盖原理、组件、框架实战与生产部署',
    icon: 'database',
    order: 1,
  },
  'ai-agent': {
    name: 'AI Agent 工程',
    description: '从 Claude Code 源码中提炼的 Agent 架构设计方法论与工程实践',
    icon: 'bot',
    order: 2,
  },
};
