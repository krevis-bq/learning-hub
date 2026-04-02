import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

import cloudflare from '@astrojs/cloudflare';

export default defineConfig({
  site: 'https://krevis-bq.github.io/learning-hub',
  integrations: [mdx()],

  markdown: {
    shikiConfig: {
      theme: 'github-dark',
    },
  },

  adapter: cloudflare(),
});