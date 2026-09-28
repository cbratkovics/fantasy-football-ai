/// <reference types="vitest" />
import path from 'node:path';

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

/**
 * Vitest configuration for the Next.js app.
 *
 * - `jsdom` so component tests can render into a DOM;
 * - `globals: true` so `describe` / `it` / `expect` are available without imports (types come
 *   from `vitest/globals` in `tsconfig.json`; tests may still import them explicitly);
 * - `@` alias mirrors the tsconfig `paths` entry;
 * - CSS is not processed (the default `css.include: []`) because no test asserts on styles and
 *   Next's global stylesheet only needs to be importable.
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    exclude: ['node_modules', '.next'],
    css: false,
    server: {
      deps: {
        // jest-dom augments vitest's `expect`; inlining keeps a single vitest instance.
        inline: ['@testing-library/jest-dom'],
      },
    },
  },
});
