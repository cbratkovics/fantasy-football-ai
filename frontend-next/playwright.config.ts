import { defineConfig, devices } from '@playwright/test'

/**
 * Browser tests for the Decision Lab route against a production build of the app.
 *
 * The web server builds and starts Next on port 3100. NEXT_PUBLIC_API_URL is pinned so the build
 * does not depend on the developer's shell; the lab itself never calls that API. Vitest picks up
 * `src/**` only, so nothing under `e2e/` runs twice.
 */
export default defineConfig({
  testDir: 'e2e',
  timeout: 90_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: 'http://localhost:3100',
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'npm run build && npm run start -- --port 3100',
    url: 'http://localhost:3100/decision-lab',
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
    env: { NEXT_PUBLIC_API_URL: 'http://localhost:7860' },
  },
})
