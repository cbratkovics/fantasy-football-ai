import { defineConfig, devices } from '@playwright/test'

/** Explicit local release review; requires the frontend :3300 and synthetic API :7861.
 * No providers, deployed API, credentials, or published-artifact edits are involved.
 */
export default defineConfig({
  testDir: 'acceptance',
  timeout: 60_000,
  expect: { timeout: 15_000 },
  workers: 1,
  retries: 0,
  reporter: 'list',
  outputDir: process.env.FFAI_ACCEPTANCE_OUTPUT || '/tmp/ffai-acceptance/browser-results',
  use: {
    baseURL: process.env.FFAI_ACCEPTANCE_FRONTEND_URL || 'http://localhost:3300',
    trace: 'retain-on-failure',
    launchOptions: process.env.FFAI_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.FFAI_CHROMIUM_EXECUTABLE } : {},
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
})
