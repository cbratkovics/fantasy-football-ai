import { readFileSync } from 'node:fs'
import { expect, test, type Page } from '@playwright/test'

const ORIGIN = 'http://localhost:3100'
const CASE = 'syn-ambiguity-floor-relaxation'
const WEEKLY_CASE = 'real-weekly-2026-w03-rb-1ed3fb'

/** Abort and record every request that leaves the app's own origin. */
async function isolate(page: Page): Promise<string[]> {
  const foreign: string[] = []
  await page.route('**/*', (route) => {
    const url = route.request().url()
    if (url.startsWith(`${ORIGIN}/`)) return route.continue()
    foreign.push(url)
    return route.abort()
  })
  return foreign
}

async function openCase(page: Page, caseId: string) {
  await page.goto(`/decision-lab?case=${caseId}`)
  await expect(page.getByTestId(`case-${caseId}`)).toHaveAttribute('aria-current', 'true')
  await expect(page.getByTestId('compute')).toBeEnabled()
}

function status(page: Page) {
  return page.getByTestId('decision-card').getByTestId('status-pill')
}

test('full flow: review → floor relaxation → experiment → record → reveal → export → import → reload', async ({ page }) => {
  const foreign = await isolate(page)
  page.on('dialog', (d) => d.accept())
  await openCase(page, CASE)

  await page.getByTestId('compute').click()
  await expect(status(page)).toHaveAttribute('data-status', 'review')
  await expect(status(page)).toContainText('Review')
  await expect(page.getByTestId('recommended-player')).toHaveText('No recommendation')

  // Relaxing the floor to 5 changes only the floor gate; the small gap keeps the review.
  const floor = page.locator('#min-floor')
  await floor.fill('5')
  await expect(page.getByTestId('stale-notice')).toBeVisible()
  await page.getByTestId('compute').click()
  await expect(status(page)).toHaveAttribute('data-status', 'review')
  const changed = page.getByTestId('what-changed')
  await expect(changed).toBeVisible()
  await expect(page.getByTestId('changed-controls')).toHaveText('min_floor')
  await expect(page.getByTestId('diff-row-floor_gate')).toContainText('meets')
  await expect(page.getByTestId('diff-row-floor_threshold')).toContainText('5')
  await expect(page.getByTestId('diff-row-status')).toHaveCount(0)
  await expect(page.getByTestId('diff-row-gap')).toHaveCount(0)
  await expect(page.getByTestId('diff-row-gap_gate')).toHaveCount(0)
  await expect(page.getByTestId('diff-row-recommended')).toHaveCount(0)

  // The case experiment recomputes as a child decision.
  await page.getByTestId('experiment-gap 0.3, floor 5').click()
  await expect(status(page)).toHaveAttribute('data-status', 'recommend')
  await expect(page.getByTestId('recommended-player')).toContainText('Synthetic A')
  await expect(page.getByTestId('baseline-line')).toContainText('Baseline prefers Synthetic B')
  await expect(page.getByTestId('parent-id').locator('code')).toHaveCount(1)
  const decisionId = await page.getByTestId('decision-id').locator('code').getAttribute('title')
  expect(decisionId).toMatch(/^[0-9a-f]{64}$/)

  // Record a hypothetical choice without a note; the recommendation was never preselected.
  await expect(page.getByTestId('choose-SYN-A')).not.toBeChecked()
  await page.getByTestId('choose-SYN-A').check()
  await expect(page.getByTestId('kind-hypothetical_replay')).toBeChecked()
  await page.getByTestId('record-choice').click()
  await expect(page.getByTestId('action-state')).toContainText('Action recorded')
  await expect(page.getByTestId('saved-badge')).toHaveText('Recorded decision')

  // Reveal outcomes: A scored 8, B scored 16.
  await page.getByRole('button', { name: 'Outcomes', exact: true }).click()
  await page.getByTestId('reveal-outcomes').click()
  const metrics = page.getByTestId('outcome-metrics')
  await expect(metrics.getByTestId('metric-chosen')).toHaveText('8')
  await expect(metrics.getByTestId('metric-regret')).toHaveText('8')
  await expect(metrics.getByTestId('metric-vs-baseline')).toHaveText('-8')
  await expect(metrics.getByTestId('metric-model-vs-baseline')).toHaveText('-8')
  await expect(page.getByTestId('decision-id').locator('code')).toHaveAttribute('title', decisionId as string)

  // Export, clear, import: the same decision id comes back.
  await page.getByRole('button', { name: /Saved decisions/ }).click()
  await expect(page.getByTestId(`saved-${decisionId}`)).toBeVisible()
  const downloadPromise = page.waitForEvent('download')
  await page.getByTestId('export-all').click()
  const download = await downloadPromise
  const path = await download.path()
  expect(path).not.toBeNull()
  const exported = JSON.parse(readFileSync(path as string, 'utf8')) as { receipts: Array<{ decision_id: string }> }
  expect(exported.receipts.map((r) => r.decision_id)).toContain(decisionId)

  await page.getByTestId('clear-all').click()
  await expect(page.getByTestId('saved-empty')).toBeVisible()
  await expect(page.getByTestId('saved-badge')).toHaveText('Unsaved exploration')

  await page.getByTestId('import-file').setInputFiles(path as string)
  await expect(page.getByTestId('import-report')).toContainText(`${exported.receipts.length} imported`)
  await expect(page.getByTestId('import-report')).toContainText('0 rejected')
  await expect(page.getByTestId(`saved-${decisionId}`)).toBeVisible()

  await page.reload()
  await expect(page.getByTestId('case-library')).toBeVisible()
  await page.getByRole('button', { name: /Saved decisions/ }).click()
  await expect(page.getByTestId(`saved-${decisionId}`)).toBeVisible()
  await expect(page.getByTestId(`saved-${decisionId}`)).toContainText('attached')

  expect(foreign).toEqual([])
})

test('stale result: changing the slot or a parameter hides the recommendation until recompute', async ({ page }) => {
  const foreign = await isolate(page)
  await openCase(page, 'syn-threshold-equality')
  await page.getByTestId('compute').click()
  await expect(status(page)).toHaveAttribute('data-status', 'recommend')
  await expect(page.getByTestId('recommended-player')).toContainText('Synthetic A')

  await page.getByTestId('slot-WR').click()
  await expect(page.getByTestId('decision-stale')).toContainText('stale for the inputs above')
  await expect(page.getByTestId('recommended-player')).toHaveCount(0)
  await expect(page.getByTestId('action-panel')).toHaveCount(0)

  await page.getByTestId('slot-RB').click()
  await page.locator('#min-gap').fill('3')
  await expect(page.getByTestId('decision-stale')).toBeVisible()
  await page.getByTestId('compute').click()
  await expect(status(page)).toHaveAttribute('data-status', 'review')
  await expect(page.getByTestId('recommended-player')).toHaveText('No recommendation')
  expect(foreign).toEqual([])
})

test('published weekly case: review on availability until every alternative is assumed available', async ({ page }) => {
  const foreign = await isolate(page)
  await openCase(page, WEEKLY_CASE)
  await expect(page.getByTestId('context-published_weekly')).toHaveAttribute('aria-pressed', 'true')
  await page.getByTestId('compute').click()
  await expect(status(page)).toHaveAttribute('data-status', 'review')
  await expect(page.getByTestId('explanation')).toContainText('Availability is not verified')

  const selects = page.locator('[data-testid^="avail-"]')
  const n = await selects.count()
  expect(n).toBe(3)
  for (let i = 0; i < n; i += 1) {
    await selects.nth(i).selectOption('assumed_available')
  }
  await expect(page.getByTestId('stale-notice')).toBeVisible()
  await page.getByTestId('compute').click()
  await expect(status(page)).toHaveAttribute('data-status', 'recommend')
  await expect(page.getByTestId('limitations').locator('[data-limitation="availability_user_assumed"]')).toBeVisible()
  await expect(page.getByTestId('limitations').locator('[data-limitation="snapshot_not_game_day_verified"]')).toBeVisible()
  await expect(page.getByTestId('changed-controls')).toHaveText('availability')

  // No outcomes are published for this snapshot.
  await page.getByTestId('choose-00-0039139').check()
  await page.getByTestId('record-choice').click()
  await page.getByRole('button', { name: 'Outcomes', exact: true }).click()
  await page.getByTestId('reveal-outcomes').click()
  await expect(page.getByTestId('outcome-none')).toContainText('No outcomes are published for this snapshot yet')
  expect(foreign).toEqual([])
})

test('keyboard: Tab reaches the first case, Enter selects it, Compute is keyboard operable', async ({ page }) => {
  const foreign = await isolate(page)
  await page.goto('/decision-lab')
  await expect(page.getByTestId('case-library')).toBeVisible()

  const focusedTestId = () => page.evaluate(() => (document.activeElement as HTMLElement | null)?.dataset.testid ?? '')
  let reached = false
  for (let i = 0; i < 60 && !reached; i += 1) {
    await page.keyboard.press('Tab')
    const id = await focusedTestId()
    if (id.startsWith('case-')) reached = true
  }
  expect(reached).toBe(true)
  const selectedId = await focusedTestId()
  await page.keyboard.press('Enter')
  await expect(page.getByTestId(selectedId)).toHaveAttribute('aria-current', 'true')
  await expect(page.getByTestId('compute')).toBeEnabled()

  await page.getByTestId('compute').focus()
  expect(await focusedTestId()).toBe('compute')
  await page.keyboard.press('Enter')
  await expect(page.getByTestId('decision-result')).toBeVisible()
  await expect(status(page)).toHaveAttribute('data-status', /recommend|review|hold/)
  expect(foreign).toEqual([])
})

test('mobile viewport: no horizontal page scroll and the table scrolls inside its own container', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await openCase(page, CASE)
  const noOverflow = async () => {
    const widths = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, inner: window.innerWidth }))
    expect(widths.scroll).toBeLessThanOrEqual(widths.inner + 1)
  }
  await noOverflow()
  await page.getByTestId('compute').click()
  await expect(page.getByTestId('alternatives-table')).toBeVisible()
  await noOverflow()
  const scroller = page.getByTestId('alternatives-scroll')
  const box = await scroller.evaluate((el) => ({ overflowX: getComputedStyle(el).overflowX, scrollWidth: el.scrollWidth, clientWidth: el.clientWidth }))
  expect(box.overflowX).toBe('auto')
  expect(box.scrollWidth).toBeGreaterThan(box.clientWidth)
})
