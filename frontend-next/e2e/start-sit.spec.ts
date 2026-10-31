import { readFileSync } from 'node:fs'
import path from 'node:path'
import { expect, test, type Page } from '@playwright/test'

const ROOT = path.resolve(process.cwd(), '../artifacts/decision_lab')
const manifest = JSON.parse(readFileSync(path.join(ROOT, 'manifest.json'), 'utf8'))
const latest = manifest.snapshots.find((s: { snapshot_id: string }) => s.snapshot_id === manifest.latest_weekly_snapshot_id)
const snapshot = JSON.parse(readFileSync(path.join(ROOT, latest.inputs.path), 'utf8'))
const backs = snapshot.rows
  .filter((r: { position: string; projection: number | null; baseline: number | null; baseline_provenance: string }) =>
    r.position === 'RB' && r.projection !== null && r.baseline !== null && ['player_history', 'position_history'].includes(r.baseline_provenance))
  .sort((a: { projection: number }, b: { projection: number }) => b.projection - a.projection)
const choices = [backs[0], backs[backs.length - 1]] as Array<{ player_id: string; display_name: string; position: string; team: string }>
const STORE = 'ffai.decision-lab/decision_receipt/1.0/receipts'

async function blockApi(page: Page) {
  await page.route('**/*', async (route) => {
    if (new URL(route.request().url()).origin === 'http://localhost:3100') return route.continue()
    return route.abort()
  })
}

async function nominate(page: Page) {
  for (const player of choices) {
    await page.getByRole('textbox', { name: 'Search players to compare' }).fill(player.player_id)
    await page.getByRole('button', { name: `Add ${player.display_name} · ${player.position} · ${player.team}`, exact: true }).click()
  }
}

test('mobile quick comparison uses verified same-origin evidence during an API outage and never records a suggestion', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await blockApi(page)
  await page.goto('/start-sit')
  const compare = page.getByRole('button', { name: 'Compare nominated alternatives' })
  await expect(compare).toBeDisabled()
  await nominate(page)
  await expect(compare).toBeDisabled()
  await page.getByLabel('Starting slot').selectOption('RB')
  await compare.click()
  await expect(page.getByTestId('quick-result')).toContainText('Check player status')
  for (const player of choices) {
    await expect(page.getByLabel(`Availability assumption for ${player.display_name}`)).toHaveValue('unknown')
    await page.getByLabel(`Availability assumption for ${player.display_name}`).selectOption('assumed_available')
  }
  await expect(page.getByText('Comparison changed. Recompute before using the result; saved choices remain unchanged.')).toBeVisible()
  await compare.focus()
  await page.keyboard.press('Enter')
  await expect(page.getByTestId('quick-result')).toContainText('Suggested starter')
  await expect(page.getByTestId('quick-result')).toContainText(choices[0].display_name)
  await expect(page.getByTestId(`choose-${choices[0].player_id}`)).not.toBeChecked()
  await expect(page.getByTestId(`choose-${choices[1].player_id}`)).not.toBeChecked()
  expect(await page.evaluate((key) => localStorage.getItem(key), STORE)).toBeNull()
  await expect(page.getByText(/PPR decisions only/)).toBeVisible()
  await expect(page.getByText(/not a verified target-week roster/)).toBeVisible()
})

test('corrupt weekly evidence fails closed without API or synthetic substitution', async ({ page }) => {
  await blockApi(page)
  await page.route('**/decision-lab/inputs/*.json', (route) => route.fulfill({
    status: 200, contentType: 'application/json', body: JSON.stringify({ ...snapshot, rows: [] }),
  }))
  await page.goto('/start-sit')
  await expect(page.getByRole('heading', { name: 'Recommendation unavailable' })).toBeVisible()
  await expect(page.getByRole('alert').filter({ hasText: 'Recommendation unavailable' })).toContainText('No other week, API forecast, or synthetic data was substituted.')
  await expect(page.getByRole('button', { name: 'Compare nominated alternatives' })).toHaveCount(0)
})
