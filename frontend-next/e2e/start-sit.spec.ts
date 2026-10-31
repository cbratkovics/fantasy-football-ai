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

async function blockApi(page: Page, baseURL = 'http://localhost:3100') {
  await page.route('**/*', async (route) => {
    if (new URL(route.request().url()).origin === new URL(baseURL).origin) return route.continue()
    return route.abort()
  })
}

async function nominate(page: Page) {
  for (const player of choices) {
    await page.getByRole('combobox', { name: 'Search players to compare' }).fill(player.player_id)
    await page.getByRole('combobox', { name: 'Search players to compare' }).press('Enter')
  }
}

test('mobile quick comparison uses verified same-origin evidence during an API outage and never records a suggestion', async ({ page, baseURL }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await blockApi(page, baseURL)
  await page.goto('/start-sit')
  const compare = page.getByRole('button', { name: 'Compare assuming these players can play' })
  await expect(compare).toBeDisabled()
  await expect(page.getByLabel('Starting slot')).toHaveValue('FLEX')
  await expect(page.getByRole('button', { name: 'Choose player 1' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Choose player 2' })).toBeVisible()
  const picker = await page.getByRole('combobox', { name: 'Search players to compare' }).boundingBox()
  expect(picker && picker.y < 844).toBeTruthy()
  await nominate(page)
  await expect(compare).toBeEnabled()
  await page.getByText('Adjust player assumptions', { exact: true }).click()
  for (const player of choices) {
    await expect(page.getByLabel(`Availability assumption for ${player.display_name}`)).toHaveValue('unknown')
  }
  await page.getByRole('button', { name: 'Compare nominated alternatives' }).click()
  await expect(page.getByTestId('quick-result')).toContainText('Check availability before choosing')
  await compare.focus()
  await page.keyboard.press('Enter')
  await expect(page.getByTestId('quick-result')).toContainText('has the higher projection')
  await expect(page.getByTestId('quick-result')).toContainText('Assuming the selected players are available')
  await expect(page.getByTestId('quick-result')).toContainText(choices[0].display_name)
  await page.getByText('Record my choice or save a hypothetical comparison', { exact: true }).click()
  await expect(page.getByTestId(`choose-${choices[0].player_id}`)).not.toBeChecked()
  await expect(page.getByTestId(`choose-${choices[1].player_id}`)).not.toBeChecked()
  expect(await page.evaluate((key) => localStorage.getItem(key), STORE)).toBeNull()
  await expect(page.getByText(/Live availability is not verified/).first()).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
})

test('corrupt weekly evidence fails closed without API or synthetic substitution', async ({ page, baseURL }) => {
  await blockApi(page, baseURL)
  await page.route('**/decision-lab/inputs/*.json', (route) => route.fulfill({
    status: 200, contentType: 'application/json', body: JSON.stringify({ ...snapshot, rows: [] }),
  }))
  await page.goto('/start-sit')
  await expect(page.getByRole('heading', { name: 'Recommendation unavailable' })).toBeVisible()
  await expect(page.getByRole('alert').filter({ hasText: 'Recommendation unavailable' })).toContainText('No other week, API forecast, or synthetic data was substituted.')
  await expect(page.getByRole('button', { name: 'Compare assuming these players can play' })).toHaveCount(0)
})
