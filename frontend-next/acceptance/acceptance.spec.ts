/** Explicit local integration suite (see docs/START_SIT_ACCEPTANCE.md).
 * Kept outside default e2e because this review suite requires the changed local API fixture.
 * No tests are skipped: run the complete suite with the documented acceptance config.
 * These envelopes do not demonstrate provider verification.
 */
import { expect, test, type Page } from '@playwright/test'
import { weeklyBundle } from '../src/lib/start-sit/__tests__/fixtures'
import { contentId } from '../src/lib/decision-lab/canonical'

const bundle = weeklyBundle((s) => { s.rows[1].projection = 13 })
const STORE = 'ffai.decision-lab/decision_receipt/1.0/receipts'
const API = process.env.FFAI_ACCEPTANCE_API_URL || 'http://127.0.0.1:7861'

function companion(state: 'verified' | 'partial' = 'verified') {
  const source = {
    source_id: 'synthetic-browser-fixture-not-provider', sha256: 'a'.repeat(64), state,
    reasons: ['LOCAL SYNTHETIC FIXTURE'], provider_updated_at_utc: null,
    first_observed_at_utc: '2099-09-01T00:00:00Z', fetched_at_utc: '2099-09-01T00:00:00Z',
    effective_at_utc: null, coverage_complete: false, observed_weeks: [1], target_rows: 3,
  }
  return {
    schema_version: 'context_snapshot/1.0', mode: 'live',
    target: { season: 2099, season_type: 'REG', week: 1, state: 'target' },
    requested_as_of_utc: '2099-09-01T00:00:00Z', observed_at_utc: '2099-09-01T00:00:00Z',
    forecast: { snapshot_id: bundle.inputs.snapshot_id, sha256: contentId(bundle.inputs), season: 2099, week: 1 },
    sources: { schedules: source, rosters: source, injuries: { ...source, state: 'unavailable', target_rows: 0 } },
    players: bundle.inputs.rows.map((p) => ({
      player_id: p.player_id, provider_id: p.player_id, name: p.display_name, position: p.position,
      current_team: p.team, identity_state: 'resolved',
      game: { game_id: 'SYN-GAME', opponent: 'SYN-OTHER', kickoff_utc: '2099-09-01T01:00:00Z', state: 'scheduled' },
      forecast: { state: 'present', forecast_team: p.team },
      availability: { state: 'unknown', designation: null, basis: 'no_provider_evidence' }, reasons: [],
    })),
  }
}
function seal(value: object) { return { ...value, context_snapshot_id: contentId(value) } }
async function evidence(page: Page) {
  await page.clock.install({ time: new Date('2099-09-01T00:30:00Z') })
  await page.route('**/decision-lab/**', (route) => {
    const key = new URL(route.request().url()).pathname.split('/decision-lab/')[1]
    return route.fulfill({ status: bundle.files[key] ? 200 : 404, contentType: 'application/json', body: bundle.files[key] ?? '{}' })
  })
}
async function compare(page: Page) {
  await page.getByLabel('Starting slot').selectOption('RB')
  for (const letter of ['A', 'B']) {
    await page.getByRole('combobox', { name: 'Search players to compare' }).fill(`SYN-${letter}`)
    await page.getByRole('combobox', { name: 'Search players to compare' }).press('Enter')
  }
  await page.getByText('Adjust player assumptions', { exact: true }).click()
  for (const letter of ['A', 'B']) await expect(page.getByLabel(`Availability assumption for Synthetic ${letter}`)).toHaveValue('unknown')
  await page.getByRole('button', { name: 'Compare nominated alternatives' }).click()
  const bulk = page.getByRole('button', { name: 'Compare assuming these players can play' })
  if (await bulk.isEnabled()) {
    await expect(page.getByTestId('quick-result')).toContainText('Check availability before choosing')
    await bulk.focus()
  } else {
    // Readiness restrictions survive explicit assumptions. Inspecting the policy result does
    // not enter the separate historical-action flow or enable an ordinary current action.
    await expect(bulk).toBeDisabled()
    for (const letter of ['A', 'B']) await page.getByLabel(`Availability assumption for Synthetic ${letter}`).selectOption('assumed_available')
    await page.getByRole('button', { name: 'Compare nominated alternatives' }).focus()
  }
  await page.keyboard.press('Enter')
}

test('local changed API supplies requested-format actuals to the running frontend', async ({ page }) => {
  const hosts: string[] = []
  page.on('request', (r) => {
    const url = new URL(r.url())
    if (/^\/predictions\/\d+\/\d+$/.test(url.pathname)) hosts.push(url.origin)
  })
  await page.goto('/predictions')
  const receiver = page.getByRole('row').filter({ hasText: 'LOCAL FIXTURE receiver' })
  await expect(receiver).toBeVisible()
  for (const [format, expected] of [['ppr', '12.0'], ['half', '9.0'], ['standard', '6.0']]) {
    if (format !== 'ppr') {
      const response = page.waitForResponse((r) => r.url().includes('/predictions/2026/8') && new URL(r.url()).searchParams.get('scoring') === format)
      await page.getByRole('button', { name: format, exact: true }).click()
      expect((await (await response).json()).scoring_rules_id).toBe('ffai-nflverse-v1')
    } else { // The initial request already completed; validate it directly against the same local API.
      const api = await page.request.get(`${API}/predictions/2026/8?scoring=ppr`)
      expect((await api.json()).scoring_rules_id).toBe('ffai-nflverse-v1')
    }
    await expect(page.getByRole('columnheader', { name: `Actual (${format.toUpperCase()})` })).toBeVisible()
    await expect(receiver.locator('td').nth(4)).toHaveText(expected)
    await expect(page.getByRole('row').filter({ hasText: 'LOCAL FIXTURE zero' }).locator('td').nth(4)).toHaveText('0.0')
    await expect(page.getByRole('row').filter({ hasText: 'LOCAL FIXTURE negative' }).locator('td').nth(4)).toHaveText('-2.0')
  }
  await expect(page.getByRole('row').filter({ hasText: 'LOCAL FIXTURE missing evidence' })).toContainText('missing realized conversion evidence')
  expect(hosts.length).toBeGreaterThanOrEqual(3)
  expect(hosts.every((host) => host === API)).toBe(true)
})

test('missing context allows only intentional conditional comparison without recording a choice', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await evidence(page)
  await page.route('**/context/snapshot.json', (r) => r.fulfill({ status: 404, body: '{}' }))
  await page.goto('/start-sit')
  await compare(page)
  await expect(page.getByText(/Live availability is not verified/).first()).toBeVisible()
  await expect(page.getByTestId('quick-result')).toContainText('has the higher projection')
  await expect(page.getByTestId('quick-result')).toContainText('Assuming the selected players are available')
  expect(await page.evaluate((key) => localStorage.getItem(key), STORE)).toBeNull()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
})

test('valid partial coverage remains unverified and uses visibly separate hypothetical exploration', async ({ page }) => {
  await evidence(page)
  await page.route('**/context/snapshot.json', (r) => r.fulfill({ status: 200, json: seal(companion('partial')) }))
  await page.goto('/start-sit')
  await compare(page)
  await expect(page.getByTestId('quick-result')).toContainText('This comparison cannot support a current lineup change')
  await expect(page.getByTestId('action-panel')).toHaveCount(0)
  await page.getByRole('button', { name: 'Explore as historical hypothetical' }).click()
  await expect(page.getByTestId('kind-hypothetical_replay')).toBeChecked()
  await expect(page.getByTestId('kind-self_reported_real')).toHaveCount(0)
})

test('known impossibility and passed kickoff survive assumptions and a failed refresh', async ({ page }) => {
  await evidence(page)
  const c = companion()
  c.players[0].game.state = 'bye'
  let fail = false
  await page.route('**/context/snapshot.json', (r) => r.fulfill(fail ? { status: 503, body: '{}' } : { status: 200, json: seal(c) }))
  await page.goto('/start-sit')
  await compare(page)
  await expect(page.getByTestId('context-assessment')).toContainText(/bye/i)
  await expect(page.getByTestId('quick-result')).toContainText('This comparison cannot support a current lineup change')
  await page.clock.fastForward(31 * 60 * 1000)
  await expect(page.getByTestId('context-assessment')).toContainText(/kickoff.*passed|deadline/i)
  fail = true
  await page.evaluate(() => window.dispatchEvent(new Event('focus')))
  await expect(page.getByTestId('context-assessment')).toContainText(/bye/i)
  await expect(page.getByTestId('quick-result')).toContainText('This comparison cannot support a current lineup change')
})

test('malformed and incompatible companions cannot be turned into missing-context permission', async ({ page }) => {
  await evidence(page)
  let incompatible = false
  await page.route('**/context/snapshot.json', (r) => {
    const c = companion()
    if (incompatible) c.forecast.snapshot_id = 'wrong-release'
    return r.fulfill({ status: 200, json: incompatible ? seal(c) : { ...seal(c), context_snapshot_id: 'tampered' } })
  })
  await page.goto('/start-sit')
  await compare(page)
  await expect(page.getByTestId('quick-result')).toContainText('This comparison cannot support a current lineup change')
  await expect(page.getByTestId('action-panel')).toHaveCount(0)
  incompatible = true
  await page.evaluate(() => window.dispatchEvent(new Event('focus')))
  await expect(page.getByTestId('context-assessment')).toContainText(/cannot be verified|does not match|incompatible/i)
  await expect(page.getByTestId('action-panel')).toHaveCount(0)
})
