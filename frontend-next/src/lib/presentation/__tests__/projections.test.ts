import { describe, expect, it } from 'vitest'
import { projectionDefaults, projectionGroup, rangeScale, actualCoverage, comparisonDestination } from '../projections'
import { createMemoryFetch } from '@/lib/decision-lab/bundle'
import { weeklyBundle } from '@/lib/start-sit/__tests__/fixtures'
import type { PredictionPeriod, PredictionRecord, PredictionsResponse } from '@/lib/api/types'

const periods: PredictionPeriod[] = [8, 3].map((week) => ({ season: 2026, week, model_version: 'm', feature_version: 'f', generated_at_utc: '2026-10-27T12:00:00Z', n: 4 }))
const row = (id: string, position: string, prediction: number, actual: number | null = null): PredictionRecord => ({ player_id: id, name: id, position, prediction, actual, team: 'SYN', candidate: 'rf', model_version: 'm', receptions_estimate: 2, floor: prediction - 5, ceiling: prediction + 5, actual_null_reason: actual === null ? 'outcome_unavailable' : null, actual_scoring_format: 'ppr' })
describe('projection view defaults and presentation', () => {
  it('defaults to latest published PPR FLEX with no manufactured periods', () => {
    expect(projectionDefaults(periods, new URLSearchParams(), null)).toMatchObject({ period: '2026-8', scoring: 'ppr', position: 'FLEX' })
  })
  it('valid explicit query precedes compatible preferences, while stale saved periods get a notice', () => {
    expect(projectionDefaults(periods, new URLSearchParams('season=2026&week=3&scoring=half&position=QB'), { period: '2026-8', scoring: 'standard', position: 'ALL' })).toMatchObject({ period: '2026-3', scoring: 'half', position: 'QB' })
    expect(projectionDefaults(periods, new URLSearchParams(), { period: '2024-18' })).toMatchObject({ period: '2026-8', notice: expect.stringContaining('saved period') })
    expect(projectionDefaults(periods, new URLSearchParams('season=2026&week=18'), null)).toMatchObject({ period: null, notice: expect.stringContaining('no published') })
    expect(projectionDefaults(periods, new URLSearchParams('season=2026&week=08'), null).period).toBe('2026-8')
  })
  it('FLEX excludes QB, keeps negative projections, and has a shared scale independent of row order', () => {
    const rows = [row('qb', 'QB', 30), row('wr', 'WR', -2), row('rb', 'RB', 12)]
    expect(projectionGroup(rows, 'FLEX').map((r) => r.player_id)).toEqual(['rb', 'wr'])
    expect(rangeScale(rows)).toEqual({ min: -7, max: 35 })
    expect(rangeScale(rows.slice().reverse())).toEqual({ min: -7, max: 35 })
  })
  it('distinguishes no outcomes from format/conversion and conflict withholding; zero and negative are observed', () => {
    expect(actualCoverage([row('none', 'WR', 10)], 'ppr')).toMatchObject({ showColumn: false, message: 'Results have not been recorded in this snapshot.' })
    expect(actualCoverage([{ ...row('none', 'WR', 10), actual_null_reason: 'conversion_evidence_missing' }], 'half')).toMatchObject({ showColumn: false, message: 'Actual scores are unavailable in this scoring format.' })
    expect(actualCoverage([{ ...row('conflict', 'WR', 10), actual_null_reason: 'outcome_revision_mismatch' }], 'ppr').message).toMatch(/withheld/)
    expect(actualCoverage([row('zero', 'RB', 10, 0), row('negative', 'WR', 9, -2)], 'ppr')).toMatchObject({ observed: 2, showColumn: true })
    expect(actualCoverage([{ ...row('old', 'WR', 10, 12), actual_scoring_format: undefined }], 'standard')).toMatchObject({ observed: 0, showColumn: false })
    expect(actualCoverage([{ ...row('legacy-null', 'WR', 10), actual_null_reason: undefined }], 'ppr').message).toBe('Actual scores are unavailable; this API does not declare why.')
  })
})

function transferFixture(historical = false) {
  const b = weeklyBundle((s) => { s.feature_version = 'test-features'; if (historical) { s.mode = 'historical_replay'; s.publication.status = 'recorded' } })
  const s = b.inputs
  const data: PredictionsResponse = {
    season: s.season, week: s.week, scoring: 'ppr', model_version: s.model_version,
    feature_version: s.feature_version ?? '', generated_at_utc: s.generated_at_utc!, data_through: s.data_cutoff!,
    n: s.rows.length, scoring_derivation: '', interval_method: '',
    predictions: s.rows.map((r) => ({ ...row(r.player_id, r.position, r.projection!), name: r.display_name, candidate: r.candidate, model_version: r.model_version })),
  }
  return { b, data }
}
describe('comparison transfer verifies one exact PPR release', () => {
  it('transfers canonical IDs to latest Start/Sit, without projection numbers', async () => {
    const { b, data } = transferFixture()
    const href = await comparisonDestination(data, ['SYN-A', 'SYN-B'], { fetchImpl: createMemoryFetch(b.files) })
    expect(href).toContain('/start-sit?')
    expect(new URL(href, 'https://example.test').searchParams.get('players')).toBe('SYN-A,SYN-B')
    expect(href).not.toContain('projection')
  })
  it('opens the exact historical period in Lab rather than latest weekly Start/Sit', async () => {
    const { b, data } = transferFixture(true)
    expect(await comparisonDestination(data, ['SYN-A', 'SYN-B'], { fetchImpl: createMemoryFetch(b.files) })).toContain('/decision-lab?')
  })
  it('rejects different releases, missing alternatives, and non-PPR source input', async () => {
    const { b, data } = transferFixture()
    const options = { fetchImpl: createMemoryFetch(b.files) }
    await expect(comparisonDestination({ ...data, model_version: 'different' }, ['SYN-A', 'SYN-B'], options)).rejects.toThrow('different releases')
    await expect(comparisonDestination(data, ['SYN-A', 'MISSING'], options)).rejects.toThrow('No alternative was removed')
    await expect(comparisonDestination({ ...data, scoring: 'half' }, ['SYN-A', 'SYN-B'], options)).rejects.toThrow('PPR evidence')
  })
})
