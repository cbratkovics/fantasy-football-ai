import { describe, expect, it } from 'vitest'
import { createMemoryFetch } from '@/lib/decision-lab/bundle'
import { buildDecisionInputs } from '@/lib/decision-lab/receipts'
import { evaluate } from '@/lib/decision-lab/policy'
import { compareWeekly, loadLatestWeekly } from '../adapter'
import { weeklyBundle } from './fixtures'

const assumed = {
  'SYN-A': { availability: 'assumed_available' as const, availability_basis: 'user_assumed' as const },
  'SYN-B': { availability: 'assumed_available' as const, availability_basis: 'user_assumed' as const },
  'SYN-C': { availability: 'assumed_available' as const, availability_basis: 'user_assumed' as const },
}
const options = { slot: 'RB', playerIds: ['SYN-A', 'SYN-B'], overrides: assumed }

describe('quick comparison uses the v1 policy boundary', () => {
  it('has exactly the Lab normalized inputs, identity, result and reasons', () => {
    const { inputs: snapshot } = weeklyBundle()
    const quick = compareWeekly(snapshot, options)
    const lab = buildDecisionInputs(snapshot, options)
    expect(quick.inputs).toEqual(lab)
    expect(quick.result).toEqual(evaluate(lab))
  })
  it.each([['SYN-A', 'SYN-B'], ['SYN-B', 'SYN-A']])('ties do not select by nomination order: %s %s', (a, b) => {
    const { inputs } = weeklyBundle((s) => { s.rows[1].projection = 14 })
    const r = compareWeekly(inputs, { ...options, playerIds: [a, b] }).result
    expect(r.status).toBe('review')
    expect(r.recommended_player_id).toBeNull()
    expect(r.reasons.map((x) => x.code)).toContain('projection_tie')
  })
  it.each([[0.4, 'review'], [0.5, 'recommend']])('preserves the canonical gap boundary %s', (gap, status) => {
    const { inputs } = weeklyBundle((s) => { s.rows[1].projection = 14 - Number(gap) })
    expect(compareWeekly(inputs, options).result.status).toBe(status)
  })
  it('unknown availability is REVIEW with no assumed user confirmation', () => {
    const { inputs } = weeklyBundle()
    const r = compareWeekly(inputs, { ...options, overrides: {} }).result
    expect(r.status).toBe('review')
    expect(r.reasons.map((x) => x.code)).toContain('availability_unresolved')
    expect(r.recommended_player_id).toBeNull()
  })
  it.each(['assumed_available', 'realized_stats_row'] as const)('does not silently trust a weekly %s availability default', (availability) => {
    const { inputs } = weeklyBundle((s) => { s.rows.forEach((r) => { r.availability_default = availability }) })
    expect(() => compareWeekly(inputs, { ...options, overrides: {} })).toThrow(/Weekly availability/)
  })
  it('requires a slot, excludes QB from ordinary FLEX, and allows RB/WR', () => {
    const { inputs } = weeklyBundle((s) => { s.rows[1].position = 'WR'; s.rows[2].position = 'QB'; s.rows[1].projection = 13 })
    expect(compareWeekly(inputs, { ...options, slot: '' }).result.status).toBe('hold')
    const r = compareWeekly(inputs, { ...options, slot: 'FLEX', playerIds: ['SYN-A', 'SYN-B', 'SYN-C'] }).result
    expect(r.status).toBe('recommend')
    expect(r.ranking.find((x) => x.player_id === 'SYN-C')?.exclusion).toBe('excluded_slot_ineligible')
    expect(r.counts.comparable).toBe(2)
  })
  it('validates duplicate, oversized and one-option nominations outside the UI', () => {
    const { inputs } = weeklyBundle()
    expect(compareWeekly(inputs, { ...options, playerIds: ['SYN-A', 'SYN-A'] }).result.status).toBe('hold')
    expect(compareWeekly(inputs, { ...options, playerIds: Array(9).fill('SYN-A') }).result.reasons.map((r) => r.code)).toContain('choice_set_too_large')
    expect(compareWeekly(inputs, { ...options, playerIds: ['SYN-A'] }).result.status).toBe('review')
  })
  it('never drops a missing nomination or joins another snapshot', () => {
    const { inputs } = weeklyBundle()
    expect(() => compareWeekly(inputs, { ...options, playerIds: ['SYN-A', 'missing', 'SYN-B'] })).toThrow(/not in snapshot/)
    inputs.rows[1].projection = null
    const r = compareWeekly(inputs, options).result
    expect(r.status).toBe('review')
    expect(r.recommended_player_id).toBeNull()
    expect(r.reasons.map((x) => x.code)).toContain('projection_missing')
  })
  it('validates direct snapshot objects before reducing rows', () => {
    const duplicate = weeklyBundle().inputs
    duplicate.rows.push({ ...duplicate.rows[0], projection: 99, ceiling: 100 })
    duplicate.population.n_rows += 1
    expect(() => compareWeekly(duplicate, options)).toThrow(/duplicate/)
    const mixed = weeklyBundle().inputs
    mixed.rows[0].model_version = 'different-model'
    expect(() => compareWeekly(mixed, options)).toThrow(/model_version differs/)
    const nonfinite = weeklyBundle().inputs
    nonfinite.rows[0].projection = Infinity
    expect(() => compareWeekly(nonfinite, options)).toThrow(/projection/)
  })
  it.each(['standard', 'half', 'custom'])('rejects unsupported scoring %s even outside the UI', (scoring) => {
    expect(() => compareWeekly(weeklyBundle().inputs, { ...options, scoring })).toThrow(/only the implemented PPR rules/)
  })
})

describe('latest weekly loader', () => {
  it('loads only the identified verified published weekly evidence', async () => {
    const b = weeklyBundle()
    const result = await loadLatestWeekly({ fetchImpl: createMemoryFetch(b.files) })
    expect(result.ok).toBe(true)
    if (result.ok) expect(result.snapshot.snapshot_id).toBe('weekly-fixture')
  })
  it('fails closed on a corrupt snapshot with no API or synthetic fallback', async () => {
    const b = weeklyBundle()
    b.files[b.manifest.snapshots[0].inputs.path] += ' '
    const result = await loadLatestWeekly({ fetchImpl: createMemoryFetch(b.files) })
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.kind).toBe('byte_digest')
  })
  it('rejects a mixed snapshot period even with valid digests', async () => {
    const b = weeklyBundle()
    b.manifest.snapshots[0].week = 2
    b.files['manifest.json'] = JSON.stringify(b.manifest)
    const result = await loadLatestWeekly({ fetchImpl: createMemoryFetch(b.files) })
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.kind).toBe('compat')
  })
  it('rejects a weekly availability assumption encoded in evidence instead of chosen by the user', async () => {
    const b = weeklyBundle((s) => { s.rows[0].availability_default = 'assumed_available' })
    const result = await loadLatestWeekly({ fetchImpl: createMemoryFetch(b.files) })
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.kind).toBe('compat')
  })
})
