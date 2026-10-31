import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { PredictionsBoard } from '../PredictionsBoard'
import { getLatestWeek, getPredictions } from '@/lib/api/players'
import type { PredictionsResponse, ScoringFormat } from '@/lib/api/types'

vi.mock('@/lib/api/players', () => ({ getLatestWeek: vi.fn(), getPredictions: vi.fn() }))
afterEach(cleanup)

function response(scoring: ScoringFormat): PredictionsResponse {
  const actual = { standard: 6, half: 9, ppr: 12 }[scoring]
  return {
    season: 2026, week: 8, scoring,
    scoring_derivation: 'PPR projection minus estimated receptions for other formats',
    interval_method: 'validation residual offsets',
    scoring_rules_id: 'ffai-nflverse-v1',
    scoring_rules_description: '4 points per passing TD; no bonuses.',
    model_version: 'model', feature_version: 'asof_v1', generated_at_utc: '2026-10-27T12:00:00Z',
    data_through: { season: 2026, week: 7 }, n: 4,
    predictions: [
      { player_id: 'receiver', name: 'Receiver', team: 'BUF', position: 'WR', prediction: 10, floor: 4, ceiling: 20, receptions_estimate: 2, candidate: 'rf', model_version: 'model', actual, actual_scoring_format: scoring, actual_source: 'artifact_stats', actual_null_reason: null },
      { player_id: 'zero', name: 'Zero', team: 'BUF', position: 'WR', prediction: 4, floor: 0, ceiling: 8, receptions_estimate: 0, candidate: 'rf', model_version: 'model', actual: 0, actual_scoring_format: scoring, actual_source: 'artifact_stats', actual_null_reason: null },
      { player_id: 'negative', name: 'Negative', team: 'BUF', position: 'QB', prediction: 3, floor: 0, ceiling: 8, receptions_estimate: 0, candidate: 'rf', model_version: 'model', actual: -2, actual_scoring_format: scoring, actual_source: 'artifact_stats', actual_null_reason: null },
      { player_id: 'unknown', name: 'Unknown', team: 'BUF', position: 'WR', prediction: 2, floor: 0, ceiling: 8, receptions_estimate: 2, candidate: 'rf', model_version: 'model', actual: null, actual_scoring_format: scoring, actual_source: null, actual_null_reason: 'conversion_evidence_missing' },
    ],
  }
}

beforeEach(() => {
  vi.mocked(getLatestWeek).mockResolvedValue({ season: 2026, week: 8 })
  vi.mocked(getPredictions).mockImplementation(async (_season, _week, format = 'ppr') => response(format))
})

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}><PredictionsBoard /></QueryClientProvider>)
}

describe('published actual scoring', () => {
  it.each([['ppr', '12.0'], ['half', '9.0'], ['standard', '6.0']] as const)('renders %s actuals and their declared rules', async (format, expected) => {
    const user = userEvent.setup()
    mount()
    await screen.findByRole('link', { name: 'Receiver' })
    if (format !== 'ppr') await user.click(screen.getByRole('button', { name: format }))
    const row = (await screen.findByRole('link', { name: 'Receiver' })).closest('tr')!
    await within(row).findByText(expected)
    expect(screen.getByRole('columnheader', { name: `Actual (${format.toUpperCase()})` })).toBeInTheDocument()
    expect(screen.getByText('4 points per passing TD; no bonuses.')).toBeInTheDocument()
    if (format !== 'ppr') expect(screen.getByText(/Derived forecast estimate/)).toBeInTheDocument()
  })

  it('preserves observed zero and negative points and explains absent conversion evidence', async () => {
    mount()
    const zero = (await screen.findByRole('link', { name: 'Zero' })).closest('tr')!
    const negative = screen.getByRole('link', { name: 'Negative' }).closest('tr')!
    expect(within(zero).getByText('0.0')).toBeInTheDocument()
    expect(within(negative).getByText('-2.0')).toBeInTheDocument()
    expect(screen.getByText('Actual unavailable: missing realized conversion evidence')).toBeInTheDocument()
    expect(screen.queryByText('not scored')).not.toBeInTheDocument()
  })

  it.each([
    ['half', undefined], ['standard', undefined], ['half', 'ppr'], ['standard', 'half'], ['ppr', 'standard'],
  ] as const)('withholds %s actuals with unverified or conflicting format %s', async (format, actualFormat) => {
    vi.mocked(getPredictions).mockImplementation(async (_season, _week, requested = 'ppr') => {
      const result = response(requested)
      result.predictions[0] = { ...result.predictions[0], actual: 12, actual_scoring_format: actualFormat }
      return result
    })
    const user = userEvent.setup()
    mount()
    await screen.findByRole('link', { name: 'Receiver' })
    if (format !== 'ppr') await user.click(screen.getByRole('button', { name: format }))
    const row = (await screen.findByRole('link', { name: 'Receiver' })).closest('tr')!
    expect(await within(row).findByText('Actual unavailable: scoring format is unverified or does not match')).toBeInTheDocument()
    expect(within(row).queryByText('12.0')).not.toBeInTheDocument()
  })

  it('preserves the original PPR actual from an older API without format metadata', async () => {
    vi.mocked(getPredictions).mockImplementation(async () => {
      const result = response('ppr')
      delete result.predictions[0].actual_scoring_format
      return result
    })
    mount()
    const row = (await screen.findByRole('link', { name: 'Receiver' })).closest('tr')!
    expect(within(row).getByText('12.0')).toBeInTheDocument()
  })

})
