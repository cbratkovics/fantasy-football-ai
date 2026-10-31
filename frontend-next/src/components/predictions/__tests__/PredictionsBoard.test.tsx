import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { PredictionsBoard } from '../PredictionsBoard'
import { getPredictionPeriods, getPredictions } from '@/lib/api/players'
import type { PredictionsResponse, ScoringFormat } from '@/lib/api/types'

vi.mock('@/lib/api/players', () => ({ getPredictionPeriods: vi.fn(), getPredictions: vi.fn() }))
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
  vi.clearAllMocks()
  localStorage.clear()
  window.history.replaceState({}, '', '/predictions')
  vi.mocked(getPredictionPeriods).mockResolvedValue({ latest: null, periods: [{ season: 2026, week: 8, model_version: 'model', feature_version: 'asof_v1', generated_at_utc: '2026-10-27T12:00:00Z', n: 4 }] })
  vi.mocked(getPredictions).mockImplementation(async (_season, _week, format = 'ppr') => response(format))
})

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}><PredictionsBoard /></QueryClientProvider>)
}

describe('published actual scoring', () => {
  it('starts with PPR and FLEX and offers only indexed publication periods', async () => {
    mount()
    await screen.findByRole('table')
    expect(screen.getByRole('button', { name: 'FLEX' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'ppr' })).toHaveAttribute('aria-pressed', 'true')
    expect(within(screen.getByLabelText('Published period')).getAllByRole('option')).toHaveLength(1)
    expect(within(screen.getByRole('table')).queryByRole('link', { name: 'Negative' })).not.toBeInTheDocument()
  })

  it('keeps projection rank when searching for one lower-ranked player', async () => {
    const user = userEvent.setup()
    mount()
    await screen.findByRole('table')
    await user.type(screen.getByLabelText('Find a player or forecast team'), 'Zero')
    const row = within(screen.getByRole('table')).getByRole('link', { name: 'Zero' }).closest('tr')!
    expect(within(row).getAllByRole('cell')[0]).toHaveTextContent('2')
  })

  it('suppresses repetitive no-outcome cells but distinguishes conversion withholding', async () => {
    vi.mocked(getPredictions).mockImplementation(async (_s, _w, format = 'ppr') => {
      const data = response(format)
      data.predictions = data.predictions.map((r) => ({ ...r, actual: null, actual_null_reason: format === 'ppr' ? 'outcome_unavailable' : 'conversion_evidence_missing' }))
      return data
    })
    const user = userEvent.setup()
    mount()
    expect(await screen.findByText('Results have not been recorded in this snapshot.')).toBeInTheDocument()
    expect(screen.queryByRole('columnheader', { name: /Actual/ })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'half' }))
    expect(await screen.findByText('Actual scores are unavailable in this scoring format.')).toBeInTheDocument()
    expect(screen.queryByText(/Games have not been played/)).not.toBeInTheDocument()
  })

  it('explains an unpublished explicit period and waits for an intentional valid selection', async () => {
    window.history.replaceState({}, '', '/predictions?season=2026&week=18')
    mount()
    await screen.findByText(/That requested period has no published projection file/)
    expect(getPredictions).not.toHaveBeenCalled()
    await userEvent.setup().selectOptions(screen.getByLabelText('Published period'), '2026-8')
    expect(await screen.findByRole('table')).toBeInTheDocument()
  })

  it.each([['ppr', '12.0'], ['half', '9.0'], ['standard', '6.0']] as const)('renders %s actuals and their declared rules', async (format, expected) => {
    const user = userEvent.setup()
    mount()
    await within(await screen.findByRole('table')).findByRole('link', { name: 'Receiver' })
    if (format !== 'ppr') await user.click(screen.getByRole('button', { name: format }))
    const row = (await within(await screen.findByRole('table')).findByRole('link', { name: 'Receiver' })).closest('tr')!
    await within(row).findByText(expected)
    expect(screen.getByRole('columnheader', { name: `Actual (${format.toUpperCase()})` })).toBeInTheDocument()
    await user.click(screen.getByText('Scoring, model range and source details'))
    expect(screen.getByText(/4 points per passing TD; no bonuses/)).toBeVisible()
    if (format !== 'ppr') expect(screen.getByText(/Derived forecast estimate/)).toBeInTheDocument()
  })

  it('preserves observed zero and negative points and explains absent conversion evidence', async () => {
    mount()
    await within(await screen.findByRole('table')).findByRole('link', { name: 'Zero' })
    await userEvent.setup().click(screen.getByRole('button', { name: 'ALL' }))
    const zero = within(screen.getByRole('table')).getByRole('link', { name: 'Zero' }).closest('tr')!
    const negative = within(screen.getByRole('table')).getByRole('link', { name: 'Negative' }).closest('tr')!
    expect(within(zero).getByText('0.0')).toBeInTheDocument()
    expect(within(negative).getByText('-2.0')).toBeInTheDocument()
    expect(within(screen.getByRole('table')).getByText('Actual unavailable: missing realized conversion evidence')).toBeInTheDocument()
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
    await within(await screen.findByRole('table')).findByRole('link', { name: 'Receiver' })
    if (format !== 'ppr') await user.click(screen.getByRole('button', { name: format }))
    const row = (await within(await screen.findByRole('table')).findByRole('link', { name: 'Receiver' })).closest('tr')!
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
    const row = (await within(await screen.findByRole('table')).findByRole('link', { name: 'Receiver' })).closest('tr')!
    expect(within(row).getByText('12.0')).toBeInTheDocument()
  })

})
