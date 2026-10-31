import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { PerformanceArtifact } from '@/lib/api/types'
import { getPerformance } from '@/lib/api/players'
import { PerformanceDashboard } from '../PerformanceDashboard'

vi.mock('@/lib/api/players', async (original) => ({ ...await original<typeof import('@/lib/api/players')>(), getPerformance: vi.fn() }))
const values = (mae: number, n: number, rate: number) => ({ mae, n, median_ae: mae / 2, rmse: mae * 2, within_3_rate: rate, within_5_rate: rate + 0.1 })
function evaluation(season: number): PerformanceArtifact {
  return {
    artifact_version: '2.1', eval_id: `test-eval-${season}`, season, kind: 'out_of_sample_season',
    split: { strategy: 'forward_time_holdout', test_start_season: season, test_start_week: 1 },
    metrics: values(4, 100, 0.46), baseline: values(5, 100, 0.43),
    cohorts: { RB: { ...values(3, 30, 0.6), baseline: values(2, 30, 0.7) } },
    rolling_origin: { candidate: 'rf', folds: [], mean_mae: 4.1, mean_baseline_mae: 5.1, strategy: 'earlier weeks only' },
    model: { version: 'test-model', feature_version: 'test', candidate: {}, challenger: {}, trained_at_utc: '2026-01-01T00:00:00Z', input_sha256: 'a'.repeat(64) },
    generated_at_utc: '2026-02-01T00:00:00Z', code_commit: 'test-commit',
    input: { path: 'tests/fixture.csv', sha256: 'a'.repeat(64), n_rows: 100 },
    metric_definitions: { mae: 'mean absolute error', baseline: 'causal trailing mean' }, policy_sweep: [],
  } as PerformanceArtifact
}
function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}><PerformanceDashboard /></QueryClientProvider>)
}
beforeEach(() => { localStorage.clear(); window.history.replaceState({}, '', '/performance'); vi.mocked(getPerformance).mockResolvedValue({ model_version: 'test-model', feature_version: 'test', evaluations: [evaluation(2024), evaluation(2025)] }) })
afterEach(() => { cleanup(); vi.clearAllMocks() })

describe('readable model results', () => {
  it('defaults to newest season and average miss with source details closed', async () => {
    mount()
    const finding = await screen.findByTestId('evaluation-finding')
    expect(screen.getByLabelText('Evaluation')).toHaveValue('test-eval-2025')
    expect(screen.getByLabelText('Measure')).toHaveValue('mae')
    expect(finding).toHaveTextContent('Average miss: 4.00 PPR points')
    expect(finding).toHaveTextContent('Historical baseline: 43.0%')
    expect(finding).toHaveTextContent('100')
    expect(screen.getByText('View methodology and source evidence').closest('details')).not.toHaveAttribute('open')
    expect(screen.queryByText('test-model / test')).not.toBeInTheDocument()
  })
  it('updates headline, baseline comparison, rate, sample and chart with position/metric changes', async () => {
    const user = userEvent.setup(); mount()
    await screen.findByTestId('evaluation-finding')
    await user.selectOptions(screen.getByLabelText('Position group'), 'RB')
    const finding = screen.getByTestId('evaluation-finding')
    expect(finding).toHaveTextContent('Average miss: 3.00 PPR points')
    expect(finding).toHaveTextContent('1.00 PPR points higher')
    expect(finding).toHaveTextContent('Historical baseline: 70.0%')
    expect(finding).toHaveTextContent('30')
    await user.selectOptions(screen.getByLabelText('Measure'), 'within_3_rate')
    expect(within(finding).getByRole('heading', { level: 2 })).toHaveTextContent('Within 3 points: 60.0%')
    expect(finding).toHaveTextContent('10.0 percentage points lower')
    expect(screen.getByLabelText('Within 3 points comparison')).toHaveTextContent('Historical baseline70.0%')
  })
  it('honors a deep link over saved settings and refuses an invalid requested evaluation', async () => {
    localStorage.setItem('ffai.display.evaluation/1', JSON.stringify({ evaluationId: 'test-eval-2025', position: 'ALL', metric: 'rmse' }))
    window.history.replaceState({}, '', '/performance?evaluation=test-eval-2024&position=RB&metric=mae')
    const view = mount(); await screen.findByTestId('evaluation-finding')
    expect(screen.getByLabelText('Evaluation')).toHaveValue('test-eval-2024')
    expect(screen.getByLabelText('Position group')).toHaveValue('RB')
    view.unmount(); window.history.replaceState({}, '', '/performance?evaluation=missing')
    mount()
    await screen.findByRole('alert')
    expect(screen.queryByTestId('evaluation-finding')).not.toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('not silently substituted')
  })
  it('clears stale saved evaluation with a readable notice', async () => {
    localStorage.setItem('ffai.display.evaluation/1', JSON.stringify({ evaluationId: 'retired', position: 'ALL', metric: 'mae' }))
    mount(); await screen.findByTestId('evaluation-finding')
    expect(screen.getByRole('status')).toHaveTextContent('saved evaluation is no longer available')
    expect(screen.getByLabelText('Evaluation')).toHaveValue('test-eval-2025')
    expect(JSON.parse(localStorage.getItem('ffai.display.evaluation/1')!)).toEqual({ evaluationId: 'test-eval-2025', position: 'ALL', metric: 'mae' })
  })
})
