import { describe, expect, it } from 'vitest'
import type { MetricBlock, PerformanceArtifact } from '@/lib/api/types'
import { cohortMetrics, compareEvaluationMetric, displayMetric, selectEvaluation, supportedEvaluations } from '../evaluation'

const block = (mae: number, n = 100, rate = 0.4): MetricBlock => ({ n, mae, median_ae: mae / 2, rmse: mae * 2, within_3_rate: rate, within_5_rate: rate + 0.1 })
const artifact = (season: number, mae: number, id = `eval-${season}`): PerformanceArtifact => ({
  artifact_version: '2.1', eval_id: id, season, kind: 'out_of_sample_season', split: { strategy: 'forward_time_holdout', test_start_season: season, test_start_week: 1 },
  metrics: block(mae), baseline: block(5), cohorts: { RB: { ...block(3, 30), baseline: block(4, 30) } },
  generated_at_utc: '2026-01-01T00:00:00Z', code_commit: 'test', input: { path: 'tests/fixture.csv', sha256: 'a'.repeat(64), n_rows: 100 },
  model: { version: 'test', feature_version: 'test', candidate: { RB: 'rf' }, challenger: { RB: 'xgb' }, trained_at_utc: '2025-01-01T00:00:00Z', input_sha256: 'a'.repeat(64) },
  rolling_origin: { candidate: 'rf', strategy: 'earlier data', folds: [], mean_mae: mae, mean_baseline_mae: 5 }, metric_definitions: { mae: 'mean absolute error' }, policy_sweep: [],
})

describe('evaluation display defaults', () => {
  it('selects the newest supported historical season, not the lowest error', () => {
    expect(selectEvaluation([artifact(2024, 1), artifact(2025, 8)], null, 'default').artifact?.eval_id).toBe('eval-2025')
  })
  it('honors an explicit older period', () => {
    expect(selectEvaluation([artifact(2025, 3), artifact(2024, 8)], 'eval-2024', 'query').artifact?.season).toBe(2024)
  })
  it('does not silently replace an unknown explicit evaluation; stale saved state gets a notice', () => {
    expect(selectEvaluation([artifact(2025, 3)], 'missing', 'query').artifact).toBeUndefined()
    const saved = selectEvaluation([artifact(2025, 3)], 'missing', 'saved')
    expect(saved.artifact?.season).toBe(2025)
    expect(saved.staleSaved).toBe(true)
    expect(saved.notice).toMatch(/saved evaluation/)
  })
  it('excludes partial/unsupported evaluation metadata from complete-season defaults', () => {
    const partial = artifact(2026, 0.1); partial.split.test_start_week = 7
    const weekly = { ...artifact(2027, 0.1), kind: 'weekly' } as unknown as PerformanceArtifact
    expect(supportedEvaluations([weekly, partial, artifact(2025, 6)]).map((a) => a.season)).toEqual([2025])
  })
})
describe('historical comparison presentation', () => {
  it('uses the published aggregate instead of averaging position errors', () => {
    const a = artifact(2025, 8)
    expect(cohortMetrics(a, 'ALL')?.model.mae).toBe(8)
    expect(cohortMetrics(a, 'RB')?.model.n).toBe(30)
    expect(cohortMetrics(a, 'missing')).toBeNull()
  })
  it('describes model-better, baseline-better and exact tie equally', () => {
    expect(compareEvaluationMetric(block(4), block(5), 'mae').description).toContain('1.00 PPR points lower')
    expect(compareEvaluationMetric(block(6), block(5), 'mae').description).toContain('1.00 PPR points higher')
    expect(compareEvaluationMetric(block(5), block(5), 'mae').state).toBe('tied')
    expect(compareEvaluationMetric(block(5), block(5), 'mae').description).toContain('same recorded error')
  })
  it('computes relative improvement from unrounded compatible inputs', () => {
    const result = compareEvaluationMetric(block(4.4909), block(4.8046), 'mae')
    expect(result.relativeImprovement).toBeCloseTo((4.8046 - 4.4909) / 4.8046, 12)
    expect(result.description).toContain('0.31 PPR points lower')
  })
  it('handles a zero baseline denominator without an invented percentage', () => {
    const result = compareEvaluationMetric(block(1), block(0), 'mae')
    expect(result.state).toBe('baseline_better')
    expect(result.relativeImprovement).toBeNull()
  })
  it('labels rate differences as percentage points', () => {
    const result = compareEvaluationMetric(block(3, 100, 0.4603), block(4, 100, 0.432), 'within_3_rate')
    expect(result.description).toContain('2.8 percentage points higher')
    expect(result.relativeImprovement).toBeNull()
    expect(displayMetric(result.model, 'within_3_rate')).toBe('46.0%')
  })
  it('rejects missing, nonfinite, impossible-rate and different populations', () => {
    expect(compareEvaluationMetric(undefined, block(4), 'mae').state).toBe('unavailable')
    expect(compareEvaluationMetric(block(NaN), block(4), 'mae').state).toBe('unavailable')
    expect(compareEvaluationMetric(block(3, 100), block(4, 99), 'mae').state).toBe('unavailable')
    expect(compareEvaluationMetric(block(3, 100, 1.5), block(4), 'within_3_rate').state).toBe('unavailable')
    expect(compareEvaluationMetric(block(3, 0), block(4, 0), 'mae').state).toBe('unavailable')
  })
  it('does not turn a small nonzero difference into apparent equality', () => {
    expect(compareEvaluationMetric(block(4.001), block(4.002), 'mae').description).toContain('less than 0.01')
  })
})
