import type { MetricBlock, PerformanceArtifact } from '@/lib/api/types'

export type EvaluationMetric = 'mae' | 'median_ae' | 'rmse' | 'within_3_rate' | 'within_5_rate'
export const EVALUATION_METRICS: EvaluationMetric[] = ['mae', 'median_ae', 'rmse', 'within_3_rate', 'within_5_rate']
export const METRIC_LABELS: Record<EvaluationMetric, string> = {
  mae: 'Average absolute error', median_ae: 'Median absolute error', rmse: 'Root mean squared error',
  within_3_rate: 'Within 3 points', within_5_rate: 'Within 5 points',
}
export const METRIC_EXPLANATIONS: Record<EvaluationMetric, string> = {
  mae: 'The average distance between a projection and the recorded PPR score. It is not a guaranteed range.',
  median_ae: 'The middle absolute error: half of evaluated errors were no larger than this value.',
  rmse: 'An error measure that gives larger misses more weight. Lower is better.',
  within_3_rate: 'The share of evaluated projections within 3 PPR points of the recorded score. It is not a probability for your next decision.',
  within_5_rate: 'The share of evaluated projections within 5 PPR points of the recorded score. It is not a probability for your next decision.',
}

export function finiteMetric(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
}

/** The two registered kinds describe complete historical season evaluations, never rolling weeks.
 * Order by period and kind, not by the error or win-rate values being displayed.
 */
export function supportedEvaluations(evaluations: PerformanceArtifact[]): PerformanceArtifact[] {
  return evaluations.filter((a) => ['frozen_test', 'out_of_sample_season'].includes(a.kind)
    && Number.isInteger(a.season) && a.season > 0 && a.split?.test_start_season === a.season
    && a.split?.test_start_week === 1 && (finiteMetric(a.metrics?.n) ?? 0) > 0)
    .slice().sort((a, b) => b.season - a.season
      || Number(b.kind === 'out_of_sample_season') - Number(a.kind === 'out_of_sample_season')
      || a.eval_id.localeCompare(b.eval_id))
}

export function selectEvaluation(evaluations: PerformanceArtifact[], requestedId: string | null, origin: 'query' | 'saved' | 'default' | 'user') {
  const supported = supportedEvaluations(evaluations)
  const requested = requestedId ? supported.find((a) => a.eval_id === requestedId) : undefined
  if (requested) return { artifact: requested, notice: null, staleSaved: false }
  if (requestedId && origin !== 'saved') return {
    artifact: undefined, notice: 'That evaluation is unavailable or is not a supported complete-season evaluation. Choose one of the available evaluations below.', staleSaved: false,
  }
  return {
    artifact: supported[0], staleSaved: Boolean(requestedId),
    notice: requestedId ? 'Your saved evaluation is no longer available. The latest supported historical evaluation is shown.' : null,
  }
}

export function cohortMetrics(artifact: PerformanceArtifact, cohort: string): { model: MetricBlock; baseline: MetricBlock } | null {
  // ALL is the artifact aggregate. Never average the position rows to manufacture one.
  if (cohort === 'ALL') return { model: artifact.metrics, baseline: artifact.baseline }
  const found = artifact.cohorts?.[cohort]
  return found ? { model: found, baseline: found.baseline } : null
}

export function metricValue(block: MetricBlock | undefined, metric: EvaluationMetric): number | null {
  const value = finiteMetric(block?.[metric])
  return metric.endsWith('_rate') && value !== null && value > 1 ? null : value
}

export function displayMetric(value: number | null, metric: EvaluationMetric): string {
  if (value === null) return 'Unavailable'
  return metric.endsWith('_rate') ? `${(value * 100).toFixed(1)}%` : value.toFixed(2)
}

export interface EvaluationComparison {
  state: 'model_better' | 'baseline_better' | 'tied' | 'unavailable'
  model: number | null
  baseline: number | null
  difference: number | null
  relativeImprovement: number | null
  description: string
}

/** Presentation math over the same compatible rows; this does not alter evaluation artifacts. */
export function compareEvaluationMetric(model: MetricBlock | undefined, baseline: MetricBlock | undefined, metric: EvaluationMetric): EvaluationComparison {
  const m = metricValue(model, metric), b = metricValue(baseline, metric)
  const unavailable: EvaluationComparison = { state: 'unavailable', model: m, baseline: b, difference: null, relativeImprovement: null, description: 'A compatible model and historical-baseline comparison is unavailable for this selection.' }
  if (m === null || b === null || !model || !baseline || finiteMetric(model.n) === null || model.n <= 0 || model.n !== baseline.n) return unavailable
  const rate = metric.endsWith('_rate')
  const improvement = rate ? m - b : b - m
  const state = improvement === 0 ? 'tied' : improvement > 0 ? 'model_better' : 'baseline_better'
  const absolute = Math.abs(m - b)
  const scaled = rate ? absolute * 100 : absolute
  const precision = rate ? 1 : 2
  const minimum = Math.pow(10, -precision)
  const delta = scaled > 0 && scaled < minimum ? `less than ${minimum.toFixed(precision)}` : scaled.toFixed(precision)
  const description = state === 'tied'
    ? `The model and historical baseline have the same ${metric.endsWith('_rate') ? 'recorded rate' : 'recorded error'} in this evaluated population.`
    : rate
      ? `The model's within-tolerance rate was ${delta} percentage points ${m > b ? 'higher' : 'lower'} in this evaluated population.`
      : `The model's ${metric === 'mae' ? 'average error' : metric === 'median_ae' ? 'median error' : 'root mean squared error'} was ${delta} PPR points ${m < b ? 'lower' : 'higher'} in this evaluated population.`
  return {
    state, model: m, baseline: b, difference: absolute,
    relativeImprovement: !rate && b > 0 ? (b - m) / b : null,
    description,
  }
}
