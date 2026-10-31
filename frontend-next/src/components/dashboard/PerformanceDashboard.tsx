'use client'

import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { MetricBlock } from '@/lib/api/types'
import { EVAL_KIND_EXPLANATION, evaluationLabel, getPerformance } from '@/lib/api/players'
import { fmtInt, fmtNum, fmtUtc } from '@/lib/format'
import { formatLocalDate, readPreference, writePreference } from '@/lib/presentation/display'
import { EVALUATION_METRICS, METRIC_LABELS, METRIC_EXPLANATIONS, cohortMetrics, compareEvaluationMetric, displayMetric, metricValue, selectEvaluation, supportedEvaluations, type EvaluationMetric } from '@/lib/presentation/evaluation'
import { ErrorState, LoadingState } from '@/components/ui/States'
import { RollingOriginChart } from './RollingOriginChart'
import { DecisionsPanel } from './DecisionsPanel'

const PREF_KEY = 'ffai.display.evaluation/1'
const FOCUS = 'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#176e5a]'
interface View { evaluationId: string | null; origin: 'query' | 'saved' | 'default' | 'user'; position: string; metric: EvaluationMetric; notice: string | null }
function initialView(): View {
  const query = new URLSearchParams(typeof window === 'undefined' ? '' : window.location.search)
  const raw = readPreference(PREF_KEY)
  const saved = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {}
  const explicit = query.get('evaluation') ?? query.get('eval')
  const savedId = typeof saved.evaluationId === 'string' ? saved.evaluationId : null
  const metric = query.get('metric') ?? saved.metric
  const validMetric = EVALUATION_METRICS.includes(metric as EvaluationMetric)
  return {
    evaluationId: explicit ?? savedId, origin: explicit !== null ? 'query' : savedId !== null ? 'saved' : 'default',
    position: query.get('position') ?? (typeof saved.position === 'string' ? saved.position : 'ALL'),
    metric: validMetric ? metric as EvaluationMetric : 'mae',
    notice: metric && !validMetric ? 'That saved or requested measure is unavailable. Average absolute error is shown.' : null,
  }
}

/** Presentation only: all measurements remain from one selected historical artifact. */
export function PerformanceDashboard() {
  const perf = useQuery({ queryKey: ['performance'], queryFn: getPerformance })
  const [view, setView] = useState<View>(initialView)
  const evaluations = useMemo(() => supportedEvaluations(perf.data?.evaluations ?? []), [perf.data])
  const selection = selectEvaluation(evaluations, view.evaluationId, view.origin)
  const a = selection.artifact
  const positions = a ? ['ALL', ...Object.keys(a.cohorts ?? {})] : ['ALL']
  const position = positions.includes(view.position) ? view.position : view.origin === 'saved' ? 'ALL' : view.position
  const selected = a ? cohortMetrics(a, position) : null
  const comparison = compareEvaluationMetric(selected?.model, selected?.baseline, view.metric)
  const within = compareEvaluationMetric(selected?.model, selected?.baseline, 'within_3_rate')
  const positionLabel = position === 'ALL' ? 'All positions' : position
  const notice = view.notice ?? selection.notice ?? (view.position !== position ? 'Your saved position group is unavailable. All positions are shown.' : null)
  useEffect(() => {
    if (selection.staleSaved && a) writePreference(PREF_KEY, { evaluationId: a.eval_id, position, metric: view.metric })
  }, [selection.staleSaved, a, position, view.metric])
  const update = (change: Partial<View>) => {
    const next = { ...view, ...change, evaluationId: change.evaluationId ?? a?.eval_id ?? view.evaluationId, origin: a ? 'user' as const : view.origin, notice: null }
    setView(next)
    writePreference(PREF_KEY, { evaluationId: next.evaluationId, position: next.position, metric: next.metric })
  }
  if (perf.isPending) return <LoadingState label="Loading historical model results…" />
  if (perf.isError) return <ErrorState error={perf.error} context="Historical evaluation data could not be loaded" />
  if (!evaluations.length) return <section className="product-card"><h2 className="text-xl font-semibold">No complete historical evaluation is available</h2><p className="mt-2">Model results will appear when a supported season evaluation is published.</p><a href="/predictions" className={`mt-3 inline-flex min-h-[44px] items-center font-semibold text-[#145744] underline ${FOCUS}`}>Browse player projections</a></section>

  return <div className="space-y-5" data-testid="performance-dashboard">
    <section className="product-card" aria-label="Choose historical results">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <label className="col-span-2 grid min-w-0 gap-2 text-base font-medium sm:col-span-1" htmlFor="evaluation-period">Evaluation<select id="evaluation-period" className={`product-field min-h-[44px] min-w-0 w-full rounded-lg border border-[#c7ccc5] bg-white px-3 py-2 text-base ${FOCUS}`} value={a?.eval_id ?? ''} onChange={(e) => update({ evaluationId: e.target.value, origin: 'user' })}>{!a && <option value="">Choose an available evaluation</option>}{evaluations.map((e) => <option key={e.eval_id} value={e.eval_id}>{evaluationLabel(e)}</option>)}</select></label>
        <label className="grid min-w-0 gap-2 text-base font-medium" htmlFor="evaluation-position">Position group<select id="evaluation-position" className={`product-field min-h-[44px] min-w-0 w-full rounded-lg border border-[#c7ccc5] bg-white px-3 py-2 text-base ${FOCUS}`} value={position} onChange={(e) => update({ position: e.target.value })}>{!positions.includes(position) && <option value={position}>{position} — unavailable</option>}{positions.map((p) => <option key={p} value={p}>{p === 'ALL' ? 'All positions' : p}</option>)}</select></label>
        <label className="grid min-w-0 gap-2 text-base font-medium" htmlFor="evaluation-metric">Measure<select id="evaluation-metric" className={`product-field min-h-[44px] min-w-0 w-full rounded-lg border border-[#c7ccc5] bg-white px-3 py-2 text-base ${FOCUS}`} value={view.metric} onChange={(e) => update({ metric: e.target.value as EvaluationMetric })}>{EVALUATION_METRICS.map((m) => <option key={m} value={m}>{m === 'mae' ? 'Average miss' : m === 'median_ae' ? 'Median miss' : m === 'rmse' ? 'RMS error' : METRIC_LABELS[m]}</option>)}</select></label>
      </div>
      {notice && <p className="mt-3 text-sm text-[#4b5b54]" role="status">{notice}</p>}
      {a && <p className="mt-3 text-sm text-[#4b5b54]">{a.season} historical season · {positionLabel} · Implemented PPR scoring</p>}
    </section>
    {!a && <section className="product-card" role="alert"><h2 className="text-xl font-semibold">Choose an available evaluation</h2><p className="mt-2">{selection.notice}</p><p className="mt-2 text-sm">Another evaluation was not silently substituted for the requested one.</p></section>}
    {a && !selected && <section className="product-card" role="alert"><h2 className="text-xl font-semibold">This position group is unavailable</h2><p className="mt-2">Choose a position group above. No aggregate was substituted for your selection.</p></section>}
    {a && selected && <>
      <section className="product-card" aria-label="Selected evaluation finding" data-testid="evaluation-finding">
        <p className="text-sm font-medium text-[#4b5b54]">{positionLabel} · {a.season} · {a.kind === 'frozen_test' ? 'Held-out test season' : 'Complete out-of-sample season'}</p>
        <h2 className="mt-2 text-2xl font-semibold leading-snug text-[#173c30]">{view.metric === 'mae' ? 'Average miss' : METRIC_LABELS[view.metric]}: {displayMetric(comparison.model, view.metric)}{!view.metric.endsWith('_rate') && comparison.model !== null ? ' PPR points' : ''}</h2>
        <p className="mt-3 max-w-4xl text-base leading-7">{comparison.state === 'unavailable' ? comparison.description : <>Compared with <strong>{displayMetric(comparison.baseline, view.metric)}{!view.metric.endsWith('_rate') ? ' PPR points' : ''}</strong> for the historical baseline. {comparison.description}</>}</p>
        <div className="mt-6 grid gap-6 lg:grid-cols-[1.1fr_.9fr]">
          <div><ComparisonBars model={comparison.model} baseline={comparison.baseline} metric={view.metric} /><p className="mt-3 text-sm leading-6 text-[#4b5b54]">{METRIC_EXPLANATIONS[view.metric]}</p></div>
          <div className="grid gap-5 border-t border-[#d8ddd6] pt-5 sm:grid-cols-2 lg:border-l lg:border-t-0 lg:pl-6 lg:pt-0">
            <div><h3 className="text-base font-medium">Within 3 PPR points</h3><p className="mt-2 text-2xl font-semibold tabular-nums">{displayMetric(within.model, 'within_3_rate')}</p><p className="mt-1 text-sm text-[#4b5b54]">Historical baseline: {displayMetric(within.baseline, 'within_3_rate')}</p><p className="mt-2 text-sm leading-6">A historical frequency, not the chance your next decision succeeds.</p></div>
            <div><h3 className="text-base font-medium">Evaluated player-weeks</h3><p className="mt-2 text-2xl font-semibold tabular-nums">{fmtInt(selected.model.n)}</p><p className="mt-1 text-sm text-[#4b5b54]">{positionLabel}, {a.season} season</p><p className="mt-2 text-sm leading-6">Repeated observations of players, not unique users or lineup decisions.</p></div>
          </div>
        </div>
        <p className="mt-5 border-t border-[#d8ddd6] pt-4 text-sm leading-6 text-[#4b5b54]">Historical performance on this evaluated population does not establish current availability or lineup-win performance. This scores a frozen model, separate from the refit-by-week experiment below.</p>
      </section>
      <Disclosure title="Understand the historical baseline and this comparison">
        <p>The historical baseline uses a causal trailing mean of the player&apos;s earlier recorded points, with a position-history fallback. A week&apos;s outcomes are added only after that week is scored. It is not a fixed last-three-games average.</p>
        <p>{EVAL_KIND_EXPLANATION[a.kind]}. The values above come from the selected artifact&apos;s aggregate or exact position block; position averages are never pooled to invent an overall result.</p>
        <p>{comparison.state === 'unavailable' ? 'The selected values are missing or refer to different sample sizes.' : view.metric.endsWith('_rate') ? comparison.description : comparison.relativeImprovement === null ? 'Relative error reduction is unavailable because the baseline denominator is zero.' : `Relative error ${comparison.relativeImprovement >= 0 ? 'reduction' : 'increase'}: ${Math.abs(comparison.relativeImprovement * 100).toFixed(1)}%, calculated from unrounded compatible values.`}</p>
        <p>These are sample descriptions, not evidence of statistical significance, calibrated confidence, or a future guarantee.</p>
      </Disclosure>
      <Disclosure title="More results by position and season">
        <p className="text-sm">Values belong to their named season and position group. Model and baseline rows use the same player-weeks.</p>
        <div className="overflow-x-auto"><table className="w-full min-w-[650px] text-left text-sm tabular-nums"><caption className="mb-3 text-left text-base font-semibold">All measures · {a.season}</caption><thead><tr><th className="py-3 pr-4">Position / source</th><th className="px-3 text-right">Player-weeks</th>{EVALUATION_METRICS.map((m) => <th key={m} className="px-3 text-right">{METRIC_LABELS[m]}</th>)}</tr></thead><tbody>{['ALL', ...Object.keys(a.cohorts ?? {})].map((c) => { const blocks = cohortMetrics(a, c); return blocks ? <MetricRows key={c} cohort={c} model={blocks.model} baseline={blocks.baseline} /> : null })}</tbody></table></div>
        {evaluations.length > 1 && <div className="overflow-x-auto"><table className="w-full min-w-[500px] text-left text-sm tabular-nums"><caption className="mb-3 text-left text-base font-semibold">Complete historical evaluations · all positions</caption><thead><tr><th className="py-3 pr-4">Evaluation</th><th className="px-3 text-right">Average miss</th><th className="px-3 text-right">Baseline miss</th><th className="px-3 text-right">Player-weeks</th></tr></thead><tbody>{evaluations.map((e) => <tr key={e.eval_id} className="border-t border-[#d8ddd6]"><td className="py-3 pr-4"><button className={`min-h-[44px] text-left font-medium text-[#145744] underline ${FOCUS}`} onClick={() => update({ evaluationId: e.eval_id, origin: 'user' })}>{evaluationLabel(e)}</button></td><td className="px-3 text-right">{displayMetric(metricValue(e.metrics, 'mae'), 'mae')}</td><td className="px-3 text-right">{displayMetric(metricValue(e.baseline, 'mae'), 'mae')}</td><td className="px-3 text-right">{fmtInt(e.metrics.n)}</td></tr>)}</tbody></table></div>}
      </Disclosure>
      <Disclosure title="Rolling-origin experiment: refit before each scored week">
        <p>This experiment refits using only earlier weeks. Its fold averages are not pooled with the frozen-model results above.</p>
        <p>Mean fold error: <strong>{fmtNum(a.rolling_origin.mean_mae, 2)} PPR points</strong>; baseline {fmtNum(a.rolling_origin.mean_baseline_mae, 2)} across {a.rolling_origin.folds.length} folds. These folds are for all positions and do not change with the position selector.</p>
        <RollingOriginChart key={a.eval_id} folds={a.rolling_origin.folds} label={evaluationLabel(a)} />
        <details><summary className={`cursor-pointer font-medium ${FOCUS}`}>View every scored fold</summary><div className="mt-3 max-h-80 overflow-auto"><table className="w-full min-w-[420px] text-left text-sm tabular-nums"><thead><tr><th className="py-3">Season / week</th><th className="text-right">Player-weeks</th><th className="text-right">Model error</th><th className="text-right">Baseline error</th></tr></thead><tbody>{a.rolling_origin.folds.map((f) => <tr key={`${f.season}-${f.week}`} className="border-t border-[#d8ddd6]"><td className="py-3">{f.season} · week {f.week}</td><td className="text-right">{fmtInt(f.n)}</td><td className="text-right">{fmtNum(f.mae, 2)}</td><td className="text-right">{fmtNum(f.baseline_mae, 2)}</td></tr>)}</tbody></table></div></details>
      </Disclosure>
      <Disclosure title="View methodology and source evidence">
        <p>Evaluation generated {formatLocalDate(a.generated_at_utc)}. Exact UTC: {fmtUtc(a.generated_at_utc)}.</p>
        <dl className="grid gap-4 text-sm sm:grid-cols-2"><Evidence label="Artifact ID" value={a.eval_id} /><Evidence label="Model / features" value={`${a.model.version} / ${a.model.feature_version}`} /><Evidence label="Input file" value={a.input.path} /><Evidence label="Input SHA-256" value={a.input.sha256} /><Evidence label="Code commit" value={a.code_commit} /><Evidence label="Split" value={`${a.split.strategy}; begins season ${a.split.test_start_season}, week ${a.split.test_start_week}`} /></dl>
        <p>Model trained {formatLocalDate(a.model.trained_at_utc)}. The evaluation input contains {fmtInt(a.input.n_rows)} player-week rows; headline samples use the exact selected metric block.</p>
        <h3 className="text-base font-semibold">Metric definitions from the artifact</h3><dl className="space-y-3 text-sm">{Object.entries(a.metric_definitions).map(([name, definition]) => <div key={name}><dt className="font-mono">{name}</dt><dd>{definition}</dd></div>)}</dl>
        <p className="text-sm">Implemented PPR: 1/reception, passing yards /25, four-point passing TD, −2/interception, rushing/receiving yards /10, six-point rushing/receiving/special-teams TD, two-point conversion +2 and lost fumble −2. No yardage bonuses; custom league compatibility is unverified.</p>
      </Disclosure>
      <Disclosure title="Optional historical floor-filter experiment">
        <p className="text-sm">A separate educational filter over historical outcomes, not a current lineup recommendation. Its own controls do not change the evaluation above.</p>
        <DecisionsPanel seasons={Array.from(new Set(evaluations.map((e) => e.season))).sort()} />
      </Disclosure>
    </>}
  </div>
}
function Disclosure({ title, children }: { title: string; children: React.ReactNode }) {
  const [opened, setOpened] = useState(false)
  return <details onToggle={(event) => { if (event.currentTarget.open) setOpened(true) }} className="product-details min-w-0 rounded-xl border border-[#c7ccc5] bg-white p-5"><summary className={`cursor-pointer text-base font-semibold ${FOCUS}`}>{title}</summary>{opened && <div className="mt-4 min-w-0 space-y-4 break-words text-base leading-7">{children}</div>}</details>
}
function Evidence({ label, value }: { label: string; value: string }) {
  return <div className="min-w-0"><dt className="font-semibold">{label}</dt><dd className="mt-1 break-all font-mono">{value}</dd></div>
}
function MetricRows({ cohort, model, baseline }: { cohort: string; model: MetricBlock; baseline: MetricBlock }) {
  return <>{([['Model', model], ['Historical baseline', baseline]] as const).map(([name, block]) => <tr key={name} className="border-t border-[#d8ddd6]"><th scope="row" className="py-3 pr-4 font-medium">{cohort === 'ALL' ? 'All positions' : cohort} · {name}</th><td className="px-3 text-right">{fmtInt(block.n)}</td>{EVALUATION_METRICS.map((m) => <td key={m} className="px-3 text-right">{displayMetric(metricValue(block, m), m)}</td>)}</tr>)}</>
}
function ComparisonBars({ model, baseline, metric }: { model: number | null; baseline: number | null; metric: EvaluationMetric }) {
  const max = metric.endsWith('_rate') ? 1 : Math.max(model ?? 0, baseline ?? 0, 0.01) * 1.08
  return <div className="space-y-4" aria-label={`${METRIC_LABELS[metric]} comparison`}>{([['Model', model, '#176e5a'], ['Historical baseline', baseline, '#647b6b']] as const).map(([name, value, color]) => <div key={name}><div className="mb-2 flex justify-between gap-3 text-base"><span className="font-medium">{name}</span><strong className="tabular-nums">{displayMetric(value, metric)}</strong></div><div className="h-3 overflow-hidden rounded-full bg-[#e6e9e2]" aria-hidden="true">{value !== null && <div className="h-full rounded-full" style={{ width: `${Math.max(0, Math.min(100, value / max * 100))}%`, backgroundColor: color }} />}</div></div>)}<p className="text-sm text-[#4b5b54]">{metric.endsWith('_rate') ? 'Higher is better. Shared scale: 0–100%.' : 'Lower is better. Both bars use the same scale from zero.'}</p></div>
}
