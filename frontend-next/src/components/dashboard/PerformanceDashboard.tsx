'use client'

import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChartBarIcon, CircleStackIcon, ScaleIcon, ShieldCheckIcon } from '@heroicons/react/24/outline'
import { EVAL_KIND_EXPLANATION, EVAL_KIND_LABEL, evaluationLabel, getPerformance, sortEvaluations } from '@/lib/api/players'
import type { MetricBlock, PerformanceArtifact } from '@/lib/api/types'
import { fmtInt, fmtNum, fmtPct, fmtUtc, shortHash } from '@/lib/format'
import { ErrorState, LoadingState } from '@/components/ui/States'
import { RollingOriginChart } from './RollingOriginChart'

type Cohort = 'ALL' | 'QB' | 'RB' | 'WR' | 'TE'
type MetricKey = 'mae' | 'median_ae' | 'rmse' | 'within_3_rate' | 'within_5_rate'

const COHORTS: Cohort[] = ['ALL', 'QB', 'RB', 'WR', 'TE']
const METRIC_KEYS: MetricKey[] = ['mae', 'median_ae', 'rmse', 'within_3_rate', 'within_5_rate']
const METRIC_LABEL: Record<MetricKey, string> = {
  mae: 'MAE',
  median_ae: 'Median AE',
  rmse: 'RMSE',
  within_3_rate: 'Within ±3',
  within_5_rate: 'Within ±5',
}
const isRate = (k: MetricKey) => k.endsWith('_rate')
const show = (k: MetricKey, v: number | undefined) => (isRate(k) ? fmtPct(v) : fmtNum(v, 2))

interface CohortRow {
  cohort: Cohort
  model: MetricBlock
  baseline: MetricBlock
}

function cohortRows(artifact: PerformanceArtifact): CohortRow[] {
  const rows: CohortRow[] = [{ cohort: 'ALL', model: artifact.metrics, baseline: artifact.baseline }]
  for (const c of COHORTS.slice(1)) {
    const block = artifact.cohorts?.[c]
    if (block) rows.push({ cohort: c, model: block, baseline: block.baseline })
  }
  return rows
}

export function PerformanceDashboard() {
  const perf = useQuery({ queryKey: ['performance'], queryFn: getPerformance })
  const [metric, setMetric] = useState<MetricKey>('mae')
  const [cohort, setCohort] = useState<Cohort>('ALL')
  // Which artifact is shown. null = "not chosen yet", which resolves to the highest season.
  const [evalId, setEvalId] = useState<string | null>(null)

  const evaluations = useMemo(() => sortEvaluations(perf.data?.evaluations ?? []), [perf.data])
  const a: PerformanceArtifact | undefined = evaluations.find((e) => e.eval_id === evalId) ?? evaluations[0]

  const rows = useMemo(() => (a ? cohortRows(a) : []), [a])
  const selected = rows.find((r) => r.cohort === cohort) ?? rows[0]
  const chartRows = cohort === 'ALL' ? rows.filter((r) => r.cohort !== 'ALL') : rows.filter((r) => r.cohort === cohort)

  return (
    <>
      <header className="evaluation-hero">
        <div className="eyebrow">
          <span /> Model evaluation · committed artifact
        </div>
        <div className="evaluation-title-row">
          <h1>
            Evidence, not
            <br />
            <em>just output.</em>
          </h1>
          <div>
            <p>
              The champion model is scored on a forward-time holdout against a causal baseline that only ever sees a player&apos;s earlier realised points.
              Read every figure with its sample size and cohort in view.
            </p>
            {a && (
              <span>
                {a.split.strategy.replace(/_/g, ' ').toUpperCase()} · {evaluationLabel(a).toUpperCase()} · FROM WEEK {a.split.test_start_week}
              </span>
            )}
          </div>
        </div>
      </header>

      <section className="evaluation-dashboard" aria-label="Evaluation results">
        {perf.isPending && <LoadingState label="Reading the evaluation artifact…" />}
        {perf.isError && <ErrorState error={perf.error} context="GET /performance" />}

        {a && selected && (
          <>
            <div className="evaluation-toolbar evaluation-selector">
              <div>
                <small>Evaluation</small>
                <div role="tablist" aria-label="Evaluation season">
                  {evaluations.map((e) => (
                    <button
                      key={e.eval_id}
                      role="tab"
                      aria-selected={e.eval_id === a.eval_id}
                      className={e.eval_id === a.eval_id ? 'active' : ''}
                      onClick={() => setEvalId(e.eval_id)}
                    >
                      {evaluationLabel(e)}
                    </button>
                  ))}
                </div>
              </div>
              <p className="max-w-xl text-xs leading-relaxed text-[#52605d]">
                <b className="text-ink">{evaluationLabel(a)}</b>: {EVAL_KIND_EXPLANATION[a.kind] ?? a.kind}.
              </p>
            </div>

            <p className="mb-6 border border-ink bg-acid px-4 py-3 font-mono text-[11px] font-bold text-ink">
              Every figure on this page is read from the committed evaluation artifact {a.eval_id}.
            </p>

            {evaluations.length > 1 && <EvaluationComparison evaluations={evaluations} activeId={a.eval_id} onSelect={setEvalId} />}

            <div className="evaluation-kpis">
              <article>
                <div>
                  <ScaleIcon />
                  <span>MAE · {selected.cohort}</span>
                </div>
                <strong>{fmtNum(selected.model.mae, 2)}</strong>
                <p>baseline {fmtNum(selected.baseline.mae, 2)} · fantasy points</p>
              </article>
              <article>
                <div>
                  <ChartBarIcon />
                  <span>Median AE · {selected.cohort}</span>
                </div>
                <strong>{fmtNum(selected.model.median_ae, 2)}</strong>
                <p>baseline {fmtNum(selected.baseline.median_ae, 2)}</p>
              </article>
              <article>
                <div>
                  <ShieldCheckIcon />
                  <span>Within ±3 · {selected.cohort}</span>
                </div>
                <strong>{fmtPct(selected.model.within_3_rate)}</strong>
                <p>baseline {fmtPct(selected.baseline.within_3_rate)}</p>
              </article>
              <article>
                <div>
                  <CircleStackIcon />
                  <span>Sample · {selected.cohort}</span>
                </div>
                <strong>{fmtInt(selected.model.n)}</strong>
                <p>player-game rows in the {a.season} {EVAL_KIND_LABEL[a.kind] ?? a.kind}</p>
              </article>
            </div>

            <div className="evaluation-toolbar">
              <div>
                <small>Measure</small>
                <div role="group" aria-label="Evaluation measure">
                  {METRIC_KEYS.map((k) => (
                    <button key={k} className={metric === k ? 'active' : ''} onClick={() => setMetric(k)}>
                      {METRIC_LABEL[k]}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <small>Cohort</small>
                <div role="group" aria-label="Position cohort">
                  {COHORTS.map((c) => (
                    <button key={c} className={cohort === c ? 'active' : ''} onClick={() => setCohort(c)}>
                      {c}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="evaluation-grid">
              <article className="evaluation-card chart-card">
                <div className="card-heading">
                  <div>
                    <small>01 / MODEL VS BASELINE</small>
                    <h2>{METRIC_LABEL[metric]}</h2>
                  </div>
                  <span>HIGHER IS {isRate(metric) ? 'BETTER' : 'WORSE'}</span>
                </div>
                <PairedBars rows={chartRows} metric={metric} />
                <p className="chart-note">
                  <b>{METRIC_LABEL[metric]}</b> = <code>{a.metric_definitions[metric]}</code>. Dark bars are the champion ({a.rolling_origin.candidate}); light bars are the
                  baseline ({a.baseline.name ?? 'baseline'}).
                </p>
              </article>

              <article className="evaluation-card detail-card">
                <div className="card-heading">
                  <div>
                    <small>02 / COHORT READOUT</small>
                    <h2>All metrics, with baseline</h2>
                  </div>
                  <span>{rows.length} ROWS</span>
                </div>
                <div className="mt-6 overflow-x-auto">
                  <table className="w-full min-w-[520px] text-left font-mono text-xs">
                    <thead className="text-[9px] uppercase tracking-widest text-[#61706c]">
                      <tr>
                        <th className="py-2 pr-3">Cohort</th>
                        <th className="py-2 pr-3 text-right">n</th>
                        {METRIC_KEYS.map((k) => (
                          <th key={k} className="py-2 pr-3 text-right">
                            {METRIC_LABEL[k]}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => (
                        <RowPair key={r.cohort} row={r} />
                      ))}
                    </tbody>
                  </table>
                </div>
              </article>
            </div>

            <div className="method-grid">
              <article>
                <small>03 / ROLLING ORIGIN</small>
                <h2>Refit before every scored week.</h2>
                <p>{a.rolling_origin.strategy}</p>
                <div className="callout">
                  <span>!</span>
                  <p>
                    <b>Mean fold MAE {fmtNum(a.rolling_origin.mean_mae, 2)}</b> vs baseline {fmtNum(a.rolling_origin.mean_baseline_mae, 2)} across{' '}
                    {a.rolling_origin.folds.length} folds for candidate {a.rolling_origin.candidate}.
                  </p>
                </div>
                <RollingOriginChart key={a.eval_id} folds={a.rolling_origin.folds} label={evaluationLabel(a)} />
                <div className="mt-4 max-h-64 overflow-auto border border-[#d1d3cd]">
                  <table className="w-full text-left font-mono text-[11px]">
                    <thead className="sticky top-0 bg-[#f8f7f2] text-[9px] uppercase tracking-widest text-[#61706c]">
                      <tr>
                        <th className="px-3 py-2">Season</th>
                        <th className="px-3 py-2">Week</th>
                        <th className="px-3 py-2 text-right">n</th>
                        <th className="px-3 py-2 text-right">MAE</th>
                        <th className="px-3 py-2 text-right">Baseline MAE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {a.rolling_origin.folds.map((f) => (
                        <tr key={`${f.season}-${f.week}`} className="border-t border-[#e4e5dd]">
                          <td className="px-3 py-1.5">{f.season}</td>
                          <td className="px-3 py-1.5">{f.week}</td>
                          <td className="px-3 py-1.5 text-right">{fmtInt(f.n)}</td>
                          <td className="px-3 py-1.5 text-right font-bold">{fmtNum(f.mae, 2)}</td>
                          <td className="px-3 py-1.5 text-right">{fmtNum(f.baseline_mae, 2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </article>
              <article>
                <small>04 / METRIC DEFINITIONS</small>
                <h2>Verbatim from the artifact.</h2>
                <ul>
                  {Object.entries(a.metric_definitions).map(([k, v], i) => (
                    <li key={k}>
                      <span>{String(i + 1).padStart(2, '0')}</span>
                      <div>
                        <b>{k}</b>
                        <p>{v}</p>
                      </div>
                    </li>
                  ))}
                </ul>
              </article>
            </div>

            <div className="method-grid">
              <article>
                <small>05 / PROVENANCE</small>
                <h2>What was evaluated.</h2>
                <ul>
                  <li><span>ID</span><div><b>eval_id</b><p className="break-all">{a.eval_id}</p></div></li>
                  <li><span>KD</span><div><b>kind / season</b><p>{a.kind} · {a.season} — {EVAL_KIND_EXPLANATION[a.kind] ?? a.kind}</p></div></li>
                  <li><span>MV</span><div><b>model version</b><p>{a.model.version} · features {a.model.feature_version}</p></div></li>
                  <li><span>CH</span><div><b>champion / challenger</b><p>{describeCandidates(a.model.candidate)} / {describeCandidates(a.model.challenger)}</p></div></li>
                  <li><span>GT</span><div><b>code commit</b><p>{shortHash(a.code_commit)}</p></div></li>
                  <li><span>IN</span><div><b>input</b><p className="break-all">{a.input.path} · sha256 {shortHash(a.input.sha256)} · {fmtInt(a.input.n_rows)} rows</p></div></li>
                  <li><span>TS</span><div><b>trained / evaluated</b><p>{fmtUtc(a.model.trained_at_utc)} / {fmtUtc(a.generated_at_utc)}</p></div></li>
                </ul>
              </article>
              <article>
                <small>06 / HOW TO READ IT</small>
                <h2>Compare to the baseline, not to zero.</h2>
                <p>
                  The baseline is the honest yardstick: a trailing mean of the player&apos;s own earlier realised points that never sees the week it is scored on. A
                  model earns its place only where it beats that column for the cohort you care about.
                </p>
                <p>
                  Positional cohorts share one model version and one split, so their figures can be compared directly. Sample sizes differ by position; use them
                  when weighing the differences.
                </p>
                {evaluations.length > 1 && (
                  <p>
                    Every evaluation listed above scores the same frozen model artifact ({a.model.version}); only the scored season and its role in model
                    selection differ.
                  </p>
                )}
                {a.policy_sweep.length === 0 && <p>The artifact&apos;s policy sweep is empty for this evaluation, so no threshold analysis is shown.</p>}
              </article>
            </div>
            <p className="evaluation-disclaimer">
              {a.kind === 'frozen_test' ? 'Held-out' : 'Out-of-sample'} evaluation of a committed artifact. Not a guarantee of future performance.
            </p>
          </>
        )}
      </section>
    </>
  )
}

/** Compact side-by-side of the headline figures per evaluation. Values only; the reader draws the comparison. */
function EvaluationComparison({
  evaluations,
  activeId,
  onSelect,
}: {
  evaluations: PerformanceArtifact[]
  activeId: string
  onSelect: (id: string) => void
}) {
  return (
    <div className="mb-6 overflow-x-auto border border-[#bec2b8] bg-[#f8f7f2]">
      <table className="w-full min-w-[560px] text-left font-mono text-xs">
        <thead className="text-[9px] uppercase tracking-widest text-[#61706c]">
          <tr>
            <th className="px-4 py-2">Evaluation</th>
            <th className="px-4 py-2 text-right">n</th>
            <th className="px-4 py-2 text-right">MAE · ALL</th>
            <th className="px-4 py-2 text-right">Baseline MAE</th>
            <th className="px-4 py-2 text-right">Within ±3</th>
            <th className="px-4 py-2 text-right">Baseline ±3</th>
            <th className="px-4 py-2 text-right">Folds</th>
          </tr>
        </thead>
        <tbody>
          {evaluations.map((e) => {
            const active = e.eval_id === activeId
            return (
              <tr key={e.eval_id} className={`border-t border-[#d1d3cd] ${active ? 'bg-white font-bold' : 'text-[#52605d]'}`}>
                <td className="px-4 py-2">
                  <button type="button" className="text-left underline-offset-2 hover:underline" onClick={() => onSelect(e.eval_id)} aria-pressed={active}>
                    {evaluationLabel(e)}
                  </button>
                </td>
                <td className="px-4 py-2 text-right">{fmtInt(e.metrics.n)}</td>
                <td className="px-4 py-2 text-right">{fmtNum(e.metrics.mae, 2)}</td>
                <td className="px-4 py-2 text-right">{fmtNum(e.baseline.mae, 2)}</td>
                <td className="px-4 py-2 text-right">{fmtPct(e.metrics.within_3_rate)}</td>
                <td className="px-4 py-2 text-right">{fmtPct(e.baseline.within_3_rate)}</td>
                <td className="px-4 py-2 text-right">{e.rolling_origin.folds.length}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function describeCandidates(c: Record<string, string> | undefined) {
  if (!c) return '—'
  const values = Array.from(new Set(Object.values(c)))
  return values.length === 1 ? values[0] : Object.entries(c).map(([k, v]) => `${k}:${v}`).join(' ')
}

function RowPair({ row }: { row: CohortRow }) {
  return (
    <>
      <tr className="border-t border-[#d1d3cd]">
        <td className="py-2 pr-3 font-bold">{row.cohort}</td>
        <td className="py-2 pr-3 text-right">{fmtInt(row.model.n)}</td>
        {METRIC_KEYS.map((k) => (
          <td key={k} className="py-2 pr-3 text-right font-bold">
            {show(k, row.model[k])}
          </td>
        ))}
      </tr>
      <tr className="text-[#6e7976]">
        <td className="pb-2 pr-3 text-[9px] uppercase tracking-widest">baseline</td>
        <td className="pb-2 pr-3 text-right">{fmtInt(row.baseline.n)}</td>
        {METRIC_KEYS.map((k) => (
          <td key={k} className="pb-2 pr-3 text-right">
            {show(k, row.baseline[k])}
          </td>
        ))}
      </tr>
    </>
  )
}

function PairedBars({ rows, metric }: { rows: CohortRow[]; metric: MetricKey }) {
  const values = rows.flatMap((r) => [r.model[metric], r.baseline[metric]])
  const max = isRate(metric) ? 1 : Math.max(...values, 0.01) * 1.12
  return (
    <div className="bar-chart" role="img" aria-label={`${METRIC_LABEL[metric]} by cohort, model versus baseline`}>
      {rows.map((r) => (
        <div className="bar-column" key={r.cohort}>
          <span>{show(metric, r.model[metric])}</span>
          <div style={{ gap: 4 }}>
            <i style={{ height: `${Math.max(2, (r.model[metric] / max) * 100)}%` }} title={`model ${show(metric, r.model[metric])}`} />
            <i style={{ height: `${Math.max(2, (r.baseline[metric] / max) * 100)}%`, background: '#b9d3c8' }} title={`baseline ${show(metric, r.baseline[metric])}`} />
          </div>
          <b>{r.cohort}</b>
        </div>
      ))}
    </div>
  )
}
