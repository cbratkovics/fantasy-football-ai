'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getDecisions } from '@/lib/api/marts'
import type { DecisionSummaryRow, DecisionsResponse } from '@/lib/api/types'
import { fmtInt, fmtNum, fmtPct, fmtUtc, shortHash } from '@/lib/format'
import { COHORTS, UNITS } from '@/lib/project'
import { PERFORMANCE_COPY } from '@/content/performance'
import { ErrorState, LoadingState } from '@/components/ui/States'
import Link from 'next/link'

const DEFAULT_MIN_FLOOR = 6
const SUMMARY_ORDER = COHORTS
const COPY = PERFORMANCE_COPY.decisions

/**
 * Floor-policy outcomes from the gold mart fct_decision_policy, swept over the exported
 * min_floor grid. Every figure is read from the API; the only client-side work is choosing
 * which threshold and season to display.
 */
export function DecisionsPanel({ seasons }: { seasons: number[] }) {
  const [minFloor, setMinFloor] = useState<number>(DEFAULT_MIN_FLOOR)
  const [season, setSeason] = useState<number | undefined>(undefined)
  const decisions = useQuery({
    queryKey: ['marts', 'decisions', minFloor, season],
    queryFn: () => getDecisions({ min_floor: minFloor, season }),
  })
  const d: DecisionsResponse | undefined = decisions.data
  const grid = d?.available_min_floors ?? [minFloor]
  const summary = [...(d?.summary ?? [])].sort((x, y) => SUMMARY_ORDER.indexOf(x.cohort) - SUMMARY_ORDER.indexOf(y.cohort))
  const overall = summary.find((s) => s.cohort === 'ALL')

  return (
    <section id="decision-policy" className="method-grid scroll-mt-24" aria-label="Decision policy from the gold marts">
      <article style={{ gridColumn: '1 / -1' }}>
        <small>07 / DECISIONS · READ FROM GOLD MARTS EXPORTED BY THE WEEKLY BUILD</small>
        <h2>{COPY.title}</h2>
        <p>
          {COPY.intro} <code>fct_decision_policy</code>, a gold table built by dbt and exported to parquet by the weekly job; the API
          serves it in-process. The evaluation figures above still come from the committed artifacts.
          {' '}<Link className="font-bold underline" href="/data-platform?model=model.ffai_dbt.fct_decision_policy.v1#model-inspector">Inspect this model</Link>.
        </p>

        <div className="evaluation-toolbar">
          <div>
            <small>Min floor</small>
            <div role="group" aria-label="Minimum floor threshold">
              {grid.map((v) => (
                <button key={v} className={v === minFloor ? 'active' : ''} onClick={() => setMinFloor(v)}>
                  {v}
                </button>
              ))}
            </div>
          </div>
          <div>
            <small>Season</small>
            <div role="group" aria-label="Season">
              <button className={season === undefined ? 'active' : ''} onClick={() => setSeason(undefined)}>
                all
              </button>
              {seasons.map((s) => (
                <button key={s} className={s === season ? 'active' : ''} onClick={() => setSeason(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        </div>

        {decisions.isPending && <LoadingState label="Reading fct_decision_policy…" />}
        {decisions.isError && <ErrorState error={decisions.error} context="GET /marts/decisions" />}

        {d && overall && (
          <>
            <div className="evaluation-kpis">
              <article>
                <div><span>Recommended · ALL</span></div>
                <strong>{fmtPct(overall.recommendation_rate)}</strong>
                <p>{fmtInt(overall.recommendations)} of {fmtInt(overall.eligible_decisions)} {COPY.scoredRows} {minFloor}</p>
              </article>
              <article>
                <div><span>Hit rate · ALL</span></div>
                <strong>{fmtPct(overall.hit_rate)}</strong>
                <p>{COPY.hitRate}</p>
              </article>
              <article>
                <div><span>Downside · ALL</span></div>
                <strong>{fmtPct(overall.downside_rate)}</strong>
                <p>{COPY.downside}</p>
              </article>
              <article>
                <div><span>Rec. MAE · ALL</span></div>
                <strong>{fmtNum(overall.recommendation_mae, 2)}</strong>
                <p>mean regret {fmtNum(overall.mean_regret, 1)} {UNITS} {COPY.regret}</p>
              </article>
            </div>

            <div className="mt-6 overflow-x-auto border border-[#d1d3cd] bg-white">
              <table className="w-full min-w-[640px] text-left font-mono text-xs">
                <thead className="bg-[#f8f7f2] text-[9px] uppercase tracking-widest text-[#61706c]">
                  <tr>
                    <th className="px-3 py-2">Cohort</th>
                    <th className="px-3 py-2 text-right">Eligible</th>
                    <th className="px-3 py-2 text-right">Recommended</th>
                    <th className="px-3 py-2 text-right">Rate</th>
                    <th className="px-3 py-2 text-right">With outcome</th>
                    <th className="px-3 py-2 text-right">Rec. MAE</th>
                    <th className="px-3 py-2 text-right">Mean regret</th>
                    <th className="px-3 py-2 text-right">Hit rate</th>
                    <th className="px-3 py-2 text-right">Downside</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.map((s: DecisionSummaryRow) => (
                    <tr key={s.cohort} className="border-t border-[#e4e5dd]">
                      <td className="px-3 py-2 font-bold">{s.cohort}</td>
                      <td className="px-3 py-2 text-right">{fmtInt(s.eligible_decisions)}</td>
                      <td className="px-3 py-2 text-right">{fmtInt(s.recommendations)}</td>
                      <td className="px-3 py-2 text-right">{fmtPct(s.recommendation_rate)}</td>
                      <td className="px-3 py-2 text-right">{fmtInt(s.recommendations_with_outcome)}</td>
                      <td className="px-3 py-2 text-right">{fmtNum(s.recommendation_mae, 2)}</td>
                      <td className="px-3 py-2 text-right">{fmtNum(s.mean_regret, 1)}</td>
                      <td className="px-3 py-2 text-right">{fmtPct(s.hit_rate)}</td>
                      <td className="px-3 py-2 text-right">{fmtPct(s.downside_rate)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <ul>
              <li><span>PL</span><div><b>policy</b><p>{d.policy}</p></div></li>
              <li><span>RL</span><div><b>replacement level</b><p>{d.replacement_level}</p></div></li>
              <li>
                <span>EX</span>
                <div>
                  <b>export</b>
                  <p>
                    {d.mart} v{d.mart_version} · {fmtInt(d.export.row_counts[d.mart])} rows · exported {fmtUtc(d.export.exported_at_utc)} from target {d.export.target ?? '—'} ·
                    commit {shortHash(d.export.code_commit)} · {fmtInt(d.n)} season-week-position rows at this threshold
                  </p>
                </div>
              </li>
            </ul>
          </>
        )}
      </article>
    </section>
  )
}
