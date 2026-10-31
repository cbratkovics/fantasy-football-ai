'use client'

import { useMemo } from 'react'
import { line, scaleLinear, scalePoint } from 'd3'
import type { RollingFold } from '@/lib/api/types'

const WIDTH = 640
const HEIGHT = 220
const MARGIN = { top: 12, right: 12, bottom: 34, left: 40 }

/**
 * Fold-by-fold MAE for the candidate and the causal baseline. Works for any evaluation artifact
 * (frozen test or out-of-sample season): the x-axis is whatever (season, week) folds the artifact carries.
 */
export function RollingOriginChart({ folds, label }: { folds: RollingFold[]; label?: string }) {
  const m = useMemo(() => {
    const key = (f: RollingFold) => `${f.season}·${f.week}`
    const keys = folds.map(key)
    const x = scalePoint<string>().domain(keys).range([MARGIN.left, WIDTH - MARGIN.right]).padding(0.5)
    const y = scaleLinear()
      .domain([0, Math.max(1, ...folds.flatMap((f) => [f.mae, f.baseline_mae]))])
      .nice()
      .range([HEIGHT - MARGIN.bottom, MARGIN.top])
    const mk = (get: (f: RollingFold) => number) => line<RollingFold>().x((f) => x(key(f)) ?? 0).y((f) => y(get(f)))(folds) ?? ''
    return { keys, x, y, key, model: mk((f) => f.mae), baseline: mk((f) => f.baseline_mae), ticks: y.ticks(4) }
  }, [folds])

  if (folds.length === 0) return <p className="mt-4 text-sm text-[#4b5b54]">No rolling-origin folds in this evaluation.</p>
  const labelEvery = Math.max(1, Math.ceil(m.keys.length / 9))

  return (
    <figure className="mt-5 min-w-0 overflow-x-auto rounded-lg border border-[#d1d3cd] bg-white p-3">
      <figcaption className="mb-3 flex flex-wrap justify-between gap-2 text-sm text-[#4b5b54]">
        <span>Fold MAE by scored week{label ? ` · ${label}` : ''}</span>
        <span>
          <i className="mr-1 inline-block h-2 w-4 bg-moss align-middle" />model <i className="ml-3 mr-1 inline-block h-2 w-4 bg-[#b9d3c8] align-middle" />baseline
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-auto w-full min-w-[560px]" role="img" aria-label={`Rolling-origin average absolute error per fold, model versus baseline${label ? `, ${label}` : ''}`}>
        {m.ticks.map((t) => (
          <g key={t}>
            <line x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={m.y(t)} y2={m.y(t)} stroke="#e4e5dd" />
            <text x={MARGIN.left - 6} y={m.y(t)} dy="0.32em" textAnchor="end" fontSize="13" fill="#4b5b54">{t}</text>
          </g>
        ))}
        <path d={m.baseline} fill="none" stroke="#647b6b" strokeWidth={2} strokeDasharray="5 4" />
        <path d={m.model} fill="none" stroke="#176e5a" strokeWidth={2} />
        {folds.map((f) => (
          <circle key={m.key(f)} cx={m.x(m.key(f))} cy={m.y(f.mae)} r={2.5} fill="#176e5a" />
        ))}
        {m.keys.map((k, i) =>
          i % labelEvery === 0 ? (
            <text key={k} x={m.x(k)} y={HEIGHT - 12} textAnchor="middle" fontSize="12" fill="#4b5b54">{k}</text>
          ) : null,
        )}
      </svg>
    </figure>
  )
}
