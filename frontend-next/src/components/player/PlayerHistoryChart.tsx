'use client'

import { useMemo } from 'react'
import { area, line, scaleLinear, scalePoint } from 'd3'
import type { HistorySource, PlayerHistoryRow } from '@/lib/api/types'

/** Human label per history source. Mirrors schemas.PlayerHistoryRow.source. */
export const SOURCE_LABEL: Record<HistorySource, string> = {
  frozen_test: 'frozen test',
  out_of_sample_season: 'out-of-sample',
  weekly: 'weekly',
}

/** Badge classes per source, shared by the chart legend and the history table. */
export const SOURCE_STYLE: Record<HistorySource, string> = {
  frozen_test: 'bg-sand text-ink',
  out_of_sample_season: 'bg-[#cfe3d9] text-ink',
  weekly: 'bg-ink text-acid',
}

const SOURCE_ORDER: HistorySource[] = ['frozen_test', 'out_of_sample_season', 'weekly']
const SOURCE_TINT: Record<HistorySource, string> = {
  frozen_test: '#e5e5dd',
  out_of_sample_season: '#cfe3d9',
  weekly: '#101b19',
}

const WIDTH = 900
const HEIGHT = 280
const MARGIN = { top: 16, right: 16, bottom: 40, left: 44 }

/** Prediction vs actual per week, with the floor–ceiling band. Pure SVG built from d3 scales. */
export function PlayerHistoryChart({ history }: { history: PlayerHistoryRow[] }) {
  const model = useMemo(() => {
    const rows = history.filter((r) => r.prediction !== null)
    const keys = rows.map((r) => `${r.season}·${r.week}`)
    const values = rows.flatMap((r) => [r.prediction ?? 0, r.actual ?? 0, r.ceiling ?? 0, r.floor ?? 0])
    const x = scalePoint<string>().domain(keys).range([MARGIN.left, WIDTH - MARGIN.right]).padding(0.5)
    const y = scaleLinear().domain([0, Math.max(1, ...values)]).nice().range([HEIGHT - MARGIN.bottom, MARGIN.top])
    const key = (r: PlayerHistoryRow) => `${r.season}·${r.week}`
    const band = area<PlayerHistoryRow>()
      .defined((r) => r.floor !== null && r.ceiling !== null)
      .x((r) => x(key(r)) ?? 0)
      .y0((r) => y(r.floor ?? 0))
      .y1((r) => y(r.ceiling ?? 0))
    const predLine = line<PlayerHistoryRow>().x((r) => x(key(r)) ?? 0).y((r) => y(r.prediction ?? 0))
    const actualRows = rows.filter((r) => r.actual !== null)
    const actualLine = line<PlayerHistoryRow>().x((r) => x(key(r)) ?? 0).y((r) => y(r.actual ?? 0))
    // Contiguous runs of the same source, so each can be tinted along the x-axis and listed in the legend.
    const step = keys.length > 1 ? (x(keys[1]) ?? 0) - (x(keys[0]) ?? 0) : WIDTH - MARGIN.left - MARGIN.right
    const runs: Array<{ source: HistorySource; x0: number; x1: number }> = []
    for (const r of rows) {
      const cx = x(key(r)) ?? 0
      const last = runs[runs.length - 1]
      if (last && last.source === r.source) last.x1 = cx + step / 2
      else runs.push({ source: r.source, x0: cx - step / 2, x1: cx + step / 2 })
    }
    const sources = SOURCE_ORDER.filter((s) => rows.some((r) => r.source === s))
    return { rows, keys, x, y, key, runs, sources, bandPath: band(rows) ?? '', predPath: predLine(rows) ?? '', actualPath: actualLine(actualRows) ?? '', actualRows, ticks: y.ticks(5) }
  }, [history])

  if (model.rows.length === 0) return null
  const labelEvery = Math.max(1, Math.ceil(model.keys.length / 12))

  return (
    <figure className="border border-[#c5c7c0] bg-white p-4">
      <figcaption className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <span className="font-mono text-[9px] font-bold uppercase tracking-widest text-moss">Prediction vs actual · floor–ceiling band</span>
        <span className="flex flex-wrap gap-4 font-mono text-[9px] uppercase tracking-widest text-[#61706c]">
          <span><i className="mr-1 inline-block h-2 w-4 bg-moss align-middle" />prediction</span>
          <span><i className="mr-1 inline-block h-2 w-4 bg-ember align-middle" />actual</span>
          <span><i className="mr-1 inline-block h-2 w-4 bg-[#cfe3d9] align-middle" />floor–ceiling</span>
        </span>
      </figcaption>
      <div className="mb-2 flex flex-wrap items-center gap-3 font-mono text-[9px] uppercase tracking-widest text-[#61706c]" aria-label="Row sources">
        <span>Source</span>
        {model.sources.map((s) => (
          <span key={s} className={`px-1.5 py-0.5 ${SOURCE_STYLE[s]}`}>
            {SOURCE_LABEL[s]}
          </span>
        ))}
      </div>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-auto w-full min-w-[560px]" role="img" aria-label="Weekly prediction versus actual points">
          {model.ticks.map((t) => (
            <g key={t}>
              <line x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={model.y(t)} y2={model.y(t)} stroke="#e4e5dd" />
              <text x={MARGIN.left - 8} y={model.y(t)} dy="0.32em" textAnchor="end" fontSize="10" fontFamily="monospace" fill="#61706c">{t}</text>
            </g>
          ))}
          {model.runs.map((run) => (
            <rect
              key={`${run.source}-${run.x0}`}
              x={run.x0}
              y={HEIGHT - MARGIN.bottom + 2}
              width={Math.max(0, run.x1 - run.x0)}
              height={4}
              fill={SOURCE_TINT[run.source]}
            >
              <title>{SOURCE_LABEL[run.source]}</title>
            </rect>
          ))}
          <path d={model.bandPath} fill="#cfe3d9" fillOpacity={0.7} />
          <path d={model.predPath} fill="none" stroke="#176e5a" strokeWidth={2} />
          <path d={model.actualPath} fill="none" stroke="#ff6b35" strokeWidth={2} strokeDasharray="4 3" />
          {model.rows.map((r) => (
            <circle key={`p-${model.key(r)}`} cx={model.x(model.key(r))} cy={model.y(r.prediction ?? 0)} r={3} fill="#176e5a" />
          ))}
          {model.actualRows.map((r) => (
            <circle key={`a-${model.key(r)}`} cx={model.x(model.key(r))} cy={model.y(r.actual ?? 0)} r={3} fill="#ff6b35" />
          ))}
          {model.keys.map((k, i) =>
            i % labelEvery === 0 ? (
              <text key={k} x={model.x(k)} y={HEIGHT - 14} textAnchor="middle" fontSize="9" fontFamily="monospace" fill="#61706c">{k}</text>
            ) : null,
          )}
        </svg>
      </div>
    </figure>
  )
}
