'use client'

import { useMemo } from 'react'
import { area, line, scaleLinear, scalePoint } from 'd3'
import type { PlayerHistoryRow } from '@/lib/api/types'

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
    return { rows, keys, x, y, key, bandPath: band(rows) ?? '', predPath: predLine(rows) ?? '', actualPath: actualLine(actualRows) ?? '', actualRows, ticks: y.ticks(5) }
  }, [history])

  if (model.rows.length === 0) return null
  const labelEvery = Math.max(1, Math.ceil(model.keys.length / 12))

  return (
    <figure className="border border-[#c5c7c0] bg-white p-4">
      <figcaption className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <span className="font-mono text-[9px] font-bold uppercase tracking-widest text-moss">Prediction vs actual · floor–ceiling band</span>
        <span className="flex gap-4 font-mono text-[9px] uppercase tracking-widest text-[#61706c]">
          <span><i className="mr-1 inline-block h-2 w-4 bg-moss align-middle" />prediction</span>
          <span><i className="mr-1 inline-block h-2 w-4 bg-ember align-middle" />actual</span>
          <span><i className="mr-1 inline-block h-2 w-4 bg-[#cfe3d9] align-middle" />floor–ceiling</span>
        </span>
      </figcaption>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-auto w-full min-w-[560px]" role="img" aria-label="Weekly prediction versus actual points">
          {model.ticks.map((t) => (
            <g key={t}>
              <line x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={model.y(t)} y2={model.y(t)} stroke="#e4e5dd" />
              <text x={MARGIN.left - 8} y={model.y(t)} dy="0.32em" textAnchor="end" fontSize="10" fontFamily="monospace" fill="#61706c">{t}</text>
            </g>
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
