'use client'

import { useMemo } from 'react'
import { scaleLinear, scalePoint } from 'd3'
import type { Tier } from '@/lib/api/types'

interface TierChartProps {
  tiers: Tier[]
  position: string
}

const WIDTH = 900
const HEIGHT = 260
const MARGIN = { top: 16, right: 16, bottom: 36, left: 44 }

/** Dot plot of prior-season points per game by tier, drawn from the tiers artifact passed in as props. */
export function TierChart({ tiers, position }: TierChartProps) {
  const { x, y, points, ticks } = useMemo(() => {
    const all = tiers.flatMap((t) => t.players.map((p) => ({ tier: t.tier, ppg: p.ppg_prev, prob: p.tier_probability })))
    const maxPpg = Math.max(1, ...all.map((d) => d.ppg))
    const x = scalePoint<number>()
      .domain(tiers.map((t) => t.tier))
      .range([MARGIN.left, WIDTH - MARGIN.right])
      .padding(0.6)
    const y = scaleLinear().domain([0, maxPpg]).nice().range([HEIGHT - MARGIN.bottom, MARGIN.top])
    return { x, y, points: all, ticks: y.ticks(5) }
  }, [tiers])

  if (tiers.length === 0) return null

  return (
    <figure className="border border-[#c5c7c0] bg-white p-4">
      <figcaption className="mb-2 flex items-center justify-between">
        <span className="font-mono text-[9px] font-bold uppercase tracking-widest text-moss">{position} · prior-season PPG by tier</span>
        <span className="font-mono text-[9px] uppercase tracking-widest text-[#8c9a96]">dot opacity = tier probability</span>
      </figcaption>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-auto w-full min-w-[560px]" role="img" aria-label={`Prior-season points per game for each ${position} tier`}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={y(t)} y2={y(t)} stroke="#e4e5dd" />
              <text x={MARGIN.left - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize="10" fontFamily="monospace" fill="#61706c">
                {t}
              </text>
            </g>
          ))}
          {tiers.map((t) => (
            <text key={t.tier} x={x(t.tier)} y={HEIGHT - 12} textAnchor="middle" fontSize="10" fontFamily="monospace" fontWeight="bold" fill="#101b19">
              T{t.tier}
            </text>
          ))}
          {points.map((d, i) => (
            <circle
              key={i}
              cx={(x(d.tier) ?? 0) + ((i % 7) - 3) * 4}
              cy={y(d.ppg)}
              r={4}
              fill="#176e5a"
              fillOpacity={0.25 + 0.75 * d.prob}
            />
          ))}
        </svg>
      </div>
    </figure>
  )
}
