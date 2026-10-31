'use client'

import { useMemo } from 'react'
import { scaleLinear, scalePoint } from 'd3'
import type { Tier } from '@/lib/api/types'

interface TierChartProps {
  tiers: Tier[]
  position: string
  inputSeason: number | null
}

const WIDTH = 900
const HEIGHT = 310
const MARGIN = { top: 20, right: 20, bottom: 62, left: 68 }

/** Prior-season values only. Uniform dots avoid implying that assignment strength is success. */
export function TierChart({ tiers, position, inputSeason }: TierChartProps) {
  const { x, y, points, ticks } = useMemo(() => {
    const all = tiers.flatMap((t) => t.players.map((p) => ({ tier: t.tier, ppg: p.ppg_prev, name: p.name ?? p.player_id })))
    const min = Math.min(0, ...all.map((d) => d.ppg))
    const max = Math.max(1, ...all.map((d) => d.ppg))
    const x = scalePoint<number>().domain(tiers.map((t) => t.tier)).range([MARGIN.left, WIDTH - MARGIN.right]).padding(0.6)
    const y = scaleLinear().domain([min, max]).nice().range([HEIGHT - MARGIN.bottom, MARGIN.top])
    return { x, y, points: all, ticks: y.ticks(5) }
  }, [tiers])

  if (tiers.length === 0) return null

  return (
    <figure className="product-card p-5">
      <figcaption className="mb-3">
        <h2 className="text-xl font-semibold">How the groups compare</h2>
        <p className="mt-2 text-sm leading-6 text-[#52605d]">Each dot is a {position} player&apos;s {inputSeason ?? 'input-season'} PPR points per game. Horizontal placement shows group membership, not an exact player rank.</p>
      </figcaption>
      <div className="overflow-x-auto" tabIndex={0} aria-label="Scrollable historical group chart">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-auto w-full min-w-[560px]" role="img" aria-label={`${position} groups based on ${inputSeason ?? 'prior-season'} PPR points per game`}>
          {ticks.map((t) => <g key={t}>
            <line x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={y(t)} y2={y(t)} stroke="#e1e5db" />
            <text x={MARGIN.left - 10} y={y(t)} dy="0.32em" textAnchor="end" fontSize="15" fill="#45584e">{t}</text>
          </g>)}
          {tiers.map((t) => <text key={t.tier} x={x(t.tier)} y={HEIGHT - 35} textAnchor="middle" fontSize="16" fontWeight="600" fill="#152d23">Group {t.tier}</text>)}
          <text x={WIDTH / 2} y={HEIGHT - 8} textAnchor="middle" fontSize="15" fill="#45584e">Historical group (ordered by mean input-season PPR/game)</text>
          <text transform={`translate(20, ${HEIGHT / 2 - 15}) rotate(-90)`} textAnchor="middle" fontSize="15" fill="#45584e">Input-season PPR/game</text>
          {points.map((d, i) => <circle key={`${d.tier}-${i}`} cx={(x(d.tier) ?? 0) + ((i % 7) - 3) * 4} cy={y(d.ppg)} r={4} fill="#176e5a" fillOpacity={0.75}><title>{d.name}: {d.ppg.toFixed(1)} PPR/game, group {d.tier}</title></circle>)}
        </svg>
      </div>
      <p className="mt-2 text-sm text-[#52605d]">The grouped player list above supplies the same values without relying on the chart.</p>
    </figure>
  )
}
