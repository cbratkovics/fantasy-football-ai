'use client'

import Link from 'next/link'
import { DECISION_LAB_COPY } from '@/content/decision-lab'
import { describeAvailability } from '@/lib/decision-lab/explain'
import type { Receipt } from '@/lib/decision-lab/types'
import { Card, FOCUS, Help } from './Bits'
import { num } from './draft'

const COPY = DECISION_LAB_COPY.table

/**
 * One row per nominated alternative in the order the policy ranked them (comparable rows by rank,
 * then excluded rows). Nothing is re-ranked here; the ranking comes from the receipt's result.
 */
export function AlternativesTable({ receipt }: { receipt: Receipt }) {
  const r = receipt.result
  const real = receipt.inputs.snapshot.mode !== 'synthetic'
  const teams: Record<string, string | null> = {}
  receipt.inputs.alternatives.forEach((a) => {
    teams[a.player_id] = a.team
  })
  const rows = r.ranking
    .slice()
    .sort((a, b) => {
      if (a.rank !== null && b.rank !== null) return a.rank - b.rank
      if (a.rank !== null) return -1
      if (b.rank !== null) return 1
      return a.player_id < b.player_id ? -1 : a.player_id > b.player_id ? 1 : 0
    })
  return (
    <Card title={COPY.title} id="alternatives" testId="alternatives-table">
      <Help>{COPY.help}</Help>
      <div className="mt-3 overflow-x-auto" data-testid="alternatives-scroll">
        <table className="table w-full min-w-[840px] text-left font-mono text-xs">
          <thead className="bg-[#f8f7f2]">
            <tr>
              {[COPY.rank, COPY.player, COPY.team, COPY.position, COPY.projection, COPY.floor, COPY.ceiling, COPY.baseline, COPY.availability, COPY.comparable, COPY.tie].map((h) => (
                <th key={h} scope="col" className="border-b border-[#c5c7c0] px-2 py-2 text-[9px] font-bold uppercase tracking-widest text-[#5c6966]">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const recommended = r.recommended_player_id === row.player_id
              return (
                <tr key={row.player_id} data-testid={`alt-row-${row.player_id}`} className={`border-b border-[#e3e3dc] bg-white ${recommended ? 'font-bold' : ''}`}>
                  <td className="px-2 py-2 whitespace-nowrap">
                    {row.rank !== null ? `${row.rank}${recommended ? ' ▲' : ''}` : `${COPY.excluded}: ${row.exclusion ?? '—'}`}
                  </td>
                  <td className="px-2 py-2 whitespace-nowrap">
                    {real ? (
                      <Link href={`/player/${encodeURIComponent(row.player_id)}`} className={`underline ${FOCUS}`}>
                        {row.display_name ?? row.player_id}
                      </Link>
                    ) : (
                      row.display_name ?? row.player_id
                    )}
                    <span className="ml-2 text-[10px] text-[#8c9a96]">{row.player_id}</span>
                  </td>
                  <td className="px-2 py-2">{teams[row.player_id] ?? '—'}</td>
                  <td className="px-2 py-2">{row.position ?? '—'}</td>
                  <td className="px-2 py-2" data-testid={`proj-${row.player_id}`}>
                    {num(row.projection)}
                  </td>
                  <td className="px-2 py-2" data-testid={`floor-${row.player_id}`}>
                    {num(row.floor)}
                  </td>
                  <td className="px-2 py-2">{num(row.ceiling)}</td>
                  <td className="px-2 py-2 whitespace-nowrap">
                    {num(row.baseline)} <span className="text-[10px] text-[#8c9a96]">{row.baseline_provenance}</span>
                  </td>
                  <td className="px-2 py-2">{describeAvailability(row.availability, row.availability_basis)}</td>
                  <td className="px-2 py-2">{row.comparable ? COPY.yes : COPY.no}</td>
                  <td className="px-2 py-2">{row.tied_with_next ? COPY.yes : COPY.no}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
