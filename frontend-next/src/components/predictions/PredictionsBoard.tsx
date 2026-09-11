'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { getLatestWeek, getPredictions } from '@/lib/api/players'
import { POSITIONS, SCORING_FORMATS, type Position, type ScoringFormat } from '@/lib/api/types'
import { fmtNum } from '@/lib/format'
import { Segmented, TextInput } from '@/components/ui/Controls'
import { ErrorState, LoadingState, EmptyState } from '@/components/ui/States'
import { Provenance, generatedLine } from '@/components/ui/Provenance'

type PositionFilter = 'ALL' | Position
const POSITION_FILTERS: readonly PositionFilter[] = ['ALL', ...POSITIONS]
const WEEKS = Array.from({ length: 18 }, (_, i) => i + 1)

export function PredictionsBoard() {
  const latest = useQuery({ queryKey: ['latest-week'], queryFn: getLatestWeek })
  const [week, setWeek] = useState<number | null>(null)
  const [scoring, setScoring] = useState<ScoringFormat>('ppr')
  const [position, setPosition] = useState<PositionFilter>('ALL')
  const [search, setSearch] = useState('')

  useEffect(() => {
    if (latest.data && week === null) setWeek(latest.data.week)
  }, [latest.data, week])

  const season = latest.data?.season
  const activeWeek = week ?? latest.data?.week

  const predictions = useQuery({
    queryKey: ['predictions', season, activeWeek, scoring, position],
    queryFn: () => getPredictions(season as number, activeWeek as number, scoring, position === 'ALL' ? undefined : position),
    enabled: season !== undefined && activeWeek !== undefined,
  })

  const rows = useMemo(() => {
    const list = predictions.data?.predictions ?? []
    const q = search.trim().toLowerCase()
    if (!q) return list
    return list.filter((r) => (r.name ?? '').toLowerCase().includes(q) || (r.team ?? '').toLowerCase().includes(q))
  }, [predictions.data, search])

  if (latest.isPending) return <LoadingState label="Reading the manifest for the latest published week…" />
  if (latest.isError) return <ErrorState error={latest.error} context="GET /manifest" />

  const data = predictions.data

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 border border-[#c5c7c0] bg-[#f8f7f2] p-5 lg:flex-row lg:flex-wrap lg:items-center lg:justify-between">
        <div className="flex flex-col gap-4 sm:flex-row sm:flex-wrap sm:items-center">
          <Segmented label="Scoring" options={SCORING_FORMATS} value={scoring} onChange={setScoring} />
          <Segmented label="Position" options={POSITION_FILTERS} value={position} onChange={setPosition} />
          <div className="flex items-center gap-3">
            <label htmlFor="week" className="font-mono text-[9px] font-bold uppercase tracking-widest">
              Week
            </label>
            <select
              id="week"
              value={activeWeek ?? ''}
              onChange={(e) => setWeek(Number(e.target.value))}
              className="h-9 border border-[#bfc3bb] bg-white px-2 font-mono text-xs"
            >
              {WEEKS.map((w) => (
                <option key={w} value={w}>
                  {season} · week {w}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="w-full lg:w-72">
          <TextInput placeholder="Filter by player or team" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Filter predictions" />
        </div>
      </div>

      {predictions.isPending && <LoadingState label={`Loading season ${season} week ${activeWeek}…`} />}
      {predictions.isError && <ErrorState error={predictions.error} context={`GET /predictions/${season}/${activeWeek}`} />}

      {data && (
        <>
          <Provenance
            items={[
              ['Model', data.model_version],
              ['Features', data.feature_version],
              ['Data through', `season ${data.data_through.season} · week ${data.data_through.week}`],
              ['Rows', `${data.n} (${data.scoring.toUpperCase()})`],
            ]}
            note={generatedLine(data.model_version, data.feature_version, data.data_through.season, data.data_through.week, data.generated_at_utc)}
          />

          {rows.length === 0 ? (
            <EmptyState message="No rows match this filter." />
          ) : (
            <div className="overflow-x-auto border border-[#c5c7c0] bg-white">
              <table className="w-full min-w-[720px] text-left text-sm">
                <thead className="bg-[#f8f7f2] font-mono text-[9px] uppercase tracking-widest text-[#61706c]">
                  <tr>
                    <th className="px-4 py-3">#</th>
                    <th className="px-4 py-3">Player</th>
                    <th className="px-4 py-3">Team</th>
                    <th className="px-4 py-3">Pos</th>
                    <th className="px-4 py-3 text-right">Prediction</th>
                    <th className="px-4 py-3 text-right">Floor – Ceiling</th>
                    <th className="px-4 py-3">Range</th>
                    <th className="px-4 py-3 text-right">Actual</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, index) => {
                    const max = data.predictions[0]?.ceiling || 1
                    return (
                      <tr key={r.player_id} className="border-t border-[#e4e5dd] hover:bg-[#f8f7f2]">
                        <td className="px-4 py-3 font-mono text-xs text-[#8c9a96]">{index + 1}</td>
                        <td className="px-4 py-3 font-semibold">
                          <Link href={`/player/${encodeURIComponent(r.player_id)}`} className="hover:text-moss hover:underline">
                            {r.name ?? r.player_id}
                          </Link>
                        </td>
                        <td className="px-4 py-3 font-mono text-xs">{r.team ?? '—'}</td>
                        <td className="px-4 py-3 font-mono text-xs">{r.position}</td>
                        <td className="px-4 py-3 text-right font-mono text-sm font-bold">{fmtNum(r.prediction, 1)}</td>
                        <td className="px-4 py-3 text-right font-mono text-xs text-[#52605d]">
                          {fmtNum(r.floor, 1)} – {fmtNum(r.ceiling, 1)}
                        </td>
                        <td className="px-4 py-3">
                          <RangeBar floor={r.floor} prediction={r.prediction} ceiling={r.ceiling} max={max} />
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-xs">{r.actual === null ? <span className="text-[#8c9a96]">not scored</span> : fmtNum(r.actual, 1)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          <aside className="grid gap-4 border border-[#c5c7c0] bg-[#f8f7f2] p-5 text-xs leading-relaxed text-[#52605d] md:grid-cols-2">
            <div>
              <p className="font-mono text-[9px] font-bold uppercase tracking-widest text-moss">Scoring derivation</p>
              <p className="mt-2">{data.scoring_derivation}</p>
            </div>
            <div>
              <p className="font-mono text-[9px] font-bold uppercase tracking-widest text-moss">Interval method</p>
              <p className="mt-2">{data.interval_method}</p>
            </div>
          </aside>
        </>
      )}
    </div>
  )
}

function RangeBar({ floor, prediction, ceiling, max }: { floor: number; prediction: number; ceiling: number; max: number }) {
  const pct = (v: number) => `${Math.max(0, Math.min(100, (v / max) * 100))}%`
  return (
    <div className="relative h-2 w-28 bg-[#e4e5dd]" aria-hidden="true">
      <div className="absolute top-0 h-full bg-[#b9d3c8]" style={{ left: pct(floor), width: `calc(${pct(ceiling)} - ${pct(floor)})` }} />
      <div className="absolute top-[-2px] h-3 w-0.5 bg-ink" style={{ left: pct(prediction) }} />
    </div>
  )
}
