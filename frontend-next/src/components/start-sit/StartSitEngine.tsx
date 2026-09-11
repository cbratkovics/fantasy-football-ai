'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { XMarkIcon } from '@heroicons/react/24/outline'
import { getLatestWeek, getPredictions } from '@/lib/api/players'
import { SCORING_FORMATS, type PredictionRecord, type ScoringFormat } from '@/lib/api/types'
import { fmtNum } from '@/lib/format'
import { Segmented, TextInput } from '@/components/ui/Controls'
import { EmptyState, ErrorState, LoadingState } from '@/components/ui/States'
import { Provenance, generatedLine } from '@/components/ui/Provenance'

export function StartSitEngine() {
  const latest = useQuery({ queryKey: ['latest-week'], queryFn: getLatestWeek })
  const [scoring, setScoring] = useState<ScoringFormat>('ppr')
  const [selected, setSelected] = useState<string[]>([])
  const [search, setSearch] = useState('')

  const predictions = useQuery({
    queryKey: ['predictions', latest.data?.season, latest.data?.week, scoring, 'ALL'],
    queryFn: () => getPredictions(latest.data!.season, latest.data!.week, scoring),
    enabled: !!latest.data,
  })

  const byId = useMemo(() => new Map((predictions.data?.predictions ?? []).map((p) => [p.player_id, p])), [predictions.data])
  const chosen = selected.map((id) => byId.get(id)).filter((p): p is PredictionRecord => !!p)
  const best = chosen.reduce<PredictionRecord | null>((acc, p) => (acc === null || p.prediction > acc.prediction ? p : acc), null)
  const maxCeiling = Math.max(1, ...chosen.map((p) => p.ceiling))

  const matches = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return []
    return (predictions.data?.predictions ?? [])
      .filter((p) => !selected.includes(p.player_id) && ((p.name ?? '').toLowerCase().includes(q) || (p.team ?? '').toLowerCase().includes(q)))
      .slice(0, 8)
  }, [predictions.data, search, selected])

  if (latest.isPending) return <LoadingState label="Reading the manifest for the latest published week…" />
  if (latest.isError) return <ErrorState error={latest.error} context="GET /manifest" />

  const data = predictions.data

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 border border-[#c5c7c0] bg-[#f8f7f2] p-5 lg:flex-row lg:items-center lg:justify-between">
        <Segmented label="Scoring" options={SCORING_FORMATS} value={scoring} onChange={setScoring} />
        <div className="relative w-full lg:w-96">
          <TextInput
            placeholder={data ? `Add a player from season ${data.season} week ${data.week}` : 'Loading players…'}
            value={search}
            disabled={!data}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Search players to compare"
          />
          {matches.length > 0 && (
            <ul className="absolute z-10 mt-1 w-full border border-[#bfc3bb] bg-white shadow-lg">
              {matches.map((p) => (
                <li key={p.player_id}>
                  <button
                    type="button"
                    className="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-[#f8f7f2]"
                    onClick={() => {
                      setSelected((s) => [...s, p.player_id])
                      setSearch('')
                    }}
                  >
                    <span><b>{p.name ?? p.player_id}</b> <span className="font-mono text-[10px] text-[#8c9a96]">{p.position} · {p.team ?? '—'}</span></span>
                    <span className="font-mono text-xs">{fmtNum(p.prediction, 1)}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {predictions.isPending && <LoadingState label="Loading the latest predictions…" />}
      {predictions.isError && <ErrorState error={predictions.error} context={`GET /predictions/${latest.data.season}/${latest.data.week}`} />}

      {data && (
        <>
          <Provenance
            items={[
              ['Week', `season ${data.season} · week ${data.week}`],
              ['Scoring', data.scoring.toUpperCase()],
              ['Model', data.model_version],
              ['Players available', String(data.n)],
            ]}
            note={generatedLine(data.model_version, data.feature_version, data.data_through.season, data.data_through.week, data.generated_at_utc)}
          />

          {chosen.length === 0 && <EmptyState message="Search above and add two or more players to compare." />}

          {chosen.length > 0 && (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {chosen.map((p) => {
                const isBest = best?.player_id === p.player_id && chosen.length > 1
                return (
                  <article key={p.player_id} className={`relative border p-5 ${isBest ? 'border-ink bg-ink text-[#dbe5e1]' : 'border-[#c5c7c0] bg-white'}`}>
                    <button type="button" aria-label={`Remove ${p.name}`} onClick={() => setSelected((s) => s.filter((id) => id !== p.player_id))} className="absolute right-3 top-3 opacity-60 hover:opacity-100">
                      <XMarkIcon className="h-4 w-4" />
                    </button>
                    <p className={`font-mono text-[9px] font-bold uppercase tracking-widest ${isBest ? 'text-acid' : 'text-moss'}`}>{isBest ? 'Start · highest point estimate' : chosen.length > 1 ? 'Sit' : 'Selected'}</p>
                    <h3 className="mt-2 text-xl font-semibold">
                      <Link href={`/player/${encodeURIComponent(p.player_id)}`} className="hover:underline">{p.name ?? p.player_id}</Link>
                    </h3>
                    <p className={`font-mono text-[10px] ${isBest ? 'text-[#9fb0aa]' : 'text-[#8c9a96]'}`}>{p.position} · {p.team ?? '—'} · {p.candidate}</p>
                    <dl className="mt-5 grid grid-cols-3 gap-3">
                      <div><dt className="font-mono text-[9px] uppercase tracking-widest opacity-60">Floor</dt><dd className="mt-1 font-mono text-lg">{fmtNum(p.floor, 1)}</dd></div>
                      <div><dt className="font-mono text-[9px] uppercase tracking-widest opacity-60">Prediction</dt><dd className="mt-1 font-mono text-2xl font-bold">{fmtNum(p.prediction, 1)}</dd></div>
                      <div><dt className="font-mono text-[9px] uppercase tracking-widest opacity-60">Ceiling</dt><dd className="mt-1 font-mono text-lg">{fmtNum(p.ceiling, 1)}</dd></div>
                    </dl>
                    <div className={`relative mt-4 h-2 ${isBest ? 'bg-[#2e3b38]' : 'bg-[#e4e5dd]'}`} aria-hidden="true">
                      <div className="absolute top-0 h-full bg-[#b9d3c8]" style={{ left: `${(p.floor / maxCeiling) * 100}%`, width: `${((p.ceiling - p.floor) / maxCeiling) * 100}%` }} />
                      <div className={`absolute top-[-3px] h-3.5 w-0.5 ${isBest ? 'bg-acid' : 'bg-ink'}`} style={{ left: `${(p.prediction / maxCeiling) * 100}%` }} />
                    </div>
                    {p.actual !== null && <p className="mt-3 font-mono text-[10px] opacity-70">Scored: {fmtNum(p.actual, 1)}</p>}
                  </article>
                )
              })}
            </div>
          )}

          {chosen.length > 1 && best && (
            <aside className="grid gap-4 border border-[#c5c7c0] bg-[#f8f7f2] p-5 text-xs leading-relaxed text-[#52605d] md:grid-cols-2">
              <div>
                <p className="font-mono text-[9px] font-bold uppercase tracking-widest text-moss">Recommendation rule</p>
                <p className="mt-2">
                  Start <b>{best.name}</b> because it carries the highest point estimate among the players you selected. If another player&apos;s floor is higher and you need a safe
                  result, that is a legitimate reason to override this rule.
                </p>
              </div>
              <div>
                <p className="font-mono text-[9px] font-bold uppercase tracking-widest text-moss">Interval method</p>
                <p className="mt-2">{data.interval_method}</p>
              </div>
            </aside>
          )}
        </>
      )}
    </div>
  )
}
