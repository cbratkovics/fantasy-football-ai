'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { getTiers } from '@/lib/api/tiers'
import { POSITIONS, type Position } from '@/lib/api/types'
import { fmtInt, fmtNum, fmtPct, fmtUtc } from '@/lib/format'
import { tierPeriod } from '@/lib/presentation/tiers'
import { formatLocalDate, readPreference, writePreference } from '@/lib/presentation/display'
import { Segmented, TextInput } from '@/components/ui/Controls'
import { ErrorState, LoadingState } from '@/components/ui/States'
import { PageShell } from '@/components/ui/PageShell'
import { TierChart } from './TierChart'

export function TierBoard() {
  const [position, setPosition] = useState<Position>('RB')
  const [notice, setNotice] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [expandedGroups, setExpandedGroups] = useState<Record<number, boolean>>({})
  const userChanged = useRef(false)
  useEffect(() => {
    if (userChanged.current) return
    const explicit = new URLSearchParams(window.location.search).get('position')
    const saved = readPreference('ffai.tiers/view/1') as { position?: unknown } | null
    if (explicit !== null) {
      if (POSITIONS.includes(explicit as Position)) setPosition(explicit as Position)
      else setNotice('That position is not supported. Choose QB, RB, WR, or TE below.')
    } else if (saved?.position !== undefined) {
      if (POSITIONS.includes(saved.position as Position)) setPosition(saved.position as Position)
      else { writePreference('ffai.tiers/view/1', null); setNotice('An outdated position preference was cleared.') }
    }
  }, [])
  const choosePosition = (next: Position) => {
    userChanged.current = true
    setPosition(next)
    setExpandedGroups({})
    setNotice(null)
    writePreference('ffai.tiers/view/1', { position: next })
  }
  const tiers = useQuery({ queryKey: ['tiers', position], queryFn: () => getTiers(position) })
  const data = tiers.data
  const period = data ? tierPeriod(data) : null

  return (
    <PageShell wide crumb="Historical tiers" title={period?.title ?? 'Player groups'}
      lede="Explore historical groups based on prior-season production.">
      <div className="space-y-5">
        <section className="product-card flex flex-wrap items-center justify-between gap-4 p-5">
          <Segmented label="Position" options={POSITIONS} value={position} onChange={choosePosition} />
          <div className="w-full sm:w-64"><TextInput value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Search historical players" placeholder="Find a player or historical team" /></div>
        </section>
        {notice && <p role="status" className="text-sm text-[#52605d]">{notice}</p>}
        {tiers.isPending && <LoadingState label={`Loading ${position} player groups…`} />}
        {tiers.isError && <ErrorState error={tiers.error} context={`GET /tiers/${position}`} />}
        {data && period && <>
          <section aria-label="Tier season context" className="space-y-1 px-1">
            <h2 className="text-lg font-semibold">{period.targetSeason} season · {data.tiers.length} {position} groups</h2>
            <p className="text-sm leading-6 text-[#52605d]">
              {period.inputSeason !== null ? `Based on ${period.inputSeason} production` : 'Based on the artifact’s recorded inputs'}
              {period.evaluationSeason !== null ? ` · evaluated on ${period.evaluationSeason} results.` : ' · evaluation not recorded.'}
              {' '}Team labels are historical; group order is not an exact player rank.
            </p>
          </section>

          <div className="grid items-start gap-5 lg:grid-cols-2">
            {data.tiers.map((tier) => {
              const query = search.trim().toLowerCase()
              const matching = tier.players.filter((p) => !query || `${p.name ?? p.player_id} ${p.team ?? ''}`.toLowerCase().includes(query))
              const shown = expandedGroups[tier.tier] ? matching : matching.slice(0, 6)
              return <section key={tier.tier} className="product-card overflow-hidden bg-white">
                <header className="flex flex-wrap items-center justify-between gap-2 border-b border-[#d9ddd4] bg-[#f4f5ef] px-5 py-4">
                  <h2 className="text-lg font-semibold">Group {tier.tier}</h2>
                  <span className="text-sm text-[#52605d]">{query ? `${matching.length} matching of ${tier.players.length}` : `${shown.length} of ${tier.players.length}`} players</span>
                </header>
                <ul className="divide-y divide-[#e5e8df]">
                  {shown.map((p) => (
                    <li key={p.player_id} className="flex min-h-[76px] items-center justify-between gap-3 px-5 py-3">
                      <div className="min-w-0">
                        <Link href={`/player/${encodeURIComponent(p.player_id)}`} className="block text-base font-semibold underline-offset-4 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-moss">{p.name ?? p.player_id}</Link>
                        <p className="mt-1 text-sm text-[#52605d]">{position} · {p.team ?? 'team unknown'} <span className="sr-only">historical affiliation</span></p>
                      </div>
                      <div className="shrink-0 text-right">
                        <p className="text-base font-semibold tabular-nums">{fmtNum(p.ppg_prev, 1)} <span className="text-sm font-normal">PPR/game</span></p>
                        <p className="mt-1 text-sm text-[#52605d]">{p.games_prev} games{period.inputSeason !== null ? ` · ${period.inputSeason}` : ' · input season'}</p>
                      </div>
                    </li>
                  ))}
                </ul>
                {matching.length === 0 && <p className="px-5 py-4 text-sm text-[#52605d]">No players in this group match the search.</p>}
                {matching.length > 6 && <div className="border-t border-[#d9ddd4] px-5 py-3"><button type="button" className="product-button secondary w-full" onClick={() => setExpandedGroups((prev) => ({ ...prev, [tier.tier]: !prev[tier.tier] }))}>{expandedGroups[tier.tier] ? 'Show fewer players' : `Show all ${matching.length} players in group ${tier.tier}`}</button></div>}
              </section>
            })}
          </div>

          <Link href="/predictions" className="product-button secondary">View latest published projections →</Link>

          <details className="product-details">
            <summary>How these groups were built</summary>
            <div className="space-y-4 pt-4 text-base leading-7">
              <p>The model groups players with similar prior-season profiles and orders the groups by their mean prior-season PPR points per game. It does not forecast injuries, matchups, wins, or draft success.</p>
              <p>These historical groups are not current draft rankings or weekly start recommendations. A later generation date does not change the season they describe.</p>
              <p><b>Inputs:</b> {data.inputs}</p>
              <p><b>Method:</b> {data.method}</p>
              <h3 className="font-semibold">Assignment strength is a clustering measure</h3>
              <p>The mixture&apos;s assignment probability describes how strongly a player&apos;s input profile fits their assigned group. It is not their chance of winning, performing well, or being a good draft pick.</p>
              <div className="max-h-80 overflow-auto rounded-lg border border-[#d9ddd4]">
                <table className="w-full text-left text-sm"><thead><tr className="bg-[#f4f5ef]"><th className="p-3">Player</th><th className="p-3">Group</th><th className="p-3">Assignment strength</th></tr></thead>
                  <tbody>{data.tiers.flatMap((t) => t.players.map((p) => <tr key={p.player_id} className="border-t border-[#e5e8df]"><td className="p-3">{p.name ?? p.player_id}</td><td className="p-3">{t.tier}</td><td className="p-3 tabular-nums">{fmtPct(p.tier_probability, 1)}</td></tr>))}</tbody>
                </table>
              </div>
            </div>
          </details>

          <details className="product-details">
            <summary>Historical evaluation and source evidence</summary>
            <div className="space-y-4 pt-4 text-base leading-7">
              {data.evaluation ? <>
                <p>Evaluation period: <b>{period.evaluationSeason}</b>. These measures describe historical groups, not current lineup or draft outcomes.</p>
                <dl className="grid gap-5 sm:grid-cols-3">
                  <div><dt className="text-sm text-[#52605d]">Evaluated players</dt><dd className="text-2xl font-semibold tabular-nums">{fmtInt(data.evaluation.n)}</dd></div>
                  <div><dt className="text-sm text-[#52605d]">Rank correlation</dt><dd className="text-2xl font-semibold tabular-nums">{fmtNum(data.evaluation.spearman, 3)}</dd></div>
                  <div><dt className="text-sm text-[#52605d]">Within historical group band</dt><dd className="text-2xl font-semibold tabular-nums">{fmtPct(data.evaluation.within_band_rate)}</dd></div>
                </dl>
                <p>Rank correlation compares group order with realized PPR points per game. Within-band rate counts players whose realized season fell inside their group&apos;s stored band. Neither is a probability of future success.</p>
              </> : <p>No historical evaluation is recorded for this position.</p>}
              <p>Target season: {period.targetSeason}. Input season: {period.inputSeason ?? 'not specified'}. Generated {formatLocalDate(data.generated_at_utc)}.</p>
              <p className="break-all font-mono text-sm">Artifact: {data.tier_version}<br />Exact generation time: {fmtUtc(data.generated_at_utc)}</p>
            </div>
          </details>

          <TierChart tiers={data.tiers} position={position} inputSeason={period.inputSeason} />
        </>}
      </div>
    </PageShell>
  )
}
