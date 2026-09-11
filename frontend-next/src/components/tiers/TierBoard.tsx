'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { getTiers } from '@/lib/api/tiers'
import { POSITIONS, type Position } from '@/lib/api/types'
import { fmtInt, fmtNum, fmtPct, fmtUtc } from '@/lib/format'
import { Segmented } from '@/components/ui/Controls'
import { ErrorState, LoadingState } from '@/components/ui/States'
import { Provenance } from '@/components/ui/Provenance'
import { TierChart } from './TierChart'

export function TierBoard() {
  const [position, setPosition] = useState<Position>('RB')
  const tiers = useQuery({ queryKey: ['tiers', position], queryFn: () => getTiers(position) })

  return (
    <div className="space-y-6">
      <div className="border border-[#c5c7c0] bg-[#f8f7f2] p-5">
        <Segmented label="Position" options={POSITIONS} value={position} onChange={setPosition} />
      </div>

      {tiers.isPending && <LoadingState label={`Loading ${position} tiers…`} />}
      {tiers.isError && <ErrorState error={tiers.error} context={`GET /tiers/${position}`} />}

      {tiers.data && (
        <>
          <Provenance
            items={[
              ['Tier version', tiers.data.tier_version],
              ['Season', String(tiers.data.season)],
              ['Tiers', String(tiers.data.tiers.length)],
              ['Generated', fmtUtc(tiers.data.generated_at_utc)],
            ]}
          />

          <div className="grid gap-4 md:grid-cols-2">
            <article className="border border-[#c5c7c0] bg-[#f8f7f2] p-5 text-xs leading-relaxed text-[#52605d]">
              <p className="font-mono text-[9px] font-bold uppercase tracking-widest text-moss">Method</p>
              <p className="mt-2">{tiers.data.method}</p>
              <p className="mt-4 font-mono text-[9px] font-bold uppercase tracking-widest text-moss">Inputs</p>
              <p className="mt-2">{tiers.data.inputs}</p>
            </article>
            <article className="border border-[#c5c7c0] bg-ink p-5 text-[#dbe5e1]">
              <p className="font-mono text-[9px] font-bold uppercase tracking-widest text-acid">
                Evaluated on realised {tiers.data.season} PPR/game
              </p>
              {tiers.data.evaluation ? (
                <dl className="mt-4 grid grid-cols-3 gap-4">
                  <div>
                    <dt className="font-mono text-[9px] uppercase tracking-widest text-[#84918d]">n</dt>
                    <dd className="mt-1 text-2xl font-semibold text-white">{fmtInt(tiers.data.evaluation.n)}</dd>
                  </div>
                  <div>
                    <dt className="font-mono text-[9px] uppercase tracking-widest text-[#84918d]">Spearman</dt>
                    <dd className="mt-1 text-2xl font-semibold text-white">{fmtNum(tiers.data.evaluation.spearman, 3)}</dd>
                  </div>
                  <div>
                    <dt className="font-mono text-[9px] uppercase tracking-widest text-[#84918d]">Within band</dt>
                    <dd className="mt-1 text-2xl font-semibold text-white">{fmtPct(tiers.data.evaluation.within_band_rate)}</dd>
                  </div>
                </dl>
              ) : (
                <p className="mt-4 text-xs text-[#9fb0aa]">The tiers artifact carries no evaluation block for this position.</p>
              )}
              <p className="mt-4 text-[11px] leading-relaxed text-[#9fb0aa]">
                Spearman is the rank correlation between tier order and realised points per game; within band is the share of players whose realised
                season landed inside their tier&apos;s band. Both are computed by the tiers job and stored in the artifact.
              </p>
            </article>
          </div>

          <TierChart tiers={tiers.data.tiers} position={position} />

          <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
            {tiers.data.tiers.map((tier) => (
              <section key={tier.tier} className="border border-[#c5c7c0] bg-white">
                <header className="flex items-center justify-between border-b border-[#e4e5dd] bg-[#f8f7f2] px-4 py-3">
                  <h2 className="font-mono text-xs font-bold uppercase tracking-widest">Tier {tier.tier}</h2>
                  <span className="font-mono text-[10px] text-[#61706c]">{tier.players.length} players</span>
                </header>
                <ol className="divide-y divide-[#eeeee8]">
                  {tier.players.map((p) => (
                    <li key={p.player_id} className="grid grid-cols-[1fr_auto] items-center gap-3 px-4 py-2.5 text-sm">
                      <div>
                        <Link href={`/player/${encodeURIComponent(p.player_id)}`} className="font-semibold hover:text-moss hover:underline">
                          {p.name ?? p.player_id}
                        </Link>
                        <span className="ml-2 font-mono text-[10px] text-[#8c9a96]">{p.team ?? '—'}</span>
                        <div className="mt-1 font-mono text-[10px] text-[#61706c]">
                          {fmtNum(p.ppg_prev, 1)} ppg · {p.games_prev} games prior season
                        </div>
                      </div>
                      <div className="w-20 text-right">
                        <div className="font-mono text-[10px] text-[#61706c]">{fmtPct(p.tier_probability, 0)}</div>
                        <div className="mt-1 h-1.5 bg-[#e4e5dd]">
                          <div className="h-full bg-moss" style={{ width: `${Math.round(p.tier_probability * 100)}%` }} />
                        </div>
                      </div>
                    </li>
                  ))}
                </ol>
              </section>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
