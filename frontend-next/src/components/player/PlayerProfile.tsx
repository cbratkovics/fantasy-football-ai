'use client'

import { useQuery } from '@tanstack/react-query'
import { getPlayer } from '@/lib/api/players'
import { getPlayerWeek } from '@/lib/api/marts'
import { fmtInt, fmtNum, fmtUtc, shortHash } from '@/lib/format'
import { PageShell } from '@/components/ui/PageShell'
import { ErrorState, LoadingState } from '@/components/ui/States'
import { Provenance } from '@/components/ui/Provenance'
import { PROJECT, withinLabel } from '@/lib/project'
import { PERFORMANCE_COPY } from '@/content/performance'
import { PlayerHistoryChart, SOURCE_LABEL, SOURCE_STYLE } from './PlayerHistoryChart'

const [BAND_1] = PROJECT.withinK
const COPY = PERFORMANCE_COPY.player

export function PlayerProfile({ playerId }: { playerId: string }) {
  // Identity from the artifact-backed route; the history itself from the gold mart.
  const player = useQuery({ queryKey: ['player', playerId], queryFn: () => getPlayer(playerId) })
  const marts = useQuery({ queryKey: ['marts', 'player_week', playerId], queryFn: () => getPlayerWeek(playerId) })
  const data = player.data
  const week = marts.data
  const name = week?.name ?? data?.name ?? playerId
  const position = week?.position ?? data?.position ?? ''
  const team = week?.team ?? data?.team ?? null

  return (
    <PageShell
      wide
      crumb={name}
      eyebrow={COPY.eyebrow}
      title={
        <>
          {name}{' '}
          <em className="font-serif font-normal text-moss">
            {position} {team ? `· ${team}` : ''}
          </em>
        </>
      }
      lede={COPY.lede}
    >
      {(player.isPending || marts.isPending) && <LoadingState label="Loading player history from the marts…" />}
      {player.isError && <ErrorState error={player.error} context={`GET /players/${playerId}`} />}
      {marts.isError && <ErrorState error={marts.error} context={`GET /marts/player_week/${playerId}`} />}
      {week && (
        <div className="space-y-6">
          <Provenance
            items={[
              [`${PROJECT.entity.name} id`, week.player_id],
              ['Mart', `${week.mart} · ${fmtInt(week.n)} rows`],
              ['Candidate', typeof week.candidate === 'string' ? week.candidate : 'champion per position'],
              ['Exported', `${fmtUtc(week.export.exported_at_utc)} · target ${week.export.target ?? '—'} · commit ${shortHash(week.export.code_commit)}`],
              ['Model', data?.model_version ?? '—'],
            ]}
          />

          <PlayerHistoryChart history={week.rows} />

          <div className="overflow-x-auto border border-[#c5c7c0] bg-white">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead className="bg-[#f8f7f2] font-mono text-[9px] uppercase tracking-widest text-[#61706c]">
                <tr>
                  <th className="px-4 py-3">{PROJECT.period.season}</th>
                  <th className="px-4 py-3">{PROJECT.period.name}</th>
                  <th className="px-4 py-3 text-right">Prediction</th>
                  <th className="px-4 py-3 text-right">Floor</th>
                  <th className="px-4 py-3 text-right">Ceiling</th>
                  <th className="px-4 py-3 text-right">Actual</th>
                  <th className="px-4 py-3 text-right">Error</th>
                  <th className="px-4 py-3 text-right">Baseline</th>
                  <th className="px-4 py-3">{withinLabel(BAND_1)}</th>
                  <th className="px-4 py-3">Source</th>
                  <th className="px-4 py-3">Model</th>
                </tr>
              </thead>
              <tbody>
                {week.rows.map((row) => {
                  const err = row.actual !== null ? row.actual - row.prediction : null
                  return (
                    <tr key={`${row.season}-${row.week}-${row.model_version}-${row.candidate}`} className="border-t border-[#e4e5dd] font-mono text-xs">
                      <td className="px-4 py-2.5">{row.season}</td>
                      <td className="px-4 py-2.5">{row.week}</td>
                      <td className="px-4 py-2.5 text-right font-bold">{fmtNum(row.prediction, 1)}</td>
                      <td className="px-4 py-2.5 text-right text-[#52605d]">{fmtNum(row.prediction_floor, 1)}</td>
                      <td className="px-4 py-2.5 text-right text-[#52605d]">{fmtNum(row.prediction_ceiling, 1)}</td>
                      <td className="px-4 py-2.5 text-right">
                        {row.actual === null ? <span className="text-[#8c9a96]">not scored</span> : fmtNum(row.actual, 1)}
                        {row.actual_source === 'artifact' && <span className="ml-1 text-[9px] text-[#8c9a96]" title="actual recorded in the prediction artifact">†</span>}
                      </td>
                      <td className={`px-4 py-2.5 text-right ${err === null ? 'text-[#8c9a96]' : err < 0 ? 'text-ember' : 'text-moss'}`}>
                        {err === null ? '—' : `${err > 0 ? '+' : ''}${fmtNum(err, 1)}`}
                      </td>
                      <td className="px-4 py-2.5 text-right text-[#52605d]">{fmtNum(row.baseline, 1)}</td>
                      <td className="px-4 py-2.5">{row.within_3 === null ? '—' : row.within_3 ? 'yes' : 'no'}</td>
                      <td className="px-4 py-2.5">
                        <span className={`px-1.5 py-0.5 text-[10px] ${SOURCE_STYLE[row.source] ?? 'bg-ink text-acid'}`}>{SOURCE_LABEL[row.source] ?? row.source}</span>
                      </td>
                      <td className="px-4 py-2.5 text-[#8c9a96]">{row.model_version}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="font-mono text-[9px] uppercase tracking-widest text-[#8c9a96]">
            † actual taken from the prediction artifact where the stats row is not in the warehouse; the two agree wherever both exist.
          </p>
        </div>
      )}
    </PageShell>
  )
}
