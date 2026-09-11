'use client'

import { useQuery } from '@tanstack/react-query'
import { getPlayer } from '@/lib/api/players'
import { fmtNum, fmtUtc } from '@/lib/format'
import { PageShell } from '@/components/ui/PageShell'
import { ErrorState, LoadingState } from '@/components/ui/States'
import { Provenance } from '@/components/ui/Provenance'
import { PlayerHistoryChart } from './PlayerHistoryChart'

export function PlayerProfile({ playerId }: { playerId: string }) {
  const player = useQuery({ queryKey: ['player', playerId], queryFn: () => getPlayer(playerId) })
  const data = player.data

  return (
    <PageShell
      wide
      crumb={data?.name ?? playerId}
      eyebrow="Player history · frozen test + weekly rows"
      title={
        data ? (
          <>
            {data.name ?? playerId}{' '}
            <em className="font-serif font-normal text-moss">
              {data.position ?? ''} {data.team ? `· ${data.team}` : ''}
            </em>
          </>
        ) : (
          playerId
        )
      }
      lede="Rows marked frozen_test come from the champion's held-out test predictions and carry realised points. Rows marked weekly come from published prediction files and gain an actual only once that week is scored."
    >
      {player.isPending && <LoadingState label="Loading player history…" />}
      {player.isError && <ErrorState error={player.error} context={`GET /players/${playerId}`} />}
      {data && (
        <div className="space-y-6">
          <Provenance
            items={[
              ['Player id', data.player_id],
              ['Model', data.model_version],
              ['Features', data.feature_version],
              ['Served', fmtUtc(data.generated_at_utc)],
            ]}
          />

          <PlayerHistoryChart history={data.history} />

          <div className="overflow-x-auto border border-[#c5c7c0] bg-white">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead className="bg-[#f8f7f2] font-mono text-[9px] uppercase tracking-widest text-[#61706c]">
                <tr>
                  <th className="px-4 py-3">Season</th>
                  <th className="px-4 py-3">Week</th>
                  <th className="px-4 py-3 text-right">Prediction</th>
                  <th className="px-4 py-3 text-right">Floor</th>
                  <th className="px-4 py-3 text-right">Ceiling</th>
                  <th className="px-4 py-3 text-right">Actual</th>
                  <th className="px-4 py-3 text-right">Error</th>
                  <th className="px-4 py-3">Source</th>
                  <th className="px-4 py-3">Model</th>
                </tr>
              </thead>
              <tbody>
                {data.history.map((row) => {
                  const err = row.actual !== null && row.prediction !== null ? row.actual - row.prediction : null
                  return (
                    <tr key={`${row.season}-${row.week}-${row.source}`} className="border-t border-[#e4e5dd] font-mono text-xs">
                      <td className="px-4 py-2.5">{row.season}</td>
                      <td className="px-4 py-2.5">{row.week}</td>
                      <td className="px-4 py-2.5 text-right font-bold">{fmtNum(row.prediction, 1)}</td>
                      <td className="px-4 py-2.5 text-right text-[#52605d]">{fmtNum(row.floor, 1)}</td>
                      <td className="px-4 py-2.5 text-right text-[#52605d]">{fmtNum(row.ceiling, 1)}</td>
                      <td className="px-4 py-2.5 text-right">{row.actual === null ? <span className="text-[#8c9a96]">not scored</span> : fmtNum(row.actual, 1)}</td>
                      <td className={`px-4 py-2.5 text-right ${err === null ? 'text-[#8c9a96]' : err < 0 ? 'text-ember' : 'text-moss'}`}>
                        {err === null ? '—' : `${err > 0 ? '+' : ''}${fmtNum(err, 1)}`}
                      </td>
                      <td className="px-4 py-2.5">
                        <span className={`px-1.5 py-0.5 text-[10px] ${row.source === 'frozen_test' ? 'bg-sand text-ink' : 'bg-ink text-acid'}`}>{row.source}</span>
                      </td>
                      <td className="px-4 py-2.5 text-[#8c9a96]">{row.model_version ?? '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </PageShell>
  )
}
