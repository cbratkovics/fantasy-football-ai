'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'
import { useQueries } from '@tanstack/react-query'
import { getTiers } from '@/lib/api/tiers'
import { POSITIONS, type Position } from '@/lib/api/types'
import { fmtNum, fmtPct } from '@/lib/format'
import { Segmented, TextInput } from '@/components/ui/Controls'
import { ErrorState, LoadingState } from '@/components/ui/States'
import { Provenance } from '@/components/ui/Provenance'

interface BoardPlayer {
  player_id: string
  name: string
  team: string
  position: Position
  tier: number
  ppg_prev: number
  games_prev: number
  tier_probability: number
}

interface Pick {
  overall: number
  round: number
  team: number
  player: BoardPlayer
}

// Roster caps are structural draft rules, not performance claims.
const ROSTER_CAP: Record<Position, number> = { QB: 2, RB: 6, WR: 6, TE: 2 }
const TEAM_COUNTS = ['8', '10', '12'] as const
const ROUNDS = 12
type PositionFilter = 'ALL' | Position
const POSITION_FILTERS: readonly PositionFilter[] = ['ALL', ...POSITIONS]

const byBoardOrder = (a: BoardPlayer, b: BoardPlayer) => a.tier - b.tier || b.ppg_prev - a.ppg_prev || a.name.localeCompare(b.name)

function snakeTeam(overall: number, teams: number): { round: number; team: number } {
  const round = Math.floor((overall - 1) / teams) + 1
  const slot = (overall - 1) % teams
  return { round, team: round % 2 === 1 ? slot : teams - 1 - slot }
}

function aiChoose(available: BoardPlayer[], roster: BoardPlayer[]): BoardPlayer | undefined {
  const counts = roster.reduce<Record<string, number>>((acc, p) => ({ ...acc, [p.position]: (acc[p.position] ?? 0) + 1 }), {})
  return available.find((p) => (counts[p.position] ?? 0) < ROSTER_CAP[p.position]) ?? available[0]
}

export function DraftSimulator() {
  const queries = useQueries({ queries: POSITIONS.map((p) => ({ queryKey: ['tiers', p], queryFn: () => getTiers(p) })) })
  const [teamCount, setTeamCount] = useState<(typeof TEAM_COUNTS)[number]>('10')
  const [userSlot, setUserSlot] = useState(0)
  const [picks, setPicks] = useState<Pick[]>([])
  const [started, setStarted] = useState(false)
  const [filter, setFilter] = useState<PositionFilter>('ALL')
  const [search, setSearch] = useState('')

  const board = useMemo<BoardPlayer[]>(() => {
    if (queries.some((q) => !q.data)) return []
    return queries
      .flatMap((q) =>
        (q.data?.tiers ?? []).flatMap((t) =>
          t.players.map((p) => ({
            player_id: p.player_id,
            name: p.name ?? p.player_id,
            team: p.team ?? '—',
            position: q.data?.position as Position,
            tier: t.tier,
            ppg_prev: p.ppg_prev,
            games_prev: p.games_prev,
            tier_probability: p.tier_probability,
          })),
        ),
      )
      .sort(byBoardOrder)
  }, [queries])

  const teams = Number(teamCount)
  const totalPicks = teams * ROUNDS
  const taken = useMemo(() => new Set(picks.map((p) => p.player.player_id)), [picks])
  const available = useMemo(() => board.filter((p) => !taken.has(p.player_id)), [board, taken])
  const nextOverall = picks.length + 1
  const onClock = nextOverall <= totalPicks ? snakeTeam(nextOverall, teams) : null
  const complete = started && nextOverall > totalPicks
  const rosterOf = (team: number) => picks.filter((p) => p.team === team).map((p) => p.player)

  const shown = useMemo(() => {
    const q = search.trim().toLowerCase()
    return available.filter((p) => (filter === 'ALL' || p.position === filter) && (!q || p.name.toLowerCase().includes(q) || p.team.toLowerCase().includes(q))).slice(0, 60)
  }, [available, filter, search])

  /** Run AI picks until it is the user's turn or the draft is complete. */
  function advance(current: Pick[]) {
    const next = [...current]
    let remaining = board.filter((p) => !next.some((x) => x.player.player_id === p.player_id))
    while (next.length < totalPicks) {
      const overall = next.length + 1
      const { round, team } = snakeTeam(overall, teams)
      if (team === userSlot) break
      const roster = next.filter((p) => p.team === team).map((p) => p.player)
      const choice = aiChoose(remaining, roster)
      if (!choice) break
      next.push({ overall, round, team, player: choice })
      remaining = remaining.filter((p) => p.player_id !== choice.player_id)
    }
    setPicks(next)
  }

  function start() {
    setStarted(true)
    advance([])
  }

  function reset() {
    setStarted(false)
    setPicks([])
  }

  function userPick(player: BoardPlayer) {
    if (!onClock || onClock.team !== userSlot) return
    advance([...picks, { overall: nextOverall, round: onClock.round, team: onClock.team, player }])
  }

  if (queries.some((q) => q.isPending)) return <LoadingState label="Loading tiers for QB, RB, WR, TE…" />
  const failed = queries.find((q) => q.isError)
  if (failed) return <ErrorState error={failed.error} context="GET /tiers/{position}" />

  const tiersMeta = queries[0].data
  const userRoster = rosterOf(userSlot)

  return (
    <div className="space-y-6">
      {tiersMeta && (
        <Provenance
          items={[
            ['Tier version', tiersMeta.tier_version],
            ['Season', String(tiersMeta.season)],
            ['Board size', `${board.length} players`],
            ['Format', `${teams} teams · ${ROUNDS} rounds · snake`],
          ]}
          note={`Inputs: ${tiersMeta.inputs}`}
        />
      )}

      <div className="flex flex-col gap-4 border border-[#c5c7c0] bg-[#f8f7f2] p-5 lg:flex-row lg:flex-wrap lg:items-center lg:justify-between">
        <div className="flex flex-col gap-4 sm:flex-row sm:flex-wrap sm:items-center">
          <Segmented label="Teams" options={TEAM_COUNTS} value={teamCount} onChange={(v) => { if (!started) { setTeamCount(v); setUserSlot(0) } }} />
          <div className="flex items-center gap-3">
            <label htmlFor="slot" className="font-mono text-[9px] font-bold uppercase tracking-widest">Your slot</label>
            <select id="slot" disabled={started} value={userSlot} onChange={(e) => setUserSlot(Number(e.target.value))} className="h-9 border border-[#bfc3bb] bg-white px-2 font-mono text-xs disabled:opacity-50">
              {Array.from({ length: teams }, (_, i) => (
                <option key={i} value={i}>Pick {i + 1}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="flex gap-3">
          {!started ? (
            <button type="button" onClick={start} className="portfolio-button primary">Start draft</button>
          ) : (
            <button type="button" onClick={reset} className="portfolio-button secondary">Reset</button>
          )}
        </div>
      </div>

      {started && (
        <div className="border border-ink bg-ink px-4 py-3 font-mono text-xs text-acid">
          {complete
            ? `Draft complete · ${picks.length} picks`
            : onClock?.team === userSlot
              ? `You are on the clock · round ${onClock.round} · pick ${nextOverall} of ${totalPicks}`
              : `Waiting…`}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[1.2fr_.8fr]">
        <section className="border border-[#c5c7c0] bg-white">
          <header className="flex flex-col gap-3 border-b border-[#e4e5dd] bg-[#f8f7f2] p-4 sm:flex-row sm:items-center sm:justify-between">
            <h2 className="font-mono text-xs font-bold uppercase tracking-widest">Available · {available.length}</h2>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
              <Segmented label="Pos" options={POSITION_FILTERS} value={filter} onChange={setFilter} />
              <div className="w-full sm:w-48"><TextInput placeholder="Search" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Search available players" /></div>
            </div>
          </header>
          <ol className="max-h-[640px] divide-y divide-[#eeeee8] overflow-y-auto">
            {shown.map((p) => (
              <li key={p.player_id} className="grid grid-cols-[auto_1fr_auto_auto] items-center gap-3 px-4 py-2 text-sm">
                <span className="w-8 bg-sand px-1.5 py-0.5 text-center font-mono text-[10px] font-bold">T{p.tier}</span>
                <div>
                  <Link href={`/player/${encodeURIComponent(p.player_id)}`} className="font-semibold hover:text-moss hover:underline">{p.name}</Link>
                  <span className="ml-2 font-mono text-[10px] text-[#8c9a96]">{p.position} · {p.team}</span>
                </div>
                <span className="font-mono text-[10px] text-[#61706c]">{fmtNum(p.ppg_prev, 1)} ppg · {fmtPct(p.tier_probability, 0)}</span>
                <button
                  type="button"
                  disabled={!started || complete || onClock?.team !== userSlot}
                  onClick={() => userPick(p)}
                  className="border border-ink px-3 py-1 font-mono text-[10px] font-bold uppercase hover:bg-ink hover:text-acid disabled:cursor-not-allowed disabled:opacity-30"
                >
                  Draft
                </button>
              </li>
            ))}
            {shown.length === 0 && <li className="p-6 text-center text-sm text-[#6e7875]">No available players match.</li>}
          </ol>
        </section>

        <div className="space-y-6">
          <section className="border border-[#c5c7c0] bg-white">
            <header className="border-b border-[#e4e5dd] bg-[#f8f7f2] px-4 py-3">
              <h2 className="font-mono text-xs font-bold uppercase tracking-widest">Your roster · {userRoster.length}/{ROUNDS}</h2>
            </header>
            <ul className="divide-y divide-[#eeeee8]">
              {userRoster.map((p) => (
                <li key={p.player_id} className="flex items-center justify-between px-4 py-2 text-sm">
                  <span><b>{p.name}</b> <span className="font-mono text-[10px] text-[#8c9a96]">{p.position} · {p.team}</span></span>
                  <span className="font-mono text-[10px] text-[#61706c]">T{p.tier} · {fmtNum(p.ppg_prev, 1)}</span>
                </li>
              ))}
              {userRoster.length === 0 && <li className="p-6 text-center text-sm text-[#6e7875]">No picks yet.</li>}
            </ul>
          </section>

          <section className="border border-[#c5c7c0] bg-white">
            <header className="border-b border-[#e4e5dd] bg-[#f8f7f2] px-4 py-3">
              <h2 className="font-mono text-xs font-bold uppercase tracking-widest">Draft log</h2>
            </header>
            <ol className="max-h-80 divide-y divide-[#eeeee8] overflow-y-auto">
              {[...picks].reverse().map((p) => (
                <li key={p.overall} className={`grid grid-cols-[48px_1fr_auto] items-center gap-3 px-4 py-1.5 text-xs ${p.team === userSlot ? 'bg-[#eef7e3]' : ''}`}>
                  <span className="font-mono text-[10px] text-[#8c9a96]">{p.round}.{String(((p.overall - 1) % teams) + 1).padStart(2, '0')}</span>
                  <span><b>{p.player.name}</b> <span className="font-mono text-[10px] text-[#8c9a96]">{p.player.position}</span></span>
                  <span className="font-mono text-[10px] text-[#61706c]">{p.team === userSlot ? 'You' : `Team ${p.team + 1}`}</span>
                </li>
              ))}
              {picks.length === 0 && <li className="p-6 text-center text-sm text-[#6e7875]">The log fills in once the draft starts.</li>}
            </ol>
          </section>
        </div>
      </div>
    </div>
  )
}
