'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'
import { useQueries } from '@tanstack/react-query'
import { getTiers } from '@/lib/api/tiers'
import { POSITIONS, type Position } from '@/lib/api/types'
import { fmtNum } from '@/lib/format'
import { Segmented, TextInput } from '@/components/ui/Controls'
import { ErrorState, LoadingState } from '@/components/ui/States'
import { tierPeriod } from '@/lib/presentation/tiers'
import { formatLocalDate } from '@/lib/presentation/display'

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
  const period = tiersMeta ? tierPeriod(tiersMeta) : null
  const compatible = queries.every((q) => q.data?.season === tiersMeta?.season && q.data?.tier_version === tiersMeta?.tier_version)
  if (!compatible) return <ErrorState error={new Error('Player groups do not belong to one compatible historical artifact.')} context="Historical draft exercise" />

  return (
    <div className="space-y-6">
      {tiersMeta && period && <section className="product-card border-l-4 border-l-[#176e5a] p-5">
        <h2 className="text-xl font-semibold">{period.targetSeason} historical exercise</h2>
        <p className="mt-2 text-base leading-7">This board uses {period.inputSeason ?? 'prior-season'} production to group players for {period.targetSeason}. It is not a current draft ranking. Team labels are historical affiliations.</p>
        <p className="mt-2 text-sm text-[#52605d]">{board.length} grouped players · {teams} simulated teams · {ROUNDS} rounds · snake order.</p>
        <Link href="/predictions" className="mt-3 inline-flex min-h-[44px] items-center font-semibold underline underline-offset-4">View latest published projections →</Link>
      </section>}

      <details className="product-details"><summary>How this historical exercise works</summary><div className="space-y-3 pt-4 text-base leading-7">
        <p>Simulated opponents choose by group number, then prior-season PPR points per game, subject to simple roster caps. Group numbers from different positions are not a calibrated cross-position ranking. No hidden strategy or current injury information is used.</p>
        <p>Grouping assignment strength is how well an input profile fits its assigned mixture component. It is not a player&apos;s probability of success. The detailed strengths and historical evaluation remain available on <Link className="underline" href="/tiers">Historical player groups</Link>.</p>
        {tiersMeta && <><p>Inputs: {tiersMeta.inputs}</p><p>Generated {formatLocalDate(tiersMeta.generated_at_utc)}. The generation date is separate from the target season.</p><p className="break-all font-mono text-sm">Artifact: {tiersMeta.tier_version}</p></>}
      </div></details>

      <div className="product-card flex flex-col gap-4 p-5 lg:flex-row lg:flex-wrap lg:items-center lg:justify-between">
        <div className="flex flex-col gap-4 sm:flex-row sm:flex-wrap sm:items-center">
          <Segmented label="Teams" options={TEAM_COUNTS} value={teamCount} onChange={(v) => { if (!started) { setTeamCount(v); setUserSlot(0) } }} />
          <div className="flex items-center gap-3">
            <label htmlFor="slot" className="text-sm font-semibold">Your slot</label>
            <select id="slot" disabled={started} value={userSlot} onChange={(e) => setUserSlot(Number(e.target.value))} className="product-field min-h-[44px] disabled:opacity-50">
              {Array.from({ length: teams }, (_, i) => (
                <option key={i} value={i}>Pick {i + 1}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="flex gap-3">
          {!started ? (
            <button type="button" onClick={start} className="product-button primary">Start historical exercise</button>
          ) : (
            <button type="button" onClick={reset} className="product-button secondary">Reset exercise</button>
          )}
        </div>
      </div>

      {started && (
        <div className="product-card border-l-4 border-l-[#176e5a] p-4 text-base">
          {complete
            ? `Exercise complete · ${picks.length} picks`
            : onClock?.team === userSlot
              ? `You are on the clock · round ${onClock.round} · pick ${nextOverall} of ${totalPicks}`
              : `Waiting…`}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[1.2fr_.8fr]">
        <section className="product-card overflow-hidden bg-white">
          <header className="flex flex-col gap-3 border-b border-[#e4e5dd] bg-[#f8f7f2] p-4 sm:flex-row sm:items-center sm:justify-between">
            <h2 className="text-lg font-semibold">Historical board · {available.length} remaining</h2>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
              <Segmented label="Pos" options={POSITION_FILTERS} value={filter} onChange={setFilter} />
              <div className="w-full sm:w-48"><TextInput placeholder="Search" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Search available players" /></div>
            </div>
          </header>
          <ol className="max-h-[640px] divide-y divide-[#eeeee8] overflow-y-auto">
            {shown.map((p) => (
              <li key={p.player_id} className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 px-4 py-3 text-base">
                <span className="w-8 bg-sand px-1.5 py-0.5 text-center text-sm font-semibold">T{p.tier}</span>
                <div className="min-w-0">
                  <Link href={`/player/${encodeURIComponent(p.player_id)}`} className="font-semibold hover:text-moss hover:underline">{p.name}</Link>
                  <p className="mt-1 text-sm text-[#52605d]">{p.position} · {p.team} · {fmtNum(p.ppg_prev, 1)} prior-season PPR/game</p>
                </div>
                <button
                  type="button"
                  disabled={!started || complete || onClock?.team !== userSlot}
                  onClick={() => userPick(p)}
                  className="product-button min-h-[44px] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Pick
                </button>
              </li>
            ))}
            {shown.length === 0 && <li className="p-6 text-center text-sm text-[#6e7875]">No available players match.</li>}
          </ol>
        </section>

        <div className="space-y-6">
          <section className="product-card overflow-hidden bg-white">
            <header className="border-b border-[#e4e5dd] bg-[#f8f7f2] px-4 py-3">
              <h2 className="text-lg font-semibold">Your practice roster · {userRoster.length}/{ROUNDS}</h2>
            </header>
            <ul className="divide-y divide-[#eeeee8]">
              {userRoster.map((p) => (
                <li key={p.player_id} className="flex items-center justify-between px-4 py-2 text-sm">
                  <span><b>{p.name}</b> <span className="text-sm text-[#52605d]">{p.position} · {p.team}</span></span>
                  <span className="text-sm text-[#52605d]">T{p.tier} · {fmtNum(p.ppg_prev, 1)}</span>
                </li>
              ))}
              {userRoster.length === 0 && <li className="p-6 text-center text-sm text-[#6e7875]">No picks yet.</li>}
            </ul>
          </section>

          <section className="product-card overflow-hidden bg-white">
            <header className="border-b border-[#e4e5dd] bg-[#f8f7f2] px-4 py-3">
              <h2 className="text-lg font-semibold">Exercise picks</h2>
            </header>
            <ol className="max-h-80 divide-y divide-[#eeeee8] overflow-y-auto">
              {[...picks].reverse().map((p) => (
                <li key={p.overall} className={`grid grid-cols-[48px_minmax(0,1fr)_auto] items-center gap-3 px-4 py-3 text-sm ${p.team === userSlot ? 'bg-[#eef7e3]' : ''}`}>
                  <span className="text-sm text-[#52605d]">{p.round}.{String(((p.overall - 1) % teams) + 1).padStart(2, '0')}</span>
                  <span><b>{p.player.name}</b> <span className="text-sm text-[#52605d]">{p.player.position}</span></span>
                  <span className="text-sm text-[#52605d]">{p.team === userSlot ? 'You' : `Team ${p.team + 1}`}</span>
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
