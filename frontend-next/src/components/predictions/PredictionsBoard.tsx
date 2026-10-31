'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { getPredictionPeriods, getPredictions } from '@/lib/api/players'
import { type PredictionRecord, type ScoringFormat } from '@/lib/api/types'
import { fmtNum } from '@/lib/format'
import { ActualPoints, actualPresentation } from '@/components/ui/ActualPoints'
import { Segmented } from '@/components/ui/Controls'
import { EmptyState, LoadingState } from '@/components/ui/States'
import { formatLocalDate, formatPeriod } from '@/lib/presentation/display'
import {
  actualCoverage, comparisonDestination, periodKey, projectionDefaults, projectionGroup,
  PROJECTION_POSITIONS, PROJECTIONS_PREFERENCE, rangeScale, type ProjectionSettings,
} from '@/lib/presentation/projections'

const PAGE_SIZE = 25
const SCORING: ScoringFormat[] = ['ppr', 'half', 'standard']

export function PredictionsBoard() {
  const index = useQuery({ queryKey: ['prediction-periods'], queryFn: getPredictionPeriods })
  const [settings, setSettings] = useState<ProjectionSettings | null>(null)
  const [search, setSearch] = useState('')
  const [limit, setLimit] = useState(PAGE_SIZE)
  const [selected, setSelected] = useState<string[]>([])
  const [destination, setDestination] = useState<{ href: string; key: string } | null>(null)
  const [transferError, setTransferError] = useState<string | null>(null)
  const [checking, setChecking] = useState(false)
  const transferRequest = useRef(0)
  const initialized = useRef(false)

  useEffect(() => {
    if (!index.data || initialized.current) return
    initialized.current = true
    let saved: unknown = null
    try { saved = JSON.parse(localStorage.getItem(PROJECTIONS_PREFERENCE) ?? 'null') } catch { /* optional preferences */ }
    setSettings(projectionDefaults(index.data.periods, new URLSearchParams(window.location.search), saved))
  }, [index.data])
  useEffect(() => {
    if (settings) try { localStorage.setItem(PROJECTIONS_PREFERENCE, JSON.stringify({ period: settings.period, scoring: settings.scoring, position: settings.position })) } catch { /* optional preferences */ }
  }, [settings])
  const period = index.data?.periods.find((p) => periodKey(p) === settings?.period)
  const scoring = settings?.scoring ?? 'ppr'
  const position = settings?.position ?? 'FLEX'
  const predictions = useQuery({
    queryKey: ['predictions', period?.season, period?.week, scoring, 'ALL'],
    queryFn: () => getPredictions(period!.season, period!.week, scoring),
    enabled: period !== undefined,
  })
  const data = predictions.data
  const transferKey = JSON.stringify([data?.season, data?.week, data?.model_version, data?.feature_version,
    data?.generated_at_utc, data?.data_through, scoring, selected,
    selected.map((id) => data?.predictions.find((r) => r.player_id === id))])
  const readyDestination = destination?.key === transferKey ? destination.href : null
  const group = useMemo(() => projectionGroup(data?.predictions ?? [], position), [data, position])
  const rows = useMemo(() => {
    const q = search.trim().toLowerCase()
    return group.map((row, i) => ({ row, rank: i + 1 })).filter(({ row }) => !q || `${row.name ?? ''} ${row.team ?? ''} ${row.player_id}`.toLowerCase().includes(q))
  }, [group, search])
  const shown = rows.slice(0, limit)
  const scale = rangeScale(group)
  const coverage = actualCoverage(group, scoring)
  const update = (next: Partial<ProjectionSettings>) => {
    setSettings((old) => old ? { ...old, ...next, notice: null } : old)
    setLimit(PAGE_SIZE)
    setDestination(null)
    setTransferError(null)
    transferRequest.current += 1
    setChecking(false)
  }
  const toggle = (id: string) => {
    if (!selected.includes(id) && selected.length === 8) return
    setSelected((ids) => ids.includes(id) ? ids.filter((p) => p !== id) : [...ids, id])
    transferRequest.current += 1
    setDestination(null)
    setTransferError(null)
    setChecking(false)
  }
  const prepareComparison = async () => {
    if (!data || !period) return
    const request = ++transferRequest.current
    setChecking(true)
    setTransferError(null)
    try {
      // Compare from the PPR source regardless of browsing format. Keep the browsing choice.
      const ppr = scoring === 'ppr' ? data : await getPredictions(period.season, period.week, 'ppr')
      const href = await comparisonDestination(ppr, selected)
      if (request === transferRequest.current) setDestination({ href, key: transferKey })
    } catch (error) {
      if (request === transferRequest.current) setTransferError(error instanceof Error ? error.message : String(error))
    } finally { if (request === transferRequest.current) setChecking(false) }
  }

  if (index.isPending || (index.data && settings === null)) return <LoadingState label="Loading published periods…" />
  if (index.isError) return <div role="alert" className="product-card"><h2 className="text-xl font-semibold">Player projections are unavailable</h2><p className="mt-2">The API could not provide its published period index. No guessed week or replacement values are shown.</p><Link className="product-button mt-4 inline-flex" href="/start-sit">Use the published comparison</Link></div>
  if (!index.data?.periods.length) return <EmptyState message="No published projection periods are available. You can explore historical and synthetic evidence in Decision Lab." />

  const addButton = (r: PredictionRecord) => <button type="button" className="product-button secondary" aria-pressed={selected.includes(r.player_id)} aria-label={`${selected.includes(r.player_id) ? 'Remove' : 'Add'} ${r.name ?? r.player_id} ${selected.includes(r.player_id) ? 'from' : 'to'} comparison`} disabled={!selected.includes(r.player_id) && selected.length >= 8} onClick={() => toggle(r.player_id)}>{selected.includes(r.player_id) ? 'Added ✓' : 'Add to comparison'}</button>
  const actual = (r: PredictionRecord) => {
    const value = <ActualPoints actual={r.actual} reason={r.actual_null_reason} source={r.actual_source} actualFormat={r.actual_scoring_format} expectedFormat={scoring} />
    return actualPresentation(r.actual, r.actual_scoring_format, scoring, r.actual_null_reason).value === null
      ? <details className="text-sm"><summary className="cursor-pointer">Unavailable</summary><div className="mt-2 text-left">{value}</div></details>
      : value
  }
  return <div className="space-y-5">
    <section className="product-card space-y-4" aria-label="Projection filters">
      <div className="flex flex-col items-stretch gap-4 sm:flex-row sm:flex-wrap sm:items-end">
        <label className="min-w-0 w-full text-sm font-semibold sm:w-auto sm:flex-1" htmlFor="projection-period">Published period<select id="projection-period" className="product-field mt-1 w-full" value={period ? periodKey(period) : ''} onChange={(e) => update({ period: e.target.value })}>
          {!period && <option value="">Choose an available period</option>}
          {index.data.periods.map((p, i) => <option key={periodKey(p)} value={periodKey(p)}>{i === 0 ? 'Latest published: ' : ''}{formatPeriod(p.season, p.week)}</option>)}
        </select></label>
        <Segmented label="Scoring" options={SCORING} value={scoring} onChange={(value) => update({ scoring: value })} />
      </div>
      <Segmented label="Position group" options={PROJECTION_POSITIONS} value={position} onChange={(value) => update({ position: value })} />
      <label className="block text-sm font-semibold" htmlFor="projection-search">Find a player or forecast team<input id="projection-search" className="product-field mt-1 w-full" type="search" placeholder="Player name, team or ID" value={search} onChange={(e) => { setSearch(e.target.value); setLimit(PAGE_SIZE) }} /></label>
      {settings?.notice && <p role="status" className="text-sm text-[#52605d]">{settings.notice}</p>}
      {position === 'FLEX' && <p className="text-sm text-[#52605d]">FLEX includes running backs, receivers and tight ends. Team labels are from the forecast.</p>}
    </section>

    {selected.length > 0 && <section className="product-card border-[#35634b]" aria-label="Comparison tray"><h2 className="text-lg font-semibold">Your comparison · {selected.length} of 8 players</h2><p className="mt-1 text-sm text-[#52605d]">Selections are nominations, not a verified roster. Values will be loaded from the exact verified evidence.</p><ul className="my-3 flex flex-wrap gap-2">{selected.map((id) => <li key={id}><button type="button" className="product-button secondary" onClick={() => toggle(id)}>Remove {data?.predictions.find((r) => r.player_id === id)?.name ?? id}</button></li>)}</ul>{selected.length < 2 && <p className="mb-3 text-sm">Add one more player to compare.</p>}{selected.length === 8 && <p className="mb-3 text-sm">Eight players selected. Remove one before adding another.</p>}{!readyDestination && <button type="button" className="product-button" disabled={selected.length < 2 || checking || !data} onClick={() => void prepareComparison()}>{checking ? 'Checking matching evidence…' : scoring === 'ppr' ? 'Prepare comparison' : 'Compare in PPR'}</button>}{readyDestination && <Link className="product-button inline-flex" href={readyDestination}>{readyDestination.startsWith('/decision-lab') ? 'Explore this published period' : 'Open PPR comparison'}</Link>}{scoring !== 'ppr' && <p className="mt-2 text-sm">Decisions use PPR. Your {scoring === 'half' ? 'half-PPR' : 'standard'} browsing preference is kept.</p>}{transferError && <p role="alert" className="mt-3 text-sm text-[#93361f]">{transferError}</p>}</section>}
    {predictions.isPending && period && <LoadingState label="Loading this published period…" />}
    {predictions.isError && <div role="alert" className="product-card"><h2 className="font-semibold">This published period could not be loaded</h2><p>No replacement week was selected. Retry or choose another available period.</p><button className="product-button secondary mt-3" onClick={() => void predictions.refetch()}>Retry projections</button></div>}
    {data && <>
      <div className="flex flex-wrap items-end justify-between gap-2"><p className="text-sm text-[#52605d]">Showing {shown.length} of {rows.length} {position === 'ALL' ? 'all-position' : position} projections{search ? ' matching your search' : ''} · {data.n} published overall</p><p className="text-sm text-[#52605d]">Updated {formatLocalDate(data.generated_at_utc)}</p></div>
      <p className="text-sm text-[#52605d]">{position === 'ALL' ? 'Cross-position projection ordering; not lineup advice.' : `Projection order within ${position}; search keeps each player’s original rank.`} Model range: actual scores can fall outside it.</p>
      {!coverage.showColumn && <div className="rounded-lg bg-[#ebece5] p-3 text-sm">{coverage.message} <details className="mt-2"><summary className="cursor-pointer font-semibold">Inspect missing-result reasons</summary><ul className="mt-2 list-disc pl-5">{coverage.reasons.map((r) => <li key={r}>{r.replaceAll('_', ' ')}</li>)}</ul></details></div>}
      {shown.length === 0 ? <EmptyState message="No players match these filters. Clear the search or choose another position group." /> : <>
        <div className="hidden overflow-x-auto rounded-xl border border-[#c5c7c0] bg-white md:block">
          <table className="w-full text-left text-sm"><thead className="bg-[#f3f2ec] text-[#374a43]"><tr><th className="px-4 py-3">Rank</th><th className="px-4 py-3">Player</th><th className="px-4 py-3 text-right">Projected points</th><th className="px-4 py-3">Model range</th>{coverage.showColumn && <th className="px-4 py-3 text-right">Actual ({scoring.toUpperCase()})</th>}<th className="px-4 py-3"><span className="sr-only">Compare players</span></th></tr></thead>
            <tbody>{shown.map(({ row: r, rank }) => <tr key={r.player_id} className="border-t border-[#e4e5dd]"><td className="px-4 py-4 tabular-nums text-[#52605d]">{rank}</td><td className="px-4 py-4"><Link className="font-semibold underline-offset-4 hover:underline" href={`/player/${encodeURIComponent(r.player_id)}`}>{r.name ?? r.player_id}</Link><p className="mt-1 text-sm text-[#52605d]">{r.position} · {r.team ?? 'Team unknown'}</p></td><td className="px-4 py-4 text-right text-lg font-semibold tabular-nums">{fmtNum(r.prediction, 1)}</td><td className="px-4 py-4"><span className="tabular-nums">{fmtNum(r.floor, 1)}–{fmtNum(r.ceiling, 1)}</span><RangeBar row={r} scale={scale} /></td>{coverage.showColumn && <td className="max-w-[190px] px-4 py-4 text-right tabular-nums">{actual(r)}</td>}<td className="px-4 py-4">{addButton(r)}</td></tr>)}</tbody>
          </table>
        </div>
        <div className="grid gap-3 md:hidden" aria-label="Projection cards">{shown.map(({ row: r, rank }) => <article key={r.player_id} className="product-card space-y-3"><div className="flex items-start justify-between gap-3"><div><p className="text-sm text-[#52605d]">#{rank} · {r.position} · {r.team ?? 'Team unknown'}</p><Link className="text-lg font-semibold" href={`/player/${encodeURIComponent(r.player_id)}`}>{r.name ?? r.player_id}</Link></div><p className="text-right text-xl font-semibold tabular-nums">{fmtNum(r.prediction, 1)}<span className="block text-sm font-normal">points</span></p></div><p className="text-sm">Model range <b className="tabular-nums">{fmtNum(r.floor, 1)}–{fmtNum(r.ceiling, 1)}</b></p><RangeBar row={r} scale={scale} />{coverage.showColumn && <div className="text-sm">Actual ({scoring.toUpperCase()}): {actual(r)}</div>}{addButton(r)}</article>)}</div>
      </>}
      {shown.length < rows.length && <button type="button" className="product-button secondary w-full" onClick={() => setLimit((n) => n + PAGE_SIZE)}>Show {Math.min(PAGE_SIZE, rows.length - shown.length)} more players</button>}
      {coverage.showColumn && <p className="text-sm text-[#52605d]">{coverage.message}</p>}
      <details className="product-details"><summary>Scoring, model range and source details</summary><div className="mt-3 space-y-3 text-sm"><p>{scoring === 'ppr' ? 'PPR-trained forecast.' : 'Derived forecast estimate: the PPR projection is adjusted using estimated receptions. Scoring reconciliation does not validate forecast accuracy.'} {data.scoring_derivation}</p><p>Actuals use realized evidence, never estimated receptions. {data.scoring_rules_description ?? 'The implemented scoring rules may differ from custom leagues.'}</p><p>Range method: {data.interval_method}. This is a residual-offset construction, not an individualized calibrated interval or safe minimum.</p><dl><dt>Statistics through</dt><dd>{formatPeriod(data.data_through.season, data.data_through.week)}</dd><dt>Exact forecast time</dt><dd>{data.generated_at_utc}</dd><dt>Model / features</dt><dd className="break-all font-mono">{data.model_version} / {data.feature_version}</dd><dt>Scoring rules</dt><dd>{data.scoring_rules_id ?? 'ffai-nflverse-v1'}</dd></dl></div></details>
    </>}

  </div>
}

function RangeBar({ row, scale }: { row: PredictionRecord; scale: { min: number; max: number } }) {
  const pct = (value: number) => Math.max(0, Math.min(100, 100 * (value - scale.min) / (scale.max - scale.min)))
  return <div className="relative mt-2 h-2 w-full min-w-[90px] rounded bg-[#e4e5dd]" aria-hidden="true"><div className="absolute h-full rounded bg-[#8caf9b]" style={{ left: `${pct(row.floor)}%`, width: `${Math.max(0, pct(row.ceiling) - pct(row.floor))}%` }} /><div className="absolute -top-1 h-4 w-0.5 bg-[#173e2d]" style={{ left: `${pct(row.prediction)}%` }} /></div>
}
