'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Link from 'next/link'
import { DEFAULT_BASE_URL, type FetchLike } from '@/lib/decision-lab/bundle'
import { contentId } from '@/lib/decision-lab/canonical'
import { SLOTS } from '@/lib/decision-lab/policy'
import { newReceipt, recordAction, declineAction, type AvailabilityOverride } from '@/lib/decision-lab/receipts'
import { loadReceipts, saveReceipt } from '@/lib/decision-lab/storage'
import type { ActionKind, Receipt } from '@/lib/decision-lab/types'
import { compareWeekly, loadLatestWeekly, type WeeklyEvidence } from '@/lib/start-sit/adapter'
import { fmtNum } from '@/lib/format'
import { formatLocalDate, formatPeriod, selectionHref } from '@/lib/presentation/display'
import { decisionSummary } from '@/lib/presentation/decision'
import { LoadingState } from '@/components/ui/States'
import { ActionPanel } from '@/components/decision-lab/ActionPanel'
import type { Readiness } from '@/lib/start-sit/readiness'
import ContextEvidence from '@/components/start-sit/ContextEvidence'

const DRAFT_KEY = 'ffai.start-sit/draft/2'
export interface StartSitEngineProps { fetchImpl?: FetchLike; baseUrl?: string; now?: () => string }

export function StartSitEngine({ fetchImpl, baseUrl = DEFAULT_BASE_URL, now = () => new Date().toISOString() }: StartSitEngineProps) {
  const [evidence, setEvidence] = useState<WeeklyEvidence | null>(null)
  const [checkingEvidence, setCheckingEvidence] = useState(true)
  const evidenceRequest = useRef(0)
  const acceptedSnapshotDigest = useRef<string | null>(null)
  const [slot, setSlot] = useState('FLEX')
  const [selected, setSelected] = useState<string[]>([])
  const [cardCount, setCardCount] = useState(2)
  const [overrides, setOverrides] = useState<Record<string, AvailabilityOverride>>({})
  const [search, setSearch] = useState('')
  const [activeOption, setActiveOption] = useState(0)
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [hydrated, setHydrated] = useState(false)
  const [requestProblem, setRequestProblem] = useState<string | null>(null)
  const [requestedSnapshot, setRequestedSnapshot] = useState<string | null>(null)
  const [computed, setComputed] = useState<{ receipt: Receipt; key: string } | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [contextState, setContextState] = useState<Readiness['state']>('unavailable')
  const [historicalExploration, setHistoricalExploration] = useState(false)
  const contextBlocked = contextState === 'blocked' || contextState === 'incompatible'
  const searchInput = useRef<HTMLInputElement>(null)
  const refresh = useCallback(async () => {
    const request = ++evidenceRequest.current
    setCheckingEvidence(true)
    try {
      const next = await loadLatestWeekly({ baseUrl, fetchImpl })
      if (request !== evidenceRequest.current) return
      if (next.ok) {
        const digest = contentId(next.snapshot)
        if (acceptedSnapshotDigest.current !== null && acceptedSnapshotDigest.current !== digest) {
          setOverrides({}); setHistoricalExploration(false)
          setMessage('The published evidence changed. Availability assumptions were cleared; compare again when ready.')
        }
        acceptedSnapshotDigest.current = digest
      }
      setEvidence(next)
    } catch (e) {
      if (request === evidenceRequest.current) setEvidence({ ok: false, kind: 'shape', evidence_status: 'corrupt', error: 'Weekly evidence could not be verified', detail: e instanceof Error ? e.message : String(e) })
    } finally { if (request === evidenceRequest.current) setCheckingEvidence(false) }
  }, [baseUrl, fetchImpl])
  useEffect(() => { void refresh(); return () => { evidenceRequest.current += 1 } }, [refresh])
  useEffect(() => {
    const visible = () => { if (document.visibilityState === 'visible') void refresh() }
    window.addEventListener('focus', refresh); document.addEventListener('visibilitychange', visible)
    return () => { window.removeEventListener('focus', refresh); document.removeEventListener('visibilitychange', visible) }
  }, [refresh])
  const snapshot = evidence?.ok ? evidence.snapshot : null
  useEffect(() => {
    if (!snapshot || hydrated) return
    const query = new URLSearchParams(window.location.search)
    const requested = query.get('snapshot')
    const ids = query.has('players') ? (query.get('players') ?? '').split(',').filter(Boolean) : null
    const querySlot = query.get('slot')
    if (requested || ids || querySlot) {
      setRequestedSnapshot(requested)
      if (querySlot && querySlot in SLOTS) setSlot(querySlot)
      if ((requested && requested !== snapshot.snapshot_id) || (ids && !requested)) setRequestProblem('This selection belongs to different or unidentified evidence. Open its matching snapshot in Decision Lab; no week was substituted.')
      else if ((querySlot && !(querySlot in SLOTS)) || (ids && (ids.length > 8 || new Set(ids).size !== ids.length))) setRequestProblem('This comparison link has an invalid slot or choice set. No selection was substituted.')
      else { if (querySlot) setSlot(querySlot); if (ids) { setSelected(ids); setCardCount(Math.max(2, ids.length)) } }
    } else {
      try {
        const saved = JSON.parse(sessionStorage.getItem(DRAFT_KEY) ?? 'null')
        if (saved && saved.digest === contentId(snapshot) && saved.snapshot_id === snapshot.snapshot_id && typeof saved.slot === 'string' && saved.slot in SLOTS && Array.isArray(saved.selected) && saved.selected.length <= 8 && new Set(saved.selected).size === saved.selected.length && saved.selected.every((id: unknown) => typeof id === 'string' && snapshot.rows.some((r) => r.player_id === id))) {
          setSlot(saved.slot); setSelected(saved.selected); setCardCount(Math.max(2, saved.selected.length))
        } else if (saved || sessionStorage.getItem('ffai.start-sit/draft/1')) setMessage('An older comparison preference was cleared. Choose players for this published week.')
      } catch { /* Optional view state never prevents comparison. */ }
    }
    setHydrated(true)
  }, [snapshot, hydrated])
  useEffect(() => {
    if (hydrated && snapshot && !requestProblem) try { sessionStorage.setItem(DRAFT_KEY, JSON.stringify({ slot, selected, snapshot_id: snapshot.snapshot_id, digest: contentId(snapshot) })) } catch { /* optional */ }
  }, [hydrated, snapshot, slot, selected, requestProblem])
  useEffect(() => {
    if (hydrated && requestedSnapshot && snapshot && requestedSnapshot !== snapshot.snapshot_id) {
      setRequestProblem('The requested published period is no longer the latest snapshot. Open its exact evidence in Decision Lab; no week was substituted.')
    }
  }, [hydrated, requestedSnapshot, snapshot])
  const byId = useMemo(() => new Map((snapshot?.rows ?? []).map((r) => [r.player_id, r])), [snapshot])
  const missing = selected.filter((id) => !byId.has(id))
  const matches = useMemo(() => {
    const q = search.trim().toLowerCase()
    return (snapshot?.rows ?? []).filter((r) => !selected.includes(r.player_id) && (!q || `${r.display_name ?? ''} ${r.player_id} ${r.team ?? ''}`.toLowerCase().includes(q)))
      .sort((a, b) => Number(SLOTS[slot]?.includes(b.position)) - Number(SLOTS[slot]?.includes(a.position)) || (a.display_name ?? a.player_id).localeCompare(b.display_name ?? b.player_id)).slice(0, 10)
  }, [snapshot, search, slot, selected])
  const key = JSON.stringify([snapshot ? contentId(snapshot) : null, slot, selected, overrides])
  const current = computed?.key === key ? computed.receipt : null
  const stale = computed !== null && current === null
  useEffect(() => { setHistoricalExploration(false) }, [key])
  const changeSelection = (ids: string[]) => { setSelected(ids); setOverrides({}); setHistoricalExploration(false); setActionError(null) }
  const addPlayer = (id: string) => { if (selected.length >= 8 || selected.includes(id)) return; changeSelection([...selected, id]); setCardCount(Math.max(cardCount, selected.length + 1)); setSearch(''); setActiveOption(0); setShowSuggestions(false) }
  const compute = (assumeUnknown = false) => {
    if (!snapshot || checkingEvidence || requestProblem) return
    if (assumeUnknown && contextBlocked) { setMessage('Current context blocks this action. Use the explicit historical exploration path.'); return }
    const nextOverrides = { ...overrides }
    if (assumeUnknown) for (const id of selected) {
      const row = byId.get(id)
      if (row && SLOTS[slot]?.includes(row.position) && (!nextOverrides[id] || nextOverrides[id].availability === 'unknown')) nextOverrides[id] = { availability: 'assumed_available', availability_basis: 'user_assumed' }
    }
    try {
      const { inputs } = compareWeekly(snapshot, { slot, playerIds: selected, overrides: nextOverrides })
      const receipt = loadReceipts().receipts[contentId(inputs)] ?? newReceipt(inputs, { createdAtUtc: now() })
      const computedKey = JSON.stringify([contentId(snapshot), slot, selected, nextOverrides])
      setOverrides(nextOverrides); setComputed({ receipt, key: computedKey }); setMessage(null); setActionError(null)
    } catch (e) { setMessage(e instanceof Error ? e.message : String(e)) }
  }
  const persist = (receipt: Receipt) => {
    const saved = saveReceipt(receipt)
    if (saved.ok) { setComputed({ receipt: saved.receipt, key }); setActionError(null) } else setActionError(saved.message)
  }
  const record = (id: string, kind: ActionKind, note: string | null) => {
    if (!current || checkingEvidence) return
    if (contextBlocked && (!historicalExploration || kind !== 'hypothetical_replay')) { setActionError('Only explicit historical hypothetical actions can be recorded under this restriction.'); return }
    try { persist(recordAction(current, id, kind, now(), note)) } catch (e) { setActionError(String(e)) }
  }
  const decline = (note: string | null) => { if (!current || checkingEvidence) return; try { persist(declineAction(current, now(), note)) } catch (e) { setActionError(String(e)) } }
  if (!evidence) return <LoadingState label="Loading the published week…" />
  if (!evidence.ok || !snapshot) return <div role="alert" className="product-card"><h2>Recommendation unavailable</h2><p>{!evidence.ok ? evidence.error : 'Published evidence is missing.'}</p><p>No other week, API forecast, or synthetic data was substituted.</p><details className="product-details mt-4"><summary>Evidence details</summary>{!evidence.ok && <p>{evidence.detail}</p>}</details><button className="product-button mt-4" onClick={() => void refresh()}>Retry evidence load</button> <Link href="/decision-lab" className="underline">Explore historical examples</Link></div>
  if (requestProblem) return <section role="alert" className="product-card"><h2>This comparison link needs different evidence</h2><p className="my-3">{requestProblem}</p>{requestedSnapshot && <Link className="product-button mr-3" href={selectionHref('/decision-lab', requestedSnapshot, new URLSearchParams(window.location.search).get('players')?.split(',').filter(Boolean) ?? [], slot)}>Open matching evidence in Decision Lab</Link>}<a className="product-button secondary" href="/start-sit">Use latest published week</a></section>
  const names = Object.fromEntries(snapshot.rows.map((r) => [r.player_id, r.display_name ?? r.player_id]))
  const summary = current ? decisionSummary(current.result, names) : null
  const canCompare = selected.length >= 2 && !missing.length && !checkingEvidence
  return <div className="space-y-5">
    <div className="product-context-strip"><strong>Latest published: {formatPeriod(snapshot.season, snapshot.week)}</strong><span>PPR</span><span>Updated {formatLocalDate(snapshot.generated_at_utc)}</span><details><summary className="cursor-pointer underline">Details</summary><p className="mt-2">Forecast and statistics periods differ: stats through {snapshot.data_cutoff ? formatPeriod(snapshot.data_cutoff.season, snapshot.data_cutoff.week) : 'unknown'}. Published evidence is dated, not game-day verified.</p></details></div>
    {message && <p role="status" className="product-notice">{message}</p>}
    <div className="comparison-layout">
      <section className="product-card space-y-5" aria-label="Choose players">
        <div className="flex flex-wrap items-center justify-between gap-3"><h2>Choose your options</h2><label className="flex items-center gap-3 font-semibold" htmlFor="start-sit-slot">Starting slot <select id="start-sit-slot" className="product-field" value={slot} onChange={(e) => { setSlot(e.target.value); setOverrides({}) }}>{Object.keys(SLOTS).map((s) => <option key={s}>{s}</option>)}</select></label></div>
        <div><label className="mb-2 block font-semibold" htmlFor="player-search">Search players</label><input ref={searchInput} id="player-search" role="combobox" aria-label="Search players to compare" aria-autocomplete="list" aria-expanded={showSuggestions} aria-controls="player-options" aria-activedescendant={showSuggestions && matches[activeOption] ? `player-option-${activeOption}` : undefined} className="product-field w-full" placeholder="Player name or forecast team" value={search} disabled={selected.length >= 8} onFocus={() => setShowSuggestions(true)} onChange={(e) => { setSearch(e.target.value); setActiveOption(0); setShowSuggestions(true) }} onKeyDown={(e) => { if (e.key === 'Escape') setShowSuggestions(false); if (e.key === 'ArrowDown') { e.preventDefault(); setShowSuggestions(true); setActiveOption((n) => Math.min(n + 1, matches.length - 1)) } if (e.key === 'ArrowUp') { e.preventDefault(); setActiveOption((n) => Math.max(0, n - 1)) } if (e.key === 'Enter' && showSuggestions && matches[activeOption]) { e.preventDefault(); addPlayer(matches[activeOption].player_id) } }} />
          {showSuggestions && <div id="player-options" role="listbox" aria-label="Matching published projections" className="player-search-results mt-2">{matches.length ? matches.map((r, i) => <div role="option" aria-selected={i === activeOption} id={`player-option-${i}`} key={r.player_id}><button type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => addPlayer(r.player_id)}><span>Add {r.display_name ?? r.player_id} · {r.position} · {r.team ?? '—'}</span><span className="product-muted">{SLOTS[slot]?.includes(r.position) ? 'Fits slot' : 'Outside slot'}</span></button></div>) : <p className="p-3">No published players match. Try another name or team.</p>}</div>}
        </div>
        <div className="player-slots" aria-label="Nominated alternatives">{Array.from({ length: Math.max(cardCount, selected.length) }, (_, i) => {
          const id = selected[i], r = id ? byId.get(id) : null
          if (!id) return <button key={`empty-${i}`} className="player-choice empty" onClick={() => searchInput.current?.focus()}><span className="text-2xl" aria-hidden="true">＋</span><strong>Choose player {i + 1}</strong><span className="text-sm">Search above to add an option</span></button>
          return <article className="player-choice" key={id}><div className="flex items-start justify-between gap-2"><h3>{r?.display_name ?? id}</h3><button className="min-h-[44px] px-1 text-sm underline" aria-label={`Remove ${r?.display_name ?? id}`} onClick={() => changeSelection(selected.filter((pid) => pid !== id))}>Remove</button></div>{r ? <><p className="product-muted">{r.position} · {r.team ?? 'Team unknown'} <span>(forecast team)</span></p><p className="points">{fmtNum(r.projection, 1)} <span className="text-sm font-normal">PPR points</span></p><p className="mt-2 text-sm">Model range {fmtNum(r.floor, 1)}–{fmtNum(r.ceiling, 1)}</p>{!SLOTS[slot]?.includes(r.position) && <p className="product-notice product-blocker mt-2">Outside this slot. This player stays selected and is excluded by the policy.</p>}{overrides[id]?.availability === 'unavailable' && <p className="product-notice mt-2">Excluded by you</p>}</> : <p className="product-notice product-blocker mt-2">Missing projection. No smaller choice set will be substituted.</p>}</article>
        })}</div>
        {cardCount < 8 && <button className="product-button secondary" onClick={() => { setCardCount((n) => n + 1); searchInput.current?.focus() }}>Add another player</button>}
        <p className="product-muted">Choose two to eight options. FLEX accepts RB, WR and TE. Actual scores can fall outside the model range.</p>
        <div className="border-t border-[#cbd1c9] pt-4"><button className="product-button w-full" disabled={!canCompare || contextBlocked} onClick={() => compute(true)}>Compare assuming these players can play</button><p className="product-muted mt-2">Live availability is not verified. This comparison assumes the selected players are eligible for your lineup.</p>{selected.length < 2 && <p className="product-muted mt-1">Choose at least two players to compare.</p>}{missing.length > 0 && <p role="alert">A selected projection is missing. Change the selection explicitly to continue.</p>}{checkingEvidence && <p className="product-muted">Checking published evidence; actions are paused.</p>}</div>
        <details className="product-details"><summary>Adjust player assumptions</summary><p className="product-muted mb-3">Selection or evidence changes clear confirmations. An exclusion is never changed by the bulk comparison.</p>{selected.map((id) => <div className="mb-3" key={id}><label htmlFor={`availability-${id}`} className="block mb-1">Availability assumption for {names[id] ?? id}</label><select id={`availability-${id}`} className="product-field w-full" value={overrides[id]?.availability ?? 'unknown'} onChange={(e) => setOverrides((prev) => ({ ...prev, [id]: e.target.value === 'assumed_available' ? { availability: 'assumed_available', availability_basis: 'user_assumed' } : e.target.value === 'unavailable' ? { availability: 'unavailable', availability_basis: 'user_excluded' } : { availability: 'unknown', availability_basis: 'not_verified' } }))}><option value="unknown">Unknown — check player status</option><option value="assumed_available">I assume available for this dated comparison</option><option value="unavailable">I exclude this player</option></select></div>)}<button className="product-button secondary" disabled={!canCompare} onClick={() => compute()}>Compare nominated alternatives</button></details>
      </section>
      <aside className="space-y-4">
        <ContextEvidence snapshot={snapshot} playerIds={selected} onStateChange={setContextState} />
        {current && summary ? <section className="product-card" data-testid="quick-result" aria-live="polite"><p className="product-muted mb-2">{contextBlocked ? 'Current lineup change blocked' : 'Your comparison'}</p><h2 className="result-leading">{contextBlocked ? 'This comparison cannot support a current lineup change' : summary.heading}</h2>{!contextBlocked && current.result.status === 'recommend' && <p className="mt-3 font-semibold text-[#174b3d]">Assuming the selected players are available</p>}{contextBlocked ? <p className="mt-3">Check the restriction above. You can inspect the original policy result as an explicit historical hypothetical.</p> : <ul className="support-list">{summary.support.map((text) => <li key={text}>{text}</li>)}</ul>}<details className="product-details mt-5"><summary>Why this result?</summary>{contextBlocked && <p>Original dated policy result: {summary.heading}</p>}<p className="product-muted my-2">The comparison margin is a policy threshold, not statistical significance or a probability. The historical baseline uses only earlier realized points.</p><ul className="list-disc pl-5 space-y-2">{current.result.explanation.map((text, i) => <li key={i}>{text}</li>)}</ul><ul className="mt-4">{current.result.ranking.map((r) => <li key={r.player_id}>{names[r.player_id] ?? r.player_id}: {r.exclusion ? 'Excluded from this slot or by your assumption' : r.comparable ? 'Comparable' : 'Unresolved'}</li>)}</ul></details><Link className="product-button secondary mt-4" href={selectionHref('/decision-lab', snapshot.snapshot_id, selected, slot)}>Explore this comparison in Decision Lab</Link></section> : <section className="product-card"><h2>Your comparison will appear here</h2><p className="mt-3 text-[#52645b]">Choose two players, select a slot, then compare under an explicit availability assumption.</p>{stale && <p role="status" className="product-notice mt-4">Comparison changed. Recompute before using the result; saved choices remain unchanged.</p>}</section>}
        {contextBlocked && <section className="product-card"><h2>Explore without changing a lineup</h2><p className="product-muted my-3">Historical hypothetical exploration preserves the restriction and does not record a real action.</p><button className="product-button secondary" disabled={!canCompare} onClick={() => { compute(); setHistoricalExploration(true) }}>Explore as historical hypothetical</button></section>}
        {current && (!contextBlocked || historicalExploration) && <details className="product-details"><summary>Record my choice or save a hypothetical comparison</summary><p className="product-muted">Computing a suggestion records nothing. Choose a player and action kind yourself.</p><fieldset disabled={checkingEvidence}><ActionPanel key={`${current.decision_id}/${contextBlocked}`} receipt={current} onRecord={record} onDecline={decline} error={actionError} hypotheticalOnly={contextBlocked} /></fieldset></details>}
      </aside>
    </div>
    <details className="product-details"><summary>Scoring rules, model range and source evidence</summary><p>PPR: 1 per reception; 1/25 passing yards; 4 per passing TD; −2 per interception; 1/10 rushing or receiving yards; 6 per rushing/receiving/special-teams TD; 2 per two-point conversion; −2 per lost fumble. No bonuses. Custom PPR leagues may differ.</p><p>Model range uses position-level validation-residual offsets around the forecast; these are not individualized calibrated intervals, guaranteed bounds, or a safe minimum.</p><p>Nomination means user-nominated ownership, not a verified league roster. {snapshot.population.description}</p><p>Baseline reconciliation: {snapshot.baseline.reconciled === true ? 'matched' : 'not established for this snapshot'}. {snapshot.baseline.note}</p><p className="break-all">Snapshot {snapshot.snapshot_id} · {contentId(snapshot)}</p><p>Exact forecast time: {snapshot.generated_at_utc}. Model {snapshot.model_version} · feature {snapshot.feature_version}.</p><Link href="/decision-lab?view=saved" className="underline">Find saved decisions in Decision Lab</Link></details>
  </div>
}
