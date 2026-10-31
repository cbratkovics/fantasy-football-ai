'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Link from 'next/link'
import { DEFAULT_BASE_URL, type FetchLike } from '@/lib/decision-lab/bundle'
import { contentId } from '@/lib/decision-lab/canonical'
import { SLOTS } from '@/lib/decision-lab/policy'
import { newReceipt, recordAction, declineAction, type AvailabilityOverride } from '@/lib/decision-lab/receipts'
import { loadReceipts, saveReceipt } from '@/lib/decision-lab/storage'
import type { ActionKind, Receipt } from '@/lib/decision-lab/types'
import { compareWeekly, comparisonHeading, loadLatestWeekly, type WeeklyEvidence } from '@/lib/start-sit/adapter'
import { fmtNum, fmtUtc } from '@/lib/format'
import { TextInput } from '@/components/ui/Controls'
import { LoadingState } from '@/components/ui/States'
import { ActionPanel } from '@/components/decision-lab/ActionPanel'
import type { Readiness } from '@/lib/start-sit/readiness'
import ContextEvidence from '@/components/start-sit/ContextEvidence'

const DRAFT_KEY = 'ffai.start-sit/draft/1'
const FOCUS = 'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-moss'

export interface StartSitEngineProps {
  fetchImpl?: FetchLike
  baseUrl?: string
  now?: () => string
}

/** A convenient dated comparison using the Lab's verified evidence, input builder and policy. */
export function StartSitEngine({ fetchImpl, baseUrl = DEFAULT_BASE_URL, now = () => new Date().toISOString() }: StartSitEngineProps) {
  const [evidence, setEvidence] = useState<WeeklyEvidence | null>(null)
  const [checkingEvidence, setCheckingEvidence] = useState(true)
  const evidenceRequest = useRef(0)
  const acceptedSnapshotDigest = useRef<string | null>(null)
  const [slot, setSlot] = useState('')
  const [selected, setSelected] = useState<string[]>([])
  const [overrides, setOverrides] = useState<Record<string, AvailabilityOverride>>({})
  const [search, setSearch] = useState('')
  const [hydrated, setHydrated] = useState(false)
  const [computed, setComputed] = useState<{ receipt: Receipt; key: string } | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [contextState, setContextState] = useState<Readiness['state']>('unavailable')
  const [historicalExploration, setHistoricalExploration] = useState(false)
  const contextBlocked = contextState === 'blocked' || contextState === 'incompatible'

  const refresh = useCallback(async () => {
    const request = ++evidenceRequest.current
    setCheckingEvidence(true)
    try {
      const next = await loadLatestWeekly({ baseUrl, fetchImpl })
      if (request === evidenceRequest.current) {
        if (next.ok) {
          const digest = contentId(next.snapshot)
          if (acceptedSnapshotDigest.current !== null && acceptedSnapshotDigest.current !== digest) {
            // Assumptions apply to the evidence the user reviewed, not every future release.
            setOverrides({})
            setHistoricalExploration(false)
            setMessage('Availability assumptions were cleared because the published evidence changed. Confirm them for this dated comparison.')
          }
          acceptedSnapshotDigest.current = digest
        }
        setEvidence(next)
      }
    } catch (e) {
      if (request === evidenceRequest.current) setEvidence({
        ok: false, kind: 'shape', evidence_status: 'corrupt',
        error: 'Weekly evidence could not be verified',
        detail: e instanceof Error ? e.message : String(e),
      })
    } finally {
      if (request === evidenceRequest.current) setCheckingEvidence(false)
    }
  }, [baseUrl, fetchImpl])
  useEffect(() => {
    void refresh()
    return () => { evidenceRequest.current += 1 }
  }, [refresh])
  // Re-read the manifest after returning to the tab. A changed digest invalidates the computation.
  useEffect(() => {
    const visible = () => { if (document.visibilityState === 'visible') void refresh() }
    window.addEventListener('focus', refresh)
    document.addEventListener('visibilitychange', visible)
    return () => { window.removeEventListener('focus', refresh); document.removeEventListener('visibilitychange', visible) }
  }, [refresh])
  useEffect(() => {
    try {
      const d = JSON.parse(sessionStorage.getItem(DRAFT_KEY) ?? 'null') as unknown
      if (d && typeof d === 'object') {
        const saved = d as { slot?: unknown; selected?: unknown }
        if (typeof saved.slot === 'string' && saved.slot in SLOTS) setSlot(saved.slot)
        if (Array.isArray(saved.selected) && saved.selected.every((p) => typeof p === 'string')) setSelected(saved.selected)
      }
    } catch { /* unavailable storage does not prevent comparison */ }
    setHydrated(true)
  }, [])
  useEffect(() => {
    if (hydrated) try { sessionStorage.setItem(DRAFT_KEY, JSON.stringify({ slot, selected })) } catch { /* optional */ }
  }, [hydrated, slot, selected])

  const snapshot = evidence?.ok ? evidence.snapshot : null
  const byId = useMemo(() => new Map((snapshot?.rows ?? []).map((r) => [r.player_id, r])), [snapshot])
  const missing = selected.filter((id) => !byId.has(id))
  const matches = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return []
    return (snapshot?.rows ?? []).filter((r) => !selected.includes(r.player_id) && `${r.display_name ?? ''} ${r.player_id} ${r.team ?? ''}`.toLowerCase().includes(q)).slice(0, 12)
  }, [snapshot, search, selected])
  const key = JSON.stringify([snapshot ? contentId(snapshot) : null, slot, selected, overrides])
  const current = !checkingEvidence && computed?.key === key ? computed.receipt : null
  const stale = computed !== null && !checkingEvidence && current === null
  useEffect(() => { setHistoricalExploration(false) }, [key])

  const compute = () => {
    if (!snapshot || checkingEvidence) return
    try {
      const { inputs } = compareWeekly(snapshot, { slot, playerIds: selected, overrides })
      const receipt = loadReceipts().receipts[contentId(inputs)] ?? newReceipt(inputs, { createdAtUtc: now() })
      setComputed({ receipt, key })
      setMessage(null)
      setActionError(null)
    } catch (e) { setMessage(e instanceof Error ? e.message : String(e)) }
  }
  const persist = (receipt: Receipt) => {
    const saved = saveReceipt(receipt)
    if (saved.ok) { setComputed({ receipt: saved.receipt, key }); setActionError(null) }
    else setActionError(saved.message)
  }
  const record = (id: string, kind: ActionKind, note: string | null) => {
    if (!current) return
    if (contextBlocked && (!historicalExploration || kind !== 'hypothetical_replay')) {
      setActionError('Current actionability is blocked. Only explicitly chosen historical hypothetical actions can be recorded.')
      return
    }
    try { persist(recordAction(current, id, kind, now(), note)) } catch (e) { setActionError(String(e)) }
  }
  const decline = (note: string | null) => {
    if (!current) return
    try { persist(declineAction(current, now(), note)) } catch (e) { setActionError(String(e)) }
  }

  if (evidence === null) return <LoadingState label="Verifying the published weekly evidence…" />
  if (!evidence.ok || snapshot === null) return (
    <div role="alert" className="border border-[#c5c7c0] bg-white p-5">
      <h2 className="text-lg font-bold">Recommendation unavailable</h2>
      <p className="mt-2 text-sm">{!evidence.ok ? `${evidence.error}: ${evidence.detail}` : 'Weekly evidence is missing.'}</p>
      <p className="mt-2 text-xs">No other week, API forecast, or synthetic data was substituted. <Link href="/decision-lab" className="underline">Open explicit historical or synthetic exploration in Decision Lab.</Link></p>
      <button type="button" className={`mt-3 border p-2 ${FOCUS}`} onClick={() => void refresh()}>Retry evidence load</button>
    </div>
  )

  return (
    <div className="space-y-6">
      <section className="border border-[#c5c7c0] bg-[#f8f7f2] p-5">
        <h2 className="font-semibold">Dated comparison · season {snapshot.season}, week {snapshot.week}</h2>
        <p className="mt-2 text-sm">Latest published evidence is not automatically current. This is a conditional one-slot comparison; roster, injuries, league ownership and lineup legality are unverified.</p>
        <p className="mt-2 text-xs">PPR decisions only: 1 point/reception; 1/25 passing yards; 4/pass TD; −2/interception; 1/10 rushing or receiving yards; 6/rushing, receiving or special-teams TD; 2/two-point conversion; −2/fumble lost. No yardage bonuses. Custom PPR compatibility is unverified. <Link className="underline" href="/predictions">Browse derived half-PPR and standard estimates.</Link></p>
        <dl className="mt-4 grid gap-2 text-xs sm:grid-cols-2">
          <div><dt className="font-bold">Forecast generated</dt><dd>{fmtUtc(snapshot.generated_at_utc)}</dd></div>
          <div><dt className="font-bold">Stats cutoff</dt><dd>{snapshot.data_cutoff ? `Season ${snapshot.data_cutoff.season}, week ${snapshot.data_cutoff.week}` : 'Unknown'}</dd></div>
          <div><dt className="font-bold">Published projections</dt><dd>{snapshot.rows.length}; not a verified target-week roster</dd></div>
          <div><dt className="font-bold">Baseline reconciliation</dt><dd>{snapshot.baseline.reconciled === true ? 'Matched published evaluation' : 'Not reconciled for this snapshot; per-row provenance retained'}</dd></div>
        </dl>
        <details className="mt-3 break-all text-xs"><summary className={`cursor-pointer ${FOCUS}`}>Evidence identity and limitations</summary><p className="mt-2">{snapshot.snapshot_id} · {contentId(snapshot)}</p><p>{snapshot.population.description}</p><p>{snapshot.baseline.note}</p><p>Residual-offset floor/ceiling ranges are neither guaranteed bounds nor probabilities. No percentage confidence is inferred from the projection gap.</p></details>
      </section>

      <ContextEvidence snapshot={snapshot} playerIds={selected} onStateChange={setContextState} />

      <section className="space-y-4 border border-[#c5c7c0] bg-white p-5">
        <label className="block text-sm font-bold" htmlFor="start-sit-slot">Starting slot</label>
        <select id="start-sit-slot" required value={slot} onChange={(e) => setSlot(e.target.value)} className={`w-full border border-[#bfc3bb] p-2 sm:w-48 ${FOCUS}`}>
          <option value="">Choose a slot</option>
          {Object.keys(SLOTS).map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <p className="text-xs">Nominate two to eight distinct alternatives. Ordinary FLEX allows RB, WR and TE. Nomination means user-nominated ownership only, with no imported league roster.</p>
        <TextInput aria-label="Search players to compare" placeholder="Search published names or player IDs" value={search} onChange={(e) => setSearch(e.target.value)} disabled={selected.length >= 8} />
        {matches.length > 0 && <ul aria-label="Matching published projections" className="divide-y border border-[#bfc3bb]">{matches.map((r) => <li key={r.player_id}><button className={`w-full px-3 py-2 text-left text-sm ${FOCUS}`} type="button" onClick={() => { setSelected((ids) => [...ids, r.player_id]); setSearch('') }}>Add {r.display_name ?? r.player_id} · {r.position} · {r.team ?? '—'}</button></li>)}</ul>}
        <ul className="divide-y" aria-label="Nominated alternatives">
          {selected.map((id) => {
            const r = byId.get(id)
            return <li key={id} className="space-y-2 py-3 text-sm">
              <div className="flex items-start justify-between gap-2"><b>{r?.display_name ?? id} · {r?.position ?? 'projection missing'}</b><button type="button" className={`text-xs underline ${FOCUS}`} onClick={() => setSelected((ids) => ids.filter((pid) => pid !== id))}>Remove {r?.display_name ?? id}</button></div>
              {r && <><p className="text-xs">Projection {fmtNum(r.projection, 1)} · residual range {fmtNum(r.floor, 1)}–{fmtNum(r.ceiling, 1)} · baseline {fmtNum(r.baseline, 1)} ({r.baseline_provenance})</p>
                <label className="block text-xs" htmlFor={`availability-${id}`}>Availability assumption for {r.display_name ?? id}</label><select id={`availability-${id}`} className={`w-full border p-2 text-xs ${FOCUS}`} value={overrides[id]?.availability ?? 'unknown'} onChange={(e) => setOverrides((prev) => ({ ...prev, [id]: e.target.value === 'assumed_available' ? { availability: 'assumed_available', availability_basis: 'user_assumed' } : e.target.value === 'unavailable' ? { availability: 'unavailable', availability_basis: 'user_excluded' } : { availability: 'unknown', availability_basis: 'not_verified' } }))}><option value="unknown">Unknown — check player status</option><option value="assumed_available">I assume available for this dated comparison</option><option value="unavailable">I exclude this player</option></select></>}
              {!r && <p className="text-red-700">Missing from this snapshot. Comparison remains incomplete until you explicitly remove this alternative or compatible evidence is supplied.</p>}
            </li>
          })}
        </ul>
        {missing.length > 0 && <p role="alert" className="text-sm">Recommendation unavailable: selected projections are missing ({missing.join(', ')}). No reduced choice set will be ranked.</p>}
        <button type="button" className={`border border-ink bg-ink px-4 py-2 text-sm text-white disabled:opacity-50 ${FOCUS}`} disabled={checkingEvidence || !slot || selected.length < 2 || missing.length > 0} onClick={compute}>Compare nominated alternatives</button>
        {selected.length === 8 && <p className="text-xs">Eight alternatives maximum. Remove one before adding another.</p>}
        {checkingEvidence && <p role="status">Rechecking weekly evidence. The previous result is suspended until verification completes.</p>}
        {stale && <p role="status" className="text-sm">Comparison changed. Recompute before using the result; saved choices remain unchanged.</p>}
        {message && <p role="alert">{message}</p>}
      </section>

      {current && <>
        <section aria-live="polite" data-testid="quick-result" className="border border-[#c5c7c0] bg-[#f8f7f2] p-5">
          <h2 className="text-xl font-bold">{contextBlocked ? 'Current recommendation unavailable' : comparisonHeading(current.result)}</h2>
          {contextBlocked && <p className="mt-2 text-sm">Known or incompatible context blocks current advice. The original policy result below is a dated conditional comparison, not an instruction to change a lineup.</p>}
          {contextBlocked && <p className="mt-2 text-sm font-bold">Original dated policy result: {comparisonHeading(current.result)}</p>}
          {current.result.recommended_player_id && <p className="mt-2 text-lg">{byId.get(current.result.recommended_player_id)?.display_name ?? current.result.recommended_player_id}</p>}
          <p className="mt-2 text-sm">Conditional dated comparison only. Gap: {fmtNum(current.result.leading.gap, 2)} points; default minimum: {fmtNum(current.result.gates.min_projection_gap, 2)}. Availability assumptions are your assertions, not provider verification.</p>
          <p className="mt-2 text-xs">Baseline agreement: {current.result.baseline.agrees_with_model_leader === true ? 'agrees' : current.result.baseline.agrees_with_model_leader === false ? 'disagrees' : current.result.baseline.status}.</p>
          <ul className="mt-3 list-disc space-y-1 pl-5 text-sm">{current.result.explanation.map((r, i) => <li key={i}>{r}</li>)}</ul>
          <ul className="mt-3 text-xs">{current.result.ranking.map((r) => <li className="py-1" key={r.player_id}>{r.display_name ?? r.player_id}: {r.exclusion ?? (r.comparable ? 'comparable' : 'unresolved')} · {fmtNum(r.projection, 1)} points</li>)}</ul>
          <p className="mt-3 text-xs">No action is recorded by computing a recommendation. <Link href="/decision-lab" className="underline">Open Decision Lab for receipt import/export and replay.</Link></p>
        </section>
        {contextBlocked && !historicalExploration && <button type="button" className={`border border-ink p-3 text-sm ${FOCUS}`} onClick={() => setHistoricalExploration(true)}>Explore as historical hypothetical</button>}
        {(!contextBlocked || historicalExploration) && <>
          {contextBlocked && <p className="text-sm">Historical hypothetical exploration only. The original v1 receipt remains a conditional comparison; current-context facts are not persisted into its identity or events.</p>}
          <ActionPanel key={`${current.decision_id}/${contextBlocked}`} receipt={current} onRecord={record} onDecline={decline} error={actionError} hypotheticalOnly={contextBlocked} />
        </>}
      </>}
    </div>
  )
}
