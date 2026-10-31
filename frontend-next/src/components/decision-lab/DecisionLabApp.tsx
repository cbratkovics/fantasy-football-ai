'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { DECISION_LAB_COPY } from '@/content/decision-lab'
import {
  DEFAULT_BASE_URL,
  loadCases,
  loadInputsSnapshot,
  loadManifest,
  loadOutcomeSnapshot,
  policySpecDigestMatches,
  safeJoin,
  snapshotEntry,
  type BundleFailure,
  type FetchLike,
  type InputsLoad,
  type PolicySpecCheck,
} from '@/lib/decision-lab/bundle'
import { experimentInputs } from '@/lib/decision-lab/cases'
import { decisionId } from '@/lib/decision-lab/policy'
import { contentId } from '@/lib/decision-lab/canonical'
import { formatLocalDate, formatPeriod } from '@/lib/presentation/display'
import { attachOutcome, buildDecisionInputs, declineAction, newReceipt, recordAction } from '@/lib/decision-lab/receipts'
import SPEC_DIGEST from '@/lib/decision-lab/spec_digest.json'
import { clearAll, deleteReceipt, exportAll, importReceipts, loadReceipts, saveReceipt, type ImportReport, type LoadedReceipts } from '@/lib/decision-lab/storage'
import type { ActionKind, CasesFile, DecisionInputs, Experiment, LabCase, LabManifest, Mode, Receipt } from '@/lib/decision-lab/types'
import { ActionPanel } from './ActionPanel'
import { AlternativesTable } from './AlternativesTable'
import { AssumptionControls, type DiffView } from './AssumptionControls'
import { Alert, Button, Card, FOCUS, Help, MicroLabel, Notice } from './Bits'
import { CaseLibrary } from './CaseLibrary'
import { DecisionCard } from './DecisionCard'
import { changedControls, diffResults, draftFromCase, draftFromExperiment, draftFromInputs, draftKey, emptyDraft, defaultParamDraft, parseParams, SLOT_OPTIONS, type Draft } from './draft'
import { EvidencePanel } from './EvidencePanel'
import { Nomination } from './Nomination'
import { OutcomePanel, type OutcomeUiState } from './OutcomePanel'
import { SavedDecisionsPanel, type PanelMessage } from './SavedDecisionsPanel'

const COPY = DECISION_LAB_COPY
const MODES: Mode[] = ['published_weekly', 'historical_replay', 'synthetic']
const MODE_NAMES: Record<Mode, string> = { published_weekly: 'Published week', historical_replay: 'Historical replay', synthetic: 'Synthetic examples' }
const PREFERENCE_KEY = 'ffai.lab.preferences/2'
const EMPTY_PLAYERS: string[] = []

type BundleState = { status: 'loading' } | { status: 'error'; failure: BundleFailure } | { status: 'ready'; manifest: LabManifest; cases: CasesFile; spec: PolicySpecCheck }

interface Computed {
  receipt: Receipt
  saved: boolean
  /** Signature of the draft the receipt was computed from; a different current key means stale. */
  key: string
  diff: DiffView | null
}

export interface DecisionLabAppProps {
  initialCaseId?: string | null
  initialSnapshotId?: string | null
  initialPlayerIds?: string[]
  initialSlot?: string | null
  /** Test seam: a same-origin `fetch` stand-in (see `createMemoryFetch`). */
  fetchImpl?: FetchLike
  baseUrl?: string
  /** Test seam: clock for receipt timestamps. */
  now?: () => string
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

/** Container and state machine for the Decision Lab route. */
export function DecisionLabApp({ initialCaseId = null, initialSnapshotId = null, initialPlayerIds = EMPTY_PLAYERS, initialSlot = null, fetchImpl, baseUrl = DEFAULT_BASE_URL, now = () => new Date().toISOString() }: DecisionLabAppProps) {
  const opts = useMemo(() => ({ baseUrl, fetchImpl }), [baseUrl, fetchImpl])
  const [bundle, setBundle] = useState<BundleState>({ status: 'loading' })
  const [build, setBuild] = useState<Record<string, unknown> | null>(null)
  const [draft, setDraft] = useState<Draft>(() => ({ ...emptyDraft('published_weekly'), slot: 'FLEX' }))
  const [note, setNote] = useState('')
  const [initialized, setInitialized] = useState(false)
  const [selectionNotice, setSelectionNotice] = useState<string | null>(null)
  const [initializationBlocked, setInitializationBlocked] = useState(false)
  const [computed, setComputed] = useState<Computed | null>(null)
  const [frozen, setFrozen] = useState(false)
  const [computeMessage, setComputeMessage] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [outcomeState, setOutcomeState] = useState<OutcomeUiState>({ status: 'idle' })
  const [saved, setSaved] = useState<LoadedReceipts>({ receipts: {}, problems: [] })
  const [savedMessage, setSavedMessage] = useState<PanelMessage | null>(null)
  const [importReport, setImportReport] = useState<ImportReport | null>(null)
  const [panels, setPanels] = useState({ outcomes: false, evidence: false, saved: false })
  const cacheRef = useRef<Record<string, InputsLoad>>({})
  const inflightRef = useRef<Record<string, Promise<InputsLoad>>>({})
  const [, bump] = useState(0)
  const deepLinked = useRef(false)
  const initialRequest = useRef({ initialCaseId, initialSnapshotId, initialPlayerIds: [...initialPlayerIds], initialSlot })

  const ensureSnapshot = useCallback(
    (manifest: LabManifest, sid: string): Promise<InputsLoad> => {
      const cached = cacheRef.current[sid]
      if (cached !== undefined) return Promise.resolve(cached)
      const inflight = inflightRef.current[sid]
      if (inflight !== undefined) return inflight
      const p = loadInputsSnapshot(manifest, sid, opts).then((res) => {
        cacheRef.current[sid] = res
        delete inflightRef.current[sid]
        bump((n) => n + 1)
        return res
      })
      inflightRef.current[sid] = p
      return p
    },
    [opts],
  )

  // Load manifest + cases (fail closed), then the volatile _build.json best-effort.
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      const m = await loadManifest(baseUrl, fetchImpl)
      if (cancelled) return
      if (!m.ok) {
        setBundle({ status: 'error', failure: m })
        return
      }
      const c = await loadCases(m.manifest, opts)
      if (cancelled) return
      if (!c.ok) {
        setBundle({ status: 'error', failure: c })
        return
      }
      setBundle({ status: 'ready', manifest: m.manifest, cases: c.cases, spec: policySpecDigestMatches(m.manifest, SPEC_DIGEST.sha256) })
      try {
        const impl: FetchLike = fetchImpl ?? ((url: string) => fetch(url, { credentials: 'same-origin', cache: 'no-store' }))
        const res = await impl(safeJoin(baseUrl, '_build.json'))
        if (res.ok) {
          const doc = JSON.parse(await res.text()) as unknown
          if (!cancelled && doc && typeof doc === 'object' && !Array.isArray(doc)) setBuild(doc as Record<string, unknown>)
        }
      } catch {
        // volatile file; absence is not an error
      }
    })()
    return () => {
      cancelled = true
    }
  }, [baseUrl, fetchImpl, opts])

  useEffect(() => {
    setSaved(loadReceipts())
  }, [])
  const refreshSaved = () => setSaved(loadReceipts())

  const ready = bundle.status === 'ready' ? bundle : null
  const cases = ready ? ready.cases.cases : []
  const labCase: LabCase | null = draft.caseId ? cases.find((c) => c.case_id === draft.caseId) ?? null : null

  const selectCase = useCallback((c: LabCase) => {
    setSelectionNotice(null)
    setInitializationBlocked(false)
    setDraft(draftFromCase(c))
    setNote('')
    setFrozen(false)
    setComputeMessage(null)
    setActionError(null)
  }, [])

  useEffect(() => {
    if (!ready || deepLinked.current) return
    deepLinked.current = true
    const { initialCaseId, initialSnapshotId, initialPlayerIds, initialSlot } = initialRequest.current
    let cancelled = false
    const initialize = async () => {
      let next = { ...emptyDraft('published_weekly'), slot: 'FLEX' }
      let notice: string | null = null
      let blocked = false
      if (initialCaseId) {
        const c = ready.cases.cases.find((x) => x.case_id === initialCaseId)
        if (c) next = draftFromCase(c)
        else { notice = 'The requested example is not in this evidence release. Choose a mode or a supported snapshot to continue; nothing was substituted.'; blocked = true }
      } else if (initialSnapshotId) {
        const requested = snapshotEntry(ready.manifest, initialSnapshotId)
        if (!requested) { notice = 'The requested snapshot is unavailable in this release. Nothing was substituted.'; blocked = true }
        else {
          next = { ...emptyDraft(requested.mode), slot: initialSlot && SLOT_OPTIONS.includes(initialSlot as typeof SLOT_OPTIONS[number]) ? initialSlot : 'FLEX', snapshotId: requested.snapshot_id, playerIds: initialPlayerIds }
          if (initialSlot && !SLOT_OPTIONS.includes(initialSlot as typeof SLOT_OPTIONS[number])) { notice = 'The requested starting slot is unsupported. Choose a supported slot explicitly to continue.'; blocked = true }
          const loaded = await ensureSnapshot(ready.manifest, requested.snapshot_id)
          if (!loaded.ok) { notice = `The requested evidence could not be verified: ${loaded.error}. Nothing was substituted.`; blocked = true }
          else if (new Set(initialPlayerIds).size !== initialPlayerIds.length || initialPlayerIds.length > 8 || initialPlayerIds.some((id) => !loaded.snapshot.rows.some((r) => r.player_id === id))) {
            notice = 'The transferred selection contains duplicate, missing, or more than eight players. Remove or reset those choices explicitly; no reduced choice set was substituted.'; blocked = true
          }
        }
      } else if (initialPlayerIds.length > 0) {
        notice = 'A transferred player selection needs an explicit evidence snapshot. Nothing was substituted.'
        blocked = true
      } else {
        let preference: { mode: Mode; snapshotId: string; snapshotDigest: string; slot: string; playerIds: string[]; params: Draft['params'] } | null = null
        try {
          const raw = JSON.parse(localStorage.getItem(PREFERENCE_KEY) ?? 'null')
          if (raw && MODES.includes(raw.mode) && typeof raw.snapshotId === 'string' && typeof raw.snapshotDigest === 'string' && SLOT_OPTIONS.includes(raw.slot) && Array.isArray(raw.playerIds) && raw.playerIds.length <= 8 && new Set(raw.playerIds).size === raw.playerIds.length && raw.playerIds.every((id: unknown) => typeof id === 'string') && raw.params && typeof raw.params.minGap === 'string' && typeof raw.params.minFloor === 'string' && typeof raw.params.floorEnabled === 'boolean' && typeof raw.params.requireBaseline === 'boolean' && typeof raw.params.modelOnly === 'boolean' && parseParams(raw.params).ok) preference = raw
          else if (raw) notice = 'Saved display choices were no longer compatible and were cleared.'
        } catch { notice = 'Saved display choices could not be read. The published-week defaults are shown.' }
        if (preference) {
          const savedEntry = snapshotEntry(ready.manifest, preference.snapshotId)
          const loaded = savedEntry && savedEntry.mode === preference.mode ? await ensureSnapshot(ready.manifest, savedEntry.snapshot_id) : null
          if (loaded?.ok && contentId(loaded.snapshot) === preference.snapshotDigest && preference.playerIds.every((id) => loaded.snapshot.rows.some((r) => r.player_id === id))) {
            next = { ...emptyDraft(preference.mode), snapshotId: preference.snapshotId, slot: preference.slot, playerIds: preference.playerIds, params: preference.params }
            notice = 'Restored your compatible display choices. Availability confirmations were not restored.'
          } else notice = 'Saved display choices no longer match this evidence release and were cleared.'
        }
        if (!next.snapshotId) {
          const latest = ready.manifest.latest_weekly_snapshot_id
          const latestEntry = latest ? snapshotEntry(ready.manifest, latest) : null
          if (latestEntry?.mode === 'published_weekly' && latestEntry.source_family === 'weekly') next.snapshotId = latestEntry.snapshot_id
          else notice = 'No published weekly snapshot is available. Choose Historical replay or Synthetic examples explicitly.'
        }
      }
      if (!initialCaseId && !initialSnapshotId && initialSlot) {
        if (SLOT_OPTIONS.includes(initialSlot as typeof SLOT_OPTIONS[number])) next.slot = initialSlot
        else { notice = 'The requested starting slot is unsupported. Choose a supported slot explicitly to continue.'; blocked = true }
      }
      if (!cancelled) { setDraft(next); setSelectionNotice(notice); setInitializationBlocked(blocked); setInitialized(true) }
    }
    void initialize()
    return () => { cancelled = true }
  }, [ready, ensureSnapshot])

  useEffect(() => {
    if (ready && draft.snapshotId && !cacheRef.current[draft.snapshotId]) void ensureSnapshot(ready.manifest, draft.snapshotId)
  }, [ready, draft.snapshotId, ensureSnapshot])

  const snapshotLoad = draft.snapshotId ? cacheRef.current[draft.snapshotId] : undefined
  const snapshot = snapshotLoad && snapshotLoad.ok ? snapshotLoad.snapshot : null
  const snapshotState: 'idle' | 'loading' | 'error' = !draft.snapshotId ? 'idle' : !snapshotLoad ? 'loading' : snapshotLoad.ok ? 'idle' : 'error'
  const snapshotError = snapshotLoad && !snapshotLoad.ok ? `${snapshotLoad.error}: ${snapshotLoad.detail}` : null
  const entry = ready && draft.snapshotId ? snapshotEntry(ready.manifest, draft.snapshotId) : null
  const contextSnapshots = ready
    ? ready.manifest.snapshots.filter((s) => s.mode === draft.mode).slice().sort((a, b) => b.season - a.season || b.week - a.week || (a.source_family < b.source_family ? -1 : 1))
    : []
  useEffect(() => {
    if (!initialized || !snapshot || initializationBlocked) return
    try { localStorage.setItem(PREFERENCE_KEY, JSON.stringify({ mode: draft.mode, snapshotId: snapshot.snapshot_id, snapshotDigest: contentId(snapshot), slot: draft.slot, playerIds: draft.playerIds, params: draft.params })) } catch { /* Optional UI preferences never block comparisons. */ }
  }, [initialized, snapshot, initializationBlocked, draft.mode, draft.slot, draft.playerIds, draft.params])
  const key = draftKey(draft)
  const stale = computed !== null && computed.key !== key
  const parsed = parseParams(draft.params)
  const canCompute = ready !== null && snapshot !== null && draft.playerIds.length >= 2 && parsed.ok && !frozen && !initializationBlocked
  const current = computed && !stale ? computed.receipt : null

  const compute = (d: Draft, inputsOverride?: DecisionInputs) => {
    if (!ready) return
    const load = d.snapshotId ? cacheRef.current[d.snapshotId] : undefined
    if (!load || !load.ok) {
      setComputeMessage(COPY.compute.needSnapshot)
      return
    }
    const p = parseParams(d.params)
    if (!p.ok) return
    if (!d.playerIds.length) {
      setComputeMessage(COPY.compute.needAlternatives)
      return
    }
    let inputs: DecisionInputs
    try {
      inputs =
        inputsOverride ??
        buildDecisionInputs(load.snapshot, {
          slot: d.slot,
          playerIds: d.playerIds,
          overrides: d.overrides,
          parameters: p.parameters,
          evidenceStatus: d.simulateCorrupt ? 'corrupt' : 'verified',
          evidenceDetail: d.simulateCorrupt ? COPY.nomination.simulateCorruptDetail : null,
        })
    } catch (e) {
      setComputeMessage(errorText(e))
      return
    }
    const id = decisionId(inputs)
    const k = draftKey(d)
    if (computed && computed.receipt.decision_id === id) {
      setComputed({ ...computed, key: k })
      setComputeMessage(COPY.compute.unchanged)
      return
    }
    // The same semantic decision may already be saved (an earlier session, an import). Its
    // creation time, note, case and parent link are immutable: open the saved record instead of
    // creating a competing one whose metadata could never be merged over it.
    const stored = loadReceipts()
    const already = Object.prototype.hasOwnProperty.call(stored.receipts, id) ? stored.receipts[id] : null
    if (already) {
      setSaved(stored)
      setComputed({ receipt: already, saved: true, key: k, diff: null })
      setFrozen(already.action.state !== 'not_recorded')
      setActionError(null)
      setOutcomeState({ status: already.outcome.state === 'attached' ? 'attached' : 'idle' })
      setComputeMessage(COPY.compute.alreadySaved)
      return
    }
    const prev = computed && computed.receipt.case_id === d.caseId && computed.receipt.inputs.snapshot.snapshot_id === inputs.snapshot.snapshot_id ? computed.receipt : null
    const receipt = newReceipt(inputs, {
      createdAtUtc: now(),
      caseId: d.caseId,
      parentDecisionId: prev ? prev.decision_id : null,
      predictionNote: note.trim() ? note.trim() : null,
    })
    const diff: DiffView | null = prev ? { rows: diffResults(prev.result, receipt.result), controls: changedControls(prev.inputs, inputs), parentId: prev.decision_id } : null
    setComputed({ receipt, saved: false, key: k, diff })
    setFrozen(false)
    setActionError(null)
    setOutcomeState({ status: 'idle' })
    setComputeMessage(null)
  }

  const applyExperiment = (exp: Experiment) => {
    if (!labCase || !snapshot) return
    const d = draftFromExperiment(labCase, exp, draftFromCase(labCase))
    setDraft(d)
    compute(d, experimentInputs(labCase, exp, snapshot))
  }

  const persistAction = (r: Receipt) => {
    if (!computed) return
    const res = saveReceipt(r)
    if (!res.ok) {
      setComputed({ ...computed, receipt: r, saved: false })
      setActionError(`${res.reason}: ${res.message}`)
    } else {
      setComputed({ ...computed, receipt: res.receipt, saved: true })
      setActionError(null)
      refreshSaved()
    }
    setFrozen(true)
  }

  const record = (chosen: string, kind: ActionKind, actionNote: string | null) => {
    if (!computed) return
    try {
      persistAction(recordAction(computed.receipt, chosen, kind, now(), actionNote))
    } catch (e) {
      setActionError(errorText(e))
    }
  }
  const decline = (actionNote: string | null) => {
    if (!computed) return
    try {
      persistAction(declineAction(computed.receipt, now(), actionNote))
    } catch (e) {
      setActionError(errorText(e))
    }
  }

  const attachTo = async (receipt: Receipt): Promise<{ ok: true; receipt: Receipt } | { ok: false; none: boolean; message: string }> => {
    if (!ready) return { ok: false, none: false, message: COPY.load.failedTitle }
    const sid = receipt.inputs.snapshot.snapshot_id
    const load = await ensureSnapshot(ready.manifest, sid)
    if (!load.ok) return { ok: false, none: false, message: `${load.error}: ${load.detail}` }
    const o = await loadOutcomeSnapshot(ready.manifest, sid, load.snapshot, opts)
    if (!o.ok) return { ok: false, none: false, message: `${o.error}: ${o.detail}` }
    if (o.snapshot === null) return { ok: false, none: true, message: COPY.outcome.none }
    try {
      const r = attachOutcome(receipt, o.snapshot, now())
      const res = saveReceipt(r)
      if (!res.ok) return { ok: false, none: false, message: `${res.reason}: ${res.message}` }
      refreshSaved()
      return { ok: true, receipt: res.receipt }
    } catch (e) {
      return { ok: false, none: false, message: errorText(e) }
    }
  }

  const reveal = async () => {
    if (!computed) return
    setOutcomeState({ status: 'loading' })
    const res = await attachTo(computed.receipt)
    if (res.ok) {
      setComputed({ ...computed, receipt: res.receipt, saved: true })
      setOutcomeState({ status: 'attached' })
    } else {
      setOutcomeState(res.none ? { status: 'none' } : { status: 'error', message: res.message })
    }
  }

  const attachSaved = async (r: Receipt) => {
    const res = await attachTo(r)
    if (res.ok) {
      setSavedMessage({ kind: 'ok', text: `Outcomes attached to ${r.decision_id.slice(0, 12)}; its decision id is unchanged.` })
      if (computed && computed.receipt.decision_id === r.decision_id) {
        setComputed({ ...computed, receipt: res.receipt, saved: true })
        setOutcomeState({ status: 'attached' })
      }
    } else {
      setSavedMessage({ kind: 'error', text: `${COPY.outcome.failed} for ${r.decision_id.slice(0, 12)}: ${res.message}` })
    }
  }

  const openSaved = (r: Receipt) => {
    const d = draftFromInputs(r.inputs, r.case_id)
    setDraft(d)
    setNote(r.prediction_note ?? '')
    setComputed({ receipt: r, saved: true, key: draftKey(d), diff: null })
    setFrozen(true)
    setActionError(null)
    setOutcomeState(r.outcome.state === 'attached' ? { status: 'attached' } : { status: 'idle' })
    setComputeMessage(COPY.saved.opened)
    if (typeof window !== 'undefined') window.scrollTo({ top: 0 })
  }

  const removeSaved = (id: string) => {
    deleteReceipt(id)
    refreshSaved()
    if (computed && computed.receipt.decision_id === id) setComputed({ ...computed, saved: false })
  }
  const clearSaved = () => {
    clearAll()
    refreshSaved()
    setImportReport(null)
    if (computed) setComputed({ ...computed, saved: false })
  }
  const importText = (text: string) => {
    const report = importReceipts(text)
    setImportReport(report)
    setSavedMessage(null)
    refreshSaved()
  }

  const changeDraft = (d: Draft) => {
    if (d.snapshotId !== draft.snapshotId) { setNote(''); d = { ...d, overrides: {}, simulateCorrupt: false } }
    else if (d.slot !== draft.slot || d.playerIds.join('|') !== draft.playerIds.join('|')) d = { ...d, overrides: Object.fromEntries(Object.entries(d.overrides).filter(([, value]) => value.availability === 'unavailable')) }
    setInitializationBlocked(false)
    setSelectionNotice(null)
    setDraft(d)
    setComputeMessage(null)
  }
  const changeMode = (mode: Mode) => {
    if (mode === draft.mode) return
    const options = ready?.manifest.snapshots.filter((s) => s.mode === mode).slice().sort((a, b) => b.season - a.season || b.week - a.week || a.snapshot_id.localeCompare(b.snapshot_id)) ?? []
    const first = mode === 'published_weekly' ? ready?.manifest.latest_weekly_snapshot_id ?? null : options[0]?.snapshot_id ?? null
    setDraft({ ...emptyDraft(mode), slot: 'FLEX', snapshotId: first })
    setSelectionNotice(null)
    setInitializationBlocked(false)
    setNote('')
    setFrozen(false)
    setComputeMessage(null)
  }

  if (bundle.status === 'loading' || (bundle.status === 'ready' && !initialized)) {
    return (
      <div role="status" className="border border-[#c5c7c0] bg-[#f8f7f2] p-8 font-mono text-xs  text-[#61706c]">
        <span className="mr-3 inline-block h-2 w-2 animate-pulse rounded-full bg-ember align-middle" />
        {COPY.load.loading}
      </div>
    )
  }
  if (bundle.status === 'error') {
    return (
      <Alert title={COPY.load.failedTitle} testId="bundle-error">
        <p className="font-mono text-xs">
          {bundle.failure.kind}: {bundle.failure.error} — {bundle.failure.detail}
        </p>
        <p className="mt-2 text-xs text-[#52605d]">{COPY.load.failedBody}</p>
      </Alert>
    )
  }
  const spec = bundle.spec
  const noteLocked = frozen || (computed !== null && !stale)

  const customSettings = JSON.stringify(draft.params) !== JSON.stringify(defaultParamDraft())
  return (
    <div className="space-y-5">
      <section className="product-card space-y-4" aria-label="Decision mode">
        <div className="flex flex-nowrap items-center justify-between gap-3 md:flex-wrap">
          <label className="min-w-0 flex-1 md:hidden"><span className="sr-only">Comparison mode</span><select value={draft.mode} onChange={(e) => changeMode(e.target.value as Mode)} className={`min-h-[44px] w-full rounded-lg border border-[#bfc3bb] bg-white px-3 text-sm font-semibold ${FOCUS}`}>{MODES.map((mode) => <option key={mode} value={mode}>{MODE_NAMES[mode]}</option>)}</select></label>
          <div role="group" aria-label="Evidence context" className="hidden flex-wrap gap-2 md:flex">
            {MODES.map((mode) => <button key={mode} type="button" data-testid={`context-${mode}`} aria-pressed={draft.mode === mode} onClick={() => changeMode(mode)} className={`min-h-[44px] rounded-lg border px-4 py-2 text-sm font-semibold ${draft.mode === mode ? 'border-[#173e35] bg-[#173e35] text-white' : 'border-[#c5cec5] bg-white text-[#27473d]'} ${FOCUS}`}>{MODE_NAMES[mode]}</button>)}
          </div>
          <Button onClick={() => { setPanels((p) => ({ ...p, saved: true })); document.getElementById('saved-decisions')?.scrollIntoView?.({ block: 'start' }) }} aria-label={`Saved decisions (${Object.keys(saved.receipts).length})`}><span className="md:hidden">Saved ({Object.keys(saved.receipts).length})</span><span className="hidden md:inline">Saved decisions ({Object.keys(saved.receipts).length})</span></Button>
        </div>
        <p className="text-sm leading-6 text-[#4f6259]" data-testid="context-description">
          {draft.mode === 'published_weekly' ? 'Conditional PPR comparison; availability is unverified.' : draft.mode === 'historical_replay' ? 'Replay the original historical evidence, with hindsight-conditioned players.' : 'Artificial examples for learning, not real forecasts.'}
        </p>
        {snapshot && <p className="text-sm"><strong>{draft.mode === 'published_weekly' ? 'Latest published: ' : ''}{formatPeriod(snapshot.season, snapshot.week)}</strong> · PPR <span className="hidden md:inline">· Updated {formatLocalDate(snapshot.generated_at_utc)}</span></p>}
        {selectionNotice && <Notice testId="selection-notice">{selectionNotice}<button type="button" className={`ml-2 underline ${FOCUS}`} onClick={() => { setDraft({ ...draft, playerIds: [], overrides: {} }); setSelectionNotice(null); setInitializationBlocked(false) }}>Reset selection</button></Notice>}
      </section>
      {spec.matches !== true && <div role="status" data-testid="spec-warning" className="rounded-lg border border-[#b68451] bg-[#fff7e9] p-4 text-sm">{spec.checked ? COPY.load.specMismatch : COPY.load.specUnchecked} {spec.warning ? `(${spec.warning})` : ''}</div>}
      <p className="text-sm text-[#4f6259] md:hidden">Choose players → Compare → Record or reveal</p>
      <ol aria-label="Exploration steps" className="hidden flex-wrap gap-x-6 gap-y-2 text-sm text-[#4f6259] md:flex"><li><b>1.</b> Choose players or an example</li><li><b>2.</b> Compare and understand the result</li><li><b>3.</b> Record a choice, then reveal available results</li></ol>
      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-5">
          <Nomination draft={draft} snapshots={contextSnapshots} snapshot={snapshot} snapshotState={snapshotState} snapshotError={snapshotError} onChange={changeDraft} disabled={frozen} />
          <div className="product-card space-y-4" data-testid="compute-card">
            <Button tone="primary" data-testid="compute" disabled={!canCompute} onClick={() => compute(draft)}>{COPY.compute.button}</Button>
            {!canCompute && !frozen && <Help>{!snapshot ? 'Choose a supported snapshot to begin.' : draft.playerIds.length < 2 ? 'Choose at least two players to make a useful comparison.' : !parsed.ok ? 'Check the advanced settings for an invalid value.' : 'Resolve the selection message before comparing.'}</Help>}
            {stale && <p role="status" data-testid="stale-notice" className="text-sm text-[#805021]">{COPY.compute.stale}</p>}
            {computeMessage && <Notice testId="compute-message">{computeMessage}</Notice>}
            <details className="product-details"><summary>Write down a prediction (optional)</summary><label className="mt-3 block text-sm font-semibold" htmlFor="prediction-note">{COPY.note.label}</label><textarea id="prediction-note" data-testid="prediction-note" value={note} rows={2} disabled={noteLocked} aria-describedby="prediction-note-help" onChange={(e) => setNote(e.target.value)} className={`mt-2 min-h-20 w-full rounded-lg border border-[#bfc3bb] bg-white p-3 text-base disabled:bg-sand ${FOCUS}`} /><Help id="prediction-note-help">{noteLocked ? COPY.note.locked : COPY.note.help}</Help></details>
          </div>
          <details className="product-details" open={draft.caseId !== null}><summary>Choose an example</summary><div className="mt-4"><CaseLibrary cases={cases.filter((c) => c.mode === draft.mode)} selectedId={draft.caseId} onSelect={selectCase} /><Button data-testid="custom-nomination" className="mt-4" onClick={() => changeDraft({ ...emptyDraft(draft.mode), slot: draft.slot, snapshotId: draft.snapshotId })}>Choose my own players</Button><Help>Examples are listed by their stable case identifier, not by favorable results. They demonstrate policy behavior, not a performance sample.</Help></div></details>
          <AssumptionControls draft={draft} snapshot={snapshot} onChange={changeDraft} disabled={snapshot === null} frozen={frozen} onUnlock={() => setFrozen(false)} labCase={labCase} onApplyExperiment={applyExperiment} diff={computed && !stale ? computed.diff : null} />
          {customSettings && <div className="flex items-center gap-3 text-sm"><strong>Custom settings</strong><Button onClick={() => changeDraft({ ...draft, params: defaultParamDraft() })}>Reset to default settings</Button></div>}
        </div>
        <div className="min-w-0 space-y-5">
          <DecisionCard receipt={computed ? computed.receipt : null} stale={stale} saved={computed ? computed.saved : false} emptyMessage="Choose two or more players, then compare. Your result and the reasons for it will appear here." />
          {current && <details className="product-details"><summary>Compare every nominated player</summary><div className="mt-4"><AlternativesTable receipt={current} /></div></details>}
          {current && <ActionPanel key={current.decision_id} receipt={current} onRecord={record} onDecline={decline} error={actionError} />}
          <OutcomePanel receipt={current} labCase={labCase} open={panels.outcomes} onToggle={() => setPanels((p) => ({ ...p, outcomes: !p.outcomes }))} state={outcomeState} onReveal={reveal} />
          <EvidencePanel manifest={bundle.manifest} entry={entry} snapshot={snapshot} receipt={current} specCheck={spec} build={build} open={panels.evidence} onToggle={() => setPanels((p) => ({ ...p, evidence: !p.evidence }))} />
        </div>
      </div>
      <div id="saved-decisions"><SavedDecisionsPanel loaded={saved} onOpen={openSaved} onAttach={attachSaved} onDelete={removeSaved} exportText={() => exportAll()} onImport={importText} onClear={clearSaved} report={importReport} message={savedMessage} open={panels.saved} onToggle={() => setPanels((p) => ({ ...p, saved: !p.saved }))} /></div>
    </div>
  )
}
