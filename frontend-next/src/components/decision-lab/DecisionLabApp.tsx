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
import { changedControls, diffResults, draftFromCase, draftFromExperiment, draftFromInputs, draftKey, emptyDraft, parseParams, type Draft } from './draft'
import { EvidencePanel } from './EvidencePanel'
import { Nomination } from './Nomination'
import { OutcomePanel, type OutcomeUiState } from './OutcomePanel'
import { SavedDecisionsPanel, type PanelMessage } from './SavedDecisionsPanel'

const COPY = DECISION_LAB_COPY
const MODES: Mode[] = ['historical_replay', 'published_weekly', 'synthetic']

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
export function DecisionLabApp({ initialCaseId = null, fetchImpl, baseUrl = DEFAULT_BASE_URL, now = () => new Date().toISOString() }: DecisionLabAppProps) {
  const opts = useMemo(() => ({ baseUrl, fetchImpl }), [baseUrl, fetchImpl])
  const [bundle, setBundle] = useState<BundleState>({ status: 'loading' })
  const [build, setBuild] = useState<Record<string, unknown> | null>(null)
  const [draft, setDraft] = useState<Draft>(() => emptyDraft('synthetic'))
  const [note, setNote] = useState('')
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
    setDraft(draftFromCase(c))
    setNote('')
    setFrozen(false)
    setComputeMessage(null)
    setActionError(null)
  }, [])

  useEffect(() => {
    if (!ready || deepLinked.current) return
    deepLinked.current = true
    if (initialCaseId) {
      const c = ready.cases.cases.find((x) => x.case_id === initialCaseId)
      if (c) selectCase(c)
    }
  }, [ready, initialCaseId, selectCase])

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
  const key = draftKey(draft)
  const stale = computed !== null && computed.key !== key
  const parsed = parseParams(draft.params)
  const canCompute = ready !== null && snapshot !== null && draft.playerIds.length > 0 && parsed.ok && !frozen
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
    if (d.snapshotId !== draft.snapshotId) setNote('')
    setDraft(d)
    setComputeMessage(null)
  }
  const changeMode = (mode: Mode) => {
    if (mode === draft.mode) return
    setDraft(emptyDraft(mode))
    setNote('')
    setFrozen(false)
    setComputeMessage(null)
  }

  if (bundle.status === 'loading') {
    return (
      <div role="status" className="border border-[#c5c7c0] bg-[#f8f7f2] p-8 font-mono text-xs uppercase tracking-widest text-[#61706c]">
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
        <p className="mt-2 text-xs text-[#6e7875]">{COPY.load.failedBody}</p>
      </Alert>
    )
  }
  const spec = bundle.spec
  const noteLocked = frozen || (computed !== null && !stale)

  return (
    <div className="space-y-6">
      {spec.matches !== true && (
        <div role="status" data-testid="spec-warning" className="border border-ember bg-[#fff1ea] p-4 text-xs text-ink">
          <p className="font-mono text-[10px] uppercase tracking-widest text-ember">policy specification</p>
          <p className="mt-1">{spec.checked ? COPY.load.specMismatch : COPY.load.specUnchecked} {spec.warning ? `(${spec.warning})` : ''}</p>
        </div>
      )}

      <div className="border border-[#c5c7c0] bg-[#f8f7f2] p-4 md:p-5">
        <MicroLabel as="span">Context</MicroLabel>
        <div role="group" aria-label="Evidence context" className="mt-2 flex flex-col border border-[#bfc3bb] sm:flex-row">
          {MODES.map((mode) => (
            <button
              key={mode}
              type="button"
              data-testid={`context-${mode}`}
              aria-pressed={draft.mode === mode}
              onClick={() => changeMode(mode)}
              className={`flex-1 border-b border-[#bfc3bb] px-3 py-2 text-left font-mono text-[10px] font-bold uppercase tracking-widest last:border-b-0 sm:border-b-0 sm:border-r sm:last:border-r-0 ${draft.mode === mode ? 'bg-ink text-acid' : 'bg-white text-[#5c6966] hover:bg-sand'} ${FOCUS}`}
            >
              {COPY.contexts[mode].label}
            </button>
          ))}
        </div>
        <p className="mt-3 text-xs leading-5 text-[#3b4a46]" data-testid="context-description">
          <b>{COPY.contexts[draft.mode].label}.</b> {COPY.contexts[draft.mode].is} <span className="text-[#6e7875]">{COPY.contexts[draft.mode].isNot}</span>
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[380px_minmax(0,1fr)]">
        <div className="min-w-0 space-y-4">
          <CaseLibrary cases={cases} selectedId={draft.caseId} onSelect={selectCase} />
          <div className="border border-dashed border-[#bfc3bb] bg-[#f8f7f2] p-4">
            <Button data-testid="custom-nomination" onClick={() => changeDraft({ ...emptyDraft(draft.mode) })}>
              {COPY.library.custom}
            </Button>
            <Help>{COPY.library.customHelp}</Help>
          </div>
        </div>

        <div className="min-w-0 space-y-6">
          <Nomination draft={draft} snapshots={contextSnapshots} snapshot={snapshot} snapshotState={snapshotState} snapshotError={snapshotError} onChange={changeDraft} disabled={frozen} />

          <AssumptionControls draft={draft} snapshot={snapshot} onChange={changeDraft} disabled={snapshot === null} frozen={frozen} onUnlock={() => setFrozen(false)} labCase={labCase} onApplyExperiment={applyExperiment} diff={computed && !stale ? computed.diff : null} />

          <Card title="Compute" id="compute" testId="compute-card">
            <div className="flex flex-col gap-1">
              <MicroLabel htmlFor="prediction-note">{COPY.note.label}</MicroLabel>
              <textarea id="prediction-note" data-testid="prediction-note" value={note} rows={2} disabled={noteLocked} aria-describedby="prediction-note-help" onChange={(e) => setNote(e.target.value)} className={`w-full border border-[#bfc3bb] bg-white p-2 text-xs disabled:bg-sand disabled:text-[#6e7875] ${FOCUS}`} />
              <Help id="prediction-note-help">{noteLocked ? COPY.note.locked : COPY.note.help}</Help>
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-3">
              <Button tone="primary" data-testid="compute" disabled={!canCompute} onClick={() => compute(draft)}>
                {COPY.compute.button}
              </Button>
              {stale && (
                <span role="status" aria-live="polite" data-testid="stale-notice" className="font-mono text-[10px] uppercase tracking-widest text-ember">
                  {COPY.compute.stale}
                </span>
              )}
            </div>
            {computeMessage && (
              <div className="mt-3">
                <Notice testId="compute-message">{computeMessage}</Notice>
              </div>
            )}
          </Card>

          <DecisionCard receipt={computed ? computed.receipt : null} stale={stale} saved={computed ? computed.saved : false} emptyMessage={COPY.compute.empty} />

          {current && <AlternativesTable receipt={current} />}
          {current && <ActionPanel key={current.decision_id} receipt={current} onRecord={record} onDecline={decline} error={actionError} />}

          <OutcomePanel receipt={current} labCase={labCase} open={panels.outcomes} onToggle={() => setPanels((p) => ({ ...p, outcomes: !p.outcomes }))} state={outcomeState} onReveal={reveal} />
          <EvidencePanel manifest={bundle.manifest} entry={entry} snapshot={snapshot} receipt={current} specCheck={spec} build={build} open={panels.evidence} onToggle={() => setPanels((p) => ({ ...p, evidence: !p.evidence }))} />
          <SavedDecisionsPanel loaded={saved} onOpen={openSaved} onAttach={attachSaved} onDelete={removeSaved} exportText={() => exportAll()} onImport={importText} onClear={clearSaved} report={importReport} message={savedMessage} open={panels.saved} onToggle={() => setPanels((p) => ({ ...p, saved: !p.saved }))} />
        </div>
      </div>
    </div>
  )
}
