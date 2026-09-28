/**
 * Pure helpers for the Decision Lab UI state. Nothing here evaluates the policy, hashes anything
 * or touches storage; the library under `@/lib/decision-lab` does that. This module only shapes
 * what the controls hold (raw strings for numbers, explicit overrides) into library inputs and
 * derives display-only diffs between two results.
 */
import { formatNumber } from '@/lib/decision-lab/canonical'
import { PARAM_SPEC } from '@/lib/decision-lab/policy'
import type { AvailabilityOverride } from '@/lib/decision-lab/receipts'
import type {
  Availability,
  DecisionInputs,
  DecisionResult,
  Experiment,
  LabCase,
  Mode,
  Parameters,
  SnapshotRow,
} from '@/lib/decision-lab/types'

export const SLOT_OPTIONS = ['QB', 'RB', 'WR', 'TE', 'FLEX'] as const
export const MAX_ALTERNATIVES = 8

export interface ParamDraft {
  minGap: string
  floorEnabled: boolean
  minFloor: string
  requireBaseline: boolean
  modelOnly: boolean
}

/** What the controls hold. Numbers stay raw strings until they are parsed for a computation. */
export interface Draft {
  caseId: string | null
  mode: Mode
  snapshotId: string | null
  slot: string
  playerIds: string[]
  overrides: Record<string, Required<AvailabilityOverride>>
  params: ParamDraft
  simulateCorrupt: boolean
}

/** '—' for a missing number, else the canonical short form (14 → "14", 0.4 → "0.4"). */
export function num(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  try {
    return formatNumber(value)
  } catch {
    return '—'
  }
}

export function paramDraftFrom(p: Parameters): ParamDraft {
  return {
    minGap: num(p.min_projection_gap),
    floorEnabled: p.min_floor !== null,
    minFloor: p.min_floor === null ? '' : num(p.min_floor),
    requireBaseline: p.require_independent_baseline,
    modelOnly: p.model_only_exploration,
  }
}

export function defaultParamDraft(): ParamDraft {
  return {
    minGap: num(PARAM_SPEC.min_projection_gap.default as number),
    floorEnabled: false,
    minFloor: '',
    requireBaseline: PARAM_SPEC.require_independent_baseline.default === true,
    modelOnly: PARAM_SPEC.model_only_exploration.default === true,
  }
}

export function emptyDraft(mode: Mode): Draft {
  return {
    caseId: null,
    mode,
    snapshotId: null,
    slot: 'RB',
    playerIds: [],
    overrides: {},
    params: defaultParamDraft(),
    simulateCorrupt: false,
  }
}

export function draftFromCase(labCase: LabCase): Draft {
  const overrides: Draft['overrides'] = {}
  Object.keys(labCase.overrides).forEach((pid) => {
    overrides[pid] = { ...labCase.overrides[pid] }
  })
  return {
    caseId: labCase.case_id,
    mode: labCase.mode,
    snapshotId: labCase.snapshot_id,
    slot: labCase.slot,
    playerIds: labCase.alternatives.slice(),
    overrides,
    params: paramDraftFrom(labCase.parameters),
    simulateCorrupt: false,
  }
}

/** The draft an experiment produces: parameters merged, overrides replaced when present. */
export function draftFromExperiment(labCase: LabCase, experiment: Experiment, base: Draft): Draft {
  const overrides: Draft['overrides'] = {}
  const source = experiment.overrides !== undefined ? experiment.overrides : labCase.overrides
  Object.keys(source).forEach((pid) => {
    overrides[pid] = { ...source[pid] }
  })
  return {
    ...base,
    overrides,
    params: paramDraftFrom({ ...labCase.parameters, ...(experiment.parameters || {}) }),
  }
}

/** The draft that reproduces a receipt's inputs (every alternative's availability is explicit). */
export function draftFromInputs(inputs: DecisionInputs, caseId: string | null): Draft {
  const overrides: Draft['overrides'] = {}
  inputs.alternatives.forEach((a) => {
    overrides[a.player_id] = { availability: a.availability, availability_basis: a.availability_basis }
  })
  return {
    caseId,
    mode: inputs.snapshot.mode,
    snapshotId: inputs.snapshot.snapshot_id,
    slot: inputs.slot,
    playerIds: inputs.alternatives.map((a) => a.player_id),
    overrides,
    params: paramDraftFrom(inputs.parameters),
    simulateCorrupt: inputs.snapshot.evidence_status === 'corrupt',
  }
}

/** Inline error for a raw numeric field, or null when it parses inside [min, max]. */
export function numberError(raw: string, min: number, max: number): string | null {
  const trimmed = raw.trim()
  if (trimmed === '') return `Enter a number between ${min} and ${max}.`
  if (!/^[-+]?(\d+\.?\d*|\.\d+)$/.test(trimmed)) return 'Enter a finite decimal number.'
  const v = Number(trimmed)
  if (!Number.isFinite(v)) return 'Enter a finite decimal number.'
  if (v < min || v > max) return `Must be between ${min} and ${max}.`
  return null
}

export type ParsedParams = { ok: true; parameters: Parameters } | { ok: false; errors: { minGap: string | null; minFloor: string | null } }

export function parseParams(p: ParamDraft): ParsedParams {
  const gapErr = numberError(p.minGap, PARAM_SPEC.min_projection_gap.min as number, PARAM_SPEC.min_projection_gap.max as number)
  const floorErr = p.floorEnabled ? numberError(p.minFloor, PARAM_SPEC.min_floor.min as number, PARAM_SPEC.min_floor.max as number) : null
  if (gapErr || floorErr) return { ok: false, errors: { minGap: gapErr, minFloor: floorErr } }
  return {
    ok: true,
    parameters: {
      min_projection_gap: Number(p.minGap.trim()),
      min_floor: p.floorEnabled ? Number(p.minFloor.trim()) : null,
      require_independent_baseline: p.requireBaseline,
      model_only_exploration: p.modelOnly,
    },
  }
}

/** A stable signature of everything that feeds a computation; a different key means stale. */
export function draftKey(d: Draft): string {
  return JSON.stringify({
    c: d.caseId,
    s: d.snapshotId,
    slot: d.slot,
    p: d.playerIds,
    o: d.overrides,
    params: d.params,
    x: d.simulateCorrupt,
  })
}

/** The availability choice a row's control shows: the bundle default or an explicit override. */
export type AvailabilityChoice = 'default' | 'assumed_available' | 'unavailable'

export function availabilityChoice(d: Draft, playerId: string): AvailabilityChoice {
  const ov = d.overrides[playerId]
  if (!ov) return 'default'
  if (ov.availability === 'unavailable') return 'unavailable'
  if (ov.availability === 'assumed_available' && ov.availability_basis === 'user_assumed') return 'assumed_available'
  // Explicit override equal to a bundle value (for example a reopened receipt): shown as default.
  return 'default'
}

export function withAvailability(d: Draft, playerId: string, choice: AvailabilityChoice, rowDefault: Availability): Draft {
  const overrides = { ...d.overrides }
  if (choice === 'default') {
    delete overrides[playerId]
  } else if (choice === 'unavailable') {
    overrides[playerId] = { availability: 'unavailable', availability_basis: 'user_excluded' }
  } else if (rowDefault === 'assumed_available') {
    delete overrides[playerId]
  } else {
    overrides[playerId] = { availability: 'assumed_available', availability_basis: 'user_assumed' }
  }
  return { ...d, overrides }
}

/** Rows of a snapshot whose position can fill the slot, filtered by a free-text query. */
export function searchRows(rows: SnapshotRow[], eligible: string[], query: string, limit = 40): SnapshotRow[] {
  const q = query.trim().toLowerCase()
  const out: SnapshotRow[] = []
  for (let i = 0; i < rows.length && out.length < limit; i += 1) {
    const r = rows[i]
    if (eligible.indexOf(r.position) < 0) continue
    if (q) {
      const hay = `${r.display_name ?? ''} ${r.player_id} ${r.team ?? ''}`.toLowerCase()
      if (hay.indexOf(q) < 0) continue
    }
    out.push(r)
  }
  return out
}

export interface DiffRow {
  key: string
  label: string
  before: string
  after: string
}

function gate(ok: boolean | null, checked: boolean): string {
  if (!checked) return 'off'
  if (ok === null) return '—'
  return ok ? 'meets' : 'fails'
}

/** Display-only rows for the fields that differ between two results. */
export function diffResults(prev: DecisionResult, next: DecisionResult): DiffRow[] {
  const rows: DiffRow[] = []
  const push = (key: string, label: string, before: string, after: string) => {
    if (before !== after) rows.push({ key, label, before, after })
  }
  push('status', 'Status', prev.status, next.status)
  push('recommended', 'Recommended', prev.recommended_player_id ?? 'none', next.recommended_player_id ?? 'none')
  const prevCodes = prev.reasons.map((r) => r.code)
  const nextCodes = next.reasons.map((r) => r.code)
  const added = nextCodes.filter((c) => prevCodes.indexOf(c) < 0)
  const removed = prevCodes.filter((c) => nextCodes.indexOf(c) < 0)
  if (added.length) rows.push({ key: 'reasons_added', label: 'Reasons added', before: '', after: added.join(', ') })
  if (removed.length) rows.push({ key: 'reasons_removed', label: 'Reasons removed', before: removed.join(', '), after: '' })
  push('gap', 'Projection gap', num(prev.leading.gap), num(next.leading.gap))
  push('gap_threshold', 'Minimum gap', num(prev.gates.min_projection_gap), num(next.gates.min_projection_gap))
  push('gap_gate', 'Gap gate', gate(prev.gates.gap_ok, prev.leading.gap !== null), gate(next.gates.gap_ok, next.leading.gap !== null))
  push('floor_threshold', 'Floor guardrail', prev.gates.min_floor === null ? 'off' : num(prev.gates.min_floor), next.gates.min_floor === null ? 'off' : num(next.gates.min_floor))
  push('floor_gate', 'Floor gate', gate(prev.gates.floor_ok, prev.gates.floor_checked), gate(next.gates.floor_ok, next.gates.floor_checked))
  push('leading_floor', 'Leading floor', num(prev.gates.leading_floor), num(next.gates.leading_floor))
  push('comparable', 'Comparable alternatives', String(prev.counts.comparable), String(next.counts.comparable))
  push('baseline', 'Baseline preference', prev.baseline.preferred_player_id ?? prev.baseline.status, next.baseline.preferred_player_id ?? next.baseline.status)
  return rows
}

/** Names of the controls whose values differ between two decision inputs. */
export function changedControls(prev: DecisionInputs, next: DecisionInputs): string[] {
  const out: string[] = []
  if (prev.slot !== next.slot) out.push('slot')
  const prevIds = prev.alternatives.map((a) => a.player_id).join(',')
  const nextIds = next.alternatives.map((a) => a.player_id).join(',')
  if (prevIds !== nextIds) out.push('alternatives')
  const prevAvail: Record<string, string> = {}
  prev.alternatives.forEach((a) => {
    prevAvail[a.player_id] = `${a.availability}/${a.availability_basis}`
  })
  const availChanged = next.alternatives.some((a) => prevAvail[a.player_id] !== undefined && prevAvail[a.player_id] !== `${a.availability}/${a.availability_basis}`)
  if (availChanged) out.push('availability')
  const pp = prev.parameters
  const np = next.parameters
  if (pp.min_projection_gap !== np.min_projection_gap) out.push('min_projection_gap')
  if (pp.min_floor !== np.min_floor) out.push('min_floor')
  if (pp.require_independent_baseline !== np.require_independent_baseline) out.push('require_independent_baseline')
  if (pp.model_only_exploration !== np.model_only_exploration) out.push('model_only_exploration')
  if (prev.snapshot.evidence_status !== next.snapshot.evidence_status) out.push('evidence')
  if (prev.snapshot.snapshot_id !== next.snapshot.snapshot_id) out.push('snapshot')
  return out
}
