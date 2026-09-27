/**
 * Decision receipts: immutable inputs and result plus appended events — the mirror of
 * `ffai/decision_lab/receipts.py` (ADR-0033).
 *
 * Two kinds of identity:
 *
 * - decision identity — `decision_id = sha256(canonical(decision_inputs))`: schema and policy
 *   versions, the snapshot reference (including the inputs snapshot's content digest), slot, every
 *   alternative with its availability fields, and the parameters. Attaching outcomes, recording an
 *   action, or a new timestamp never changes it. Changing an assumption or a parameter produces a
 *   new decision whose `parent_decision_id` links back.
 * - event identity — `event_id = sha256(canonical({decision_id, seq, event_type, at_utc,
 *   payload}))`. Events are appended, never rewritten; `action` and `outcome` on the receipt are
 *   projections of the event list and are recomputed on validation.
 *
 * The initial action state is `not_recorded` with null ids and time; the recommendation is never
 * auto-filled as the user's choice. At most one action event may exist. Outcome events must
 * reference the decision's own snapshot.
 *
 * Trust boundary: digests let a reader check a receipt against known evidence; they are not
 * signatures and prove nothing about who acted when ({@link TRUST_NOTE}).
 */
import { contentId, deepEqual } from './canonical';
import { computeMetrics } from './metrics';
import { defaultParameters, evaluate, normalizeInputs } from './policy';
import { POLICY_VERSION, SCHEMA_VERSIONS, SPEC } from './spec';
import type {
  ActionKind,
  ActionState,
  Alternative,
  Availability,
  AvailabilityBasis,
  DecisionInputs,
  EventType,
  EvidenceStatus,
  InputsSnapshot,
  OutcomeSnapshot,
  OutcomeState,
  Parameters,
  Receipt,
  ReceiptEvent,
  SnapshotRef,
} from './types';
import {
  validateDecisionInputsShape,
  validateDecisionResultShape,
  validateOutcomeSnapshotShape,
  validateShape,
  ContractError,
} from './validate';

/** `decision_receipt/1.0`. */
export const RECEIPT_SCHEMA: string = SCHEMA_VERSIONS.receipt;

/** The trust note every receipt carries; identical to the Python constant. */
export const TRUST_NOTE =
  'Digests support integrity checks against known evidence; they are not signatures and do not ' +
  'prove who acted or when. Local storage is editable and erasable; append-only is application ' +
  'behaviour, not a tamper-proof audit service.';

/** Event types that count as "the" action of a decision (at most one may exist). */
export const ACTION_EVENTS: readonly EventType[] = ['action_recorded', 'action_declined'];

/** A receipt is malformed, tampered, or an operation on it is not allowed. */
export class ReceiptError extends Error {
  /** The failing check name when raised by {@link validateReceipt}. */
  readonly check: string | null;
  /** The checks performed so far (last one failed) when raised by {@link validateReceipt}. */
  readonly checks: ReceiptCheck[];

  constructor(message: string, check: string | null = null, checks: ReceiptCheck[] = []) {
    super(message);
    this.name = 'ReceiptError';
    this.check = check;
    this.checks = checks;
  }
}

/** Two receipts share a `decision_id` but disagree. */
export class ConflictError extends ReceiptError {
  constructor(message: string) {
    super(message);
    this.name = 'ConflictError';
  }
}

/** Resolve a snapshot id to a loaded snapshot document, or `null` when unavailable. */
export type SnapshotLookup<T> = (snapshotId: string) => T | null | undefined;

/** One entry of the check list returned by {@link validateReceipt}. */
export interface ReceiptCheck {
  check: string;
  ok: boolean;
  message?: string;
  [detail: string]: unknown;
}

// --- building decision inputs from a snapshot ---------------------------------------------------

/** The `snapshot` block of `decision_inputs` for a loaded inputs snapshot. */
export function snapshotRef(
  inputsSnapshot: InputsSnapshot,
  options: { evidenceStatus?: EvidenceStatus; evidenceDetail?: string | null } = {},
): SnapshotRef {
  return {
    snapshot_id: inputsSnapshot.snapshot_id,
    mode: inputsSnapshot.mode,
    source_family: inputsSnapshot.source_family,
    season: inputsSnapshot.season,
    week: inputsSnapshot.week,
    model_version: inputsSnapshot.model_version,
    feature_version: inputsSnapshot.feature_version ?? null,
    candidate_by_position: { ...inputsSnapshot.candidate_by_position },
    scoring_format: inputsSnapshot.scoring_format,
    inputs_sha256: contentId(inputsSnapshot),
    data_cutoff: inputsSnapshot.data_cutoff ?? null,
    generated_at_utc: inputsSnapshot.generated_at_utc ?? null,
    publication: { ...inputsSnapshot.publication },
    evidence_status: options.evidenceStatus ?? 'verified',
    evidence_detail: options.evidenceDetail ?? null,
    baseline_reconciled: inputsSnapshot.baseline ? (inputsSnapshot.baseline.reconciled ?? null) : null,
  };
}

const DEFAULT_BASIS: Record<Availability, AvailabilityBasis> = {
  realized_stats_row: 'hindsight_stats_row',
  unknown: 'not_verified',
  assumed_available: 'user_assumed',
  unavailable: 'user_excluded',
};

/** Per-player availability overrides (each field optional; missing fields keep the default). */
export type AvailabilityOverride = Partial<{
  availability: Availability;
  availability_basis: AvailabilityBasis;
}>;

/** Options for {@link buildDecisionInputs}. */
export interface BuildDecisionInputsOptions {
  slot: string;
  playerIds: string[];
  overrides?: Record<string, AvailabilityOverride> | null;
  parameters?: Partial<Parameters> | null;
  evidenceStatus?: EvidenceStatus;
  evidenceDetail?: string | null;
}

/**
 * Resolve nominated players against a snapshot into a normalized `decision_inputs/1.0` document.
 * Unknown player ids throw {@link ReceiptError}; the caller decides whether that is a UI error or
 * a HOLD. Availability starts from each row's `availability_default` and user overrides replace it.
 */
export function buildDecisionInputs(
  inputsSnapshot: InputsSnapshot,
  options: BuildDecisionInputsOptions,
): DecisionInputs {
  const overrides = options.overrides || {};
  const rows: Record<string, InputsSnapshot['rows'][number]> = {};
  for (let i = 0; i < inputsSnapshot.rows.length; i += 1) {
    rows[inputsSnapshot.rows[i].player_id] = inputsSnapshot.rows[i];
  }
  const alternatives: Alternative[] = [];
  for (let i = 0; i < options.playerIds.length; i += 1) {
    const pid = options.playerIds[i];
    if (!Object.prototype.hasOwnProperty.call(rows, pid)) {
      throw new ReceiptError(`player ${pid} is not in snapshot ${inputsSnapshot.snapshot_id}`);
    }
    const r = rows[pid];
    const ov = overrides[pid] || {};
    alternatives.push({
      player_id: pid,
      display_name: r.display_name ?? null,
      team: r.team ?? null,
      position: r.position,
      model_version: r.model_version,
      candidate: r.candidate,
      season: inputsSnapshot.season,
      week: inputsSnapshot.week,
      projection: r.projection ?? null,
      floor: r.floor ?? null,
      ceiling: r.ceiling ?? null,
      baseline: r.baseline ?? null,
      baseline_provenance: r.baseline_provenance || 'unknown',
      availability: ov.availability !== undefined ? ov.availability : r.availability_default,
      availability_basis:
        ov.availability_basis !== undefined
          ? ov.availability_basis
          : DEFAULT_BASIS[r.availability_default],
      source_row_ref: r.source_row_ref ?? null,
    });
  }
  const params: Parameters = { ...defaultParameters(), ...(options.parameters || {}) };
  const doc: DecisionInputs = {
    schema_version: 'decision_inputs/1.0',
    policy_version: POLICY_VERSION,
    snapshot: snapshotRef(inputsSnapshot, {
      evidenceStatus: options.evidenceStatus,
      evidenceDetail: options.evidenceDetail,
    }),
    slot: options.slot,
    alternatives,
    parameters: params,
  };
  return normalizeInputs(doc);
}

// --- receipts ----------------------------------------------------------------------------------------

function emptyAction(): ActionState {
  return {
    state: 'not_recorded',
    action_id: null,
    at_utc: null,
    chosen_player_id: null,
    kind: null,
    note: null,
  };
}

function emptyOutcome(): OutcomeState {
  return {
    state: 'not_attached',
    outcome_snapshot_id: null,
    outcome_id: null,
    attached_at_utc: null,
    metrics: null,
  };
}

/** Options for {@link newReceipt}. */
export interface NewReceiptOptions {
  createdAtUtc: string;
  caseId?: string | null;
  parentDecisionId?: string | null;
  predictionNote?: string | null;
}

/** Evaluate the policy and wrap inputs + result into a fresh receipt (no events yet). */
export function newReceipt(inputs: DecisionInputs, options: NewReceiptOptions): Receipt {
  const norm = normalizeInputs(inputs);
  validateDecisionInputsShape(norm);
  const result = evaluate(norm);
  validateDecisionResultShape(result);
  return {
    schema_version: 'decision_receipt/1.0',
    decision_id: result.decision_id,
    parent_decision_id: options.parentDecisionId ?? null,
    created_at_utc: options.createdAtUtc,
    case_id: options.caseId ?? null,
    prediction_note: options.predictionNote ?? null,
    inputs: norm,
    result,
    result_sha256: contentId(result),
    events: [],
    action: emptyAction(),
    outcome: emptyOutcome(),
    trust: { note: TRUST_NOTE },
  };
}

/** `sha256(canonical({decision_id, seq, event_type, at_utc, payload}))`. */
export function eventId(
  decisionId: string,
  seq: number,
  eventType: string,
  atUtc: string,
  payload: Record<string, unknown>,
): string {
  return contentId({
    decision_id: decisionId,
    seq,
    event_type: eventType,
    at_utc: atUtc,
    payload,
  });
}

/** Project the event list onto the `action` and `outcome` blocks. */
export function deriveState(receipt: Pick<Receipt, 'events'>): {
  action: ActionState;
  outcome: OutcomeState;
} {
  let action = emptyAction();
  let outcome = emptyOutcome();
  for (let i = 0; i < receipt.events.length; i += 1) {
    const ev = receipt.events[i];
    const p = ev.payload;
    if (ev.event_type === 'action_recorded') {
      action = {
        state: 'recorded',
        action_id: ev.event_id,
        at_utc: ev.at_utc,
        chosen_player_id: p.chosen_player_id as string,
        kind: p.kind as ActionKind,
        note: (p.note ?? null) as string | null,
      };
    } else if (ev.event_type === 'action_declined') {
      action = {
        state: 'declined',
        action_id: ev.event_id,
        at_utc: ev.at_utc,
        chosen_player_id: null,
        kind: null,
        note: (p.note ?? null) as string | null,
      };
    } else if (ev.event_type === 'outcome_attached') {
      outcome = {
        state: 'attached',
        outcome_snapshot_id: p.outcome_snapshot_id as string,
        outcome_id: p.outcome_id as string,
        attached_at_utc: ev.at_utc,
        metrics: p.metrics as OutcomeState['metrics'],
      };
    }
  }
  return { action, outcome };
}

/** Return a new receipt with the event appended (inputs and result untouched). */
export function appendEvent(
  receipt: Receipt,
  eventType: EventType,
  payload: Record<string, unknown>,
  atUtc: string,
): Receipt {
  if (SPEC.event_types.indexOf(eventType) < 0) {
    throw new ReceiptError(`unknown event type ${eventType}`);
  }
  if (
    ACTION_EVENTS.indexOf(eventType) >= 0 &&
    receipt.events.some((e) => ACTION_EVENTS.indexOf(e.event_type) >= 0)
  ) {
    throw new ReceiptError(
      'an action is already recorded for this decision; change an assumption to start a new decision',
    );
  }
  const seq = receipt.events.length + 1;
  const ev: ReceiptEvent = {
    event_id: eventId(receipt.decision_id, seq, eventType, atUtc, payload),
    seq,
    event_type: eventType,
    at_utc: atUtc,
    payload,
  };
  const out: Receipt = { ...receipt, events: [...receipt.events, ev] };
  const state = deriveState(out);
  out.action = state.action;
  out.outcome = state.outcome;
  return out;
}

/** Record the user's explicit choice (must be a nominated alternative). */
export function recordAction(
  receipt: Receipt,
  chosenPlayerId: string,
  kind: ActionKind,
  atUtc: string,
  note: string | null = null,
): Receipt {
  const nominated = receipt.result.ranking.map((r) => r.player_id);
  if (nominated.indexOf(chosenPlayerId) < 0) {
    throw new ReceiptError(`chosen player ${chosenPlayerId} was not among the nominated alternatives`);
  }
  if (SPEC.action_kinds.indexOf(kind) < 0) {
    throw new ReceiptError(`unknown action kind ${kind}`);
  }
  return appendEvent(
    receipt,
    'action_recorded',
    { chosen_player_id: chosenPlayerId, kind, note },
    atUtc,
  );
}

/** Record that the user declined to choose. */
export function declineAction(receipt: Receipt, atUtc: string, note: string | null = null): Receipt {
  return appendEvent(receipt, 'action_declined', { note }, atUtc);
}

/** `null` when the outcome snapshot belongs to this decision's snapshot, else the reason. */
export function outcomeCompatible(
  receipt: Pick<Receipt, 'inputs'>,
  outcomeSnapshot: Partial<OutcomeSnapshot>,
): string | null {
  const snap = receipt.inputs.snapshot;
  if (outcomeSnapshot.snapshot_id !== snap.snapshot_id) {
    return `outcome snapshot ${String(outcomeSnapshot.snapshot_id)} is not ${snap.snapshot_id}`;
  }
  if (outcomeSnapshot.inputs_content_sha256 !== snap.inputs_sha256) {
    return 'outcome snapshot pairs with a different inputs snapshot content';
  }
  if (outcomeSnapshot.season !== snap.season || outcomeSnapshot.week !== snap.week) {
    return 'outcome snapshot is for a different season/week';
  }
  if (outcomeSnapshot.model_version !== snap.model_version) {
    return 'outcome snapshot is for a different model version';
  }
  if (outcomeSnapshot.scoring_format !== snap.scoring_format) {
    return 'outcome snapshot uses a different scoring format';
  }
  return null;
}

/** Append an `outcome_attached` event with metrics computed from the outcome snapshot. */
export function attachOutcome(receipt: Receipt, outcomeSnapshot: OutcomeSnapshot, atUtc: string): Receipt {
  try {
    validateOutcomeSnapshotShape(outcomeSnapshot);
  } catch (e) {
    if (e instanceof ContractError) {
      throw new ReceiptError(`cannot attach outcomes: ${e.message}`);
    }
    throw e;
  }
  const why = outcomeCompatible(receipt, outcomeSnapshot);
  if (why) {
    throw new ReceiptError(`cannot attach outcomes: ${why}`);
  }
  const oid = contentId(outcomeSnapshot);
  const metrics = computeMetrics(receipt.result, receipt.action, { ...outcomeSnapshot, outcome_id: oid });
  return appendEvent(
    receipt,
    'outcome_attached',
    { outcome_snapshot_id: outcomeSnapshot.snapshot_id, outcome_id: oid, metrics },
    atUtc,
  );
}

// --- validation, replay, merge -------------------------------------------------------------------

/** Options for {@link validateReceipt}. */
export interface ValidateReceiptOptions {
  /** Resolve the decision's inputs snapshot to verify its content digest and row values. */
  inputsLookup?: SnapshotLookup<InputsSnapshot> | null;
  /** Resolve the outcome snapshot to recompute attached metrics. */
  outcomeLookup?: SnapshotLookup<OutcomeSnapshot> | null;
}

/**
 * Every check a replay performs, in order: `schema`, `inputs_normalized`, `inputs_contract`,
 * `decision_id`, `result_sha256`, `result_replay`, `event_sequence`, `event_id`, `single_action`,
 * `events`, `action_projection`, `outcome_projection`, `chosen_membership`, `projections`, then
 * with lookups `snapshot_available`, `snapshot_digest`, `alternative_membership`,
 * `alternative_values`, `outcome_snapshot`, `outcome_available`, `outcome_digest`,
 * `outcome_compatible`, `metrics_replay`.
 *
 * Returns the check list; throws {@link ReceiptError} (with `.check` set) on the first failure.
 */
export function validateReceipt(receipt: unknown, options: ValidateReceiptOptions = {}): ReceiptCheck[] {
  const checks: ReceiptCheck[] = [];
  const ok = (name: string, detail: Record<string, unknown> = {}): void => {
    checks.push({ check: name, ok: true, ...detail });
  };
  const fail = (name: string, message: string): never => {
    checks.push({ check: name, ok: false, message });
    throw new ReceiptError(`${name}: ${message}`, name, checks);
  };

  try {
    validateShape('receipt', receipt);
  } catch (e) {
    fail('schema', e instanceof Error ? e.message : String(e));
  }
  const r = receipt as Receipt;
  ok('schema', { schema_version: r.schema_version });

  const norm = normalizeInputs(r.inputs);
  if (!deepEqual(norm, r.inputs)) {
    fail('inputs_normalized', 'alternatives are not in canonical order or versions differ');
  }
  try {
    validateDecisionInputsShape(norm);
  } catch (e) {
    fail('inputs_contract', e instanceof Error ? e.message : String(e));
  }
  const did = contentId(norm);
  if (did !== r.decision_id) {
    fail('decision_id', `recomputed ${did.slice(0, 12)} != stored ${r.decision_id.slice(0, 12)}`);
  }
  ok('decision_id', { decision_id: did });

  const recomputed = evaluate(norm);
  if (contentId(recomputed) !== r.result_sha256) {
    fail('result_sha256', 'stored result digest does not match a fresh evaluation');
  }
  if (!deepEqual(recomputed, r.result)) {
    fail('result_replay', 'stored result differs from a fresh evaluation (tampered or stale)');
  }
  ok('result_replay', {
    status: recomputed.status,
    recommended_player_id: recomputed.recommended_player_id,
  });

  for (let i = 0; i < r.events.length; i += 1) {
    const ev = r.events[i];
    if (ev.seq !== i + 1) {
      fail('event_sequence', `event ${i + 1} has seq ${ev.seq}`);
    }
    const eid = eventId(r.decision_id, ev.seq, ev.event_type, ev.at_utc, ev.payload);
    if (eid !== ev.event_id) {
      fail('event_id', `event ${i + 1} digest mismatch`);
    }
  }
  if (r.events.filter((e) => ACTION_EVENTS.indexOf(e.event_type) >= 0).length > 1) {
    fail('single_action', 'more than one action event');
  }
  ok('events', { n: r.events.length });

  const { action, outcome } = deriveState(r);
  if (!deepEqual(action, r.action)) {
    fail('action_projection', 'action block does not match the events');
  }
  if (!deepEqual(outcome, r.outcome)) {
    fail('outcome_projection', 'outcome block does not match the events');
  }
  const nominated = recomputed.ranking.map((row) => row.player_id);
  if (
    action.state === 'recorded' &&
    (action.chosen_player_id === null || nominated.indexOf(action.chosen_player_id) < 0)
  ) {
    fail('chosen_membership', 'chosen player is not a nominated alternative');
  }
  ok('projections', { action_state: action.state, outcome_state: outcome.state });

  if (options.inputsLookup) {
    const sid = r.inputs.snapshot.snapshot_id;
    const snap = options.inputsLookup(sid);
    if (!snap) {
      fail('snapshot_available', `snapshot ${sid} is not in the bundle`);
    }
    const s = snap as InputsSnapshot;
    if (contentId(s) !== r.inputs.snapshot.inputs_sha256) {
      fail('snapshot_digest', "the bundle's inputs snapshot has a different content digest");
    }
    const rows: Record<string, Record<string, unknown>> = {};
    for (let i = 0; i < s.rows.length; i += 1) {
      rows[s.rows[i].player_id] = s.rows[i] as unknown as Record<string, unknown>;
    }
    const keys = ['projection', 'floor', 'ceiling', 'baseline', 'position'];
    for (let i = 0; i < norm.alternatives.length; i += 1) {
      const a = norm.alternatives[i] as unknown as Record<string, unknown>;
      const pid = a.player_id as string;
      const row = Object.prototype.hasOwnProperty.call(rows, pid) ? rows[pid] : undefined;
      if (!row) {
        fail('alternative_membership', `${pid} is not in the snapshot`);
      }
      for (let k = 0; k < keys.length; k += 1) {
        if (!deepEqual((row as Record<string, unknown>)[keys[k]] ?? null, a[keys[k]] ?? null)) {
          fail('alternative_values', `${pid}.${keys[k]} differs from the snapshot row`);
        }
      }
    }
    ok('snapshot_digest', { snapshot_id: s.snapshot_id });
  }

  if (outcome.state === 'attached') {
    const attachEvents = r.events.filter((e) => e.event_type === 'outcome_attached');
    const ev = attachEvents[attachEvents.length - 1];
    if (ev.payload.outcome_snapshot_id !== r.inputs.snapshot.snapshot_id) {
      fail('outcome_snapshot', 'outcome attached from a different snapshot');
    }
    if (options.outcomeLookup) {
      const osnap = options.outcomeLookup(ev.payload.outcome_snapshot_id as string);
      if (!osnap) {
        fail('outcome_available', 'outcome snapshot is not in the bundle');
      }
      const o = osnap as OutcomeSnapshot;
      const oid = contentId(o);
      if (oid !== ev.payload.outcome_id) {
        fail('outcome_digest', "the bundle's outcome snapshot has a different content digest");
      }
      const why = outcomeCompatible(r, o);
      if (why) {
        fail('outcome_compatible', why);
      }
      // Metrics are recomputed with the action state as of the attach event.
      const partial = { events: r.events.filter((e) => e.seq < ev.seq) };
      const act = deriveState(partial).action;
      const m = computeMetrics(recomputed, act, { ...o, outcome_id: oid });
      if (!deepEqual(m, ev.payload.metrics)) {
        fail('metrics_replay', 'stored outcome metrics differ from a fresh computation');
      }
      ok('metrics_replay', { outcome_id: oid });
    }
  }
  return checks;
}

/**
 * Idempotent import: identical receipts merge to one; the same decision with a superset of events
 * is accepted (the longer event list wins); anything else with the same `decision_id` throws
 * {@link ConflictError}.
 */
export function merge(existing: Receipt, incoming: Receipt): Receipt {
  if (existing.decision_id !== incoming.decision_id) {
    throw new ConflictError('receipts have different decision ids');
  }
  if (!deepEqual(existing.inputs, incoming.inputs) || existing.result_sha256 !== incoming.result_sha256) {
    throw new ConflictError('same decision_id but different inputs or result');
  }
  const a = existing.events;
  const b = incoming.events;
  const longerIsExisting = a.length > b.length;
  const shorter = longerIsExisting ? b : a;
  const longer = longerIsExisting ? a : b;
  for (let i = 0; i < shorter.length; i += 1) {
    if (shorter[i].event_id !== longer[i].event_id) {
      throw new ConflictError(`event ${shorter[i].seq} differs between the two receipts`);
    }
  }
  const out: Receipt = { ...(longerIsExisting ? existing : incoming), events: longer.slice() };
  const state = deriveState(out);
  out.action = state.action;
  out.outcome = state.outcome;
  return out;
}
