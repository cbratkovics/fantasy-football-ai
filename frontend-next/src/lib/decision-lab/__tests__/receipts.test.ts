import { describe, expect, it } from 'vitest';

import { contentId } from '../canonical';
import {
  ConflictError,
  IMMUTABLE_FIELDS,
  immutableContent,
  ReceiptError,
  TRUST_NOTE,
  attachOutcome,
  buildDecisionInputs,
  declineAction,
  deriveState,
  eventId,
  merge,
  newReceipt,
  outcomeCompatible,
  recordAction,
  snapshotRef,
  validateReceipt,
} from '../receipts';
import type { Receipt, ReceiptEvent } from '../types';
import { syntheticBundle } from './fixtures';
import { clone, loadGolden } from './golden';

const golden = loadGolden();
const G = golden.receipt;

const T0 = '2099-01-01T00:00:00+00:00';
const T1 = '2099-01-01T00:01:00+00:00';
const T2 = '2099-01-01T00:02:00+00:00';

function fresh(): Receipt {
  return newReceipt(clone(G.inputs), {
    createdAtUtc: T0,
    caseId: 'syn-ambiguity-floor-relaxation',
    predictionNote: 'I expect A',
  });
}

describe('receipt lifecycle parity with the Python reference', () => {
  it('TRUST_NOTE is identical', () => {
    expect(TRUST_NOTE).toBe(G.receipt_new.trust.note);
  });

  it('newReceipt deep-equals receipt_new', () => {
    expect(fresh()).toEqual(G.receipt_new);
  });

  it('recordAction deep-equals receipt_after_action (event ids agree)', () => {
    const r1 = recordAction(fresh(), 'SYN-A', 'hypothetical_replay', T1);
    expect(r1).toEqual(G.receipt_after_action);
    expect(r1.events[0].event_id).toBe(G.invariants.event_ids[0]);
  });

  it('attachOutcome deep-equals receipt_after_outcome (metrics agree)', () => {
    const r1 = recordAction(fresh(), 'SYN-A', 'hypothetical_replay', T1);
    const r2 = attachOutcome(r1, clone(G.outcome_snapshot), T2);
    expect(r2).toEqual(G.receipt_after_outcome);
    expect(r2.events.map((e) => e.event_id)).toEqual(G.invariants.event_ids);
    expect(r2.outcome.outcome_id).toBe(G.outcome_id);
    expect(contentId(G.outcome_snapshot)).toBe(G.outcome_id);
  });

  it('declineAction deep-equals receipt_declined', () => {
    expect(declineAction(fresh(), T1, 'skip')).toEqual(G.receipt_declined);
  });

  it('decision_id and result_sha256 never change across events', () => {
    const r0 = fresh();
    const r1 = recordAction(r0, 'SYN-A', 'hypothetical_replay', T1);
    const r2 = attachOutcome(r1, clone(G.outcome_snapshot), T2);
    expect(G.invariants.decision_id_unchanged).toBe(true);
    expect(G.invariants.result_sha_unchanged).toBe(true);
    expect(r1.decision_id).toBe(r0.decision_id);
    expect(r2.decision_id).toBe(r0.decision_id);
    expect(r2.result_sha256).toBe(r0.result_sha256);
    expect(r2.inputs).toEqual(r0.inputs);
    expect(r2.result).toEqual(r0.result);
    // Immutability: earlier receipts are untouched.
    expect(r0.events).toEqual([]);
    expect(r1.events.length).toBe(1);
  });

  it('eventId is the content id of the canonical event tuple', () => {
    const ev = G.receipt_after_action.events[0];
    expect(eventId(G.receipt_after_action.decision_id, ev.seq, ev.event_type, ev.at_utc, ev.payload)).toBe(
      ev.event_id,
    );
  });
});

describe('validateReceipt', () => {
  const lookups = {
    inputsLookup: () => null,
  };

  it('passes on all four golden receipts with the expected check names', () => {
    const names = (r: Receipt) => validateReceipt(r).map((c) => c.check);
    expect(names(G.receipt_new)).toEqual(['schema', 'decision_id', 'result_replay', 'events', 'projections']);
    expect(names(G.receipt_after_action)).toEqual(['schema', 'decision_id', 'result_replay', 'events', 'projections']);
    expect(names(G.receipt_after_outcome)).toEqual(['schema', 'decision_id', 'result_replay', 'events', 'projections']);
    expect(names(G.receipt_declined)).toEqual(['schema', 'decision_id', 'result_replay', 'events', 'projections']);
    expect(validateReceipt(G.receipt_after_outcome).every((c) => c.ok)).toBe(true);
  });

  it('recomputes attached metrics through an outcome lookup', () => {
    const checks = validateReceipt(G.receipt_after_outcome, {
      outcomeLookup: () => clone(G.outcome_snapshot),
    });
    expect(checks.map((c) => c.check)).toContain('metrics_replay');
    expect(() =>
      validateReceipt(G.receipt_after_outcome, {
        outcomeLookup: () => ({ ...clone(G.outcome_snapshot), observed_at_utc: '2099-01-02T00:00:00' }),
      }),
    ).toThrow(/outcome_digest/);
    expect(() => validateReceipt(G.receipt_after_outcome, { outcomeLookup: () => null })).toThrow(/outcome_available/);
  });

  function expectFailure(receipt: unknown, check: string): void {
    try {
      validateReceipt(receipt);
    } catch (e) {
      expect(e).toBeInstanceOf(ReceiptError);
      const err = e as ReceiptError;
      expect(err.check).toBe(check);
      expect(err.message.startsWith(`${check}: `)).toBe(true);
      expect(err.checks[err.checks.length - 1]).toMatchObject({ check, ok: false });
      return;
    }
    throw new Error(`expected ${check} to fail`);
  }

  it('fails on a status edit', () => {
    const r = clone(G.receipt_new);
    expect(r.result.status).toBe('recommend');
    r.result.status = 'review';
    r.result.recommended_player_id = null;
    // The stored digest still equals a fresh evaluation; the stored result no longer does.
    expectFailure(r, 'result_replay');
    const s = clone(G.receipt_new);
    s.result.recommended_player_id = 'SYN-B';
    expectFailure(s, 'result_replay');
    // Editing the digest itself is caught first.
    const t = clone(G.receipt_new);
    t.result_sha256 = 'd'.repeat(64);
    expectFailure(t, 'result_sha256');
  });

  it('fails on a projection edit', () => {
    const r = clone(G.receipt_new);
    r.inputs.alternatives[0].projection = 99;
    expectFailure(r, 'decision_id');
  });

  it('fails on an event id edit', () => {
    const r = clone(G.receipt_after_action);
    r.events[0].event_id = 'a'.repeat(64);
    expectFailure(r, 'event_id');
  });

  it('fails on a bad sequence', () => {
    const r = clone(G.receipt_after_action);
    r.events[0].seq = 2;
    expectFailure(r, 'event_sequence');
  });

  it('fails on a second action', () => {
    const r = clone(G.receipt_after_action);
    const payload = { note: 'again' };
    const ev: ReceiptEvent = {
      event_id: eventId(r.decision_id, 2, 'action_declined', T2, payload),
      seq: 2,
      event_type: 'action_declined',
      at_utc: T2,
      payload,
    };
    r.events.push(ev);
    const state = deriveState(r);
    r.action = state.action;
    expectFailure(r, 'single_action');
    expect(() => declineAction(G.receipt_after_action, T2)).toThrow(ReceiptError);
    expect(() => recordAction(G.receipt_declined, 'SYN-A', 'hypothetical_replay', T2)).toThrow(/already recorded/);
  });

  it('fails when the chosen player was not nominated', () => {
    const r = clone(G.receipt_new);
    const payload = { chosen_player_id: 'SYN-Z', kind: 'hypothetical_replay', note: null };
    r.events.push({
      event_id: eventId(r.decision_id, 1, 'action_recorded', T1, payload),
      seq: 1,
      event_type: 'action_recorded',
      at_utc: T1,
      payload,
    });
    r.action = deriveState(r).action;
    expectFailure(r, 'chosen_membership');
    expect(() => recordAction(G.receipt_new, 'SYN-Z', 'hypothetical_replay', T1)).toThrow(/not among the nominated/);
  });

  it('fails when the action block does not match the events', () => {
    const r = clone(G.receipt_after_action);
    r.action.chosen_player_id = 'SYN-B';
    expectFailure(r, 'action_projection');
  });

  it('fails when an outcome was attached from another snapshot', () => {
    const r = clone(G.receipt_after_action);
    const metrics = clone(G.receipt_after_outcome.outcome.metrics);
    const payload = { outcome_snapshot_id: 'other-snapshot', outcome_id: G.outcome_id, metrics };
    r.events.push({
      event_id: eventId(r.decision_id, 2, 'outcome_attached', T2, payload),
      seq: 2,
      event_type: 'outcome_attached',
      at_utc: T2,
      payload,
    });
    r.outcome = deriveState(r).outcome;
    expectFailure(r, 'outcome_snapshot');
  });

  it('fails on schema violations and un-normalized inputs', () => {
    const r = clone(G.receipt_new) as unknown as Record<string, unknown>;
    r.extra = 1;
    expectFailure(r, 'schema');
    const shuffled = clone(G.receipt_new);
    shuffled.inputs.alternatives.reverse();
    expectFailure(shuffled, 'inputs_normalized');
  });

  it('verifies the inputs snapshot digest and row values through a lookup', () => {
    const bundle = syntheticBundle();
    const inputs = buildDecisionInputs(bundle.inputs, { slot: 'RB', playerIds: ['SYN-A', 'SYN-B'] });
    const r = newReceipt(inputs, { createdAtUtc: T0 });
    const checks = validateReceipt(r, { inputsLookup: () => bundle.inputs });
    expect(checks.map((c) => c.check)).toContain('snapshot_digest');
    expect(() => validateReceipt(r, lookups)).toThrow(/snapshot_available/);
    const edited = clone(bundle.inputs);
    edited.rows[0].projection = 50;
    expect(() => validateReceipt(r, { inputsLookup: () => edited })).toThrow(/snapshot_digest/);
  });
});

describe('merge', () => {
  it('is idempotent and accepts a superset of events', () => {
    expect(merge(G.receipt_new, G.receipt_new)).toEqual(G.receipt_new);
    expect(merge(G.receipt_new, G.receipt_after_outcome)).toEqual(G.receipt_after_outcome);
    expect(merge(G.receipt_after_outcome, G.receipt_new)).toEqual(G.receipt_after_outcome);
    expect(merge(G.receipt_after_action, G.receipt_after_outcome)).toEqual(G.receipt_after_outcome);
  });

  it('rejects conflicting receipts', () => {
    expect(() => merge(G.receipt_after_action, G.receipt_declined)).toThrow(ConflictError);
    const other = clone(G.receipt_new);
    other.decision_id = 'b'.repeat(64);
    expect(() => merge(G.receipt_new, other)).toThrow(ConflictError);
    const tampered = clone(G.receipt_new);
    tampered.result_sha256 = 'c'.repeat(64);
    expect(() => merge(G.receipt_new, tampered)).toThrow(ConflictError);
  });
});

describe('buildDecisionInputs / snapshotRef / outcomeCompatible', () => {
  const bundle = syntheticBundle();

  it('resolves nominated players with defaults and overrides', () => {
    const inputs = buildDecisionInputs(bundle.inputs, {
      slot: 'FLEX',
      playerIds: ['SYN-C', 'SYN-A'],
      overrides: { 'SYN-C': { availability: 'unavailable', availability_basis: 'user_excluded' } },
      parameters: { min_floor: 5 },
    });
    expect(inputs.alternatives.map((a) => a.player_id)).toEqual(['SYN-A', 'SYN-C']);
    expect(inputs.alternatives[0]).toMatchObject({
      availability: 'assumed_available',
      availability_basis: 'user_assumed',
      season: 2099,
      week: 1,
    });
    expect(inputs.alternatives[1]).toMatchObject({ availability: 'unavailable', availability_basis: 'user_excluded' });
    expect(inputs.parameters).toEqual({
      min_projection_gap: 0.5,
      min_floor: 5,
      require_independent_baseline: true,
      model_only_exploration: false,
    });
    expect(inputs.snapshot.inputs_sha256).toBe(contentId(bundle.inputs));
    expect(inputs.snapshot).toEqual(snapshotRef(bundle.inputs));
    expect(inputs.snapshot.evidence_status).toBe('verified');
  });

  it('throws on an unknown player', () => {
    expect(() => buildDecisionInputs(bundle.inputs, { slot: 'RB', playerIds: ['SYN-Z'] })).toThrow(ReceiptError);
  });

  it('carries evidence status through to a HOLD', () => {
    const inputs = buildDecisionInputs(bundle.inputs, {
      slot: 'RB',
      playerIds: ['SYN-A', 'SYN-B'],
      evidenceStatus: 'corrupt',
      evidenceDetail: 'byte digest mismatch',
    });
    const r = newReceipt(inputs, { createdAtUtc: T0 });
    expect(r.result.status).toBe('hold');
    expect(r.result.reasons[0].code).toBe('evidence_corrupt');
  });

  it('outcomeCompatible names the mismatch', () => {
    const inputs = buildDecisionInputs(bundle.inputs, { slot: 'RB', playerIds: ['SYN-A', 'SYN-B'] });
    const r = newReceipt(inputs, { createdAtUtc: T0 });
    expect(outcomeCompatible(r, bundle.outcomes)).toBeNull();
    expect(outcomeCompatible(r, { ...bundle.outcomes, snapshot_id: 'x' })).toMatch(/is not syn-bundle/);
    expect(outcomeCompatible(r, { ...bundle.outcomes, inputs_content_sha256: '0'.repeat(64) })).toMatch(/different inputs snapshot content/);
    expect(outcomeCompatible(r, { ...bundle.outcomes, week: 2 })).toMatch(/season\/week/);
    expect(outcomeCompatible(r, { ...bundle.outcomes, model_version: 'v2' })).toMatch(/model version/);
    expect(outcomeCompatible(r, { ...bundle.outcomes, scoring_format: 'half' })).toMatch(/scoring format/);
    expect(() => attachOutcome(r, { ...bundle.outcomes, week: 2 }, T2)).toThrow(ReceiptError);
    const attached = attachOutcome(r, bundle.outcomes, T2);
    expect(attached.outcome.state).toBe('attached');
    expect(attached.outcome.metrics?.chosen_actual_reason).toBe('no_action_recorded');
    expect(attached.outcome.metrics?.model_recommended_player_id).toBeNull();
  });
});

describe('merge: complete immutable content', () => {
  const G2 = loadGolden().receipt;
  const cases: Array<[string, unknown]> = [
    ['created_at_utc', '2099-06-01T00:00:00+00:00'],
    ['prediction_note', 'a different note'],
    ['prediction_note', null],
    ['case_id', 'some-other-case'],
    ['case_id', null],
    ['parent_decision_id', '9'.repeat(64)],
    ['trust', { note: 'edited' }],
    ['schema_version', 'decision_receipt/0.9'],
  ];

  it.each(cases)('rejects a different %s in either direction and for any event count', (field, value) => {
    const base = G2.receipt_after_action;
    const other = clone(base) as unknown as Record<string, unknown>;
    other[field] = value;
    const changed = other as unknown as Receipt;
    const re = new RegExp(`different immutable content: ${field}$`);
    expect(() => merge(base, changed)).toThrow(re);
    expect(() => merge(changed, base)).toThrow(re);
    // Unequal event counts never rescue a metadata mismatch: the longer history does not win.
    const longer = clone(G2.receipt_after_outcome) as unknown as Record<string, unknown>;
    longer[field] = value;
    expect(() => merge(base, longer as unknown as Receipt)).toThrow(/different immutable content/);
    expect(() => merge(longer as unknown as Receipt, G2.receipt_new)).toThrow(/different immutable content/);
  });

  it('reports every differing field', () => {
    const other = clone(G2.receipt_new) as unknown as Record<string, unknown>;
    other.created_at_utc = '2099-06-01T00:00:00+00:00';
    other.prediction_note = 'changed';
    other.case_id = null;
    expect(() => merge(G2.receipt_new, other as unknown as Receipt)).toThrow(/content: created_at_utc, case_id, prediction_note$/);
  });

  it('keeps the shared metadata for identical imports and valid extensions', () => {
    const { receipt_new: r0, receipt_after_action: r1, receipt_after_outcome: r2 } = G2;
    const pairs: Array<[Receipt, Receipt]> = [[r0, r0], [r1, r1], [r0, r1], [r1, r0], [r1, r2], [r2, r0]];
    for (const [existing, incoming] of pairs) {
      const merged = merge(existing, incoming);
      expect(immutableContent(merged)).toEqual(immutableContent(existing));
      expect(merged.events.length).toBe(Math.max(existing.events.length, incoming.events.length));
    }
    expect(merge(r0, r2).prediction_note).toBe('I expect A');
    expect(merge(r0, r2).created_at_utc).toBe(r0.created_at_utc);
    expect(new Set([...IMMUTABLE_FIELDS, 'events', 'action', 'outcome'])).toEqual(new Set(Object.keys(r2)));
  });

  it('equal event counts with a diverging event is a conflict', () => {
    expect(G2.receipt_after_action.events.length).toBe(G2.receipt_declined.events.length);
    expect(() => merge(G2.receipt_after_action, G2.receipt_declined)).toThrow(/event 1 differs/);
  });
});
