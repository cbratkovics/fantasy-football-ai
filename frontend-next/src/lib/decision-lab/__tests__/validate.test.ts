import { describe, expect, it } from 'vitest';

import {
  ContractError,
  RECEIPT_MAX_BYTES,
  validateCasesShape,
  validateInputsSnapshotShape,
  validateManifestShape,
  validateOutcomeSnapshotShape,
  validateReceiptShape,
} from '../validate';
import { syntheticBundle } from './fixtures';
import { clone, loadGolden } from './golden';

const golden = loadGolden();
const G = golden.receipt;

describe('validateReceiptShape', () => {
  it('accepts every golden receipt and returns it', () => {
    for (const r of [G.receipt_new, G.receipt_after_action, G.receipt_after_outcome, G.receipt_declined]) {
      expect(validateReceiptShape(clone(r))).toEqual(r);
    }
  });

  it('rejects unknown keys at every level', () => {
    const top = clone(G.receipt_new) as unknown as Record<string, unknown>;
    top.extra = 1;
    expect(() => validateReceiptShape(top)).toThrow(/receipt: <root>: additional properties are not allowed \('extra'/);

    const snap = clone(G.receipt_new);
    (snap.inputs.snapshot as unknown as Record<string, unknown>).actual = 12;
    expect(() => validateReceiptShape(snap)).toThrow(/inputs\/snapshot: additional properties/);

    const alt = clone(G.receipt_new);
    (alt.inputs.alternatives[0] as unknown as Record<string, unknown>).regret = 0;
    expect(() => validateReceiptShape(alt)).toThrow(/inputs\/alternatives\/0: additional/);

    const res = clone(G.receipt_new);
    (res.result.gates as unknown as Record<string, unknown>).bonus = true;
    expect(() => validateReceiptShape(res)).toThrow(/result\/gates: additional/);

    const ev = clone(G.receipt_after_action);
    (ev.events[0] as unknown as Record<string, unknown>).signed = true;
    expect(() => validateReceiptShape(ev)).toThrow(/events\/0: additional/);

    const act = clone(G.receipt_after_action);
    (act.action as unknown as Record<string, unknown>).who = 'me';
    expect(() => validateReceiptShape(act)).toThrow(/action: additional/);

    const out = clone(G.receipt_after_outcome);
    (out.outcome as unknown as Record<string, unknown>).verified = true;
    expect(() => validateReceiptShape(out)).toThrow(/outcome: additional/);
  });

  it('rejects unknown enum values', () => {
    const status = clone(G.receipt_new);
    (status.result as unknown as Record<string, unknown>).status = 'maybe';
    expect(() => validateReceiptShape(status)).toThrow(/result\/status/);

    const sev = clone(G.receipt_new);
    (sev.result.reasons[0] as unknown as Record<string, unknown>).severity = 'fatal';
    expect(() => validateReceiptShape(sev)).toThrow(/reasons\/0\/severity/);

    const avail = clone(G.receipt_new);
    (avail.inputs.alternatives[0] as unknown as Record<string, unknown>).availability = 'probably';
    expect(() => validateReceiptShape(avail)).toThrow(/availability/);

    const prov = clone(G.receipt_new);
    (prov.inputs.alternatives[0] as unknown as Record<string, unknown>).baseline_provenance = 'vibes';
    expect(() => validateReceiptShape(prov)).toThrow(/baseline_provenance/);

    const kind = clone(G.receipt_after_action);
    (kind.action as unknown as Record<string, unknown>).kind = 'real';
    expect(() => validateReceiptShape(kind)).toThrow(/action\/kind/);

    const evt = clone(G.receipt_after_action);
    (evt.events[0] as unknown as Record<string, unknown>).event_type = 'note';
    expect(() => validateReceiptShape(evt)).toThrow(/events\/0\/event_type/);

    const lim = clone(G.receipt_new);
    lim.result.limitations.push('made_up');
    expect(() => validateReceiptShape(lim)).toThrow(/limitations\/3/);
  });

  it('rejects a missing required-but-nullable field and wrong types', () => {
    const missing = clone(G.receipt_new) as unknown as { result: Record<string, unknown> };
    delete missing.result.recommended_player_id;
    expect(() => validateReceiptShape(missing)).toThrow(/'recommended_player_id' is a required property/);

    const nullOk = clone(G.receipt_new);
    nullOk.result.status = 'review';
    nullOk.result.recommended_player_id = null;
    expect(() => validateReceiptShape(nullOk)).not.toThrow();

    const noAction = clone(G.receipt_new) as unknown as Record<string, unknown>;
    delete noAction.action;
    expect(() => validateReceiptShape(noAction)).toThrow(/'action' is a required property/);

    const wrongType = clone(G.receipt_new);
    (wrongType.inputs.alternatives[0] as unknown as Record<string, unknown>).projection = '14';
    expect(() => validateReceiptShape(wrongType)).toThrow(/projection: "14" is not of type 'number', 'null'/);

    const nonFinite = clone(G.receipt_new);
    nonFinite.inputs.alternatives[0].projection = Number.NaN;
    expect(() => validateReceiptShape(nonFinite)).toThrow(ContractError);

    const badSha = clone(G.receipt_new);
    badSha.decision_id = 'XYZ';
    expect(() => validateReceiptShape(badSha)).toThrow(/decision_id: "XYZ" does not match/);

    const badTime = clone(G.receipt_new);
    badTime.created_at_utc = 'yesterday';
    expect(() => validateReceiptShape(badTime)).toThrow(/created_at_utc/);

    const fraction = clone(G.receipt_new);
    (fraction.inputs.snapshot as unknown as Record<string, unknown>).week = 1.5;
    expect(() => validateReceiptShape(fraction)).toThrow(/week: 1.5 is not of type 'integer'/);

    expect(() => validateReceiptShape(null)).toThrow(ContractError);
    expect(() => validateReceiptShape([])).toThrow(ContractError);
    expect(() => validateReceiptShape('{}')).toThrow(ContractError);
  });

  it('rejects an inconsistent result', () => {
    const r = clone(G.receipt_new);
    r.result.status = 'review';
    expect(() => validateReceiptShape(r)).toThrow(/review carries a recommended_player_id/);
    const s = clone(G.receipt_new);
    s.result.recommended_player_id = null;
    expect(() => validateReceiptShape(s)).toThrow(/recommend without recommended_player_id/);
    const t = clone(G.receipt_new);
    t.result.recommended_player_id = 'SYN-Z';
    expect(() => validateReceiptShape(t)).toThrow(/recommended_player_id not among the alternatives/);
  });

  it('rejects oversized payloads and too many events', () => {
    const big = clone(G.receipt_new);
    big.prediction_note = 'x'.repeat(RECEIPT_MAX_BYTES + 1);
    expect(() => validateReceiptShape(big)).toThrow(/exceeds the 524288-byte limit/);
    expect(() => validateReceiptShape(G.receipt_new, 'y'.repeat(RECEIPT_MAX_BYTES + 1))).toThrow(/exceeds/);

    const many = clone(G.receipt_after_action);
    const ev = many.events[0];
    for (let i = 2; i <= 257; i += 1) {
      many.events.push({ ...ev, seq: i, event_type: 'prediction_note' });
    }
    expect(() => validateReceiptShape(many)).toThrow(/events: array has 257 items; the maximum is 256/);
  });
});

describe('snapshot, cases and manifest contracts', () => {
  const bundle = syntheticBundle();

  it('accept the synthetic bundle', () => {
    expect(validateInputsSnapshotShape(clone(bundle.inputs))).toEqual(bundle.inputs);
    expect(validateOutcomeSnapshotShape(clone(bundle.outcomes), bundle.inputs)).toEqual(bundle.outcomes);
    expect(validateCasesShape(clone(bundle.cases))).toEqual(bundle.cases);
    expect(validateManifestShape(clone(bundle.manifest))).toEqual(bundle.manifest);
  });

  it('inputs snapshot: unique ids, n_rows, ppr, forbidden outcome fields, ordering', () => {
    const dup = clone(bundle.inputs);
    dup.rows[1].player_id = 'SYN-A';
    expect(() => validateInputsSnapshotShape(dup)).toThrow(/duplicate player_id SYN-A/);

    const n = clone(bundle.inputs);
    n.population.n_rows = 2;
    expect(() => validateInputsSnapshotShape(n)).toThrow(/population.n_rows != len\(rows\)/);

    const half = clone(bundle.inputs);
    half.scoring_format = 'half_ppr';
    expect(() => validateInputsSnapshotShape(half)).toThrow(/scoring_format must be ppr/);

    const leak = clone(bundle.inputs);
    (leak.rows[0] as unknown as Record<string, unknown>).actual = 20;
    expect(() => validateInputsSnapshotShape(leak)).toThrow(ContractError);

    const floor = clone(bundle.inputs);
    floor.rows[0].floor = 99;
    expect(() => validateInputsSnapshotShape(floor)).toThrow(/floor above projection/);

    const cand = clone(bundle.inputs);
    cand.rows[0].candidate = 'xgb';
    expect(() => validateInputsSnapshotShape(cand)).toThrow(/candidate xgb != rf/);

    const model = clone(bundle.inputs);
    model.rows[0].model_version = 'other';
    expect(() => validateInputsSnapshotShape(model)).toThrow(/model_version differs/);

    const pos = clone(bundle.inputs);
    pos.rows[0].position = 'K';
    expect(() => validateInputsSnapshotShape(pos)).toThrow(/unsupported position K/);
  });

  it('outcome snapshot: coverage counts, sources, compatibility', () => {
    const cov = clone(bundle.outcomes);
    cov.coverage.observed = 3;
    expect(() => validateOutcomeSnapshotShape(cov)).toThrow(/coverage counts disagree/);

    const src = clone(bundle.outcomes);
    src.rows[0].actual_source = null;
    expect(() => validateOutcomeSnapshotShape(src)).toThrow(/observed actual without a source/);

    const extra = clone(bundle.outcomes);
    extra.rows[2].player_id = 'SYN-Z';
    expect(() => validateOutcomeSnapshotShape(extra, bundle.inputs)).toThrow(/players not in the inputs snapshot: \["SYN-Z"\]/);

    const digest = clone(bundle.outcomes);
    digest.inputs_content_sha256 = '0'.repeat(64);
    expect(() => validateOutcomeSnapshotShape(digest, bundle.inputs)).toThrow(/inputs_content_sha256 does not match/);

    const period = clone(bundle.outcomes);
    period.week = 2;
    expect(() => validateOutcomeSnapshotShape(period, bundle.inputs)).toThrow(/period or model differs/);
  });

  it('cases: duplicate ids, unknown override targets, bad case_id pattern', () => {
    const dup = clone(bundle.cases);
    dup.cases.push(clone(dup.cases[0]));
    expect(() => validateCasesShape(dup)).toThrow(/duplicate case_id/);

    const ov = clone(bundle.cases);
    ov.cases[0].overrides = { 'SYN-Z': { availability: 'unavailable', availability_basis: 'user_excluded' } };
    expect(() => validateCasesShape(ov)).toThrow(/override for a player not nominated: SYN-Z/);

    const id = clone(bundle.cases);
    id.cases[0].case_id = 'Bad_ID';
    expect(() => validateCasesShape(id)).toThrow(/case_id/);
  });

  it('manifest: duplicate snapshots, unknown latest id, schema_versions drift', () => {
    const dup = clone(bundle.manifest);
    dup.snapshots.push(clone(dup.snapshots[0]));
    expect(() => validateManifestShape(dup)).toThrow(/duplicate snapshot_id/);

    const latest = clone(bundle.manifest);
    latest.latest_weekly_snapshot_id = 'nope';
    expect(() => validateManifestShape(latest)).toThrow(/latest_weekly_snapshot_id is not a listed snapshot/);

    const drift = clone(bundle.manifest);
    drift.schema_versions.receipt = 'decision_receipt/2.0';
    expect(() => validateManifestShape(drift)).toThrow(/schema_versions differ/);

    const policy = clone(bundle.manifest);
    policy.policy_version = '9.9.9';
    expect(() => validateManifestShape(policy)).toThrow(/policy_version: "1.0.0" was expected/);
  });
});
