import { describe, expect, it } from 'vitest';

import { contentId } from '../canonical';
import {
  CHOICE_MAX,
  CHOICE_MIN,
  TOL,
  decisionId,
  defaultParameters,
  evaluate,
  normalizeInputs,
  renderReason,
  validateParameters,
} from '../policy';
import { POLICY_VERSION } from '../spec';
import type { DecisionInputs } from '../types';
import { clone, loadGolden } from './golden';

const golden = loadGolden();

function goldenCase(name: string) {
  const c = golden.policy.find((p) => p.name === name);
  if (!c) throw new Error(`no golden case ${name}`);
  return c;
}

describe('policy parity with the Python reference', () => {
  it('covers the expected number of golden cases', () => {
    expect(golden.policy.length).toBe(36);
    expect(golden.policy_version).toBe(POLICY_VERSION);
  });

  for (const c of golden.policy) {
    it(`evaluate(${c.name}) deep-equals the golden result`, () => {
      const result = evaluate(clone(c.inputs));
      expect(result).toEqual(c.expected);
      // Reason order and explanation strings are part of the contract.
      expect(result.reasons.map((r) => r.code)).toEqual(c.expected.reasons.map((r) => r.code));
      expect(result.explanation).toEqual(c.expected.explanation);
      expect(result.decision_id).toBe(c.expected.decision_id);
      expect(contentId(result)).toBe(contentId(c.expected));
    });
  }

  it('does not mutate its input', () => {
    const c = goldenCase('ambiguity_floor_8');
    const inputs = clone(c.inputs);
    evaluate(inputs);
    expect(inputs).toEqual(c.inputs);
  });
});

describe('policy invariants', () => {
  it('is invariant to alternative order (same decision_id and result)', () => {
    const a = goldenCase('ambiguity_resolved_gap_0_3');
    const b = goldenCase('order_shuffled_same_result');
    expect(a.expected.decision_id).toBe(b.expected.decision_id);
    expect(evaluate(clone(a.inputs))).toEqual(evaluate(clone(b.inputs)));
    const shuffled = clone(a.inputs);
    shuffled.alternatives.reverse();
    expect(evaluate(shuffled)).toEqual(a.expected);
    expect(decisionId(shuffled)).toBe(a.expected.decision_id);
  });

  it('ambiguity regression: review at floor 8 and 5 with gap 0.5; recommend SYN-A at gap 0.3', () => {
    const floor8 = evaluate(clone(goldenCase('ambiguity_floor_8').inputs));
    expect(floor8.status).toBe('review');
    expect(floor8.recommended_player_id).toBeNull();
    expect(floor8.reasons.map((r) => r.code)).toEqual([
      'gap_below_minimum',
      'floor_below_minimum',
      'baseline_disagrees',
    ]);

    const floor5 = evaluate(clone(goldenCase('ambiguity_floor_5_gap_unchanged').inputs));
    expect(floor5.status).toBe('review');
    expect(floor5.gates.floor_ok).toBe(true);
    expect(floor5.gates.gap_ok).toBe(false);

    const gap03 = evaluate(clone(goldenCase('ambiguity_resolved_gap_0_3').inputs));
    expect(gap03.status).toBe('recommend');
    expect(gap03.recommended_player_id).toBe('SYN-A');
    expect(gap03.baseline.preferred_player_id).toBe('SYN-B');
    expect(gap03.baseline.agrees_with_model_leader).toBe(false);
    expect(gap03.reasons[0].code).toBe('recommend_highest_projection');
    expect(gap03.explanation[0]).toBe(
      'Synthetic A (SYN-A) has the highest PPR projection (14), 0.4 ahead of Synthetic B (SYN-B), conditional on the recorded eligibility assumptions.',
    );
  });

  it('gap and floor equality qualify; just-below fails', () => {
    expect(evaluate(clone(goldenCase('gap_equality_qualifies').inputs)).status).toBe('recommend');
    expect(evaluate(clone(goldenCase('floor_equality_qualifies').inputs)).status).toBe('recommend');
    expect(evaluate(clone(goldenCase('floor_just_below_fails').inputs)).status).toBe('review');
  });

  it('a tie is review even with a zero minimum gap', () => {
    const r = evaluate(clone(goldenCase('exact_tie_gap_zero').inputs));
    expect(r.status).toBe('review');
    expect(r.ranking.find((x) => x.rank === 1)?.tied_with_next).toBe(true);
    expect(r.leading.gap).toBe(0);
  });

  it('invalid evidence holds even when the gap would qualify', () => {
    const r = evaluate(clone(goldenCase('evidence_corrupt_hold_despite_low_gap').inputs));
    expect(r.status).toBe('hold');
    expect(r.recommended_player_id).toBeNull();
    expect(r.explanation[0]).toBe('Snapshot evidence failed its integrity check: byte digest mismatch.');
  });

  it('throws TypeError for a non-object', () => {
    expect(() => evaluate(null as unknown as DecisionInputs)).toThrow(TypeError);
    expect(() => evaluate([] as unknown as DecisionInputs)).toThrow(TypeError);
  });

  it('treats an empty alternatives list as review with no leader', () => {
    const r = evaluate(clone(goldenCase('empty_choice_set').inputs));
    expect(r.status).toBe('review');
    expect(r.counts).toEqual({ nominated: 0, excluded: 0, comparable: 0 });
    expect(r.leading).toEqual({ first: null, second: null, gap: null });
  });
});

describe('policy helpers', () => {
  it('exposes spec constants', () => {
    expect(TOL).toBe(1e-6);
    expect(CHOICE_MIN).toBe(2);
    expect(CHOICE_MAX).toBe(8);
    expect(defaultParameters()).toEqual({
      min_projection_gap: 0.5,
      min_floor: null,
      require_independent_baseline: true,
      model_only_exploration: false,
    });
  });

  it('normalizeInputs sorts alternatives by player id and pins versions', () => {
    const inputs = clone(goldenCase('three_way_ranking_display_tie').inputs);
    inputs.policy_version = 'stale';
    const shuffled = { ...inputs, alternatives: [...inputs.alternatives].reverse() };
    const norm = normalizeInputs(shuffled);
    expect(norm.alternatives.map((a) => a.player_id)).toEqual(['SYN-A', 'SYN-B', 'SYN-C']);
    expect(norm.policy_version).toBe(POLICY_VERSION);
    expect(norm.schema_version).toBe('decision_inputs/1.0');
  });

  it('renderReason formats yes/no, numbers, lists and null like Python', () => {
    expect(renderReason('projection_missing', { players: ['A', 'B'] })).toBe(
      'A, B have no finite PPR projection, so the comparison is incomplete.',
    );
    expect(renderReason('slot_unsupported', { value: null })).toBe('Slot — is not supported.');
    expect(renderReason('gap_meets_minimum', { gap: 0.1 + 0.2, threshold: 0 })).toBe(
      'The projection gap 0.3 meets the minimum 0.',
    );
    expect(renderReason('floor_guardrail_disabled', {})).toBe('No floor guardrail is enabled.');
    expect(() => renderReason('nope', {})).toThrow();
  });

  it('validateParameters messages match Python byte for byte', () => {
    expect(validateParameters({ min_projection_gap: -1, min_floor: null, require_independent_baseline: true, model_only_exploration: false })).toEqual([
      'min_projection_gap must be between 0 and 50',
    ]);
    expect(validateParameters({ min_projection_gap: 0.3, min_floor: '8', require_independent_baseline: true, model_only_exploration: false })).toEqual([
      'min_floor must be null or a finite number',
    ]);
    expect(validateParameters({ min_projection_gap: NaN, min_floor: 61, require_independent_baseline: 'yes', model_only_exploration: false, zeta: 1, alpha: 2 })).toEqual([
      'min_projection_gap must be a finite number',
      'min_floor must be between 0 and 60',
      'require_independent_baseline must be a boolean',
      'unknown parameter(s): alpha, zeta',
    ]);
  });
});
