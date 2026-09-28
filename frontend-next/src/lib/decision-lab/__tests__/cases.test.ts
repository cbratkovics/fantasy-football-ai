import { describe, expect, it } from 'vitest';

import { caseInputs, experimentInputs } from '../cases';
import { evaluate } from '../policy';
import type { LabCase } from '../types';
import { syntheticBundle } from './fixtures';

const b = syntheticBundle();

function exclusionCase(): LabCase {
  return {
    ...b.cases.cases[0],
    case_id: 'syn-slot-and-exclusion',
    alternatives: ['SYN-A', 'SYN-B', 'SYN-C'],
    overrides: { 'SYN-B': { availability: 'unavailable', availability_basis: 'user_excluded' } },
    parameters: { min_projection_gap: 0.5, min_floor: null, require_independent_baseline: true, model_only_exploration: false },
    experiments: [
      { label: 'remove the exclusion', overrides: {}, expected: { status: 'recommend', recommended_player_id: 'SYN-A' } },
      { label: 'tighten the gap', parameters: { min_projection_gap: 5 }, expected: { status: 'review', recommended_player_id: null } },
    ],
  };
}

describe('caseInputs / experimentInputs', () => {
  it('caseInputs applies the stored overrides and parameters', () => {
    const c = exclusionCase();
    const inputs = caseInputs(c, b.inputs);
    expect(inputs.alternatives.map((a) => [a.player_id, a.availability])).toEqual([
      ['SYN-A', 'assumed_available'],
      ['SYN-B', 'unavailable'],
      ['SYN-C', 'assumed_available'],
    ]);
    const r = evaluate(inputs);
    expect(r.status).toBe('recommend'); // A vs C: 14 vs 9.5 clears the 0.5 gap
    expect(r.reasons.map((x) => x.code)).toContain('excluded_unavailable');
  });

  it('experiment overrides replace the case overrides; parameters merge', () => {
    const c = exclusionCase();
    const removed = experimentInputs(c, c.experiments[0], b.inputs);
    expect(removed.alternatives.every((a) => a.availability === 'assumed_available')).toBe(true);
    expect(removed.parameters).toEqual(c.parameters);
    expect(evaluate(removed).status).toBe('review'); // A vs B: 0.4 gap below 0.5
    const tightened = experimentInputs(c, c.experiments[1], b.inputs);
    expect(tightened.alternatives[1].availability).toBe('unavailable'); // overrides kept
    expect(tightened.parameters).toEqual({ ...c.parameters, min_projection_gap: 5 });
    expect(evaluate(tightened).status).toBe('review');
  });

  it('the synthetic bundle case reproduces its expected block', () => {
    const c = b.cases.cases[0];
    const r = evaluate(caseInputs(c, b.inputs));
    expect(r.status).toBe(c.expected.status);
    expect(r.recommended_player_id).toBe(c.expected.recommended_player_id);
    for (const code of c.expected.reason_codes) {
      expect(r.reasons.map((x) => x.code)).toContain(code);
    }
  });
});
