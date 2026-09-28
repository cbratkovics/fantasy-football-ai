import { describe, expect, it } from 'vitest';

import { aggregate, computeMetrics, outcomeIndex } from '../metrics';
import type { OutcomeMetrics } from '../types';
import { clone, loadGolden } from './golden';

const golden = loadGolden();

describe('metrics parity with the Python reference', () => {
  it('has the expected number of golden cases', () => {
    expect(golden.metrics.length).toBe(8);
  });

  for (const c of golden.metrics) {
    it(`computeMetrics(${c.name}) deep-equals the golden block`, () => {
      const m = computeMetrics(clone(c.result), clone(c.action), clone(c.outcome_snapshot));
      expect(m).toEqual(c.expected);
    });
  }

  it('outcomeIndex quantizes and ignores null/non-numeric actuals', () => {
    const idx = outcomeIndex({
      rows: [
        { player_id: 'A', actual: 0.1 + 0.2, actual_source: 'synthetic' },
        { player_id: 'B', actual: null, actual_source: null },
      ],
    });
    expect(Array.from(idx.entries())).toEqual([['A', 0.3]]);
  });

  it('extreme outcomes never change the result they are joined to', () => {
    const c = golden.metrics.find((m) => m.name === 'extreme_outcomes_do_not_touch_result');
    if (!c) throw new Error('missing case');
    const result = clone(c.result);
    computeMetrics(result, c.action, c.outcome_snapshot);
    expect(result).toEqual(c.result);
  });
});

describe('aggregate', () => {
  const rows = golden.metrics.map((m) => m.expected);

  it('uses metric-specific denominators and null means for zero denominators', () => {
    const agg = aggregate(rows);
    expect(agg.rows).toBe(8);
    expect(agg.with_recorded_action).toBe(6);
    // Every case with a recorded action and observed chosen outcome contributes.
    expect(agg.chosen_actual_points.n).toBe(rows.filter((r) => typeof r.chosen_actual_points === 'number').length);
    expect(agg.model_beats_baseline.n).toBe(
      agg.model_beats_baseline.wins + agg.model_beats_baseline.losses + agg.model_beats_baseline.ties,
    );
    const empty = aggregate([]);
    expect(empty).toEqual({
      rows: 0,
      with_recorded_action: 0,
      chosen_actual_points: { n: 0, mean: null },
      choice_set_regret: { n: 0, mean: null },
      points_vs_baseline_choice: { n: 0, mean: null },
      model_policy_vs_baseline_choice: { n: 0, mean: null },
      model_beats_baseline: { n: 0, wins: 0, losses: 0, ties: 0 },
    });
  });

  it('only null metrics leave a zero denominator even when rows exist', () => {
    const nullRows: OutcomeMetrics[] = rows.map((r) => ({
      ...r,
      chosen_actual_points: null,
      choice_set_regret: null,
      points_vs_baseline_choice: null,
      model_policy_vs_baseline_choice: null,
    }));
    const agg = aggregate(nullRows);
    expect(agg.rows).toBe(8);
    expect(agg.chosen_actual_points).toEqual({ n: 0, mean: null });
    expect(agg.model_beats_baseline).toEqual({ n: 0, wins: 0, losses: 0, ties: 0 });
  });

  it('quantizes means', () => {
    const base = rows[0];
    const agg = aggregate([
      { ...base, choice_set_regret: 0.1 },
      { ...base, choice_set_regret: 0.2 },
      { ...base, choice_set_regret: 0.4 },
    ]);
    expect(agg.choice_set_regret).toEqual({ n: 3, mean: 0.233333 });
  });
});
