/**
 * Outcome metrics for a saved decision — the mirror of `ffai/decision_lab/metrics.py`.
 *
 * Observed outcomes never enter {@link evaluate}; they are joined here, after a decision exists,
 * to answer "what can we honestly conclude?". Every metric has its own eligibility rule and reports
 * why it is null instead of inventing a number:
 *
 * - `chosen_actual_points` — the explicitly chosen player's observed PPR points, or null with a
 *   reason (`no_action_recorded`, `declined`, `outcome_missing`).
 * - `choice_set_regret` — max observed actual over the comparable choice set minus the chosen
 *   player's actual; only when every member of the choice set has an observed outcome. A missing
 *   alternative leaves the regret unavailable — it is never imputed to zero.
 * - `points_vs_baseline_choice` — chosen actual minus the baseline-preferred player's actual.
 * - `model_policy_vs_baseline_choice` — the model policy's recommended player's actual minus the
 *   baseline-preferred player's actual; a hypothetical comparison between two comparators, not
 *   attributed to the user's action.
 *
 * Zero and negative actuals are valid observations.
 */
import { compareCodePoint, q } from './canonical';
import { SCHEMA_VERSIONS } from './spec';
import type { ActionState, DecisionResult, OutcomeMetrics, OutcomeRow } from './types';

/** `outcome_metrics/1.0`. */
export const METRICS_SCHEMA: string = SCHEMA_VERSIONS.outcome_metrics;

type Loose = Record<string, unknown>;

/** A possibly partial outcome row (only `player_id` and `actual` are read). */
export type OutcomeRowLike = Pick<OutcomeRow, 'player_id' | 'actual'> & Partial<OutcomeRow>;

/** A possibly partial outcome snapshot (only `rows`, `snapshot_id`, `outcome_id` are read). */
export interface OutcomeLike {
  rows: OutcomeRowLike[];
  snapshot_id?: string | null;
  outcome_id?: string | null;
  schema_version?: string;
}

/** A possibly partial action block (`state` defaults to `not_recorded`). */
export type ActionLike = Partial<ActionState>;

/** `{player_id: observed actual}` for the rows with an observed (numeric) outcome. */
export function outcomeIndex(outcomeSnapshot: OutcomeLike | Loose): Map<string, number> {
  const idx = new Map<string, number>();
  const rows = Array.isArray(outcomeSnapshot.rows) ? (outcomeSnapshot.rows as unknown as Loose[]) : [];
  for (let i = 0; i < rows.length; i += 1) {
    const actual = rows[i].actual;
    if (typeof actual === 'number') {
      idx.set(String(rows[i].player_id), q(actual) as number);
    }
  }
  return idx;
}

function lookup(idx: Map<string, number>, pid: string | null | undefined): number | null {
  if (pid === null || pid === undefined) {
    return null;
  }
  const v = idx.get(pid);
  return v === undefined ? null : v;
}

/** Join a policy result, the (possibly absent) human action, and an outcome snapshot. */
export function computeMetrics(
  result: DecisionResult,
  action: ActionLike,
  outcomeSnapshot: OutcomeLike,
): OutcomeMetrics {
  const actuals = outcomeIndex(outcomeSnapshot);
  const ranking = result.ranking || [];
  const choiceSet = ranking.filter((r) => r.comparable).map((r) => r.player_id);
  const nominated = ranking.map((r) => r.player_id);
  const observed = choiceSet.filter((pid) => actuals.has(pid));
  const missing = choiceSet.filter((pid) => !actuals.has(pid));
  const state: OutcomeMetrics['action_state'] = action.state ?? 'not_recorded';
  const chosen: string | null = state === 'recorded' ? (action.chosen_player_id ?? null) : null;

  const perAlt = nominated.map((pid) => ({
    player_id: pid,
    actual: lookup(actuals, pid),
    observed: actuals.has(pid),
  }));

  // chosen actual
  let chosenActual: number | null = null;
  let chosenReason: string | null;
  if (state === 'not_recorded') {
    chosenReason = 'no_action_recorded';
  } else if (state === 'declined') {
    chosenReason = 'declined';
  } else if (chosen === null || !actuals.has(chosen)) {
    chosenReason = 'outcome_missing';
  } else {
    chosenActual = actuals.get(chosen) as number;
    chosenReason = null;
  }

  // best in the comparable choice set (only meaningful when complete)
  let best: OutcomeMetrics['best_in_choice_set'] = null;
  let regret: number | null = null;
  let regretReason: string | null = null;
  if (!choiceSet.length) {
    regretReason = 'empty_choice_set';
  } else if (missing.length) {
    regretReason = 'incomplete_outcomes';
  } else {
    const bestPid = choiceSet
      .slice()
      .sort(
        (a, b) => (actuals.get(b) as number) - (actuals.get(a) as number) || compareCodePoint(a, b),
      )[0];
    best = { player_id: bestPid, actual: actuals.get(bestPid) as number };
    if (chosenActual === null) {
      regretReason = chosenReason;
    } else if (chosen === null || choiceSet.indexOf(chosen) < 0) {
      regretReason = 'chosen_outside_choice_set';
    } else {
      regret = q((actuals.get(bestPid) as number) - chosenActual);
    }
  }

  // baseline comparator
  const basePid: string | null = result.baseline ? result.baseline.preferred_player_id : null;
  const baseActual = basePid ? lookup(actuals, basePid) : null;
  let vsBaseline: number | null = null;
  let vsBaselineReason: string | null = null;
  if (basePid === null || basePid === undefined) {
    vsBaselineReason = 'no_baseline_preference';
  } else if (chosenActual === null) {
    vsBaselineReason = chosenReason;
  } else if (baseActual === null) {
    vsBaselineReason = 'baseline_outcome_missing';
  } else {
    vsBaseline = q(chosenActual - baseActual);
  }

  const modelPid: string | null = result.recommended_player_id ?? null;
  const modelActual = modelPid ? lookup(actuals, modelPid) : null;
  let modelVsBaseline: number | null = null;
  let modelVsBaselineReason: string | null = null;
  if (modelPid === null) {
    modelVsBaselineReason = 'no_model_recommendation';
  } else if (basePid === null) {
    modelVsBaselineReason = 'no_baseline_preference';
  } else if (modelActual === null || baseActual === null) {
    modelVsBaselineReason = 'outcome_missing';
  } else {
    modelVsBaseline = q(modelActual - baseActual);
  }

  return {
    schema_version: 'outcome_metrics/1.0',
    decision_id: result.decision_id ?? null,
    outcome_snapshot_id: outcomeSnapshot.snapshot_id ?? null,
    outcome_id: outcomeSnapshot.outcome_id ?? null,
    action_state: state,
    action_kind: state === 'recorded' ? (action.kind ?? null) : null,
    chosen_player_id: chosen,
    choice_set: choiceSet,
    coverage: {
      nominated: nominated.length,
      choice_set_size: choiceSet.length,
      observed: observed.length,
      missing,
    },
    chosen_actual_points: chosenActual,
    chosen_actual_reason: chosenReason,
    best_in_choice_set: best,
    choice_set_regret: regret,
    choice_set_regret_reason: regretReason,
    baseline_preferred_player_id: basePid,
    baseline_preferred_actual: baseActual,
    points_vs_baseline_choice: vsBaseline,
    points_vs_baseline_choice_reason: vsBaselineReason,
    model_recommended_player_id: modelPid,
    model_recommended_actual: modelActual,
    model_policy_vs_baseline_choice: modelVsBaseline,
    model_policy_vs_baseline_choice_reason: modelVsBaselineReason,
    per_alternative: perAlt,
    notes: [
      'A recommendation is not a recorded action; a recorded action is not proof it was carried out in a league.',
      'Hypothetical replay outcomes are not measured product impact.',
    ],
  };
}

/** `{n, mean}` over the rows where `key` is numeric; `mean` is null for a zero denominator. */
export interface MeanBlock {
  n: number;
  mean: number | null;
}

/** The result of {@link aggregate}. */
export interface AggregateMetrics {
  rows: number;
  with_recorded_action: number;
  chosen_actual_points: MeanBlock;
  choice_set_regret: MeanBlock;
  points_vs_baseline_choice: MeanBlock;
  model_policy_vs_baseline_choice: MeanBlock;
  model_beats_baseline: { n: number; wins: number; losses: number; ties: number };
}

/**
 * Aggregate several metric blocks with metric-specific denominators. Each mean uses only the rows
 * where that metric is non-null; zero denominators give `null` with the counts. Callers aggregate
 * the rows of one group at a time and never re-average group means.
 */
export function aggregate(metricRows: OutcomeMetrics[]): AggregateMetrics {
  const numeric = (key: keyof OutcomeMetrics): number[] => {
    const vals: number[] = [];
    for (let i = 0; i < metricRows.length; i += 1) {
      const v = metricRows[i][key];
      if (typeof v === 'number') {
        vals.push(v);
      }
    }
    return vals;
  };
  const mean = (key: keyof OutcomeMetrics): MeanBlock => {
    const vals = numeric(key);
    let sum = 0;
    for (let i = 0; i < vals.length; i += 1) {
      sum += vals[i];
    }
    return { n: vals.length, mean: vals.length ? q(sum / vals.length) : null };
  };
  const mvb = numeric('model_policy_vs_baseline_choice');
  return {
    rows: metricRows.length,
    with_recorded_action: metricRows.filter((m) => m.action_state === 'recorded').length,
    chosen_actual_points: mean('chosen_actual_points'),
    choice_set_regret: mean('choice_set_regret'),
    points_vs_baseline_choice: mean('points_vs_baseline_choice'),
    model_policy_vs_baseline_choice: mean('model_policy_vs_baseline_choice'),
    model_beats_baseline: {
      n: mvb.length,
      wins: mvb.filter((v) => v > 0).length,
      losses: mvb.filter((v) => v < 0).length,
      ties: mvb.filter((v) => v === 0).length,
    },
  };
}
