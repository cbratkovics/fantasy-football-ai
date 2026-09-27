/**
 * The reference decision policy — a line-by-line mirror of `ffai/decision_lab/policy.py`
 * (`policy_spec.json` `policy_version` 1.0.0).
 *
 * {@link evaluate} maps one `decision_inputs/1.0` document to one `decision_result/1.0` document.
 * It is a pure function of its argument: no clock, no file, no randomness, and it accepts only the
 * input schema — target-week actuals, error metrics, hindsight ranks and regret cannot reach it
 * because the input schema has no field for them.
 *
 * Precedence (each step keeps computing so the result carries the full evidence; the status is
 * the most severe reason found):
 *
 * 1. HOLD — invalid or incompatible evidence (corrupt/blocked snapshot, unsupported scoring format
 *    or slot, mixed periods or model versions, invalid parameters, too many alternatives,
 *    duplicates). Invalid input never becomes a weaker recommendation.
 * 2. Exclusions (unavailable, slot-ineligible) with visible reasons, then REVIEW for missing
 *    projections, unresolved availability, fewer than two comparable alternatives, or a missing
 *    independent baseline when one is required.
 * 3. Ranking by projection descending; ties broken by player id for display only and flagged.
 * 4. Gap before any floor gate; `|gap| <= tolerance` is a tie → REVIEW.
 * 5. `gap < min_projection_gap - tolerance` → REVIEW; equality qualifies.
 * 6. Floor guardrail (only when `min_floor` is not null); equality qualifies.
 * 7. Otherwise RECOMMEND the leading option, conditional on the recorded assumptions.
 *
 * Parity with Python is exact on the golden fixtures. The only representational differences are
 * for malformed inputs that the contracts reject anyway: Python renders a missing `season`/`week`
 * as `None` where JavaScript renders `null`, and Python prints a float-typed integer as `2099.0`
 * where JavaScript prints `2099` — both only inside the `mixed_periods` detail strings.
 */
import {
  CanonicalError,
  compareCodePoint,
  contentId,
  formatNumber,
  q,
  sortedKeys,
} from './canonical';
import { POLICY_VERSION, SCHEMA_VERSIONS, SPEC } from './spec';
import type { DecisionInputs, DecisionResult, Parameters, RankingRow, Reason, Severity } from './types';

/** Comparison tolerance (`policy_spec.json` → `tolerance`). */
export const TOL: number = SPEC.tolerance;
/** Slot → eligible positions. */
export const SLOTS: Record<string, string[]> = SPEC.slots;
/** Minimum comparable alternatives for a comparison. */
export const CHOICE_MIN: number = SPEC.choice_set.min;
/** Maximum nominated alternatives before HOLD. */
export const CHOICE_MAX: number = SPEC.choice_set.max;
/** Availability values that make an alternative comparable. */
export const COMPARABLE_AVAILABILITY: readonly string[] = SPEC.availability.comparable;
/** Baseline provenances that count as independent of the model. */
export const INDEPENDENT_BASELINE: readonly string[] = SPEC.independent_baseline_provenance;
/** Reason code → severity and template. */
export const REASONS = SPEC.reasons;
/** Limitation code → text. */
export const LIMITATIONS: Record<string, string> = SPEC.limitations;
/** Parameter specification (defaults and bounds). */
export const PARAM_SPEC = SPEC.parameters;
/** `decision_inputs/1.0`. */
export const INPUTS_SCHEMA: string = SCHEMA_VERSIONS.decision_inputs;
/** `decision_result/1.0`. */
export const RESULT_SCHEMA: string = SCHEMA_VERSIONS.decision_result;

type Loose = Record<string, unknown>;

/** The default policy parameters from the spec. */
export function defaultParameters(): Parameters {
  const out: Loose = {};
  const keys = Object.keys(PARAM_SPEC);
  for (let i = 0; i < keys.length; i += 1) {
    out[keys[i]] = PARAM_SPEC[keys[i]].default;
  }
  return out as unknown as Parameters;
}

/**
 * Python's `_fmt`: booleans → "yes"/"no", numbers via {@link formatNumber}, arrays joined with
 * ", ", null/undefined → "—", anything else via `String`.
 */
export function formatDetailValue(value: unknown): string {
  if (typeof value === 'boolean') {
    return value ? 'yes' : 'no';
  }
  if (typeof value === 'number') {
    return formatNumber(value);
  }
  if (Array.isArray(value)) {
    return value.map(formatDetailValue).join(', ');
  }
  if (value === null || value === undefined) {
    return '—';
  }
  return String(value);
}

/** Deterministic explanation text from a reason code and its measured detail values. */
export function renderReason(code: string, detail: Record<string, unknown>): string {
  const spec = REASONS[code];
  if (!spec) {
    throw new Error(`unknown reason code ${code}`);
  }
  let text = spec.template;
  const keys = Object.keys(detail);
  for (let i = 0; i < keys.length; i += 1) {
    const key = keys[i];
    text = text.split('{' + key + '}').join(formatDetailValue(detail[key]));
  }
  return text;
}

function reason(code: string, detail: Record<string, unknown> = {}): Reason {
  return { code, severity: REASONS[code].severity as Severity, detail };
}

function isNum(x: unknown): x is number {
  return typeof x === 'number';
}

/** `null` for non-numbers, NaN and ±Infinity; otherwise the value quantized to six decimals. */
export function finite(x: unknown): number | null {
  if (!isNum(x)) {
    return null;
  }
  try {
    return q(x);
  } catch (e) {
    if (e instanceof CanonicalError) {
      return null;
    }
    throw e;
  }
}

function label(alt: Loose): string {
  const name = alt.display_name;
  const pid = String(alt.player_id);
  return name ? `${String(name)} (${pid})` : pid;
}

function asObject(value: unknown): Loose {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Loose) : {};
}

/**
 * Alternatives sorted by `player_id` (code-point order) and versions pinned; the form that is
 * hashed. Other keys are copied through unchanged.
 */
export function normalizeInputs(inputs: DecisionInputs): DecisionInputs;
export function normalizeInputs(inputs: Loose): Loose;
export function normalizeInputs(inputs: DecisionInputs | Loose): DecisionInputs | Loose {
  const loose = inputs as Loose;
  const rawAlts = Array.isArray(loose.alternatives) ? (loose.alternatives as Loose[]) : [];
  const alts = rawAlts
    .map((a) => ({ ...a }))
    .sort((a, b) => compareCodePoint(String(a.player_id), String(b.player_id)));
  return {
    ...loose,
    schema_version: INPUTS_SCHEMA,
    policy_version: POLICY_VERSION,
    alternatives: alts,
  };
}

/**
 * The decision identity: sha256 of the canonical inputs (alternatives sorted by player id).
 * Covers schema and policy versions, the snapshot reference (including the inputs snapshot's
 * content digest), slot, every alternative row with its availability fields, and the parameters.
 * It does not cover outcomes, events, timestamps, or the computed result.
 */
export function decisionId(inputs: DecisionInputs | Loose): string {
  return contentId(normalizeInputs(inputs as Loose));
}

function pyNum(n: number | undefined): string {
  // Bounds in policy_spec.json are integers; Python prints them as `0`, `50`, `60`.
  return String(n);
}

/** Parameter problems, byte-identical to Python's `_validate_parameters` messages. */
export function validateParameters(params: Loose): string[] {
  const problems: string[] = [];
  const gap = params.min_projection_gap;
  if (finite(gap) === null) {
    problems.push('min_projection_gap must be a finite number');
  } else {
    const lo = PARAM_SPEC.min_projection_gap.min as number;
    const hi = PARAM_SPEC.min_projection_gap.max as number;
    const g = gap as number;
    if (!(lo <= g && g <= hi)) {
      problems.push(`min_projection_gap must be between ${pyNum(lo)} and ${pyNum(hi)}`);
    }
  }
  const floor = params.min_floor;
  if (floor !== null && floor !== undefined) {
    if (finite(floor) === null) {
      problems.push('min_floor must be null or a finite number');
    } else {
      const lo = PARAM_SPEC.min_floor.min as number;
      const hi = PARAM_SPEC.min_floor.max as number;
      const f = floor as number;
      if (!(lo <= f && f <= hi)) {
        problems.push(`min_floor must be between ${pyNum(lo)} and ${pyNum(hi)}`);
      }
    }
  }
  const flags = ['require_independent_baseline', 'model_only_exploration'];
  for (let i = 0; i < flags.length; i += 1) {
    if (typeof params[flags[i]] !== 'boolean') {
      problems.push(`${flags[i]} must be a boolean`);
    }
  }
  const unknown = sortedKeys(
    Object.keys(params).filter((k) => !Object.prototype.hasOwnProperty.call(PARAM_SPEC, k)),
  );
  if (unknown.length) {
    problems.push(`unknown parameter(s): ${unknown.join(', ')}`);
  }
  return problems;
}

function uniqueSorted(values: string[]): string[] {
  const seen: Record<string, true> = {};
  const out: string[] = [];
  for (let i = 0; i < values.length; i += 1) {
    if (!seen[values[i]]) {
      seen[values[i]] = true;
      out.push(values[i]);
    }
  }
  return sortedKeys(out);
}

/**
 * Apply the policy to one `decision_inputs/1.0` document. Pure; never throws on bad domain values
 * (they become HOLD reasons); throws `TypeError` only for a non-object.
 */
export function evaluate(inputs: DecisionInputs): DecisionResult {
  if (inputs === null || typeof inputs !== 'object' || Array.isArray(inputs)) {
    throw new TypeError('inputs must be an object');
  }
  const norm = normalizeInputs(inputs as unknown as Loose);
  const snapshot = asObject(norm.snapshot);
  const slot = norm.slot;
  const alts = norm.alternatives as Loose[];
  const params = asObject(norm.parameters);
  const reasons: Reason[] = [];
  const limitations: string[] = [];

  // --- 1. evidence validity → HOLD --------------------------------------------------------
  if (snapshot.evidence_status === 'corrupt') {
    reasons.push(reason('evidence_corrupt', { detail: snapshot.evidence_detail || '?' }));
  }
  const pub = asObject(snapshot.publication);
  if (pub.status === 'blocked') {
    reasons.push(reason('evidence_blocked', { detail: pub.note || 'blocked' }));
  }
  const fmt = snapshot.scoring_format;
  if (fmt !== SPEC.scoring_format) {
    reasons.push(reason('scoring_format_unsupported', { value: fmt === undefined ? null : fmt }));
  }
  const slotKnown = typeof slot === 'string' && Object.prototype.hasOwnProperty.call(SLOTS, slot);
  if (!slotKnown) {
    reasons.push(reason('slot_unsupported', { value: slot === undefined ? null : slot }));
  }
  const periods = uniqueSorted(
    alts.filter((a) => 'season' in a).map((a) => `${String(a.season)}-w${String(a.week)}`),
  );
  const snapPeriod = `${String(snapshot.season)}-w${String(snapshot.week)}`;
  if (periods.length && (periods.length > 1 || periods[0] !== snapPeriod)) {
    reasons.push(reason('mixed_periods', { detail: [...periods, `snapshot ${snapPeriod}`] }));
  }
  const versions = uniqueSorted(alts.map((a) => `${String(a.model_version)}/${String(a.candidate)}`));
  if (versions.length) {
    const cbp = asObject(snapshot.candidate_by_position);
    const expected: Record<string, true> = {};
    let expectedCount = 0;
    const cbpKeys = Object.keys(cbp);
    for (let i = 0; i < cbpKeys.length; i += 1) {
      const key = `${String(snapshot.model_version)}/${String(cbp[cbpKeys[i]])}`;
      if (!expected[key]) {
        expected[key] = true;
        expectedCount += 1;
      }
    }
    const modelVersions = uniqueSorted(versions.map((v) => v.split('/')[0]));
    const subset = versions.every((v) => expected[v] === true);
    if (modelVersions.length > 1 || (expectedCount > 0 && !subset)) {
      reasons.push(reason('mixed_model_versions', { detail: versions }));
    }
  }
  const paramProblems = validateParameters(params);
  if (paramProblems.length) {
    reasons.push(reason('parameters_invalid', { detail: paramProblems.join('; ') }));
  }
  if (alts.length > CHOICE_MAX) {
    reasons.push(reason('choice_set_too_large', { count: alts.length, limit: CHOICE_MAX }));
  }
  const seen: Record<string, true> = {};
  for (let i = 0; i < alts.length; i += 1) {
    const pid = String(alts[i].player_id);
    if (seen[pid]) {
      reasons.push(reason('duplicate_alternative', { player_id: pid }));
    }
    seen[pid] = true;
  }

  // --- 2. exclusions and choice-set checks → REVIEW ----------------------------------------
  const eligible = slotKnown ? SLOTS[slot as string] : [];
  const rankingRows: RankingRow[] = [];
  const comparable: RankingRow[] = [];
  const unresolved: string[] = [];
  const missingProjection: string[] = [];
  for (let i = 0; i < alts.length; i += 1) {
    const a = alts[i];
    const availability = a.availability;
    const row: RankingRow = {
      player_id: a.player_id as string,
      display_name: (a.display_name ?? null) as string | null,
      position: (a.position ?? null) as string | null,
      projection: finite(a.projection),
      floor: finite(a.floor),
      ceiling: finite(a.ceiling),
      baseline: finite(a.baseline),
      baseline_provenance: (a.baseline_provenance || 'unknown') as RankingRow['baseline_provenance'],
      availability: (availability ?? null) as string | null,
      availability_basis: (a.availability_basis ?? null) as string | null,
      comparable: false,
      exclusion: null,
      rank: null,
      tied_with_next: false,
    };
    if (availability === 'unavailable') {
      row.exclusion = 'excluded_unavailable';
      reasons.push(reason('excluded_unavailable', { player: label(a) }));
    } else if (typeof a.position !== 'string' || eligible.indexOf(a.position) < 0) {
      row.exclusion = 'excluded_slot_ineligible';
      reasons.push(
        reason('excluded_slot_ineligible', {
          player: label(a),
          position: a.position === undefined ? null : a.position,
          slot: slot === undefined ? null : slot,
        }),
      );
    } else {
      const comparableAvail =
        typeof availability === 'string' && COMPARABLE_AVAILABILITY.indexOf(availability) >= 0;
      if (row.projection === null) {
        missingProjection.push(label(a));
      }
      if (!comparableAvail) {
        unresolved.push(label(a));
      }
      if (row.projection !== null && comparableAvail) {
        row.comparable = true;
        comparable.push(row);
      }
      if (availability === 'assumed_available' && limitations.indexOf('availability_user_assumed') < 0) {
        limitations.push('availability_user_assumed');
      }
      if (
        availability === 'realized_stats_row' &&
        limitations.indexOf('population_hindsight_conditioned') < 0
      ) {
        limitations.push('population_hindsight_conditioned');
      }
    }
    rankingRows.push(row);
  }
  if (missingProjection.length) {
    reasons.push(reason('projection_missing', { players: missingProjection }));
  }
  if (unresolved.length) {
    reasons.push(reason('availability_unresolved', { players: unresolved }));
  }
  if (comparable.length < CHOICE_MIN) {
    reasons.push(reason('insufficient_choice_set', { count: comparable.length, minimum: CHOICE_MIN }));
  }

  // baseline availability (independent comparator) over the comparable set
  const noBaseline = comparable.filter(
    (r) => r.baseline === null || INDEPENDENT_BASELINE.indexOf(r.baseline_provenance) < 0,
  );
  const requireBaseline = params.require_independent_baseline === true;
  const modelOnly = params.model_only_exploration === true;
  if (noBaseline.length && requireBaseline && !modelOnly) {
    reasons.push(
      reason('baseline_unavailable', { players: noBaseline.map((r) => label(r as unknown as Loose)) }),
    );
  }
  if (noBaseline.length && modelOnly) {
    reasons.push(
      reason('baseline_not_independent', {
        players: noBaseline.map((r) => label(r as unknown as Loose)),
      }),
    );
    limitations.push('baseline_not_independent');
  }

  // --- 3. ranking (display order by player id on ties) -------------------------------------
  const ordered = comparable
    .slice()
    .sort(
      (x, y) =>
        (y.projection as number) - (x.projection as number) ||
        compareCodePoint(String(x.player_id), String(y.player_id)),
    );
  for (let i = 0; i < ordered.length; i += 1) {
    const r = ordered[i];
    r.rank = i + 1;
    if (
      i + 1 < ordered.length &&
      Math.abs((r.projection as number) - (ordered[i + 1].projection as number)) <= TOL
    ) {
      r.tied_with_next = true;
    }
  }

  // --- 4./5. gap before the floor gate ----------------------------------------------------
  let leading: DecisionResult['leading'] = { first: null, second: null, gap: null };
  const gates: DecisionResult['gates'] = {
    min_projection_gap: finite(params.min_projection_gap),
    gap_ok: null,
    min_floor:
      params.min_floor !== null && params.min_floor !== undefined ? finite(params.min_floor) : null,
    floor_checked: false,
    leading_floor: null,
    floor_ok: null,
  };
  if (ordered.length >= 2 && !paramProblems.length) {
    const first = ordered[0];
    const second = ordered[1];
    const gap = q((first.projection as number) - (second.projection as number)) as number;
    leading = { first: first.player_id, second: second.player_id, gap };
    const minGap = gates.min_projection_gap as number;
    if (Math.abs(gap) <= TOL) {
      gates.gap_ok = false;
      reasons.push(reason('projection_tie', { value: first.projection }));
    } else if (gap < minGap - TOL) {
      gates.gap_ok = false;
      reasons.push(
        reason('gap_below_minimum', {
          gap,
          threshold: minGap,
          first: label(first as unknown as Loose),
          second: label(second as unknown as Loose),
        }),
      );
    } else {
      gates.gap_ok = true;
      reasons.push(reason('gap_meets_minimum', { gap, threshold: minGap }));
    }
    // --- 6. floor guardrail ---------------------------------------------------------------
    const minFloor = gates.min_floor;
    gates.leading_floor = first.floor;
    if (minFloor === null) {
      reasons.push(reason('floor_guardrail_disabled'));
    } else {
      gates.floor_checked = true;
      if (first.floor === null) {
        gates.floor_ok = false;
        reasons.push(
          reason('floor_missing', { player: label(first as unknown as Loose), threshold: minFloor }),
        );
      } else if (first.floor < minFloor - TOL) {
        gates.floor_ok = false;
        reasons.push(
          reason('floor_below_minimum', {
            player: label(first as unknown as Loose),
            floor: first.floor,
            threshold: minFloor,
          }),
        );
      } else {
        gates.floor_ok = true;
        reasons.push(reason('floor_meets_minimum', { floor: first.floor, threshold: minFloor }));
      }
    }
  }

  // --- baseline preference (separate comparator) ------------------------------------------
  const provenance: Record<string, RankingRow['baseline_provenance']> = {};
  for (let i = 0; i < comparable.length; i += 1) {
    provenance[comparable[i].player_id] = comparable[i].baseline_provenance;
  }
  const baselineBlock: DecisionResult['baseline'] = {
    status: 'unavailable',
    preferred_player_id: null,
    preferred_value: null,
    agrees_with_model_leader: null,
    missing_for: noBaseline.map((r) => r.player_id),
    provenance,
  };
  if (comparable.length && !noBaseline.length) {
    const byBase = comparable
      .slice()
      .sort(
        (x, y) =>
          (y.baseline as number) - (x.baseline as number) ||
          compareCodePoint(String(x.player_id), String(y.player_id)),
      );
    const top = byBase[0];
    if (byBase.length > 1 && Math.abs((top.baseline as number) - (byBase[1].baseline as number)) <= TOL) {
      const tied = byBase.filter(
        (r) => Math.abs((r.baseline as number) - (top.baseline as number)) <= TOL,
      );
      baselineBlock.status = 'tie';
      baselineBlock.preferred_value = top.baseline;
      reasons.push(reason('baseline_tie', { players: tied.map((r) => label(r as unknown as Loose)) }));
    } else {
      baselineBlock.status = 'available';
      baselineBlock.preferred_player_id = top.player_id;
      baselineBlock.preferred_value = top.baseline;
      if (leading.first !== null) {
        const agrees = top.player_id === leading.first;
        baselineBlock.agrees_with_model_leader = agrees;
        reasons.push(
          reason(agrees ? 'baseline_agrees' : 'baseline_disagrees', {
            player: label(top as unknown as Loose),
            value: top.baseline,
          }),
        );
      }
    }
  }

  // --- status --------------------------------------------------------------------------------
  let status: DecisionResult['status'];
  if (reasons.some((r) => r.severity === 'hold')) {
    status = 'hold';
  } else if (reasons.some((r) => r.severity === 'review')) {
    status = 'review';
  } else {
    status = 'recommend';
  }
  let recommended: string | null = null;
  if (status === 'recommend') {
    const first = ordered[0];
    recommended = first.player_id;
    reasons.unshift(
      reason('recommend_highest_projection', {
        player: label(first as unknown as Loose),
        value: first.projection,
        gap: leading.gap,
        second: label(ordered[1] as unknown as Loose),
      }),
    );
  }
  const mode = snapshot.mode;
  if (mode === 'published_weekly') {
    limitations.push('snapshot_not_game_day_verified');
  }
  if (mode === 'synthetic') {
    limitations.push('synthetic_evidence');
  }
  // Anything short of a positive reconciliation is a limitation on real evidence (synthetic
  // snapshots already carry synthetic_evidence and have nothing to reconcile).
  if (mode !== 'synthetic' && snapshot.baseline_reconciled !== true && limitations.indexOf('baseline_unreconciled') < 0) {
    limitations.push('baseline_unreconciled');
  }
  limitations.push('floors_not_probabilities');

  let excluded = 0;
  for (let i = 0; i < rankingRows.length; i += 1) {
    if (rankingRows[i].exclusion) {
      excluded += 1;
    }
  }

  return {
    schema_version: 'decision_result/1.0',
    policy_version: POLICY_VERSION,
    decision_id: contentId(norm),
    status,
    recommended_player_id: recommended,
    reasons,
    ranking: rankingRows,
    leading,
    gates,
    baseline: baselineBlock,
    counts: { nominated: alts.length, excluded, comparable: comparable.length },
    limitations,
    explanation: reasons.map((r) => renderReason(r.code, r.detail)),
  };
}

/** Content identity of a result (used to detect tampered receipts). */
export function resultIdentity(result: DecisionResult): string {
  return contentId(result);
}
