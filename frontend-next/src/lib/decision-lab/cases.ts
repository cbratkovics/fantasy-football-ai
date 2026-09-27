/**
 * Case-library helpers — the mirror of `case_inputs` / `experiment_inputs` in
 * `ffai/decision_lab/cases.py`.
 *
 * Experiment semantics (shared with the Python exporter): `experiment.parameters` is a partial
 * override merged onto the case parameters; `experiment.overrides`, when present, *replaces* the
 * case overrides (an empty object therefore removes every exclusion/assumption of the case).
 * Every `expected` block in `cases.json` was computed by the Python policy on inputs built exactly
 * this way, so evaluating {@link caseInputs} / {@link experimentInputs} must reproduce it.
 */
import { buildDecisionInputs } from './receipts';
import { SCHEMA_VERSIONS } from './spec';
import type { DecisionInputs, Experiment, InputsSnapshot, LabCase } from './types';

/** `decision_lab_cases/1.0`. */
export const CASES_SCHEMA: string = SCHEMA_VERSIONS.cases;

/** Decision inputs for a case as stored (slot, alternatives, overrides, parameters). */
export function caseInputs(labCase: LabCase, snapshot: InputsSnapshot): DecisionInputs {
  return buildDecisionInputs(snapshot, {
    slot: labCase.slot,
    playerIds: labCase.alternatives.slice(),
    overrides: { ...labCase.overrides },
    parameters: { ...labCase.parameters },
  });
}

/** Decision inputs for one experiment: parameters merged, overrides replaced when present. */
export function experimentInputs(
  labCase: LabCase,
  experiment: Experiment,
  snapshot: InputsSnapshot,
): DecisionInputs {
  const overrides =
    experiment.overrides !== undefined ? { ...experiment.overrides } : { ...labCase.overrides };
  return buildDecisionInputs(snapshot, {
    slot: labCase.slot,
    playerIds: labCase.alternatives.slice(),
    overrides,
    parameters: { ...labCase.parameters, ...(experiment.parameters || {}) },
  });
}
