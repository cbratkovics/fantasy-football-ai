/**
 * Typed access to `policy_spec.json` (a byte-identical copy of `ffai/decision_lab/policy_spec.json`).
 *
 * The JSON file is the versioned specification; this module only gives it a type and exposes the
 * constants every other module reads. Never edit the JSON here — regenerate it from Python.
 */
import rawSpec from './policy_spec.json';

export interface ReasonSpec {
  severity: 'hold' | 'review' | 'exclusion' | 'info';
  template: string;
}

export interface ParameterSpec {
  default: number | boolean | null;
  min?: number;
  max?: number;
}

export interface PolicySpec {
  policy_version: string;
  spec_version: string;
  schema_versions: {
    inputs_snapshot: string;
    outcome_snapshot: string;
    decision_inputs: string;
    decision_result: string;
    outcome_metrics: string;
    receipt: string;
    manifest: string;
    cases: string;
  };
  tolerance: number;
  canonical_number_decimals: number;
  scoring_format: string;
  slots: Record<string, string[]>;
  choice_set: { min: number; max: number };
  parameters: Record<string, ParameterSpec>;
  modes: string[];
  source_families: string[];
  availability: { values: string[]; bases: string[]; comparable: string[] };
  baseline_provenance: string[];
  independent_baseline_provenance: string[];
  statuses: string[];
  action_kinds: string[];
  event_types: string[];
  reasons: Record<string, ReasonSpec>;
  limitations: Record<string, string>;
}

/** The parsed policy specification. */
export const SPEC: PolicySpec = rawSpec as unknown as PolicySpec;

/** `policy_spec.json` → `policy_version` (currently `1.0.0`). */
export const POLICY_VERSION: string = SPEC.policy_version;

/** Every document schema version, keyed as in the spec. */
export const SCHEMA_VERSIONS = SPEC.schema_versions;
