/**
 * Decision Lab document types — the TypeScript mirror of `ffai/decision_lab/contracts.py`.
 *
 * Field names and enums are identical to the Python reference; `policy_spec.json` in this
 * directory is a byte-for-byte copy of `ffai/decision_lab/policy_spec.json` (checked by
 * `tests/test_decision_lab_contracts.py`). Nothing here is optional unless the Python schema
 * allows null: required-but-nullable fields stay present as `null`.
 */

export type Mode = 'historical_replay' | 'published_weekly' | 'synthetic';
export type SourceFamily = 'frozen_test' | 'out_of_sample_season' | 'weekly' | 'synthetic';
export type Slot = 'QB' | 'RB' | 'WR' | 'TE' | 'FLEX';
export type Position = 'QB' | 'RB' | 'WR' | 'TE';
export type Status = 'hold' | 'review' | 'recommend';
export type Severity = 'hold' | 'review' | 'exclusion' | 'info';
export type Availability = 'assumed_available' | 'unavailable' | 'unknown' | 'realized_stats_row';
export type AvailabilityBasis =
  | 'user_assumed'
  | 'user_excluded'
  | 'not_verified'
  | 'hindsight_stats_row'
  | 'synthetic';
export type BaselineProvenance =
  | 'player_history'
  | 'position_history'
  | 'model_prediction_fallback'
  | 'unknown';
export type EvidenceStatus = 'verified' | 'unverified' | 'corrupt';
export type PublicationStatus = 'recorded' | 'published' | 'synthetic' | 'blocked';
export type ActionKind = 'hypothetical_replay' | 'self_reported_real';
export type EventType = 'prediction_note' | 'action_recorded' | 'action_declined' | 'outcome_attached';

export interface Period {
  season: number;
  week: number;
}

export interface Publication {
  status: PublicationStatus;
  run_id: string | null;
  action: string | null;
  note: string | null;
}

export interface SnapshotRef {
  snapshot_id: string;
  mode: Mode;
  source_family: SourceFamily;
  season: number;
  week: number;
  model_version: string;
  feature_version: string | null;
  candidate_by_position: Record<string, string>;
  scoring_format: string;
  inputs_sha256: string;
  data_cutoff: Period | null;
  generated_at_utc: string | null;
  publication: Publication;
  evidence_status: EvidenceStatus;
  evidence_detail: string | null;
  baseline_reconciled: boolean | null;
}

export interface Alternative {
  player_id: string;
  display_name: string | null;
  team: string | null;
  position: string;
  model_version: string;
  candidate: string;
  season: number;
  week: number;
  projection: number | null;
  floor: number | null;
  ceiling: number | null;
  baseline: number | null;
  baseline_provenance: BaselineProvenance;
  availability: Availability;
  availability_basis: AvailabilityBasis;
  source_row_ref: string | null;
}

export interface Parameters {
  min_projection_gap: number;
  min_floor: number | null;
  require_independent_baseline: boolean;
  model_only_exploration: boolean;
}

export interface DecisionInputs {
  schema_version: 'decision_inputs/1.0';
  policy_version: string;
  snapshot: SnapshotRef;
  slot: string;
  alternatives: Alternative[];
  parameters: Parameters;
}

export interface Reason {
  code: string;
  severity: Severity;
  detail: Record<string, unknown>;
}

export interface RankingRow {
  player_id: string;
  display_name: string | null;
  position: string | null;
  projection: number | null;
  floor: number | null;
  ceiling: number | null;
  baseline: number | null;
  baseline_provenance: BaselineProvenance;
  availability: string | null;
  availability_basis: string | null;
  comparable: boolean;
  exclusion: string | null;
  rank: number | null;
  tied_with_next: boolean;
}

export interface DecisionResult {
  schema_version: 'decision_result/1.0';
  policy_version: string;
  decision_id: string;
  status: Status;
  recommended_player_id: string | null;
  reasons: Reason[];
  ranking: RankingRow[];
  leading: { first: string | null; second: string | null; gap: number | null };
  gates: {
    min_projection_gap: number | null;
    gap_ok: boolean | null;
    min_floor: number | null;
    floor_checked: boolean;
    leading_floor: number | null;
    floor_ok: boolean | null;
  };
  baseline: {
    status: 'available' | 'unavailable' | 'tie';
    preferred_player_id: string | null;
    preferred_value: number | null;
    agrees_with_model_leader: boolean | null;
    missing_for: string[];
    provenance: Record<string, BaselineProvenance>;
  };
  counts: { nominated: number; excluded: number; comparable: number };
  limitations: string[];
  explanation: string[];
}

export interface SnapshotRow {
  player_id: string;
  display_name: string | null;
  team: string | null;
  position: string;
  model_version: string;
  candidate: string;
  projection: number | null;
  floor: number | null;
  ceiling: number | null;
  baseline: number | null;
  baseline_provenance: BaselineProvenance;
  availability_default: Availability;
  display_source: 'prediction_source' | 'dim_player_current' | 'synthetic' | null;
  source_row_ref: string | null;
}

export interface SourceFile {
  role: string;
  path: string;
  sha256: string | null;
  rows?: number | null;
}

export interface Population {
  conditioning: 'realized_stats_rows' | 'scored_eligible_players' | 'synthetic';
  description: string;
  n_rows: number;
  exclusions: string[];
}

export interface BaselineMeta {
  name: string;
  provenance_basis:
    | 'mart_column'
    | 'population_rule_and_reconciliation'
    | 'population_rule_unreconciled'
    | 'synthetic'
    | 'unknown';
  reconciled: boolean | null;
  reconciled_to: string | null;
  note: string | null;
}

export interface InputsSnapshot {
  schema_version: 'inputs_snapshot/1.0';
  snapshot_id: string;
  mode: Mode;
  source_family: SourceFamily;
  season: number;
  week: number;
  model_version: string;
  feature_version: string | null;
  candidate_by_position: Record<string, string>;
  scoring_format: string;
  data_cutoff: Period | null;
  generated_at_utc: string | null;
  publication: Publication;
  population: Population;
  baseline: BaselineMeta;
  champion_at_source: {
    model_version: string;
    candidate_by_position: Record<string, string>;
    basis: 'eval_artifact' | 'predictions_file' | 'synthetic';
  } | null;
  source: {
    mart: string | null;
    files: SourceFile[];
    mart_export: {
      exported_at_utc: string | null;
      target: string | null;
      code_commit: string | null;
      invocation_id: string | null;
    } | null;
  };
  rows: SnapshotRow[];
}

export interface OutcomeRow {
  player_id: string;
  actual: number | null;
  actual_source: 'stats' | 'artifact' | 'synthetic' | null;
}

export interface OutcomeSnapshot {
  schema_version: 'outcome_snapshot/1.0';
  snapshot_id: string;
  inputs_content_sha256: string;
  season: number;
  week: number;
  model_version: string;
  scoring_format: string;
  observed_at_utc: string | null;
  source: { mart: string | null; files: SourceFile[] };
  coverage: { scored: number; observed: number };
  rows: OutcomeRow[];
}

export interface OutcomeMetrics {
  schema_version: 'outcome_metrics/1.0';
  decision_id: string | null;
  outcome_snapshot_id: string | null;
  outcome_id: string | null;
  action_state: 'not_recorded' | 'recorded' | 'declined';
  action_kind: ActionKind | null;
  chosen_player_id: string | null;
  choice_set: string[];
  coverage: { nominated: number; choice_set_size: number; observed: number; missing: string[] };
  chosen_actual_points: number | null;
  chosen_actual_reason: string | null;
  best_in_choice_set: { player_id: string; actual: number } | null;
  choice_set_regret: number | null;
  choice_set_regret_reason: string | null;
  baseline_preferred_player_id: string | null;
  baseline_preferred_actual: number | null;
  points_vs_baseline_choice: number | null;
  points_vs_baseline_choice_reason: string | null;
  model_recommended_player_id: string | null;
  model_recommended_actual: number | null;
  model_policy_vs_baseline_choice: number | null;
  model_policy_vs_baseline_choice_reason: string | null;
  per_alternative: Array<{ player_id: string; actual: number | null; observed: boolean }>;
  notes: string[];
}

export interface ReceiptEvent {
  event_id: string;
  seq: number;
  event_type: EventType;
  at_utc: string;
  payload: Record<string, unknown>;
}

export interface ActionState {
  state: 'not_recorded' | 'recorded' | 'declined';
  action_id: string | null;
  at_utc: string | null;
  chosen_player_id: string | null;
  kind: ActionKind | null;
  note: string | null;
}

export interface OutcomeState {
  state: 'not_attached' | 'attached';
  outcome_snapshot_id: string | null;
  outcome_id: string | null;
  attached_at_utc: string | null;
  metrics: OutcomeMetrics | null;
}

export interface Receipt {
  schema_version: 'decision_receipt/1.0';
  decision_id: string;
  parent_decision_id: string | null;
  created_at_utc: string;
  case_id: string | null;
  prediction_note: string | null;
  inputs: DecisionInputs;
  result: DecisionResult;
  result_sha256: string;
  events: ReceiptEvent[];
  action: ActionState;
  outcome: OutcomeState;
  trust: { note: string };
}

export interface Experiment {
  label: string;
  parameters?: Partial<Parameters>;
  overrides?: Record<string, { availability: Availability; availability_basis: AvailabilityBasis }>;
  expected: { status: Status; recommended_player_id: string | null };
}

export interface LabCase {
  case_id: string;
  title: string;
  mode: Mode;
  snapshot_id: string;
  slot: string;
  alternatives: string[];
  overrides: Record<string, { availability: Availability; availability_basis: AvailabilityBasis }>;
  parameters: Parameters;
  setup: string;
  experiments: Experiment[];
  expected: { status: Status; recommended_player_id: string | null; reason_codes: string[] };
  outcomes_available: boolean;
  outcome_snapshot_id: string | null;
  limitations: string[];
  selection: {
    method: 'curated_synthetic' | 'deterministic_discovery';
    criteria: string;
    sealed_pattern: string | null;
  } | null;
  source_row_refs: string[];
}

export interface CasesFile {
  schema_version: 'decision_lab_cases/1.0';
  policy_version: string;
  cases: LabCase[];
  notes: string[];
}

export interface FileRef {
  path: string;
  file_sha256: string;
  content_sha256: string;
  n_rows: number;
}

export interface OutcomeFileRef extends FileRef {
  n_observed: number;
}

export interface ManifestSnapshot {
  snapshot_id: string;
  mode: Mode;
  source_family: SourceFamily;
  season: number;
  week: number;
  model_version: string;
  candidate_by_position: Record<string, string>;
  inputs: FileRef;
  outcomes: OutcomeFileRef | null;
  population: Population;
  baseline: BaselineMeta;
  reconciliation: {
    status: 'matched' | 'mismatch' | 'unavailable';
    reference: string | null;
    expected: Record<string, unknown> | null;
    observed: Record<string, unknown> | null;
    tolerance: number | null;
  } | null;
}

export interface LabManifest {
  schema_version: 'decision_lab_manifest/1.0';
  exporter_version: string;
  policy_version: string;
  schema_versions: Record<string, string>;
  code_revision: { produced_at: string | null; note: string };
  lineage: Record<string, unknown>;
  sources: SourceFile[];
  snapshots: ManifestSnapshot[];
  cases: FileRef;
  latest_weekly_snapshot_id: string | null;
  policy_spec: { path: string; sha256: string };
  digest_coverage: Record<string, string>;
}
