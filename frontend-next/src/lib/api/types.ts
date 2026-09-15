// TypeScript mirrors of ffai/serve/schemas.py plus the evaluation artifact served by /performance.

export type Position = 'QB' | 'RB' | 'WR' | 'TE'
export const POSITIONS: Position[] = ['QB', 'RB', 'WR', 'TE']

export type ScoringFormat = 'ppr' | 'half' | 'standard'
export const SCORING_FORMATS: ScoringFormat[] = ['ppr', 'half', 'standard']

export interface DataThrough {
  season: number
  week: number
}

/** How an evaluation artifact relates to model selection. Mirrors schemas.EvalKind. */
export type EvalKind = 'frozen_test' | 'out_of_sample_season'

export type HistorySource = EvalKind | 'weekly'

export interface Root {
  name: string
  version: string
  docs: string
  health: string
  performance: string
  manifest: string
}

export interface Health {
  status: 'ok'
  model_version: string
  feature_version: string
  tiers_version: string | null
  /** The frozen-test evaluation id. */
  eval_id: string | null
  /** Every registered eval_id, in manifest order. */
  evaluations: string[]
  data_through: DataThrough
  loaded_at_utc: string
  positions_loaded: string[]
}

export interface Manifest {
  manifest_version?: string
  champion?: { model_version: string; candidate: Record<string, string> | string }
  challenger?: { model_version: string; candidate: Record<string, string> | string }
  feature_version?: string
  tiers_version?: string
  eval_id?: string
  data_through?: DataThrough
  predictions?: { latest?: string }
  last_run?: { action?: string; at_utc?: string; reasons?: string[]; run_id?: string }
  updated_at_utc?: string
  [key: string]: unknown
}

export interface ManifestResponse {
  manifest: Manifest
}

export interface PredictionRecord {
  player_id: string
  name: string | null
  team: string | null
  position: string
  prediction: number
  floor: number
  ceiling: number
  receptions_estimate: number
  model_version: string
  candidate: string
  actual: number | null
}

export interface PredictionsResponse {
  season: number
  week: number
  scoring: ScoringFormat
  scoring_derivation: string
  interval_method: string
  model_version: string
  feature_version: string
  generated_at_utc: string
  data_through: DataThrough
  n: number
  predictions: PredictionRecord[]
}

export interface PlayerHistoryRow {
  season: number
  week: number
  prediction: number | null
  floor: number | null
  ceiling: number | null
  actual: number | null
  model_version: string | null
  source: HistorySource
}

export interface PlayerResponse {
  player_id: string
  name: string | null
  team: string | null
  position: string | null
  model_version: string
  feature_version: string
  generated_at_utc: string
  history: PlayerHistoryRow[]
}

export interface TierPlayer {
  player_id: string
  name: string | null
  team: string | null
  ppg_prev: number
  games_prev: number
  tier_probability: number
}

export interface Tier {
  tier: number
  players: TierPlayer[]
}

export interface TierEvaluation {
  n?: number
  spearman?: number
  within_band_rate?: number
  tier_sizes?: Record<string, number>
  [key: string]: unknown
}

export interface TiersResponse {
  position: string
  season: number
  tier_version: string
  method: string
  inputs: string
  evaluation: TierEvaluation | null
  generated_at_utc: string
  tiers: Tier[]
}

export interface MetricBlock {
  n: number
  mae: number
  median_ae: number
  rmse: number
  within_3_rate: number
  within_5_rate: number
}

/** The tolerance-band keys an artifact carries (project config withinK = [3, 5]). */
export type WithinKey = 'within_3_rate' | 'within_5_rate'

export interface BaselineBlock extends MetricBlock {
  name?: string
}

export interface CohortBlock extends MetricBlock {
  baseline: MetricBlock
}

export interface RollingFold {
  season: number
  week: number
  n: number
  mae: number
  baseline_mae: number
}

export interface PerformanceArtifact {
  artifact_version: string
  eval_id: string
  kind: EvalKind
  season: number
  generated_at_utc: string
  code_commit: string
  input: { path: string; sha256: string; n_rows: number }
  model: {
    version: string
    feature_version: string
    candidate: Record<string, string>
    challenger: Record<string, string>
    trained_at_utc: string
    input_sha256: string
  }
  split: { strategy: string; test_start_season: number; test_start_week: number }
  metrics: MetricBlock
  baseline: BaselineBlock
  cohorts: Record<string, CohortBlock>
  rolling_origin: {
    candidate: string
    strategy: string
    folds: RollingFold[]
    mean_mae: number
    mean_baseline_mae: number
  }
  metric_definitions: Record<string, string>
  policy_sweep: unknown[]
}

/** GET /performance: every registered evaluation artifact for the champion. */
export interface PerformanceResponse {
  model_version: string
  feature_version: string
  evaluations: PerformanceArtifact[]
}

// --- gold marts (parquet exported by the weekly dbt build; /marts/*; ADR-0018) ---

export const MART_SOURCE = 'gold marts exported by the weekly build' as const

export interface MartExport {
  exported_at_utc: string | null
  target: string | null
  invocation_id: string | null
  code_commit: string | null
  row_counts: Record<string, number>
}

export interface WeeklyEvalRow {
  eval_window: string
  season: number
  week: number
  model_version: string
  candidate: string
  cohort: string
  n: number
  mae: number
  median_ae: number | null
  rmse: number | null
  within_3_rate: number
  within_5_rate: number | null
  interval_coverage: number | null
  baseline_mae: number | null
  baseline_within_3_rate: number | null
  mae_minus_baseline: number | null
}

export interface WeeklyEvalResponse {
  source: typeof MART_SOURCE
  mart: 'fct_weekly_eval'
  export: MartExport
  cohort: string
  n: number
  rows: WeeklyEvalRow[]
}

export interface PlayerWeekRow {
  season: number
  week: number
  model_version: string
  candidate: string
  position: string
  source: HistorySource
  eval_window: string
  prediction: number
  prediction_floor: number | null
  prediction_ceiling: number | null
  actual: number | null
  actual_source: 'stats' | 'artifact' | null
  abs_error: number | null
  within_3: boolean | null
  within_5: boolean | null
  interval_hit: boolean | null
  baseline: number
  baseline_abs_error: number | null
}

export interface PlayerWeekResponse {
  source: typeof MART_SOURCE
  mart: 'fct_player_week'
  export: MartExport
  player_id: string
  name: string | null
  team: string | null
  position: string | null
  candidate: string | Record<string, string>
  n: number
  rows: PlayerWeekRow[]
}

export interface DecisionPolicyRow {
  season: number
  week: number
  position: string
  model_version: string
  candidate: string
  min_floor: number
  eligible_decisions: number
  recommendations: number
  reviews: number
  recommendation_rate: number | null
  review_rate: number | null
  recommendations_with_outcome: number
  recommendation_mae: number | null
  review_mae: number | null
  mean_regret: number | null
  hit_rate: number | null
  downside_rate: number | null
}

export interface DecisionSummaryRow {
  cohort: string
  eligible_decisions: number
  recommendations: number
  recommendation_rate: number | null
  recommendation_mae: number | null
  mean_regret: number | null
  hit_rate: number | null
  downside_rate: number | null
  recommendations_with_outcome: number
}

export interface DecisionsResponse {
  source: typeof MART_SOURCE
  mart: 'fct_decision_policy'
  export: MartExport
  min_floor: number
  available_min_floors: number[]
  policy: string
  replacement_level: string
  n: number
  summary: DecisionSummaryRow[]
  rows: DecisionPolicyRow[]
}
