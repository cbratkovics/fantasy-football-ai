// TypeScript mirrors of ffai/serve/schemas.py plus the evaluation artifact served by /performance.

export type Position = 'QB' | 'RB' | 'WR' | 'TE'
export const POSITIONS: Position[] = ['QB', 'RB', 'WR', 'TE']

export type ScoringFormat = 'ppr' | 'half' | 'standard'
export const SCORING_FORMATS: ScoringFormat[] = ['ppr', 'half', 'standard']

export interface DataThrough {
  season: number
  week: number
}

export interface Health {
  status: 'ok'
  model_version: string
  feature_version: string
  tiers_version: string | null
  eval_id: string | null
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
  source: 'frozen_test' | 'weekly'
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
