"""Pydantic v2 response models. Every payload carries model/feature versions and timestamps."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ScoringFormat = Literal["standard", "half", "ppr"]
EvalKind = Literal["frozen_test", "out_of_sample_season"]


class Root(BaseModel):
    name: str
    version: str
    docs: str
    health: str
    performance: str
    manifest: str
    marts: str | None = Field(default=None, description="gold marts index, when exported")


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
    feature_version: str
    tiers_version: str | None
    eval_id: str | None = Field(description="the frozen-test evaluation id")
    evaluations: list[str] = Field(default_factory=list, description="every registered eval_id")
    data_through: dict[str, int]
    loaded_at_utc: str
    positions_loaded: list[str]
    marts_exported_at_utc: str | None = Field(
        default=None, description="when the gold marts being served were exported (null if none)"
    )


class PredictionRecord(BaseModel):
    player_id: str
    name: str | None
    team: str | None
    position: str
    prediction: float = Field(description="points under the requested scoring format")
    floor: float
    ceiling: float
    receptions_estimate: float
    model_version: str
    candidate: str
    actual: float | None = Field(default=None, description="realised PPR points once attached")


class PredictionsResponse(BaseModel):
    season: int
    week: int
    scoring: ScoringFormat
    scoring_derivation: str
    interval_method: str
    model_version: str
    feature_version: str
    generated_at_utc: str
    data_through: dict[str, int]
    n: int
    predictions: list[PredictionRecord]


class PlayerHistoryRow(BaseModel):
    season: int
    week: int
    prediction: float | None
    floor: float | None = None
    ceiling: float | None = None
    actual: float | None
    model_version: str | None
    source: Literal["frozen_test", "out_of_sample_season", "weekly"]


class PlayerResponse(BaseModel):
    player_id: str
    name: str | None
    team: str | None
    position: str | None
    model_version: str
    feature_version: str
    generated_at_utc: str
    history: list[PlayerHistoryRow]


class TierPlayer(BaseModel):
    player_id: str
    name: str | None
    team: str | None
    ppg_prev: float
    games_prev: int
    tier_probability: float


class Tier(BaseModel):
    tier: int
    players: list[TierPlayer]


class TiersResponse(BaseModel):
    position: str
    season: int
    tier_version: str
    method: str
    inputs: str
    evaluation: dict[str, Any] | None
    generated_at_utc: str
    tiers: list[Tier]


class ManifestResponse(BaseModel):
    manifest: dict[str, Any]


class EvaluationSummary(BaseModel):
    eval_id: str
    kind: EvalKind
    season: int | None
    path: str


class EvaluationsResponse(BaseModel):
    model_version: str
    feature_version: str
    evaluations: list[dict[str, Any]] = Field(
        description="full evaluation artifacts, each with eval_id, kind and season"
    )


# --- gold marts (exported parquet, read in-process; ADR-0018) ---


class MartExport(BaseModel):
    """Provenance of the parquet files being served (from _export_manifest.json)."""

    exported_at_utc: str | None
    target: str | None = Field(description="dbt target the marts were built on (prod = MotherDuck)")
    invocation_id: str | None
    code_commit: str | None
    row_counts: dict[str, int]


class WeeklyEvalRow(BaseModel):
    eval_window: str
    season: int
    week: int
    model_version: str
    candidate: str
    cohort: str
    n: int
    mae: float
    median_ae: float | None
    rmse: float | None
    within_3_rate: float
    within_5_rate: float | None
    interval_coverage: float | None
    baseline_mae: float | None
    baseline_within_3_rate: float | None
    mae_minus_baseline: float | None


class WeeklyEvalResponse(BaseModel):
    source: Literal["gold marts exported by the weekly build"]
    mart: Literal["fct_weekly_eval"]
    export: MartExport
    cohort: str
    n: int
    rows: list[WeeklyEvalRow]


class PlayerWeekRow(BaseModel):
    season: int
    week: int
    model_version: str
    candidate: str
    position: str
    source: str
    eval_window: str
    prediction: float
    prediction_floor: float | None
    prediction_ceiling: float | None
    actual: float | None
    actual_source: str | None
    abs_error: float | None
    within_3: bool | None
    within_5: bool | None
    interval_hit: bool | None
    baseline: float
    baseline_abs_error: float | None


class PlayerWeekResponse(BaseModel):
    source: Literal["gold marts exported by the weekly build"]
    mart: Literal["fct_player_week"]
    export: MartExport
    player_id: str
    name: str | None
    team: str | None
    position: str | None
    candidate: str | dict[str, str] = Field(description="champion candidate filter applied")
    n: int
    rows: list[PlayerWeekRow]


class DecisionPolicyRow(BaseModel):
    season: int
    week: int
    position: str
    model_version: str
    candidate: str
    min_floor: float
    eligible_decisions: int
    recommendations: int
    reviews: int
    recommendation_rate: float | None
    review_rate: float | None
    recommendations_with_outcome: int
    recommendation_mae: float | None
    review_mae: float | None
    mean_regret: float | None
    hit_rate: float | None
    downside_rate: float | None


class DecisionSummaryRow(BaseModel):
    cohort: str = Field(description="ALL or a position; count-weighted over the returned rows")
    eligible_decisions: int
    recommendations: int
    recommendation_rate: float | None
    recommendation_mae: float | None
    mean_regret: float | None
    hit_rate: float | None
    downside_rate: float | None
    recommendations_with_outcome: int


class DecisionsResponse(BaseModel):
    source: Literal["gold marts exported by the weekly build"]
    mart: Literal["fct_decision_policy"]
    mart_version: int = Field(description="explicit dbt model version served by the API")
    export: MartExport
    min_floor: float
    available_min_floors: list[float]
    policy: str = Field(description="how policy_action is decided")
    replacement_level: str = Field(description="how replacement_level_points is defined")
    n: int
    summary: list[DecisionSummaryRow]
    rows: list[DecisionPolicyRow]
