"""Pydantic v2 response models. Every payload carries model/feature versions and timestamps."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ScoringFormat = Literal["standard", "half", "ppr"]


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
    feature_version: str
    tiers_version: str | None
    eval_id: str | None
    data_through: dict[str, int]
    loaded_at_utc: str
    positions_loaded: list[str]


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
    source: Literal["frozen_test", "weekly"]


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
