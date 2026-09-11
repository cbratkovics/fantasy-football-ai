"""Paths, seasons, and constants shared by training, evaluation, and serving.

Nothing here depends on the environment; there are no secrets. Paths are resolved relative to
the repository root so the same code runs from a checkout, from CI, and inside the API image.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.environ.get("FFAI_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
CACHE_DIR = Path(os.environ.get("FFAI_CACHE_DIR", REPO_ROOT / "data" / "cache"))
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")
SEASON_TYPE = "REG"
MIN_SEASON = 2019

# Frozen evaluation design: identical to the legacy production trainer so the rebuilt models are
# comparable to the recorded 2025-07-31 metadata.
TRAIN_SEASONS: tuple[int, ...] = (2019, 2020, 2021, 2022)
VAL_SEASON = 2023
TEST_SEASON = 2024
HISTORICAL_SEASONS: tuple[int, ...] = TRAIN_SEASONS + (VAL_SEASON, TEST_SEASON)

TARGET = "fantasy_points_ppr"
RANDOM_STATE = 42

# Regular season length by season (17 games / 18 weeks from 2021).
REGULAR_SEASON_WEEKS = {2019: 17, 2020: 17}
DEFAULT_REGULAR_SEASON_WEEKS = 18


def regular_season_weeks(season: int) -> int:
    """Number of regular-season weeks for a season."""
    return REGULAR_SEASON_WEEKS.get(season, DEFAULT_REGULAR_SEASON_WEEKS)
