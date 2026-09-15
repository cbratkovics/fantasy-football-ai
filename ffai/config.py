"""Paths, seasons, and the one project configuration shared by every layer.

Nothing here depends on the environment; there are no secrets. Paths are resolved relative to
the repository root so the same code runs from a checkout, from CI, and inside the API image.

``PROJECT`` (ADR-0028) is the single source of the project-wide constants that used to be
string literals in the package, the dbt project, the workflows, and the frontend: slug and
display name, the entity / period / cohort vocabulary, the target column and its units, the
tolerance bands, the candidate names, and the deployment names. The other layers do not import
this module at runtime (dbt and the site cannot); instead they carry a *mirror* of the values
they need and ``tests/test_project_config.py`` fails when a mirror drifts:

* ``dbt/dbt_project.yml`` ``vars`` must equal :func:`dbt_vars`
* ``frontend-next/src/lib/project.config.json`` must equal :func:`frontend_config`
  (regenerate with ``python -m ffai.config --frontend``)
* the workflows read deployment names with ``python -m ffai.config <key>``

The legacy module-level names (``POSITIONS``, ``TARGET`` …) are kept as aliases of ``PROJECT``
fields so no caller changed in this pass.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.environ.get("FFAI_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
CACHE_DIR = Path(os.environ.get("FFAI_CACHE_DIR", REPO_ROOT / "data" / "cache"))
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
SCHEMAS_DIR = ARTIFACTS_DIR / "schemas"


@dataclass(frozen=True)
class ProjectConfig:
    """Project-wide constants. Frozen: change the values here, then regenerate the mirrors."""

    # identity
    slug: str
    display_name: str
    package_name: str
    domain_summary: str
    # vocabulary
    entity_name: str  # "player"
    entity_key: str  # "player_id"
    entity_display_column: str  # "player_display_name"
    period_name: str  # "week"
    period_columns: tuple[str, ...]  # ("season", "week"); period_key = season * 100 + week
    season_name: str  # "season"
    cohort_name: str  # "position": one model per cohort
    cohorts: tuple[str, ...]
    # target and metrics
    target_column: str
    target_units: str  # "points"
    target_scoring_format: str  # the format the model predicts; others derive by rules
    scoring_formats: tuple[str, ...]
    within_k: tuple[int, ...]  # tolerance bands reported next to MAE
    candidates: tuple[str, ...]
    # data
    source_name: str
    source_library: str
    min_season: int
    train_seasons: tuple[int, ...]
    val_season: int
    test_season: int
    # warehouse
    dbt_project_name: str
    motherduck_database: str
    # deployment
    github_owner: str
    repo_url: str
    hf_space: str
    api_url: str
    site_url: str
    site_origins: tuple[str, ...]
    pages_url: str
    schedule_cron: str
    schedule_note: str
    python_version: str
    node_version: str
    env_prefix: str = "FFAI_"
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


PROJECT = ProjectConfig(
    slug="fantasy-football-ai",
    display_name="Win My League · Decision Lab",
    package_name="ffai",
    domain_summary=(
        "Weekly PPR fantasy-points projections for NFL quarterbacks, running backs, wide "
        "receivers and tight ends, with an artifact-backed evaluation trail."
    ),
    entity_name="player",
    entity_key="player_id",
    entity_display_column="player_display_name",
    period_name="week",
    period_columns=("season", "week"),
    season_name="season",
    cohort_name="position",
    cohorts=("QB", "RB", "WR", "TE"),
    target_column="fantasy_points_ppr",
    target_units="points",
    target_scoring_format="ppr",
    scoring_formats=("standard", "half", "ppr"),
    within_k=(3, 5),
    candidates=("rf", "xgb"),
    source_name="nflverse",
    source_library="nflreadpy",
    min_season=2019,
    train_seasons=(2019, 2020, 2021, 2022),
    val_season=2023,
    test_season=2024,
    dbt_project_name="ffai_dbt",
    motherduck_database="ffai",
    github_owner="cbratkovics",
    repo_url="https://github.com/cbratkovics/fantasy-football-ai",
    hf_space="cbratkovics/fantasy-football-ai",
    api_url="https://cbratkovics-fantasy-football-ai.hf.space",
    site_url="https://www.winmyleague.ai",
    site_origins=(
        "https://winmyleague.ai",
        "https://www.winmyleague.ai",
        "https://fantasy-football-ai.vercel.app",
    ),
    pages_url="https://cbratkovics.github.io/fantasy-football-ai/",
    schedule_cron="0 10 * 9-12,1 2",
    schedule_note="Tuesdays 10:00 UTC, September through January",
    python_version="3.11",
    node_version="20",
)

# --- legacy aliases (unchanged names, unchanged values) ---------------------------------------
POSITIONS: tuple[str, ...] = PROJECT.cohorts
SEASON_TYPE = "REG"
MIN_SEASON = PROJECT.min_season

# Frozen evaluation design: identical to the legacy production trainer so the rebuilt models are
# comparable to the recorded 2025-07-31 metadata.
TRAIN_SEASONS: tuple[int, ...] = PROJECT.train_seasons
VAL_SEASON = PROJECT.val_season
TEST_SEASON = PROJECT.test_season
HISTORICAL_SEASONS: tuple[int, ...] = TRAIN_SEASONS + (VAL_SEASON, TEST_SEASON)

TARGET = PROJECT.target_column
RANDOM_STATE = 42

# Regular season length by season (17 games / 18 weeks from 2021).
REGULAR_SEASON_WEEKS = {2019: 17, 2020: 17}
DEFAULT_REGULAR_SEASON_WEEKS = 18

LOCAL_ORIGINS: tuple[str, ...] = ("http://localhost:3000", "http://127.0.0.1:3000")


def regular_season_weeks(season: int) -> int:
    """Number of regular-season weeks for a season."""
    return REGULAR_SEASON_WEEKS.get(season, DEFAULT_REGULAR_SEASON_WEEKS)


def cors_origins() -> list[str]:
    """Default browser origins the API accepts (local dev plus the deployed site)."""
    return [*LOCAL_ORIGINS, *PROJECT.site_origins]


# --- mirrors ----------------------------------------------------------------------------------


def dbt_vars() -> dict[str, Any]:
    """The project vars ``dbt/dbt_project.yml`` must declare with exactly these defaults."""
    return {
        "entity_key": PROJECT.entity_key,
        "cohorts": list(PROJECT.cohorts),
        "candidates": list(PROJECT.candidates),
        "target_scoring_format": PROJECT.target_scoring_format,
        "within_k": list(PROJECT.within_k),
    }


def frontend_config() -> dict[str, Any]:
    """Labels and names the site reads instead of string literals (ADR-0028)."""
    return {
        "slug": PROJECT.slug,
        "displayName": PROJECT.display_name,
        "domainSummary": PROJECT.domain_summary,
        "repoUrl": PROJECT.repo_url,
        "apiUrl": PROJECT.api_url,
        "pagesUrl": PROJECT.pages_url,
        "entity": {"name": PROJECT.entity_name, "key": PROJECT.entity_key},
        "period": {"name": PROJECT.period_name, "season": PROJECT.season_name},
        "cohort": {"name": PROJECT.cohort_name, "values": list(PROJECT.cohorts)},
        "target": {
            "column": PROJECT.target_column,
            "units": PROJECT.target_units,
            "format": PROJECT.target_scoring_format,
            "formats": list(PROJECT.scoring_formats),
        },
        "withinK": list(PROJECT.within_k),
        "candidates": list(PROJECT.candidates),
    }


def _main(argv: list[str]) -> int:  # pragma: no cover - thin CLI
    """``python -m ffai.config --frontend | --dbt-vars | --json | <field>``."""
    if not argv or argv[0] in {"-h", "--help"}:
        print(_main.__doc__)
        return 0
    if argv[0] == "--frontend":
        print(json.dumps(frontend_config(), indent=2, ensure_ascii=False))
    elif argv[0] == "--dbt-vars":
        print(json.dumps(dbt_vars()))
    elif argv[0] == "--json":
        print(json.dumps(PROJECT.as_dict(), indent=2, default=list))
    else:
        value = getattr(PROJECT, argv[0])
        print(",".join(value) if isinstance(value, tuple) else value)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main(sys.argv[1:]))
