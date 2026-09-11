"""nflverse loaders with a local parquet cache.

The only external data source in this repository. Data is read through ``nflreadpy`` (the
maintained Python client for nflverse releases). ``nfl_data_py`` is used as a fallback only if
``nflreadpy`` cannot be imported; the loader records which library served the request so the
training metadata can state it.

Cache: ``data/cache/<name>_<first>-<last>_<YYYY-MM-DD>.parquet`` keyed by the season range and the
load date. A same-day cache hit is served without any network access; pass ``refresh=True`` to
bypass it. The cache directory is git-ignored.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ffai.config import CACHE_DIR, POSITIONS, SEASON_TYPE

try:  # pragma: no cover - import guard
    import nflreadpy as _nfl

    LIBRARY = f"nflreadpy=={_nfl.__version__}"
    _FALLBACK = False
except ImportError:  # pragma: no cover - exercised only where nflreadpy is unavailable
    import nfl_data_py as _nfl  # type: ignore[no-redef]

    LIBRARY = f"nfl_data_py=={getattr(_nfl, '__version__', 'unknown')}"
    _FALLBACK = True

# Identifier / context columns kept from the weekly player stats release.
ID_COLUMNS: tuple[str, ...] = (
    "player_id",
    "player_display_name",
    "position",
    "season",
    "week",
    "season_type",
    "team",
    "opponent_team",
)

# Raw stat columns kept. Every feature and every scoring rule is derived from these.
STAT_COLUMNS: tuple[str, ...] = (
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "passing_air_yards",
    "passing_2pt_conversions",
    "sack_fumbles_lost",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles_lost",
    "rushing_2pt_conversions",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "receiving_fumbles_lost",
    "receiving_2pt_conversions",
    "special_teams_tds",
    "fantasy_points",
    "fantasy_points_ppr",
)

# nfl_data_py (legacy nflverse release) names that differ from the current release.
_LEGACY_RENAMES = {
    "interceptions": "passing_interceptions",
    "player_display_name": "player_display_name",
}


def _seasons_list(seasons: int | Iterable[int]) -> list[int]:
    if isinstance(seasons, int):
        return [seasons]
    out = sorted({int(s) for s in seasons})
    if not out:
        raise ValueError("seasons must not be empty")
    return out


def _cache_path(name: str, seasons: list[int], load_date: dt.date | None = None) -> Path:
    load_date = load_date or dt.date.today()
    return CACHE_DIR / f"{name}_{seasons[0]}-{seasons[-1]}_{load_date.isoformat()}.parquet"


def _to_pandas(frame) -> pd.DataFrame:  # noqa: ANN001 - polars or pandas
    if isinstance(frame, pd.DataFrame):
        return frame
    return frame.to_pandas()


def _cached(name: str, seasons: list[int], fetch, refresh: bool) -> pd.DataFrame:  # noqa: ANN001
    path = _cache_path(name, seasons)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    frame = _to_pandas(fetch())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def load_weekly_stats(
    seasons: int | Iterable[int],
    *,
    positions: Iterable[str] = POSITIONS,
    refresh: bool = False,
) -> pd.DataFrame:
    """Weekly player stats, regular season only, for the fantasy positions.

    Returns one row per (player_id, season, week) with ``ID_COLUMNS`` + ``STAT_COLUMNS``. Rows are
    sorted by (player_id, season, week). Raw stat columns are left as they come from nflverse
    (``fantasy_points`` / ``fantasy_points_ppr`` are nflverse's own values and are only used to
    reconcile ``ffai.scoring``).
    """
    seasons_l = _seasons_list(seasons)

    def fetch():  # noqa: ANN202
        if _FALLBACK:
            return _nfl.import_weekly_data(seasons_l).rename(columns=_LEGACY_RENAMES)
        return _nfl.load_player_stats(seasons_l, summary_level="week")

    raw = _cached("player_stats", seasons_l, fetch, refresh)
    missing = [c for c in ID_COLUMNS + STAT_COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError(f"nflverse player stats missing expected columns: {missing}")
    df = raw[list(ID_COLUMNS + STAT_COLUMNS)]
    df = df[(df["season_type"] == SEASON_TYPE) & (df["position"].isin(list(positions)))]
    df = df.copy()
    for col in STAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["season"] = df["season"].astype("int64")
    df["week"] = df["week"].astype("int64")
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    return df


def load_rosters(seasons: int | Iterable[int], *, refresh: bool = False) -> pd.DataFrame:
    """Season rosters (one row per player-season-team) with age-related fields."""
    seasons_l = _seasons_list(seasons)

    def fetch():  # noqa: ANN202
        if _FALLBACK:
            return _nfl.import_rosters(seasons_l)
        return _nfl.load_rosters(seasons_l)

    return _cached("rosters", seasons_l, fetch, refresh)


def load_schedules(seasons: int | Iterable[int], *, refresh: bool = False) -> pd.DataFrame:
    """Game schedules with kickoff dates; used to determine the current season/week."""
    seasons_l = _seasons_list(seasons)

    def fetch():  # noqa: ANN202
        if _FALLBACK:
            return _nfl.import_schedules(seasons_l)
        return _nfl.load_schedules(seasons_l)

    return _cached("schedules", seasons_l, fetch, refresh)


def load_injuries(seasons: int | Iterable[int], *, refresh: bool = False) -> pd.DataFrame:
    """Weekly injury reports. Loaded for completeness; not used as a feature in v1."""
    seasons_l = _seasons_list(seasons)

    def fetch():  # noqa: ANN202
        if _FALLBACK:
            return _nfl.import_injuries(seasons_l)
        return _nfl.load_injuries(seasons_l)

    return _cached("injuries", seasons_l, fetch, refresh)


def current_season_week(schedules: pd.DataFrame, today: dt.date | None = None) -> tuple[int, int]:
    """Return (season, next_week_to_play) from a schedules frame.

    The "next week" is the smallest regular-season week whose games have not all been played as
    of ``today``; if every game of the latest season is done, returns (season, last_week + 1).
    """
    today = today or dt.date.today()
    reg = schedules[schedules["game_type"] == SEASON_TYPE].copy()
    reg["gameday"] = pd.to_datetime(reg["gameday"]).dt.date
    season = int(reg["season"].max())
    reg = reg[reg["season"] == season]
    pending = reg[reg["gameday"] >= today]
    if pending.empty:
        return season, int(reg["week"].max()) + 1
    return season, int(pending["week"].min())
