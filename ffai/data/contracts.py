"""Data-contract checks for the weekly stats frame.

Pure functions that return a report dict; nothing raises. The weekly pipeline HOLDs when
``report["ok"]`` is false. Checks:

* required columns present
* grain uniqueness at (player_id, season, week)
* value ranges (season >= MIN_SEASON, 1 <= week <= 22, non-negative counting stats)
* null rates on scoring-relevant columns
* freshness (the latest (season, week) is at least the expected one)
* row-count sanity versus a prior snapshot (rows may only grow; growth for the newest week must
  be within a plausible band)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ffai.config import MIN_SEASON, POSITIONS
from ffai.data.nflverse import ID_COLUMNS, STAT_COLUMNS

GRAIN: tuple[str, ...] = ("player_id", "season", "week")
# Counting stats only. Yardage (rushing/receiving/air yards) is legitimately negative at times,
# and fantasy points can be negative.
NON_NEGATIVE: tuple[str, ...] = (
    "completions",
    "attempts",
    "passing_tds",
    "passing_interceptions",
    "passing_2pt_conversions",
    "sack_fumbles_lost",
    "carries",
    "rushing_tds",
    "rushing_fumbles_lost",
    "rushing_2pt_conversions",
    "receptions",
    "targets",
    "receiving_tds",
    "receiving_fumbles_lost",
    "receiving_2pt_conversions",
    "special_teams_tds",
)
MAX_NULL_RATE = 0.02
MAX_WEEK = 22
# One regular-season week of QB/RB/WR/TE stat rows has been ~330-420 rows since 2019.
WEEK_ROWS_MIN = 200
WEEK_ROWS_MAX = 600


def _check(name: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def check_stats_contract(
    df: pd.DataFrame,
    *,
    expected_through: tuple[int, int] | None = None,
    prior_row_count: int | None = None,
) -> dict[str, Any]:
    """Run every contract check and return ``{"ok": bool, "checks": [...], "summary": {...}}``."""
    checks: list[dict[str, Any]] = []

    required = list(ID_COLUMNS + STAT_COLUMNS)
    missing = [c for c in required if c not in df.columns]
    checks.append(_check("required_columns", not missing, {"missing": missing}))
    if missing:
        return {"ok": False, "checks": checks, "summary": {"rows": int(len(df))}}

    dupes = int(df.duplicated(list(GRAIN)).sum())
    checks.append(_check("grain_unique_player_season_week", dupes == 0, {"duplicates": dupes}))

    bad_season = int((df["season"] < MIN_SEASON).sum())
    bad_week = int(((df["week"] < 1) | (df["week"] > MAX_WEEK)).sum())
    checks.append(
        _check(
            "value_ranges",
            bad_season == 0 and bad_week == 0,
            {"bad_season": bad_season, "bad_week": bad_week},
        )
    )

    negatives = {c: int((df[c] < 0).sum()) for c in NON_NEGATIVE if c in df.columns}
    negatives = {c: n for c, n in negatives.items() if n}
    checks.append(_check("non_negative_counting_stats", not negatives, negatives))

    bad_pos = sorted(set(df["position"].dropna().unique()) - set(POSITIONS))
    checks.append(_check("positions_in_scope", not bad_pos, {"unexpected": bad_pos}))

    null_rates = {c: float(df[c].isna().mean()) for c in ("fantasy_points_ppr", "position", "team")}
    high = {c: r for c, r in null_rates.items() if r > MAX_NULL_RATE}
    checks.append(_check("null_rates", not high, {"rates": null_rates, "threshold": MAX_NULL_RATE}))

    if len(df):
        latest = df.sort_values(["season", "week"]).iloc[-1]
        latest_sw = (int(latest["season"]), int(latest["week"]))
    else:
        latest_sw = (0, 0)
    if expected_through is not None:
        checks.append(
            _check(
                "freshness",
                latest_sw >= tuple(expected_through),
                {"latest": latest_sw, "expected_through": list(expected_through)},
            )
        )

    if len(df):
        newest_week_rows = int(
            ((df["season"] == latest_sw[0]) & (df["week"] == latest_sw[1])).sum()
        )
        checks.append(
            _check(
                "newest_week_row_count",
                WEEK_ROWS_MIN <= newest_week_rows <= WEEK_ROWS_MAX,
                {"rows": newest_week_rows, "band": [WEEK_ROWS_MIN, WEEK_ROWS_MAX]},
            )
        )

    if prior_row_count is not None:
        grew = len(df) >= prior_row_count
        checks.append(
            _check(
                "row_count_monotonic",
                grew,
                {"rows": int(len(df)), "prior_rows": int(prior_row_count)},
            )
        )

    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "summary": {
            "rows": int(len(df)),
            "players": int(df["player_id"].nunique()) if len(df) else 0,
            "latest": list(latest_sw),
        },
    }
