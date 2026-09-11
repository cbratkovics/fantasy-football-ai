"""Rules-based fantasy scoring from raw nflverse stat columns.

The rules reproduce nflverse's own ``fantasy_points`` (standard) and ``fantasy_points_ppr``
definitions, which are computed in nflfastR's ``calculate_player_stats``:

    standard = passing_yards / 25 + 4 * passing_tds - 2 * passing_interceptions
             + rushing_yards / 10 + 6 * rushing_tds
             + receiving_yards / 10 + 6 * receiving_tds
             + 2 * (passing + rushing + receiving 2pt conversions)
             - 2 * (sack + rushing + receiving fumbles lost)
             + 6 * special_teams_tds
    ppr      = standard + 1.0 * receptions
    half     = standard + 0.5 * receptions

``tests/test_scoring_reconciliation.py`` asserts that ``score(df, "standard")`` and
``score(df, "ppr")`` match nflverse's columns to within 0.01 on every 2019-2024 row. Half-PPR is
derived from the same rules with a 0.5 reception weight, never from a multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

ScoringFormat = Literal["standard", "half", "ppr"]
FORMATS: tuple[ScoringFormat, ...] = ("standard", "half", "ppr")


@dataclass(frozen=True)
class ScoringRules:
    """Point values per stat unit."""

    passing_yard: float = 0.04  # 1 point per 25 yards
    passing_td: float = 4.0
    passing_interception: float = -2.0
    rushing_yard: float = 0.1
    rushing_td: float = 6.0
    receiving_yard: float = 0.1
    receiving_td: float = 6.0
    reception: float = 0.0
    two_point_conversion: float = 2.0
    fumble_lost: float = -2.0
    special_teams_td: float = 6.0


RULES: dict[ScoringFormat, ScoringRules] = {
    "standard": ScoringRules(reception=0.0),
    "half": ScoringRules(reception=0.5),
    "ppr": ScoringRules(reception=1.0),
}

REQUIRED_COLUMNS: tuple[str, ...] = (
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "receptions",
    "passing_2pt_conversions",
    "rushing_2pt_conversions",
    "receiving_2pt_conversions",
    "sack_fumbles_lost",
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
    "special_teams_tds",
)


def score(df: pd.DataFrame, fmt: ScoringFormat) -> pd.Series:
    """Fantasy points per row under ``fmt`` computed from raw stat columns.

    Missing stat values are treated as zero (nflverse leaves a stat null when a player had no
    opportunities in that category).
    """
    if fmt not in RULES:
        raise ValueError(f"unknown scoring format {fmt!r}; expected one of {FORMATS}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"scoring requires columns {missing}")
    r = RULES[fmt]
    s = df[list(REQUIRED_COLUMNS)].astype("float64").fillna(0.0)
    points = (
        r.passing_yard * s["passing_yards"]
        + r.passing_td * s["passing_tds"]
        + r.passing_interception * s["passing_interceptions"]
        + r.rushing_yard * s["rushing_yards"]
        + r.rushing_td * s["rushing_tds"]
        + r.receiving_yard * s["receiving_yards"]
        + r.receiving_td * s["receiving_tds"]
        + r.reception * s["receptions"]
        + r.two_point_conversion
        * (
            s["passing_2pt_conversions"]
            + s["rushing_2pt_conversions"]
            + s["receiving_2pt_conversions"]
        )
        + r.fumble_lost
        * (s["sack_fumbles_lost"] + s["rushing_fumbles_lost"] + s["receiving_fumbles_lost"])
        + r.special_teams_td * s["special_teams_tds"]
    )
    points.name = f"points_{fmt}"
    return points


def score_all(df: pd.DataFrame) -> pd.DataFrame:
    """All three formats as columns ``points_standard``, ``points_half``, ``points_ppr``."""
    return pd.concat([score(df, f) for f in FORMATS], axis=1)


def reconcile(df: pd.DataFrame, tolerance: float = 0.01) -> pd.DataFrame:
    """Rows where the rules disagree with nflverse's ``fantasy_points`` / ``fantasy_points_ppr``.

    Returns an empty frame when everything reconciles. Rows where nflverse's own column is null
    are ignored.
    """
    out = []
    for fmt, col in (("standard", "fantasy_points"), ("ppr", "fantasy_points_ppr")):
        if col not in df.columns:
            continue
        ours = score(df, fmt)
        theirs = pd.to_numeric(df[col], errors="coerce")
        mask = theirs.notna() & ((ours - theirs).abs() > tolerance)
        if mask.any():
            diff = df.loc[mask, ["player_id", "season", "week"]].copy()
            diff["format"] = fmt
            diff["ours"] = ours[mask]
            diff["nflverse"] = theirs[mask]
            diff["abs_diff"] = (ours[mask] - theirs[mask]).abs()
            out.append(diff)
    if not out:
        return pd.DataFrame(
            columns=["player_id", "season", "week", "format", "ours", "nflverse", "abs_diff"]
        )
    return pd.concat(out, ignore_index=True)
