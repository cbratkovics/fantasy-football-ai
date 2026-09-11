"""As-of feature builder — THE feature module.

Every feature for target row ``(player_id, season, week)`` is computed from that player's stat
rows with ``(season, week)`` strictly earlier than the target. Nothing from the target row or any
later row is used. This is enforced by construction (all statistics are computed on a series that
has already been shifted by one game within the player's group) and by
``tests/test_asof_no_leakage.py`` on real data.

Feature scheme (ported from the legacy production trainer, see ``ffai/_legacy``):

* for each of the 15 base stats: ``{stat}_L1`` (previous game), ``{stat}_L3_avg`` and
  ``{stat}_L5_avg`` (mean of the previous ≤3 / ≤5 games, crossing season boundaries),
  ``{stat}_season_avg`` (mean of previous games in the same season; NaN → 0 at week 1)
* three efficiency ratios of L3 averages: ``completion_pct_L3``, ``yards_per_carry_L3``,
  ``catch_rate_L3``
* ``week_of_season`` (the target week) and ``games_played_prior`` (prior stat rows this season)

Position filters reproduce the legacy counts: QB 44, RB 40, WR 27, TE 27.

Differences from the legacy script, all deliberate:
1. Rolling / expanding windows are computed inside ``groupby("player_id")`` on the shifted
   series, so a player's first row never inherits another player's value (the legacy code
   shifted the flattened groupby result and relied on positional ``.values`` assignment).
2. Ratios are computed for every row where their inputs exist instead of only for one position;
   the position filter decides which ratios a model sees.
3. ``interceptions`` is now ``passing_interceptions`` (current nflverse column name).
4. ``games_played`` is renamed ``games_played_prior`` to say what it is.

A "game" is a stat row: weeks without a row (bye, inactive) are simply absent from the history,
exactly as in the legacy trainer.
"""

from __future__ import annotations

import pandas as pd

from ffai.config import POSITIONS, TARGET

FEATURE_VERSION = "asof_v1"

BASE_STATS: tuple[str, ...] = (
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "receptions",
    "targets",
    "fantasy_points_ppr",
    "attempts",
    "completions",
    "carries",
    "passing_interceptions",
    "passing_air_yards",
    "receiving_air_yards",
)
LAGS: tuple[str, ...] = ("L1", "L3_avg", "L5_avg", "season_avg")
RATIO_FEATURES: tuple[str, ...] = ("completion_pct_L3", "yards_per_carry_L3", "catch_rate_L3")
TEMPORAL_FEATURES: tuple[str, ...] = ("week_of_season", "games_played_prior")
KEY_COLUMNS: tuple[str, ...] = ("player_id", "season", "week")
CONTEXT_COLUMNS: tuple[str, ...] = ("position", "team", "player_display_name")
HISTORY_FLAG = "has_history"
TARGET_FLAG = "is_target"

# Substrings excluded per position, verbatim from the legacy trainer.
_EXCLUDE_NON_QB = ("passing", "completion", "attempts", "interceptions")
_EXCLUDE_NON_RUSHER = ("rushing", "carries", "yards_per_carry")
_EXCLUDE_NON_RECEIVER = ("receiving", "receptions", "targets", "catch_rate")


def all_feature_names() -> list[str]:
    """Every feature column produced by :func:`build_features`, in a stable order."""
    names = [f"{stat}_{lag}" for stat in BASE_STATS for lag in LAGS]
    names.extend(RATIO_FEATURES)
    names.extend(TEMPORAL_FEATURES)
    return names


def features_for_position(position: str) -> list[str]:
    """Feature subset for a position (QB 44, RB 40, WR 27, TE 27)."""
    if position not in POSITIONS:
        raise ValueError(f"unknown position {position!r}")
    names = all_feature_names()
    if position != "QB":
        names = [n for n in names if not any(t in n for t in _EXCLUDE_NON_QB)]
    if position not in ("RB", "QB"):
        names = [n for n in names if not any(t in n for t in _EXCLUDE_NON_RUSHER)]
    if position not in ("WR", "TE", "RB"):
        names = [n for n in names if not any(t in n for t in _EXCLUDE_NON_RECEIVER)]
    return names


FEATURES_BY_POSITION: dict[str, list[str]] = {p: features_for_position(p) for p in POSITIONS}


def _shifted(df: pd.DataFrame, stat: str, by: list[str]) -> pd.Series:
    """The stat series shifted by one row within ``by`` groups (row order = game order)."""
    return df.groupby(by, sort=False)[stat].shift(1)


def _rolling_mean(df: pd.DataFrame, shifted: pd.Series, window: int) -> pd.Series:
    tmp = df[["player_id"]].assign(_v=shifted)
    out = (
        tmp.groupby("player_id", sort=False)["_v"]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return out.reindex(df.index)


def _expanding_mean(df: pd.DataFrame, shifted: pd.Series) -> pd.Series:
    tmp = df[["player_id", "season"]].assign(_v=shifted)
    out = (
        tmp.groupby(["player_id", "season"], sort=False)["_v"]
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    return out.reindex(df.index)


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.where(den != 0, 1.0)


def build_features(
    stats: pd.DataFrame,
    targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build as-of features at (player_id, season, week) grain.

    Parameters
    ----------
    stats
        Weekly stat rows (one per player-game) with ``KEY_COLUMNS``, ``CONTEXT_COLUMNS`` and the
        ``BASE_STATS`` columns. Only rows that are actually in ``stats`` contribute history.
    targets
        Optional future rows to score: columns ``player_id``, ``season``, ``week`` (and
        optionally ``position``, ``team``, ``player_display_name``). They must not collide with
        a row already in ``stats``. Their features are computed from prior stat rows only; their
        target value is NaN and ``is_target`` is True.

    Returns
    -------
    DataFrame with ``KEY_COLUMNS`` + ``CONTEXT_COLUMNS`` + ``target`` (the realised PPR points,
    NaN for target stubs) + ``has_history`` + ``is_target`` + every feature name from
    :func:`all_feature_names`. Feature NaNs are filled with 0 *except* that rows with
    ``has_history == False`` (no prior game at all) keep NaN in ``fantasy_points_ppr_L1`` so callers
    can drop them; every other NaN is 0.
    """
    required = list(KEY_COLUMNS) + ["position"] + list(BASE_STATS)
    missing = [c for c in required if c not in stats.columns]
    if missing:
        raise KeyError(f"build_features requires columns {missing}")

    df = stats.copy()
    df[TARGET_FLAG] = False
    if targets is not None and len(targets):
        t = targets.copy()
        for c in KEY_COLUMNS:
            if c not in t.columns:
                raise KeyError(f"targets requires column {c!r}")
        collide = t.merge(df[list(KEY_COLUMNS)], on=list(KEY_COLUMNS), how="inner")
        if len(collide):
            raise ValueError(f"{len(collide)} target rows already exist in stats")
        # Carry forward the player's latest context if the target does not supply it.
        latest_ctx = (
            df.sort_values(["player_id", "season", "week"])
            .groupby("player_id", sort=False)[list(CONTEXT_COLUMNS)]
            .last()
        )
        for c in CONTEXT_COLUMNS:
            if c not in t.columns:
                t[c] = t["player_id"].map(latest_ctx[c])
        t[TARGET_FLAG] = True
        for stat in BASE_STATS:
            t[stat] = float("nan")
        df = pd.concat([df, t[df.columns.intersection(t.columns)]], ignore_index=True)

    if df.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("stats has duplicate (player_id, season, week) rows")

    df = df.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
    for stat in BASE_STATS:
        df[stat] = pd.to_numeric(df[stat], errors="coerce").astype("float64")

    feats: dict[str, pd.Series] = {}
    for stat in BASE_STATS:
        s_prev = _shifted(df, stat, ["player_id"])
        feats[f"{stat}_L1"] = s_prev
        feats[f"{stat}_L3_avg"] = _rolling_mean(df, s_prev, 3)
        feats[f"{stat}_L5_avg"] = _rolling_mean(df, s_prev, 5)
        s_prev_season = _shifted(df, stat, ["player_id", "season"])
        feats[f"{stat}_season_avg"] = _expanding_mean(df, s_prev_season)

    feats["completion_pct_L3"] = _safe_ratio(feats["completions_L3_avg"], feats["attempts_L3_avg"])
    feats["yards_per_carry_L3"] = _safe_ratio(
        feats["rushing_yards_L3_avg"], feats["carries_L3_avg"]
    )
    feats["catch_rate_L3"] = _safe_ratio(feats["receptions_L3_avg"], feats["targets_L3_avg"])
    feats["week_of_season"] = df["week"].astype("float64")
    feats["games_played_prior"] = (
        df.groupby(["player_id", "season"], sort=False).cumcount().astype("float64")
    )

    features = pd.DataFrame(feats, index=df.index)[all_feature_names()]
    has_history = features["fantasy_points_ppr_L1"].notna()
    features = features.fillna(0.0)
    features.loc[~has_history, "fantasy_points_ppr_L1"] = float("nan")

    out = df[list(KEY_COLUMNS) + list(CONTEXT_COLUMNS)].copy()
    out["target"] = df[TARGET].where(~df[TARGET_FLAG])
    out[HISTORY_FLAG] = has_history.to_numpy()
    out[TARGET_FLAG] = df[TARGET_FLAG].to_numpy()
    out = pd.concat([out, features], axis=1)
    return out


def training_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for supervised training: realised target and at least one prior game."""
    return features[(~features[TARGET_FLAG]) & features[HISTORY_FLAG]].copy()
