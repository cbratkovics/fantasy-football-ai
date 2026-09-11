"""Leakage test on real data.

For a random sample of (player, season, week) targets we recompute every feature from scratch
using only that player's rows strictly before the target, and assert equality with the module's
output. We also perturb the target week's own stats (and a later week's stats) and assert that no
feature of the target row changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffai.features import asof

N_TARGETS = 200
SEED = 20260910


def _independent_features(history: pd.DataFrame, season: int, week: int) -> dict[str, float]:
    """Reference implementation using only ``history`` rows (all strictly before the target)."""
    hist = history.sort_values(["season", "week"])
    same_season = hist[hist["season"] == season]
    out: dict[str, float] = {}
    for stat in asof.BASE_STATS:
        vals = hist[stat].astype("float64").to_numpy()
        vals_season = same_season[stat].astype("float64").to_numpy()
        out[f"{stat}_L1"] = float(vals[-1]) if len(vals) else np.nan
        out[f"{stat}_L3_avg"] = float(np.nanmean(vals[-3:])) if len(vals) else 0.0
        out[f"{stat}_L5_avg"] = float(np.nanmean(vals[-5:])) if len(vals) else 0.0
        out[f"{stat}_season_avg"] = float(np.nanmean(vals_season)) if len(vals_season) else 0.0

    def ratio(n: str, d: str) -> float:
        num, den = out[n], out[d]
        return num / (den if den != 0 else 1.0)

    out["completion_pct_L3"] = ratio("completions_L3_avg", "attempts_L3_avg")
    out["yards_per_carry_L3"] = ratio("rushing_yards_L3_avg", "carries_L3_avg")
    out["catch_rate_L3"] = ratio("receptions_L3_avg", "targets_L3_avg")
    out["week_of_season"] = float(week)
    out["games_played_prior"] = float(len(same_season))
    # Module semantics: every NaN becomes 0 except L1 when there is no history at all.
    has_history = len(vals) > 0
    for k, v in out.items():
        if isinstance(v, float) and np.isnan(v):
            out[k] = np.nan if (k == "fantasy_points_ppr_L1" and not has_history) else 0.0
    return out


def _sample_targets(stats: pd.DataFrame, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(stats), size=min(n, len(stats)), replace=False)
    return stats.iloc[np.sort(idx)][["player_id", "season", "week"]].reset_index(drop=True)


def test_features_equal_independent_recomputation(stats: pd.DataFrame) -> None:
    feats = asof.build_features(stats).set_index(["player_id", "season", "week"])
    targets = _sample_targets(stats, N_TARGETS)
    names = asof.all_feature_names()
    for pid, season, week in targets.itertuples(index=False):
        player_rows = stats[stats["player_id"] == pid]
        history = player_rows[
            (player_rows["season"] < season)
            | ((player_rows["season"] == season) & (player_rows["week"] < week))
        ]
        expected = _independent_features(history, int(season), int(week))
        got = feats.loc[(pid, season, week), names]
        for name in names:
            e, g = expected[name], float(got[name])
            if np.isnan(e):
                assert np.isnan(g), f"{pid} {season} wk{week} {name}: expected NaN, got {g}"
            else:
                assert g == pytest.approx(
                    e, abs=1e-9
                ), f"{pid} {season} wk{week} {name}: {g} != {e}"


def test_perturbing_target_or_future_week_does_not_change_features(stats: pd.DataFrame) -> None:
    base = asof.build_features(stats).set_index(["player_id", "season", "week"])
    # One target per player: perturbing weeks >= target A must not be confused with the
    # legitimate effect on a later target B of the same player.
    targets = _sample_targets(stats, 120).drop_duplicates("player_id").head(40)
    names = asof.all_feature_names()
    rng = np.random.default_rng(SEED + 1)
    perturbed = stats.copy()
    key = perturbed.set_index(["player_id", "season", "week"]).index
    for pid, season, week in targets.itertuples(index=False):
        # Perturb the target row itself and every later row of the same player.
        later = (perturbed["player_id"] == pid) & (
            (perturbed["season"] > season)
            | ((perturbed["season"] == season) & (perturbed["week"] >= week))
        )
        for stat in asof.BASE_STATS:
            perturbed.loc[later, stat] = perturbed.loc[later, stat].fillna(0) + rng.uniform(50, 500)
    after = asof.build_features(perturbed).set_index(["player_id", "season", "week"])
    assert len(key) == len(after)
    for pid, season, week in targets.itertuples(index=False):
        b = base.loc[(pid, season, week), names].to_numpy(dtype="float64")
        a = after.loc[(pid, season, week), names].to_numpy(dtype="float64")
        assert np.array_equal(
            np.nan_to_num(b, nan=-1), np.nan_to_num(a, nan=-1)
        ), f"features of {pid} {season} wk{week} changed when the target/future weeks changed"


def test_future_week_targets_use_prior_rows_only(stats: pd.DataFrame) -> None:
    last = stats.sort_values(["season", "week"]).iloc[-1]
    season, week = int(last["season"]), int(last["week"])
    players = stats[stats["season"] == season]["player_id"].drop_duplicates().head(25)
    targets = pd.DataFrame({"player_id": players, "season": season, "week": week + 1})
    out = asof.build_features(stats, targets=targets)
    stubs = out[out[asof.TARGET_FLAG]]
    assert len(stubs) == len(players)
    assert stubs["target"].isna().all()
    feats = stubs.set_index("player_id")
    for pid in players:
        history = stats[stats["player_id"] == pid]
        expected = _independent_features(history, season, week + 1)
        for name in asof.all_feature_names():
            e, g = expected[name], float(feats.loc[pid, name])
            if np.isnan(e):
                assert np.isnan(g)
            else:
                assert g == pytest.approx(e, abs=1e-9), f"{pid} {name}: {g} != {e}"
    # Non-target rows are unchanged by adding targets.
    without = asof.build_features(stats).set_index(["player_id", "season", "week"])
    with_t = out[~out[asof.TARGET_FLAG]].set_index(["player_id", "season", "week"])
    pd.testing.assert_frame_equal(
        without[asof.all_feature_names()].sort_index(),
        with_t[asof.all_feature_names()].sort_index(),
    )
