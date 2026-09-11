"""Grain and cross-player contamination checks for the as-of feature builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffai.config import POSITIONS
from ffai.features import asof

LAG_FEATURES = [n for n in asof.all_feature_names() if "_L" in n or n.endswith("_season_avg")]


def test_feature_frame_is_unique_at_player_season_week(stats: pd.DataFrame) -> None:
    out = asof.build_features(stats)
    assert not out.duplicated(["player_id", "season", "week"]).any()
    assert len(out) == len(stats)


def test_first_row_of_every_player_has_no_inherited_history(stats: pd.DataFrame) -> None:
    out = asof.build_features(stats).sort_values(["player_id", "season", "week"])
    first = out.groupby("player_id", sort=False).head(1)
    # No prior game at all → every lag feature is 0 and L1 is NaN; nothing leaks across players.
    assert (~first[asof.HISTORY_FLAG]).all()
    assert first["fantasy_points_ppr_L1"].isna().all()
    others = [c for c in LAG_FEATURES if c != "fantasy_points_ppr_L1"]
    assert np.nanmax(first[others].abs().to_numpy()) == 0.0
    assert (first["games_played_prior"] == 0).all()


def test_second_row_uses_only_the_players_own_first_game(stats: pd.DataFrame) -> None:
    out = asof.build_features(stats).sort_values(["player_id", "season", "week"])
    grouped = out.groupby("player_id", sort=False)
    second = grouped.nth(1)
    src = stats.sort_values(["player_id", "season", "week"]).groupby("player_id", sort=False).nth(0)
    src = src.set_index("player_id")
    second = second.set_index("player_id")
    common = second.index.intersection(src.index)
    for stat in ("fantasy_points_ppr", "targets", "carries"):
        expected = src.loc[common, stat].fillna(0.0).to_numpy(dtype="float64")
        assert np.allclose(second.loc[common, f"{stat}_L1"].to_numpy(dtype="float64"), expected)
        assert np.allclose(second.loc[common, f"{stat}_L3_avg"].to_numpy(dtype="float64"), expected)


def test_feature_counts_per_position() -> None:
    assert {p: len(asof.features_for_position(p)) for p in POSITIONS} == {
        "QB": 44,
        "RB": 40,
        "WR": 27,
        "TE": 27,
    }
    assert len(asof.all_feature_names()) == 65
    for p in POSITIONS:
        assert set(asof.features_for_position(p)) <= set(asof.all_feature_names())


def test_duplicate_input_rows_are_rejected(stats: pd.DataFrame) -> None:
    dup = pd.concat([stats, stats.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        asof.build_features(dup)


def test_target_collision_is_rejected(stats: pd.DataFrame) -> None:
    t = stats.head(1)[["player_id", "season", "week"]]
    with pytest.raises(ValueError, match="already exist"):
        asof.build_features(stats, targets=t)
