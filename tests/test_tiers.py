"""Tier builder: component cap and the keep-previous comparison rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffai.models import tiers


def _inputs(n_players: int, position: str = "QB", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "player_id": [f"{position}{i}" for i in range(n_players)],
            "position": position,
            "team": "T",
            "player_display_name": [f"P{i}" for i in range(n_players)],
            "ppg_prev": rng.normal(12, 5, n_players).clip(0),
            "ppg_std_prev": rng.normal(6, 2, n_players).clip(0.5),
            "games_prev": rng.integers(4, 18, n_players),
            "opportunity_share_prev": rng.uniform(0.01, 0.6, n_players),
            "season": 2024,
        }
    )


@pytest.mark.parametrize("n_players", [60, 100, 190])
def test_components_are_capped_by_players_per_component(n_players: int) -> None:
    fitted = tiers.fit_position_tiers(_inputs(n_players), "QB")
    cap = max(tiers.COMPONENT_RANGE.start, n_players // tiers.MIN_PLAYERS_PER_COMPONENT)
    assert fitted["max_components_allowed"] == cap
    assert fitted["n_components"] <= cap
    assert all(k <= cap for k in fitted["bic"])
    assert fitted["n_components"] >= tiers.COMPONENT_RANGE.start
    assert set(fitted["players"]["tier"]) == set(range(1, fitted["n_components"] + 1))


def test_worse_on_both_rule() -> None:
    before = {"spearman": 0.6, "within_band_rate": 0.3}
    assert tiers._worse_on_both(before, {"spearman": 0.5, "within_band_rate": 0.2})
    assert not tiers._worse_on_both(before, {"spearman": 0.5, "within_band_rate": 0.4})
    assert not tiers._worse_on_both(before, {"spearman": 0.7, "within_band_rate": 0.2})
    assert not tiers._worse_on_both(before, {"spearman": None, "within_band_rate": 0.1})


def test_tier_evaluation_bands() -> None:
    players = pd.DataFrame({"player_id": list("abcdef"), "tier": [1, 1, 2, 2, 3, 3]})
    realised = pd.DataFrame({"player_id": list("abcdef"), "ppg": [20, 19, 15, 14, 5, 4]})
    ev = tiers.evaluate_tiers(players, realised)
    assert ev["within_band_rate"] == 1.0 and ev["spearman"] > 0.9
    flipped = realised.assign(ppg=[4, 5, 14, 15, 19, 20])
    ev2 = tiers.evaluate_tiers(players, flipped)
    assert ev2["spearman"] < 0 and ev2["within_band_rate"] < 0.5
