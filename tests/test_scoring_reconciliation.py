"""The scoring rules must reproduce nflverse's fantasy_points / fantasy_points_ppr exactly."""

import numpy as np
import pandas as pd

from ffai import scoring


def test_rules_match_nflverse_standard_and_ppr(stats: pd.DataFrame) -> None:
    diff = scoring.reconcile(stats, tolerance=0.01)
    assert diff.empty, f"{len(diff)} rows disagree with nflverse:\n{diff.head(20).to_string()}"
    for fmt, col in (("standard", "fantasy_points"), ("ppr", "fantasy_points_ppr")):
        d = (scoring.score(stats, fmt) - stats[col]).abs()
        assert d.max() <= 0.01
        assert stats[col].notna().all()


def test_half_ppr_is_rule_based_not_a_multiplier(stats: pd.DataFrame) -> None:
    std = scoring.score(stats, "standard")
    half = scoring.score(stats, "half")
    ppr = scoring.score(stats, "ppr")
    recs = stats["receptions"].fillna(0.0)
    assert np.allclose(half - std, 0.5 * recs)
    assert np.allclose(ppr - std, 1.0 * recs)
    # A player with 10 receptions and 0 other stats: 5.0 half vs 10.0 ppr, not a fixed ratio.
    row = pd.DataFrame({c: [0.0] for c in scoring.REQUIRED_COLUMNS})
    row["receptions"] = 10.0
    assert scoring.score(row, "half").iloc[0] == 5.0
    assert scoring.score(row, "ppr").iloc[0] == 10.0
    assert scoring.score(row, "standard").iloc[0] == 0.0


def test_score_all_columns(stats: pd.DataFrame) -> None:
    out = scoring.score_all(stats.head(50))
    assert list(out.columns) == ["points_standard", "points_half", "points_ppr"]
    assert len(out) == 50
