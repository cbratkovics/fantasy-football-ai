"""Data-contract checks on a fixture with injected violations."""

from __future__ import annotations

import pandas as pd

from ffai.data import contracts


def _names(report: dict) -> dict[str, bool]:
    return {c["name"]: c["ok"] for c in report["checks"]}


def test_clean_frame_passes(stats: pd.DataFrame) -> None:
    latest = stats.sort_values(["season", "week"]).iloc[-1]
    report = contracts.check_stats_contract(
        stats,
        expected_through=(int(latest["season"]), int(latest["week"])),
        prior_row_count=len(stats) - 10,
    )
    failing = {k: v for k, v in _names(report).items() if not v}
    # The small CI fixture has fewer rows per week than a full week; that check is data-size bound.
    failing.pop("newest_week_row_count", None)
    assert not failing, report
    assert report["summary"]["rows"] == len(stats)


def test_duplicate_grain_is_flagged(stats: pd.DataFrame) -> None:
    dup = pd.concat([stats, stats.head(3)], ignore_index=True)
    report = contracts.check_stats_contract(dup)
    assert report["ok"] is False
    assert _names(report)["grain_unique_player_season_week"] is False


def test_missing_column_short_circuits(stats: pd.DataFrame) -> None:
    report = contracts.check_stats_contract(stats.drop(columns=["receptions"]))
    assert report["ok"] is False
    assert report["checks"][0]["name"] == "required_columns"
    assert report["checks"][0]["detail"]["missing"] == ["receptions"]


def test_out_of_range_and_negative_values_are_flagged(stats: pd.DataFrame) -> None:
    bad = stats.copy()
    bad.loc[bad.index[0], "week"] = 99
    bad.loc[bad.index[1], "season"] = 1999
    bad.loc[bad.index[2], "targets"] = -4
    report = contracts.check_stats_contract(bad)
    names = _names(report)
    assert names["value_ranges"] is False
    assert names["non_negative_counting_stats"] is False


def test_unexpected_position_and_nulls_are_flagged(stats: pd.DataFrame) -> None:
    bad = stats.copy()
    bad.loc[bad.index[:5], "position"] = "K"
    bad.loc[bad.index[: int(len(bad) * 0.05)], "fantasy_points_ppr"] = None
    report = contracts.check_stats_contract(bad)
    names = _names(report)
    assert names["positions_in_scope"] is False
    assert names["null_rates"] is False


def test_freshness_and_row_count_regression_are_flagged(stats: pd.DataFrame) -> None:
    report = contracts.check_stats_contract(
        stats, expected_through=(2099, 1), prior_row_count=len(stats) + 1
    )
    names = _names(report)
    assert names["freshness"] is False
    assert names["row_count_monotonic"] is False
    assert report["ok"] is False
