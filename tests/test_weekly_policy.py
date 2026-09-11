"""The weekly publish/hold/promote policy on synthetic step outcomes."""

from __future__ import annotations

import pytest

from ffai.pipeline import weekly


def test_publish_when_everything_is_clean() -> None:
    action, reasons = weekly.decide(
        contract_ok=True, drift_status="ok", promote_ok=False, promote_reason="challenger won 2/4"
    )
    assert action == weekly.ACTION_PUBLISH
    assert any("contracts ok" in r for r in reasons)
    assert any("no promotion" in r for r in reasons)


def test_hold_on_contract_failure_regardless_of_drift() -> None:
    action, reasons = weekly.decide(
        contract_ok=False,
        drift_status="ok",
        promote_ok=True,
        contract_failures=["grain_unique_player_season_week", "freshness"],
    )
    assert action == weekly.ACTION_HOLD
    assert "grain_unique_player_season_week" in reasons[0] and "freshness" in reasons[0]


def test_hold_on_drift_even_if_promotion_rule_passes() -> None:
    action, reasons = weekly.decide(
        contract_ok=True,
        drift_status="hold",
        promote_ok=True,
        drift_features=["RB:targets_L3_avg"],
    )
    assert action == weekly.ACTION_HOLD
    assert "RB:targets_L3_avg" in reasons[0]


def test_promote_carries_the_rule_reason_and_drift_warning() -> None:
    action, reasons = weekly.decide(
        contract_ok=True,
        drift_status="warn",
        promote_ok=True,
        promote_reason="challenger won all 4 recent weeks and frozen test (4.40 < 4.55)",
        drift_features=["WR:receiving_yards_L1"],
    )
    assert action == weekly.ACTION_PROMOTE
    assert reasons[0].startswith("drift warning on WR:receiving_yards_L1")
    assert "promotion rule satisfied" in reasons[1]


def test_rolling_file_roundtrip(tmp_path) -> None:
    assert weekly.read_rolling(2025, tmp_path) == {"season": 2025, "weeks": []}
    p = weekly.rolling_path(2025, tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text('{"season": 2025, "weeks": [{"week": 3}]}', encoding="utf-8")
    assert weekly.read_rolling(2025, tmp_path)["weeks"] == [{"week": 3}]


def _relax_for_fixture(monkeypatch, stats) -> None:
    """The 40-player fixture is far smaller than a real pull: drop the size-dependent contract
    and drift thresholds and the committed manifest's prior row count."""
    from ffai.data import contracts, nflverse
    from ffai.eval import drift
    from ffai.models import registry

    real_read = registry.read_manifest

    def read_without_last_run(*args, **kwargs):
        m = real_read(*args, **kwargs)
        m.pop("last_run", None)
        return m

    monkeypatch.setattr(registry, "read_manifest", read_without_last_run)
    monkeypatch.setattr(nflverse, "load_weekly_stats", lambda seasons, **kw: stats)
    monkeypatch.setattr(contracts, "WEEK_ROWS_MIN", 1)
    monkeypatch.setattr(drift, "HOLD_PSI", 99.0)
    monkeypatch.setattr(drift, "SEVERE_PSI", 99.0)


def test_first_week_of_a_season_has_nothing_to_score(monkeypatch, stats) -> None:
    """Week 1: no prior predictions file → the rolling step records the reason and the job
    still scores the week."""
    from ffai.config import ARTIFACTS_DIR

    if not (ARTIFACTS_DIR / "manifest.json").exists():
        pytest.skip("no committed manifest")
    season = int(stats["season"].max()) + 1
    _relax_for_fixture(monkeypatch, stats)
    log = weekly.run_weekly(season=season, week=1, dry_run=True, write_model_card=False)
    steps = {s["step"]: s for s in log["steps"]}
    assert steps["contracts"]["ok"]
    assert steps["rolling_eval"]["week"] is None and "week 1" in steps["rolling_eval"]["note"]
    assert log["action"] == weekly.ACTION_PUBLISH
    assert steps["score"]["week"] == 1 and steps["score"]["n"] > 0
    assert weekly.read_rolling(season, ARTIFACTS_DIR) == {"season": season, "weeks": []}


def test_partial_target_week_rows_are_dropped_before_contracts(monkeypatch, stats) -> None:
    from ffai.config import ARTIFACTS_DIR

    if not (ARTIFACTS_DIR / "manifest.json").exists():
        pytest.skip("no committed manifest")
    last = stats.sort_values(["season", "week"]).iloc[-1]
    season, week = int(last["season"]), int(last["week"])
    _relax_for_fixture(monkeypatch, stats)
    # Scoring the last fixture week: its own rows are "partial" and must not be used.
    log = weekly.run_weekly(season=season, week=week, dry_run=True, write_model_card=False)
    st = {s["step"]: s for s in log["steps"]}["stats"]
    assert st["dropped_partial_rows"] == int(
        ((stats.season == season) & (stats.week == week)).sum()
    )
    assert st["rows_before_target_week"] + st["dropped_partial_rows"] == st["loaded_rows"]


def test_describe_failure_names_the_missing_week() -> None:
    msg = weekly._describe_failure(
        {
            "name": "freshness",
            "ok": False,
            "detail": {"expected_through": [2026, 2], "latest": [2026, 1]},
        }
    )
    assert "season 2026 week 2" in msg and "season 2026 week 1" in msg
    assert weekly._describe_failure({"name": "grain_unique_player_season_week", "ok": False}) == (
        "grain_unique_player_season_week"
    )
