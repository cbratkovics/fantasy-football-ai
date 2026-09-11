"""The weekly publish/hold/promote policy on synthetic step outcomes."""

from __future__ import annotations

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
