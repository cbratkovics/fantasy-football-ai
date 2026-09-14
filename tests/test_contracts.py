"""The contracts wrapper: mapping dbt run results to the weekly run-log actions (ADR-0015).

The checks themselves are dbt tests (dbt/models/silver/_silver.yml, dbt/tests/silver/); CI
runs them with ``dbt build``. These tests cover the Python side only: the pure mapping from a
fixture ``run_results.json`` to the report the policy consumes, the var plumbing, and the
command line, without invoking dbt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ffai.config import REPO_ROOT
from ffai.data import contracts
from ffai.pipeline import weekly

FIXTURE = Path(__file__).parent / "fixtures" / "dbt_run_results.json"


@pytest.fixture(scope="module")
def report() -> dict:
    return contracts.summarise_run_results(json.loads(FIXTURE.read_text(encoding="utf-8")))


def _by_name(report: dict) -> dict[str, dict]:
    return {c["name"]: c for c in report["checks"]}


def test_failing_and_erroring_nodes_make_the_report_not_ok(report: dict) -> None:
    assert report["ok"] is False
    names = _by_name(report)
    assert names["freshness"]["ok"] is False
    assert names["newest_week_row_count"]["ok"] is False
    assert names["model:slv_predictions"]["ok"] is False
    assert names["model:slv_predictions"]["detail"]["status"] == "error"


def test_pass_warn_and_skipped_are_not_failures(report: dict) -> None:
    names = _by_name(report)
    assert names["grain_unique_player_season_week"]["ok"] is True
    assert names["not_null_slv_player_stats_player_id"]["ok"] is True
    warn = names["dbt_expectations_expect_column_values_to_be_between_slv_player_stats_week__22__1"]
    assert warn["ok"] is True and warn["detail"]["status"] == "warn"
    skipped = names["accepted_values_slv_predictions_candidate__rf__xgb"]
    assert skipped["ok"] is True and skipped["detail"]["status"] == "skipped"
    assert names["scoring_rules_standard_half_ppr"]["ok"] is True  # unit tests are checks too
    # Successful models are not listed as checks; only failing ones are.
    assert "model:slv_player_stats" not in names


def test_summary_counts_statuses(report: dict) -> None:
    dbt = report["summary"]["dbt"]
    assert dbt["status_counts"] == {
        "success": 2,
        "error": 1,
        "pass": 3,
        "fail": 2,
        "warn": 1,
        "skipped": 1,
    }
    assert dbt["invocation_id"] == "fixture-0001" and dbt["n_checks"] == 8


def test_all_pass_is_ok() -> None:
    rr = {
        "results": [
            {"status": "success", "unique_id": "model.ffai_dbt.slv_player_stats"},
            {"status": "pass", "unique_id": "test.ffai_dbt.not_null_slv_player_stats_week.ab"},
        ]
    }
    assert contracts.summarise_run_results(rr)["ok"] is True
    assert contracts.summarise_run_results({"results": []})["ok"] is True


def test_failures_map_to_a_hold_with_readable_reasons(report: dict) -> None:
    failures = [weekly._describe_failure(c) for c in report["checks"] if not c["ok"]]
    action, reasons = weekly.decide(
        contract_ok=report["ok"], drift_status="ok", promote_ok=True, contract_failures=failures
    )
    assert action == weekly.ACTION_HOLD
    assert "freshness: nflverse has no complete stats" in reasons[0]
    assert "newest_week_row_count: 1 failing row(s)" in reasons[0]
    assert "model:slv_predictions: dbt error: Binder Error" in reasons[0]


def test_check_names_follow_the_adr_mapping() -> None:
    assert contracts.check_name("test.ffai_dbt.assert_scoring_rules_reconcile_to_nflverse") == (
        "scoring_reconciliation"
    )
    assert (
        contracts.check_name(
            "test.ffai_dbt.dbt_expectations_expect_table_columns_to_contain_set_slv_player_stats_x.1"
        )
        == "required_columns"
    )
    assert (
        contracts.check_name("test.ffai_dbt.accepted_values_slv_player_stats_position__QB__RB.2")
        == "positions_in_scope"
    )
    assert contracts.check_name("test.ffai_dbt.some_other_test.3") == "some_other_test"


def test_dbt_vars_and_command_line() -> None:
    v = contracts.dbt_vars(
        stats_path=REPO_ROOT / "data" / "cache" / "player_stats_2019-2026_2026-09-10.parquet",
        target_season=2026,
        target_week=2,
        expected_through=(2026, 1),
        prior_row_count=40330,
    )
    assert v == {
        "stats_path": "data/cache/player_stats_2019-2026_2026-09-10.parquet",
        "target_season": 2026,
        "target_week": 2,
        "expected_season": 2026,
        "expected_week": 1,
        "prior_row_count": 40330,
    }
    assert contracts.dbt_vars() == {}
    cmd = contracts.dbt_command(v, target="prod")
    assert cmd[3:5] == ["build", "--select"] and cmd[5] == contracts.DBT_SELECT
    assert "--indirect-selection" in cmd and "cautious" in cmd
    assert cmd[cmd.index("--target") + 1] == "prod"
    assert json.loads(cmd[cmd.index("--vars") + 1]) == v


def test_missing_run_results_is_a_failing_dbt_check(tmp_path, monkeypatch) -> None:
    class Proc:
        returncode = 2
        stderr = "boom"

    monkeypatch.setattr(contracts.subprocess, "run", lambda *a, **k: Proc())
    rep = contracts.run_silver_contracts(project_dir=tmp_path, target="dev")
    assert rep["ok"] is False and rep["checks"][0]["name"] == "dbt"
    assert rep["checks"][0]["detail"]["returncode"] == 2
