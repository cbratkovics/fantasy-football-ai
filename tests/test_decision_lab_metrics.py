"""Outcome metrics for a saved Decision Lab choice set (``ffai.decision_lab.metrics``).

Outcomes are joined after a decision exists. Every metric has its own eligibility rule and a
named reason when it is null; nothing is imputed, and extreme outcomes never touch the result.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from ffai.decision_lab import golden, metrics, policy
from ffai.decision_lab.canonical import content_id
from ffai.decision_lab.golden import A, B, _alt, _inputs, _params

GOLDEN = json.loads(golden.GOLDEN_PATH.read_text(encoding="utf-8"))
GOLDEN_METRICS = {c["name"]: c for c in GOLDEN["metrics"]}

RESULT = policy.evaluate(_inputs([A, B], _params(0.3, 5.0)))  # recommend A, baseline prefers B
AT = "2099-01-01T00:00:00+00:00"
NO_ACTION = {
    "state": "not_recorded",
    "action_id": None,
    "at_utc": None,
    "chosen_player_id": None,
    "kind": None,
    "note": None,
}
DECLINED = {
    "state": "declined",
    "action_id": "b" * 64,
    "at_utc": AT,
    "chosen_player_id": None,
    "kind": None,
    "note": "no choice",
}


def recorded(pid: str, kind: str = "hypothetical_replay") -> dict[str, Any]:
    return {
        "state": "recorded",
        "action_id": "a" * 64,
        "at_utc": AT,
        "chosen_player_id": pid,
        "kind": kind,
        "note": None,
    }


def outcomes(rows: dict[str, float | None], outcome_id: str = "1" * 64) -> dict[str, Any]:
    return {
        "schema_version": "outcome_snapshot/1.0",
        "snapshot_id": "syn-golden",
        "outcome_id": outcome_id,
        "rows": [{"player_id": pid, "actual": actual} for pid, actual in rows.items()],
    }


FULL = outcomes({"SYN-A": 8.0, "SYN-B": 16.0})


# --- golden ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GOLDEN_METRICS))
def test_golden_metrics_case_reproduces(name: str) -> None:
    case = GOLDEN_METRICS[name]
    assert (
        metrics.compute_metrics(case["result"], case["action"], case["outcome_snapshot"])
        == case["expected"]
    )


def test_golden_metrics_cases_share_one_result() -> None:
    ids = {c["result"]["decision_id"] for c in GOLDEN["metrics"]}
    assert len(ids) == 1
    assert GOLDEN["metrics"][0]["result"] == RESULT


# --- chosen / regret / baseline comparisons ------------------------------------------------------------


def test_chosen_a_with_full_outcomes() -> None:
    m = metrics.compute_metrics(RESULT, recorded("SYN-A"), FULL)
    assert m["schema_version"] == "outcome_metrics/1.0"
    assert m["decision_id"] == RESULT["decision_id"]
    assert m["outcome_snapshot_id"] == "syn-golden" and m["outcome_id"] == "1" * 64
    assert m["action_state"] == "recorded" and m["action_kind"] == "hypothetical_replay"
    assert m["chosen_player_id"] == "SYN-A"
    assert m["choice_set"] == ["SYN-A", "SYN-B"]
    assert m["coverage"] == {"nominated": 2, "choice_set_size": 2, "observed": 2, "missing": []}
    assert m["chosen_actual_points"] == 8.0 and m["chosen_actual_reason"] is None
    assert m["best_in_choice_set"] == {"player_id": "SYN-B", "actual": 16.0}
    assert m["choice_set_regret"] == 8.0 and m["choice_set_regret_reason"] is None
    assert m["baseline_preferred_player_id"] == "SYN-B" and m["baseline_preferred_actual"] == 16.0
    assert m["points_vs_baseline_choice"] == -8.0 and m["points_vs_baseline_choice_reason"] is None
    assert m["model_recommended_player_id"] == "SYN-A" and m["model_recommended_actual"] == 8.0
    assert m["model_policy_vs_baseline_choice"] == -8.0
    assert m["model_policy_vs_baseline_choice_reason"] is None
    assert m["per_alternative"] == [
        {"player_id": "SYN-A", "actual": 8.0, "observed": True},
        {"player_id": "SYN-B", "actual": 16.0, "observed": True},
    ]
    assert len(m["notes"]) == 2 and "not a recorded action" in m["notes"][0]


def test_chosen_b_beats_the_model_recommendation() -> None:
    m = metrics.compute_metrics(RESULT, recorded("SYN-B", "self_reported_real"), FULL)
    assert m["action_kind"] == "self_reported_real"
    assert m["chosen_actual_points"] == 16.0
    assert m["choice_set_regret"] == 0.0
    assert m["points_vs_baseline_choice"] == 0.0
    # The model-vs-baseline comparison does not depend on the user's action.
    assert m["model_policy_vs_baseline_choice"] == -8.0


def test_no_action_gives_nulls_with_reasons_but_still_compares_model_to_baseline() -> None:
    m = metrics.compute_metrics(RESULT, NO_ACTION, FULL)
    assert m["action_state"] == "not_recorded" and m["action_kind"] is None
    assert m["chosen_player_id"] is None
    assert m["chosen_actual_points"] is None and m["chosen_actual_reason"] == "no_action_recorded"
    assert m["best_in_choice_set"] == {"player_id": "SYN-B", "actual": 16.0}
    assert m["choice_set_regret"] is None and m["choice_set_regret_reason"] == "no_action_recorded"
    assert m["points_vs_baseline_choice"] is None
    assert m["points_vs_baseline_choice_reason"] == "no_action_recorded"
    assert m["model_policy_vs_baseline_choice"] == -8.0
    assert m["model_policy_vs_baseline_choice_reason"] is None


def test_declined_action_reasons() -> None:
    m = metrics.compute_metrics(RESULT, DECLINED, FULL)
    assert m["action_state"] == "declined" and m["action_kind"] is None
    assert m["chosen_player_id"] is None
    assert m["chosen_actual_reason"] == "declined"
    assert m["choice_set_regret"] is None and m["choice_set_regret_reason"] == "declined"
    assert m["points_vs_baseline_choice_reason"] == "declined"
    assert m["model_policy_vs_baseline_choice"] == -8.0


def test_recorded_state_is_required_for_a_chosen_player() -> None:
    # A chosen_player_id on a non-recorded action block is ignored, never trusted.
    m = metrics.compute_metrics(RESULT, {**NO_ACTION, "chosen_player_id": "SYN-A"}, FULL)
    assert m["chosen_player_id"] is None and m["chosen_actual_reason"] == "no_action_recorded"


def test_missing_alternative_outcome_leaves_regret_null() -> None:
    partial = outcomes({"SYN-A": 8.0, "SYN-B": None}, "2" * 64)
    m = metrics.compute_metrics(RESULT, recorded("SYN-A"), partial)
    assert m["chosen_actual_points"] == 8.0 and m["chosen_actual_reason"] is None
    assert m["coverage"] == {
        "nominated": 2,
        "choice_set_size": 2,
        "observed": 1,
        "missing": ["SYN-B"],
    }
    assert m["best_in_choice_set"] is None
    assert m["choice_set_regret"] is None and m["choice_set_regret_reason"] == "incomplete_outcomes"
    assert m["baseline_preferred_actual"] is None
    assert m["points_vs_baseline_choice"] is None
    assert m["points_vs_baseline_choice_reason"] == "baseline_outcome_missing"
    assert m["model_policy_vs_baseline_choice"] is None
    assert m["model_policy_vs_baseline_choice_reason"] == "outcome_missing"
    assert m["per_alternative"][1] == {"player_id": "SYN-B", "actual": None, "observed": False}


def test_absent_row_is_the_same_as_a_null_actual() -> None:
    absent = outcomes({"SYN-A": 8.0}, "2" * 64)
    null = outcomes({"SYN-A": 8.0, "SYN-B": None}, "2" * 64)
    assert metrics.compute_metrics(RESULT, recorded("SYN-A"), absent) == metrics.compute_metrics(
        RESULT, recorded("SYN-A"), null
    )


def test_chosen_players_own_outcome_missing() -> None:
    partial = outcomes({"SYN-A": 8.0, "SYN-B": None}, "2" * 64)
    m = metrics.compute_metrics(RESULT, recorded("SYN-B"), partial)
    assert m["chosen_player_id"] == "SYN-B"
    assert m["chosen_actual_points"] is None and m["chosen_actual_reason"] == "outcome_missing"
    assert m["choice_set_regret_reason"] == "incomplete_outcomes"
    assert m["points_vs_baseline_choice_reason"] == "outcome_missing"


def test_chosen_outcome_missing_with_complete_choice_set() -> None:
    # The chosen player is nominated but excluded, and has no outcome: regret carries the reason.
    excluded = _alt("SYN-C", 12.0, 5.0, 10.0, availability="unavailable", basis="user_excluded")
    result = policy.evaluate(_inputs([A, B, excluded], _params(0.3, None)))
    m = metrics.compute_metrics(result, recorded("SYN-C"), FULL)
    assert m["choice_set"] == ["SYN-A", "SYN-B"]
    assert m["coverage"]["missing"] == []
    assert m["chosen_actual_reason"] == "outcome_missing"
    assert m["choice_set_regret"] is None and m["choice_set_regret_reason"] == "outcome_missing"
    assert m["best_in_choice_set"] == {"player_id": "SYN-B", "actual": 16.0}


def test_zero_and_negative_actuals_are_valid_observations() -> None:
    m = metrics.compute_metrics(
        RESULT, recorded("SYN-B"), outcomes({"SYN-A": 0.0, "SYN-B": -1.2}, "3" * 64)
    )
    assert m["coverage"]["observed"] == 2 and m["coverage"]["missing"] == []
    assert m["chosen_actual_points"] == -1.2 and m["chosen_actual_reason"] is None
    assert m["best_in_choice_set"] == {"player_id": "SYN-A", "actual": 0.0}
    assert m["choice_set_regret"] == 1.2 and m["choice_set_regret_reason"] is None
    assert m["points_vs_baseline_choice"] == 0.0
    assert m["model_policy_vs_baseline_choice"] == 1.2
    assert m["per_alternative"][0] == {"player_id": "SYN-A", "actual": 0.0, "observed": True}


def test_best_in_choice_set_ties_break_by_player_id() -> None:
    m = metrics.compute_metrics(RESULT, recorded("SYN-B"), outcomes({"SYN-A": 10.0, "SYN-B": 10.0}))
    assert m["best_in_choice_set"] == {"player_id": "SYN-A", "actual": 10.0}
    assert m["choice_set_regret"] == 0.0


def test_chosen_outside_the_choice_set() -> None:
    excluded = _alt("SYN-B", 13.6, 7.0, 13.0, availability="unavailable", basis="user_excluded")
    result = policy.evaluate(
        _inputs([A, excluded, _alt("SYN-C", 12.0, 5.0, 10.0)], _params(0.3, None))
    )
    assert result["recommended_player_id"] == "SYN-A"
    m = metrics.compute_metrics(
        result, recorded("SYN-B"), outcomes({"SYN-A": 8.0, "SYN-B": 16.0, "SYN-C": 3.0})
    )
    assert m["choice_set"] == ["SYN-A", "SYN-C"]
    assert m["coverage"] == {"nominated": 3, "choice_set_size": 2, "observed": 2, "missing": []}
    assert m["chosen_player_id"] == "SYN-B"
    assert m["chosen_actual_points"] == 16.0 and m["chosen_actual_reason"] is None
    assert m["best_in_choice_set"] == {"player_id": "SYN-A", "actual": 8.0}
    assert m["choice_set_regret"] is None
    assert m["choice_set_regret_reason"] == "chosen_outside_choice_set"
    # The baseline agrees with the model here (A), so the user's pick is compared against A.
    assert m["baseline_preferred_player_id"] == "SYN-A"
    assert m["points_vs_baseline_choice"] == 8.0
    assert m["model_policy_vs_baseline_choice"] == 0.0
    assert [p["player_id"] for p in m["per_alternative"]] == ["SYN-A", "SYN-B", "SYN-C"]


def test_no_baseline_preference_reasons() -> None:
    tie = policy.evaluate(
        _inputs(
            [_alt("SYN-A", 14.0, 6.0, 12.0), _alt("SYN-B", 13.0, 7.0, 12.0)], _params(0.3, None)
        )
    )
    assert tie["baseline"]["status"] == "tie"
    m = metrics.compute_metrics(tie, recorded("SYN-A"), FULL)
    assert m["baseline_preferred_player_id"] is None and m["baseline_preferred_actual"] is None
    assert m["points_vs_baseline_choice"] is None
    assert m["points_vs_baseline_choice_reason"] == "no_baseline_preference"
    assert m["model_policy_vs_baseline_choice"] is None
    assert m["model_policy_vs_baseline_choice_reason"] == "no_baseline_preference"
    assert m["choice_set_regret"] == 8.0  # regret does not need a baseline


def test_no_model_recommendation_reason() -> None:
    review = policy.evaluate(_inputs([A, B], _params(0.5, 8.0)))
    assert review["status"] == "review" and review["recommended_player_id"] is None
    m = metrics.compute_metrics(review, recorded("SYN-A"), FULL)
    assert m["model_recommended_player_id"] is None and m["model_recommended_actual"] is None
    assert m["model_policy_vs_baseline_choice"] is None
    assert m["model_policy_vs_baseline_choice_reason"] == "no_model_recommendation"
    # The user's own comparison against the baseline is still available.
    assert m["points_vs_baseline_choice"] == -8.0


def test_empty_choice_set_reason() -> None:
    empty = policy.evaluate(_inputs([], _params(0.3, None)))
    m = metrics.compute_metrics(empty, NO_ACTION, FULL)
    assert m["choice_set"] == [] and m["coverage"]["choice_set_size"] == 0
    assert m["choice_set_regret_reason"] == "empty_choice_set"
    assert m["best_in_choice_set"] is None


def test_extreme_outcomes_never_alter_the_result() -> None:
    result = copy.deepcopy(RESULT)
    before = content_id(result)
    m = metrics.compute_metrics(
        result, recorded("SYN-A"), outcomes({"SYN-A": 999.5, "SYN-B": -50.0}, "4" * 64)
    )
    assert content_id(result) == before
    assert result == RESULT
    assert m["chosen_actual_points"] == 999.5
    assert m["choice_set_regret"] == 0.0
    assert m["points_vs_baseline_choice"] == 1049.5
    assert m["model_policy_vs_baseline_choice"] == 1049.5
    assert m["decision_id"] == RESULT["decision_id"]


def test_outcome_index_ignores_non_numeric_and_bool_actuals() -> None:
    idx = metrics.outcome_index(
        {
            "rows": [
                {"player_id": "a", "actual": 1.5},
                {"player_id": "b", "actual": None},
                {"player_id": "c", "actual": True},
                {"player_id": "d", "actual": "7"},
                {"player_id": "e", "actual": 0},
            ]
        }
    )
    assert idx == {"a": 1.5, "e": 0.0}


def test_metrics_values_are_quantized() -> None:
    m = metrics.compute_metrics(
        RESULT, recorded("SYN-A"), outcomes({"SYN-A": 0.1 + 0.2, "SYN-B": 0.6})
    )
    assert m["chosen_actual_points"] == 0.3
    assert m["choice_set_regret"] == 0.3
    assert m["points_vs_baseline_choice"] == -0.3


# --- aggregate ---------------------------------------------------------------------------------------


def _row(**kw: Any) -> dict[str, Any]:
    base = {
        "action_state": "recorded",
        "chosen_actual_points": None,
        "choice_set_regret": None,
        "points_vs_baseline_choice": None,
        "model_policy_vs_baseline_choice": None,
    }
    return base | kw


def test_aggregate_uses_metric_specific_denominators() -> None:
    rows = [
        _row(
            chosen_actual_points=10.0,
            choice_set_regret=2.0,
            points_vs_baseline_choice=1.0,
            model_policy_vs_baseline_choice=3.0,
        ),
        _row(
            chosen_actual_points=20.0,
            choice_set_regret=None,
            points_vs_baseline_choice=-1.0,
            model_policy_vs_baseline_choice=-3.0,
        ),
        _row(action_state="not_recorded", model_policy_vs_baseline_choice=0.0),
        _row(action_state="declined"),
    ]
    agg = metrics.aggregate(rows)
    assert agg["rows"] == 4
    assert agg["with_recorded_action"] == 2
    assert agg["chosen_actual_points"] == {"n": 2, "mean": 15.0}
    assert agg["choice_set_regret"] == {"n": 1, "mean": 2.0}
    assert agg["points_vs_baseline_choice"] == {"n": 2, "mean": 0.0}
    assert agg["model_policy_vs_baseline_choice"] == {"n": 3, "mean": 0.0}
    assert agg["model_beats_baseline"] == {"n": 3, "wins": 1, "losses": 1, "ties": 1}


def test_aggregate_zero_denominators_are_null_with_zero_counts() -> None:
    agg = metrics.aggregate([])
    assert agg["rows"] == 0 and agg["with_recorded_action"] == 0
    for key in (
        "chosen_actual_points",
        "choice_set_regret",
        "points_vs_baseline_choice",
        "model_policy_vs_baseline_choice",
    ):
        assert agg[key] == {"n": 0, "mean": None}
    assert agg["model_beats_baseline"] == {"n": 0, "wins": 0, "losses": 0, "ties": 0}
    only_nulls = metrics.aggregate([_row(), _row(action_state="not_recorded")])
    assert only_nulls["rows"] == 2 and only_nulls["chosen_actual_points"] == {"n": 0, "mean": None}


def test_aggregate_pools_rows_and_never_re_averages_group_means() -> None:
    small = [_row(choice_set_regret=10.0)]
    large = [_row(choice_set_regret=0.0) for _ in range(9)]
    pooled = metrics.aggregate(small + large)["choice_set_regret"]
    assert pooled == {"n": 10, "mean": 1.0}
    group_means = [metrics.aggregate(g)["choice_set_regret"]["mean"] for g in (small, large)]
    assert sum(group_means) / 2 == 5.0
    assert pooled["mean"] != 5.0


def test_aggregate_ignores_non_numeric_values_and_bools() -> None:
    agg = metrics.aggregate(
        [_row(choice_set_regret="3"), _row(choice_set_regret=True), _row(choice_set_regret=3)]
    )
    # bool is an int subclass and is counted as a number by isinstance; the reference implementation
    # accepts it, and the value 1.0 enters the mean. Strings are ignored.
    assert agg["choice_set_regret"]["n"] == 2
    assert agg["choice_set_regret"]["mean"] == 2.0


def test_aggregate_wins_losses_ties_on_golden_metrics() -> None:
    rows = [c["expected"] for c in GOLDEN["metrics"]]
    agg = metrics.aggregate(rows)
    assert agg["rows"] == len(rows)
    assert agg["with_recorded_action"] == sum(1 for r in rows if r["action_state"] == "recorded")
    mb = agg["model_beats_baseline"]
    assert mb["n"] == mb["wins"] + mb["losses"] + mb["ties"]
    assert mb["n"] == agg["model_policy_vs_baseline_choice"]["n"]
    assert mb["n"] == sum(1 for r in rows if r["model_policy_vs_baseline_choice"] is not None)
