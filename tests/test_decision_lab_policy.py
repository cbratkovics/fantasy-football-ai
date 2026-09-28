"""Executable specification of the Decision Lab policy (``ffai.decision_lab.policy``, 1.0.0).

Every test states the policy rule it pins: status, recommended player, the ordered reason codes,
gap and gate values, the baseline block, limitations, and explanation text. The golden fixture
(``tests/fixtures/decision_lab/golden.json``) is the contract shared with the TypeScript mirror;
the first test fails loudly when it is stale.
"""

from __future__ import annotations

import copy
import json
import math
import random
from typing import Any

import pytest

from ffai.decision_lab import POLICY_VERSION, SCHEMA_VERSIONS, contracts, golden, policy, spec
from ffai.decision_lab.canonical import CanonicalError, content_id, format_number
from ffai.decision_lab.golden import A, B, _alt, _inputs, _params, _snapshot

SPEC = spec()
GOLDEN = json.loads(golden.GOLDEN_PATH.read_text(encoding="utf-8"))
GOLDEN_POLICY = {c["name"]: c for c in GOLDEN["policy"]}
HOLD_STATUSES = {"hold"}


def codes(result: dict[str, Any]) -> list[str]:
    return [r["code"] for r in result["reasons"]]


def evaluate(alts: list[dict[str, Any]], params: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return policy.evaluate(_inputs(alts, params, **kw))


# --- 1. golden currency ---------------------------------------------------------------------------


def test_golden_fixture_is_current() -> None:
    assert golden.check(), (
        "golden fixture is stale: run `.venv/bin/python -m ffai.decision_lab.golden --write` "
        "and commit tests/fixtures/decision_lab/golden.json"
    )
    assert GOLDEN["policy_version"] == POLICY_VERSION


@pytest.mark.parametrize("name", sorted(GOLDEN_POLICY))
def test_golden_policy_case_reproduces_exactly(name: str) -> None:
    case = GOLDEN_POLICY[name]
    result = policy.evaluate(case["inputs"])
    assert result == case["expected"]
    assert result["decision_id"] == policy.decision_id(case["inputs"])
    contracts.check_decision_result(result)


def test_golden_policy_cases_cover_every_status_and_reason_code() -> None:
    statuses = {c["expected"]["status"] for c in GOLDEN["policy"]}
    assert statuses == set(SPEC["statuses"])
    seen = {code for c in GOLDEN["policy"] for code in codes(c["expected"])}
    assert seen == set(SPEC["reasons"]), sorted(set(SPEC["reasons"]) - seen)


# --- 2. ambiguity cannot be bypassed by floor relaxation ------------------------------------------


@pytest.mark.parametrize("min_floor", [8.0, 5.0, 0.0, None])
def test_gap_below_minimum_is_review_regardless_of_floor(min_floor: float | None) -> None:
    result = evaluate([A, B], _params(0.5, min_floor))
    assert result["status"] == "review"
    assert result["recommended_player_id"] is None
    assert codes(result)[0] == "gap_below_minimum"
    assert result["leading"] == {"first": "SYN-A", "second": "SYN-B", "gap": 0.4}
    assert result["gates"]["min_projection_gap"] == 0.5
    assert result["gates"]["gap_ok"] is False
    if min_floor is None:
        assert "floor_guardrail_disabled" in codes(result)
        assert result["gates"]["floor_checked"] is False
        assert result["gates"]["floor_ok"] is None
    else:
        assert result["gates"]["floor_checked"] is True
        assert result["gates"]["min_floor"] == min_floor
        assert result["gates"]["leading_floor"] == 6.0
        assert result["gates"]["floor_ok"] is (min_floor <= 6.0)


def test_ambiguity_floor_8_reason_order_and_explanations() -> None:
    result = evaluate([A, B], _params(0.5, 8.0))
    assert codes(result) == ["gap_below_minimum", "floor_below_minimum", "baseline_disagrees"]
    assert result["explanation"] == [
        "The projection gap 0.4 between Synthetic A (SYN-A) and Synthetic B (SYN-B) is below the "
        "minimum 0.5.",
        "Synthetic A (SYN-A) leads on projection but its model floor 6 is below the guardrail 8.",
        "The causal trailing-mean baseline prefers Synthetic B (SYN-B) (13) instead; the baseline "
        "is a separate comparator, not the model.",
    ]


def test_lowering_the_gap_resolves_ambiguity_but_not_the_baseline_disagreement() -> None:
    result = evaluate([A, B], _params(0.3, 5.0))
    assert result["status"] == "recommend"
    assert result["recommended_player_id"] == "SYN-A"
    assert codes(result) == [
        "recommend_highest_projection",
        "gap_meets_minimum",
        "floor_meets_minimum",
        "baseline_disagrees",
    ]
    assert result["leading"]["gap"] == 0.4
    assert result["gates"] == {
        "min_projection_gap": 0.3,
        "gap_ok": True,
        "min_floor": 5.0,
        "floor_checked": True,
        "leading_floor": 6.0,
        "floor_ok": True,
    }
    assert result["baseline"]["status"] == "available"
    assert result["baseline"]["preferred_player_id"] == "SYN-B"
    assert result["baseline"]["preferred_value"] == 13.0
    assert result["baseline"]["agrees_with_model_leader"] is False
    assert result["baseline"]["missing_for"] == []
    assert result["counts"] == {"nominated": 2, "excluded": 0, "comparable": 2}


# --- 3. ties and floating-point boundaries --------------------------------------------------------


def test_exact_tie_is_review_even_with_zero_min_gap() -> None:
    result = evaluate(
        [_alt("SYN-Y", 10.0, 1.0, 6.0), _alt("SYN-X", 10.0, 1.0, 5.0)], _params(0.0, None)
    )
    assert result["status"] == "review"
    assert result["recommended_player_id"] is None
    assert codes(result) == ["projection_tie", "floor_guardrail_disabled", "baseline_disagrees"]
    assert result["leading"] == {"first": "SYN-X", "second": "SYN-Y", "gap": 0.0}
    assert result["gates"]["gap_ok"] is False
    rows = {r["player_id"]: r for r in result["ranking"]}
    assert rows["SYN-X"]["rank"] == 1 and rows["SYN-X"]["tied_with_next"] is True
    assert rows["SYN-Y"]["rank"] == 2 and rows["SYN-Y"]["tied_with_next"] is False
    assert result["explanation"][0] == (
        "The leading projections tie at 10; the order shown is by player id only, not evidence."
    )


def test_tie_order_is_by_player_id_only_and_baseline_preference_does_not_break_it() -> None:
    # Y has the better baseline, X has the smaller id: X ranks first, the baseline prefers Y.
    result = evaluate(
        [_alt("SYN-X", 10.0, 1.0, 5.0), _alt("SYN-Y", 10.0, 1.0, 6.0)], _params(0.0, None)
    )
    assert result["leading"]["first"] == "SYN-X"
    assert result["baseline"]["preferred_player_id"] == "SYN-Y"
    assert result["baseline"]["agrees_with_model_leader"] is False


def test_result_is_invariant_under_alternative_order() -> None:
    alts = [
        _alt("SYN-C", 12.0, 3.0, 9.0),
        _alt("SYN-A", 12.0, 3.0, 9.5),
        _alt("SYN-B", 15.0, 3.0, 9.0),
    ]
    reference = evaluate(alts, _params(0.3, None))
    rng = random.Random(7)
    for _ in range(6):
        shuffled = list(alts)
        rng.shuffle(shuffled)
        result = evaluate(shuffled, _params(0.3, None))
        assert result == reference
        assert result["decision_id"] == reference["decision_id"]
    assert [r["player_id"] for r in reference["ranking"]] == ["SYN-A", "SYN-B", "SYN-C"]
    ranks = {r["player_id"]: (r["rank"], r["tied_with_next"]) for r in reference["ranking"]}
    assert ranks == {"SYN-B": (1, False), "SYN-A": (2, True), "SYN-C": (3, False)}
    assert reference["status"] == "recommend" and reference["recommended_player_id"] == "SYN-B"


def test_floating_point_sum_ties_after_quantization() -> None:
    result = evaluate(
        [_alt("SYN-P", 0.1 + 0.2, 0.0, 1.0), _alt("SYN-Q", 0.3, 0.0, 0.5)], _params(0.0, None)
    )
    assert result["status"] == "review"
    assert codes(result)[0] == "projection_tie"
    assert result["leading"]["gap"] == 0.0
    assert {r["projection"] for r in result["ranking"]} == {0.3}


def test_values_within_tolerance_tie() -> None:
    result = evaluate(
        [_alt("SYN-A", 10.0, 1.0, 5.0), _alt("SYN-B", 10.0 + 5e-8, 1.0, 6.0)], _params(0.0, None)
    )
    assert result["status"] == "review"
    assert codes(result)[0] == "projection_tie"
    assert result["leading"]["gap"] == 0.0
    assert result["ranking"][0]["tied_with_next"] is True


def test_gap_exactly_equal_to_minimum_qualifies() -> None:
    result = evaluate([A, B], _params(0.4, None))
    assert result["status"] == "recommend"
    assert result["recommended_player_id"] == "SYN-A"
    assert codes(result) == [
        "recommend_highest_projection",
        "gap_meets_minimum",
        "floor_guardrail_disabled",
        "baseline_disagrees",
    ]
    assert result["gates"]["gap_ok"] is True
    assert result["explanation"][1] == "The projection gap 0.4 meets the minimum 0.4."


def test_gap_just_below_minimum_fails() -> None:
    result = evaluate([A, B], _params(0.401, None))
    assert result["status"] == "review"
    assert codes(result)[0] == "gap_below_minimum"


def test_floor_exactly_equal_to_guardrail_qualifies() -> None:
    result = evaluate([A, B], _params(0.3, 6.0))
    assert result["status"] == "recommend"
    assert result["recommended_player_id"] == "SYN-A"
    assert "floor_meets_minimum" in codes(result)
    assert result["gates"]["floor_ok"] is True
    assert result["gates"]["leading_floor"] == 6.0 and result["gates"]["min_floor"] == 6.0


def test_floor_lower_by_a_thousandth_fails() -> None:
    result = evaluate([A, B], _params(0.3, 6.001))
    assert result["status"] == "review"
    assert result["recommended_player_id"] is None
    assert codes(result) == ["gap_meets_minimum", "floor_below_minimum", "baseline_disagrees"]
    assert result["gates"]["floor_ok"] is False
    assert result["explanation"][1] == (
        "Synthetic A (SYN-A) leads on projection but its model floor 6 is below the guardrail 6.001."
    )


def test_floor_within_tolerance_of_guardrail_qualifies() -> None:
    result = evaluate([A, B], _params(0.3, 6.0 + 5e-7))
    assert result["status"] == "recommend"
    assert result["gates"]["floor_ok"] is True


# --- 4. precedence: HOLD beats everything -----------------------------------------------------------


HOLD_CASES: dict[str, tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str]] = {
    "evidence_corrupt": (
        [A, B],
        _params(0.0, None),
        {"snapshot": _snapshot(evidence_status="corrupt")},
        "evidence_corrupt",
    ),
    "evidence_blocked": (
        [A, B],
        _params(0.0, None),
        {"snapshot": _snapshot(publication_status="blocked", note="export failed")},
        "evidence_blocked",
    ),
    "scoring_unsupported": (
        [A, B],
        _params(0.0, None),
        {"snapshot": _snapshot(scoring="half")},
        "scoring_format_unsupported",
    ),
    "slot_unsupported": ([A, B], _params(0.0, None), {"slot": "K"}, "slot_unsupported"),
    "mixed_periods": (
        [A, _alt("SYN-B", 13.6, 7.0, 13.0, week=2)],
        _params(0.0, None),
        {},
        "mixed_periods",
    ),
    "mixed_model_versions": (
        [A, _alt("SYN-B", 13.6, 7.0, 13.0, model_version="other-model")],
        _params(0.0, None),
        {},
        "mixed_model_versions",
    ),
    "mixed_candidates": (
        [A, _alt("SYN-B", 13.6, 7.0, 13.0, candidate="xgb")],
        _params(0.0, None),
        {},
        "mixed_model_versions",
    ),
    "negative_gap": ([A, B], _params(-1.0, None), {}, "parameters_invalid"),
    "gap_above_max": ([A, B], _params(50.5, None), {}, "parameters_invalid"),
    "string_floor": ([A, B], {**_params(0.3, None), "min_floor": "8"}, {}, "parameters_invalid"),
    "bool_gap": (
        [A, B],
        {**_params(0.3, None), "min_projection_gap": True},
        {},
        "parameters_invalid",
    ),
    "unknown_key": ([A, B], {**_params(0.0, None), "min_ceiling": 1.0}, {}, "parameters_invalid"),
    "non_bool_flag": (
        [A, B],
        {**_params(0.0, None), "require_independent_baseline": "yes"},
        {},
        "parameters_invalid",
    ),
    "missing_flag": (
        [A, B],
        {"min_projection_gap": 0.0, "min_floor": None, "require_independent_baseline": True},
        {},
        "parameters_invalid",
    ),
    "choice_set_too_large": (
        [_alt(f"SYN-{i}", 10.0 + i, 1.0, 5.0) for i in range(9)],
        _params(0.0, None),
        {},
        "choice_set_too_large",
    ),
    "duplicate": ([A, dict(A), B], _params(0.0, None), {}, "duplicate_alternative"),
}


@pytest.mark.parametrize("name", sorted(HOLD_CASES))
def test_hold_has_no_recommendation_and_is_not_relaxed_by_thresholds(name: str) -> None:
    alts, params, kw, code = HOLD_CASES[name]
    result = evaluate(alts, params, **kw)
    assert result["status"] == "hold"
    assert result["recommended_player_id"] is None
    assert code in codes(result)
    hold_codes = [r["code"] for r in result["reasons"] if r["severity"] == "hold"]
    assert hold_codes and hold_codes[0] == code
    assert "recommend_highest_projection" not in codes(result)
    contracts.check_decision_result(result)
    # Relaxing thresholds to zero never turns a HOLD into anything weaker.
    if "parameters_invalid" not in code:
        relaxed = evaluate(alts, {**params, "min_projection_gap": 0.0, "min_floor": None}, **kw)
        assert relaxed["status"] == "hold"
        assert relaxed["recommended_player_id"] is None
        relaxed_floor0 = evaluate(
            alts, {**params, "min_projection_gap": 0.0, "min_floor": 0.0}, **kw
        )
        assert relaxed_floor0["status"] == "hold"


def test_corrupt_evidence_is_a_first_class_reason_ahead_of_gap_and_floor() -> None:
    result = evaluate([A, B], _params(0.0, 0.0), snapshot=_snapshot(evidence_status="corrupt"))
    assert result["status"] == "hold"
    assert result["recommended_player_id"] is None
    assert codes(result) == [
        "evidence_corrupt",
        "gap_meets_minimum",
        "floor_meets_minimum",
        "baseline_disagrees",
    ]
    assert result["reasons"][0]["severity"] == "hold"
    assert result["reasons"][0]["detail"] == {"detail": "byte digest mismatch"}
    assert (
        result["explanation"][0]
        == "Snapshot evidence failed its integrity check: byte digest mismatch."
    )
    # The evidence is still computed and shown (gap, floor) but never promoted to a recommendation.
    assert result["leading"]["gap"] == 0.4 and result["gates"]["gap_ok"] is True


def test_blocked_publication_hold_carries_the_note() -> None:
    result = evaluate(
        [A, B],
        _params(0.1, None),
        snapshot=_snapshot(publication_status="blocked", note="lab export failed"),
    )
    assert result["status"] == "hold"
    assert codes(result)[0] == "evidence_blocked"
    assert result["explanation"][0] == "The snapshot carries an evidence block: lab export failed."


def test_unsupported_scoring_and_slot_holds_render_the_value() -> None:
    result = evaluate([A, B], _params(0.1, None), snapshot=_snapshot(scoring="half"))
    assert result["explanation"][0] == (
        "Scoring format half is not supported; the lab compares PPR projections only."
    )
    result = evaluate([A, B], _params(0.1, None), slot="K")
    assert result["status"] == "hold"
    assert codes(result)[0] == "slot_unsupported"
    assert result["explanation"][0] == "Slot K is not supported."
    # No slot means no eligible position: everything is excluded, and the HOLD still dominates.
    assert all(r["exclusion"] == "excluded_slot_ineligible" for r in result["ranking"])


def test_mixed_periods_detail_lists_every_period_and_the_snapshot() -> None:
    result = evaluate([A, _alt("SYN-B", 13.6, 7.0, 13.0, week=2)], _params(0.1, None))
    assert codes(result)[0] == "mixed_periods"
    assert result["reasons"][0]["detail"]["detail"] == ["2099-w1", "2099-w2", "snapshot 2099-w1"]
    # A period that differs from the snapshot even when the alternatives agree is also a HOLD.
    result = evaluate(
        [_alt("SYN-A", 14.0, 6.0, 12.0, season=2098), _alt("SYN-B", 13.6, 7.0, 13.0, season=2098)],
        _params(0.1, None),
    )
    assert result["status"] == "hold" and codes(result)[0] == "mixed_periods"


def test_mixed_model_versions_and_candidates_hold() -> None:
    result = evaluate(
        [A, _alt("SYN-B", 13.6, 7.0, 13.0, model_version="other-model")], _params(0.1, None)
    )
    assert result["status"] == "hold" and codes(result)[0] == "mixed_model_versions"
    assert result["reasons"][0]["detail"]["detail"] == ["other-model/rf", "synthetic-model-v1/rf"]
    result = evaluate([A, _alt("SYN-B", 13.6, 7.0, 13.0, candidate="xgb")], _params(0.1, None))
    assert result["status"] == "hold" and codes(result)[0] == "mixed_model_versions"


@pytest.mark.parametrize(
    ("params", "fragment"),
    [
        (_params(-1.0, None), "min_projection_gap must be between 0 and 50"),
        ({**_params(0.3, None), "min_floor": "8"}, "min_floor must be null or a finite number"),
        ({**_params(0.3, None), "min_floor": 61.0}, "min_floor must be between 0 and 60"),
        ({**_params(0.3, None), "unknown_knob": 1}, "unknown parameter(s): unknown_knob"),
        (
            {**_params(0.3, None), "model_only_exploration": 1},
            "model_only_exploration must be a boolean",
        ),
        (
            {**_params(0.3, None), "min_projection_gap": "0.3"},
            "min_projection_gap must be a finite number",
        ),
    ],
)
def test_parameters_invalid_names_the_problem(params: dict[str, Any], fragment: str) -> None:
    result = evaluate([A, B], params)
    assert result["status"] == "hold"
    assert result["recommended_player_id"] is None
    assert codes(result) == ["parameters_invalid"]
    assert fragment in result["reasons"][0]["detail"]["detail"]
    # With invalid parameters no gate is evaluated at all, and with no model leader the baseline
    # block records its preference without an agrees/disagrees claim.
    assert result["leading"] == {"first": None, "second": None, "gap": None}
    assert result["gates"]["gap_ok"] is None and result["gates"]["floor_checked"] is False
    assert result["baseline"]["preferred_player_id"] == "SYN-B"
    assert result["baseline"]["agrees_with_model_leader"] is None


def test_nan_gap_cannot_be_hashed_and_raises() -> None:
    """Non-finite numbers cannot travel in JSON and the inputs contract rejects them, so the
    policy refuses to compute an identity for them instead of inventing one (documented)."""
    with pytest.raises(CanonicalError):
        evaluate([A, B], _params(float("nan"), None))


def test_nan_parameter_is_rejected_by_the_inputs_contract() -> None:
    with pytest.raises(contracts.ContractError, match="min_projection_gap"):
        contracts.check_decision_inputs(_inputs([A, B], _params(float("nan"), None)))


def test_choice_set_too_large_hold() -> None:
    alts = [_alt(f"SYN-{i}", 10.0 + i, 1.0, 5.0) for i in range(9)]
    result = evaluate(alts, _params(0.3, None))
    assert result["status"] == "hold"
    assert codes(result)[0] == "choice_set_too_large"
    assert result["reasons"][0]["detail"] == {"count": 9, "limit": 8}
    assert result["explanation"][0] == "9 alternatives were nominated; the maximum is 8."
    assert result["counts"]["nominated"] == 9
    # Exactly eight is allowed.
    ok = evaluate(alts[:8], _params(0.3, None))
    assert ok["status"] == "recommend" and ok["recommended_player_id"] == "SYN-7"


def test_duplicate_alternative_hold() -> None:
    result = evaluate([A, dict(A), B], _params(0.3, None))
    assert result["status"] == "hold"
    assert codes(result)[0] == "duplicate_alternative"
    assert result["reasons"][0]["detail"] == {"player_id": "SYN-A"}
    assert result["explanation"][0] == "Player SYN-A was nominated more than once."
    assert result["counts"]["nominated"] == 3


def test_evaluate_rejects_non_dict_only() -> None:
    with pytest.raises(TypeError):
        policy.evaluate([A, B])  # type: ignore[arg-type]


# --- 5. exclusions and the choice set -------------------------------------------------------------


def test_unavailable_player_is_excluded_with_a_visible_reason() -> None:
    excluded = _alt("SYN-B", 13.6, 7.0, 13.0, availability="unavailable", basis="user_excluded")
    result = evaluate([A, excluded, _alt("SYN-C", 12.0, 5.0, 10.0)], _params(0.3, None))
    assert result["status"] == "recommend"
    assert result["recommended_player_id"] == "SYN-A"
    assert codes(result) == [
        "recommend_highest_projection",
        "excluded_unavailable",
        "gap_meets_minimum",
        "floor_guardrail_disabled",
        "baseline_agrees",
    ]
    rows = {r["player_id"]: r for r in result["ranking"]}
    assert rows["SYN-B"]["comparable"] is False
    assert rows["SYN-B"]["exclusion"] == "excluded_unavailable"
    assert rows["SYN-B"]["rank"] is None
    assert rows["SYN-A"]["rank"] == 1 and rows["SYN-C"]["rank"] == 2
    assert result["counts"] == {"nominated": 3, "excluded": 1, "comparable": 2}
    assert result["leading"] == {"first": "SYN-A", "second": "SYN-C", "gap": 2.0}
    assert (
        "Synthetic B (SYN-B) excluded: marked unavailable for this slot." in result["explanation"]
    )
    # The excluded player never enters the baseline comparison either.
    assert "SYN-B" not in result["baseline"]["provenance"]


def test_one_option_after_exclusion_is_insufficient() -> None:
    excluded = _alt("SYN-B", 13.6, 7.0, 13.0, availability="unavailable", basis="user_excluded")
    result = evaluate([A, excluded], _params(0.3, None))
    assert result["status"] == "review"
    assert result["recommended_player_id"] is None
    assert codes(result) == ["excluded_unavailable", "insufficient_choice_set"]
    assert result["reasons"][1]["detail"] == {"count": 1, "minimum": 2}
    assert (
        result["explanation"][1]
        == "Only 1 comparable alternative(s) remain; a comparison needs at least 2."
    )
    assert result["leading"] == {"first": None, "second": None, "gap": None}
    assert result["counts"] == {"nominated": 2, "excluded": 1, "comparable": 1}
    # A single comparable player is still a baseline "preference" but there is no leader to agree
    # with, so no baseline_agrees / baseline_disagrees reason is emitted.
    assert result["baseline"]["status"] == "available"
    assert result["baseline"]["preferred_player_id"] == "SYN-A"
    assert result["baseline"]["agrees_with_model_leader"] is None


def test_slot_ineligible_qb_is_excluded_from_flex() -> None:
    qb = _alt("SYN-C", 20.0, 10.0, 18.0, position="QB")
    result = evaluate([A, B, qb], _params(0.3, None), slot="FLEX")
    assert result["status"] == "recommend"
    assert result["recommended_player_id"] == "SYN-A"
    row = next(r for r in result["ranking"] if r["player_id"] == "SYN-C")
    assert row["comparable"] is False and row["exclusion"] == "excluded_slot_ineligible"
    assert "excluded_slot_ineligible" in codes(result)
    reason = next(r for r in result["reasons"] if r["code"] == "excluded_slot_ineligible")
    assert reason["severity"] == "exclusion"
    assert reason["detail"] == {"player": "Synthetic C (SYN-C)", "position": "QB", "slot": "FLEX"}
    assert (
        "Synthetic C (SYN-C) excluded: position QB cannot fill the FLEX slot."
        in result["explanation"]
    )


def test_rb_wr_te_are_all_eligible_in_flex() -> None:
    alts = [
        _alt("SYN-A", 14.0, 6.0, 12.0, position="RB"),
        _alt("SYN-B", 13.0, 7.0, 11.0, position="WR"),
        _alt("SYN-C", 12.0, 5.0, 10.0, position="TE"),
    ]
    result = evaluate(alts, _params(0.3, None), slot="FLEX")
    assert result["status"] == "recommend"
    assert result["recommended_player_id"] == "SYN-A"
    assert all(r["comparable"] and r["exclusion"] is None for r in result["ranking"])
    assert result["counts"] == {"nominated": 3, "excluded": 0, "comparable": 3}
    assert "excluded_slot_ineligible" not in codes(result)


@pytest.mark.parametrize("slot", ["QB", "RB", "WR", "TE"])
def test_dedicated_slots_accept_only_their_position(slot: str) -> None:
    other = {"QB": "RB", "RB": "WR", "WR": "TE", "TE": "QB"}[slot]
    result = evaluate(
        [
            _alt("SYN-A", 14.0, 6.0, 12.0, position=slot),
            _alt("SYN-B", 13.0, 7.0, 11.0, position=other),
        ],
        _params(0.3, None),
        slot=slot,
    )
    rows = {r["player_id"]: r for r in result["ranking"]}
    assert rows["SYN-A"]["exclusion"] is None
    assert rows["SYN-B"]["exclusion"] == "excluded_slot_ineligible"
    assert result["status"] == "review" and "insufficient_choice_set" in codes(result)


def test_empty_choice_set_is_review() -> None:
    result = evaluate([], _params(0.3, None))
    assert result["status"] == "review"
    assert result["recommended_player_id"] is None
    assert codes(result) == ["insufficient_choice_set"]
    assert result["ranking"] == []
    assert result["counts"] == {"nominated": 0, "excluded": 0, "comparable": 0}
    assert result["baseline"]["status"] == "unavailable"
    assert result["baseline"]["preferred_player_id"] is None
    assert result["limitations"] == ["synthetic_evidence", "floors_not_probabilities"]


def test_single_option_is_review() -> None:
    result = evaluate([A], _params(0.3, None))
    assert result["status"] == "review"
    assert codes(result) == ["insufficient_choice_set"]
    assert result["recommended_player_id"] is None
    assert result["baseline"]["preferred_player_id"] == "SYN-A"
    assert result["leading"]["first"] is None


def test_missing_projection_is_review_and_the_player_is_not_comparable() -> None:
    result = evaluate([_alt("SYN-A", None, 6.0, 12.0), B], _params(0.3, None))
    assert result["status"] == "review"
    assert result["recommended_player_id"] is None
    assert codes(result) == ["projection_missing", "insufficient_choice_set"]
    row = next(r for r in result["ranking"] if r["player_id"] == "SYN-A")
    assert row["projection"] is None and row["comparable"] is False and row["exclusion"] is None
    assert result["explanation"][0] == (
        "Synthetic A (SYN-A) have no finite PPR projection, so the comparison is incomplete."
    )


def test_missing_projection_with_enough_comparable_players_still_reviews() -> None:
    result = evaluate(
        [_alt("SYN-A", None, 6.0, 12.0), B, _alt("SYN-C", 12.0, 5.0, 10.0)], _params(0.3, None)
    )
    assert result["status"] == "review"
    assert codes(result)[0] == "projection_missing"
    assert result["leading"] == {"first": "SYN-B", "second": "SYN-C", "gap": 1.6}


def test_unknown_availability_is_review_until_an_assumption_is_recorded() -> None:
    unknown = _alt("SYN-A", 14.0, 6.0, 12.0, availability="unknown", basis="not_verified")
    result = evaluate([unknown, B], _params(0.3, None))
    assert result["status"] == "review"
    assert result["recommended_player_id"] is None
    assert codes(result) == ["availability_unresolved", "insufficient_choice_set"]
    row = next(r for r in result["ranking"] if r["player_id"] == "SYN-A")
    assert row["comparable"] is False and row["exclusion"] is None
    assert result["explanation"][0] == (
        "Availability is not verified for Synthetic A (SYN-A); record an explicit assumption before "
        "the comparison can proceed."
    )
    assert "availability_user_assumed" in result["limitations"]  # from SYN-B


def test_assumed_available_adds_the_user_assumed_limitation() -> None:
    result = evaluate([A, B], _params(0.3, None))
    assert result["limitations"] == [
        "availability_user_assumed",
        "synthetic_evidence",
        "floors_not_probabilities",
    ]


def test_realized_stats_rows_add_the_hindsight_population_limitation() -> None:
    alts = [
        _alt(
            "SYN-A", 14.0, 6.0, 12.0, availability="realized_stats_row", basis="hindsight_stats_row"
        ),
        _alt(
            "SYN-B", 13.6, 7.0, 13.0, availability="realized_stats_row", basis="hindsight_stats_row"
        ),
    ]
    result = evaluate(alts, _params(0.3, None))
    assert result["status"] == "recommend" and result["recommended_player_id"] == "SYN-A"
    assert "population_hindsight_conditioned" in result["limitations"]
    assert "availability_user_assumed" not in result["limitations"]
    mixed = evaluate([alts[0], B], _params(0.3, None))
    assert {"population_hindsight_conditioned", "availability_user_assumed"} <= set(
        mixed["limitations"]
    )


# --- 6. the baseline is a separate comparator ---------------------------------------------------------


@pytest.mark.parametrize("provenance", ["model_prediction_fallback", "unknown"])
def test_non_independent_baseline_is_review_when_required(provenance: str) -> None:
    result = evaluate(
        [_alt("SYN-A", 14.0, 6.0, 14.0, provenance=provenance), B], _params(0.3, None)
    )
    assert result["status"] == "review"
    assert result["recommended_player_id"] is None
    assert codes(result) == [
        "baseline_unavailable",
        "gap_meets_minimum",
        "floor_guardrail_disabled",
    ]
    assert result["baseline"] == {
        "status": "unavailable",
        "preferred_player_id": None,
        "preferred_value": None,
        "agrees_with_model_leader": None,
        "missing_for": ["SYN-A"],
        "provenance": {"SYN-A": provenance, "SYN-B": "player_history"},
    }
    assert result["explanation"][0] == (
        "No independent baseline for Synthetic A (SYN-A); the policy requires one unless model-only "
        "exploration is enabled."
    )


def test_null_baseline_value_counts_as_missing() -> None:
    result = evaluate(
        [_alt("SYN-A", 14.0, 6.0, None, provenance="player_history"), B], _params(0.3, None)
    )
    assert result["status"] == "review"
    assert codes(result)[0] == "baseline_unavailable"
    assert result["baseline"]["missing_for"] == ["SYN-A"]


def test_model_only_exploration_recommends_without_a_baseline_claim() -> None:
    result = evaluate(
        [_alt("SYN-A", 14.0, 6.0, 14.0, provenance="model_prediction_fallback"), B],
        _params(0.3, None, True, True),
    )
    assert result["status"] == "recommend"
    assert result["recommended_player_id"] == "SYN-A"
    assert codes(result) == [
        "recommend_highest_projection",
        "baseline_not_independent",
        "gap_meets_minimum",
        "floor_guardrail_disabled",
    ]
    assert "baseline_agrees" not in codes(result) and "baseline_disagrees" not in codes(result)
    assert result["baseline"]["status"] == "unavailable"
    assert result["baseline"]["preferred_player_id"] is None
    assert result["baseline"]["agrees_with_model_leader"] is None
    assert result["baseline"]["missing_for"] == ["SYN-A"]
    assert "baseline_not_independent" in result["limitations"]


def test_baseline_not_required_recommends_and_claims_nothing() -> None:
    result = evaluate(
        [_alt("SYN-A", 14.0, 6.0, None, provenance="unknown"), B], _params(0.3, None, False, False)
    )
    assert result["status"] == "recommend" and result["recommended_player_id"] == "SYN-A"
    assert "baseline_unavailable" not in codes(result)
    assert "baseline_not_independent" not in codes(result)
    assert "baseline_not_independent" not in result["limitations"]
    assert (
        result["baseline"]["status"] == "unavailable"
        and result["baseline"]["preferred_player_id"] is None
    )


def test_baseline_tie_expresses_no_preference() -> None:
    result = evaluate(
        [_alt("SYN-A", 14.0, 6.0, 12.0), _alt("SYN-B", 13.0, 7.0, 12.0)], _params(0.3, None)
    )
    assert result["status"] == "recommend" and result["recommended_player_id"] == "SYN-A"
    assert codes(result) == [
        "recommend_highest_projection",
        "gap_meets_minimum",
        "floor_guardrail_disabled",
        "baseline_tie",
    ]
    assert result["baseline"]["status"] == "tie"
    assert result["baseline"]["preferred_player_id"] is None
    assert result["baseline"]["preferred_value"] == 12.0
    assert result["baseline"]["agrees_with_model_leader"] is None
    assert result["explanation"][-1] == (
        "The baseline values tie between Synthetic A (SYN-A), Synthetic B (SYN-B); it expresses no "
        "preference."
    )


def test_baseline_agrees_flag() -> None:
    result = evaluate([_alt("SYN-A", 14.0, 6.0, 15.0), B], _params(0.3, None))
    assert result["status"] == "recommend" and result["recommended_player_id"] == "SYN-A"
    assert codes(result)[-1] == "baseline_agrees"
    assert result["baseline"]["preferred_player_id"] == "SYN-A"
    assert result["baseline"]["preferred_value"] == 15.0
    assert result["baseline"]["agrees_with_model_leader"] is True
    assert result["explanation"][-1] == (
        "The causal trailing-mean baseline also prefers Synthetic A (SYN-A) (15)."
    )


def test_baseline_never_changes_the_recommended_player() -> None:
    for base_a, base_b in [(12.0, 13.0), (1.0, 50.0), (13.0, 12.0), (0.0, 0.0)]:
        result = evaluate(
            [_alt("SYN-A", 14.0, 6.0, base_a), _alt("SYN-B", 13.6, 7.0, base_b)], _params(0.3, None)
        )
        assert result["status"] == "recommend"
        assert result["recommended_player_id"] == "SYN-A"
        assert result["leading"]["first"] == "SYN-A"
        expected = "SYN-B" if base_b > base_a else ("SYN-A" if base_a > base_b else None)
        assert result["baseline"]["preferred_player_id"] == expected


# --- 7. floor guardrail ---------------------------------------------------------------------------


def test_missing_floor_with_guardrail_is_review() -> None:
    result = evaluate([_alt("SYN-A", 14.0, None, 12.0), B], _params(0.3, 5.0))
    assert result["status"] == "review"
    assert result["recommended_player_id"] is None
    assert codes(result) == ["gap_meets_minimum", "floor_missing", "baseline_disagrees"]
    assert result["gates"] == {
        "min_projection_gap": 0.3,
        "gap_ok": True,
        "min_floor": 5.0,
        "floor_checked": True,
        "leading_floor": None,
        "floor_ok": False,
    }
    assert result["explanation"][1] == (
        "Synthetic A (SYN-A) leads on projection but has no model floor while a floor guardrail of 5 "
        "is enabled."
    )


def test_missing_floor_without_guardrail_recommends() -> None:
    result = evaluate([_alt("SYN-A", 14.0, None, 12.0), B], _params(0.3, None))
    assert result["status"] == "recommend" and result["recommended_player_id"] == "SYN-A"
    assert codes(result) == [
        "recommend_highest_projection",
        "gap_meets_minimum",
        "floor_guardrail_disabled",
        "baseline_disagrees",
    ]
    assert result["gates"]["floor_checked"] is False
    assert result["gates"]["leading_floor"] is None and result["gates"]["floor_ok"] is None
    assert "No floor guardrail is enabled." in result["explanation"]


def test_a_lower_ranked_player_with_a_better_floor_is_never_recommended() -> None:
    # B clears the guardrail (7 >= 6.5) but A leads on projection: the answer is REVIEW, not B.
    result = evaluate([A, B], _params(0.3, 6.5))
    assert result["status"] == "review"
    assert result["recommended_player_id"] is None
    assert codes(result) == ["gap_meets_minimum", "floor_below_minimum", "baseline_disagrees"]
    assert result["leading"]["first"] == "SYN-A"
    assert result["gates"]["leading_floor"] == 6.0
    # Only the leader's floor is ever checked.
    result = evaluate(
        [_alt("SYN-A", 14.0, 9.0, 12.0), _alt("SYN-B", 13.6, 1.0, 13.0)], _params(0.3, 8.0)
    )
    assert result["status"] == "recommend" and result["recommended_player_id"] == "SYN-A"


def test_floor_gate_is_evaluated_after_the_gap_gate() -> None:
    # Both gates fail: both reasons are present, gap first.
    result = evaluate([A, B], _params(0.5, 8.0))
    assert codes(result)[:2] == ["gap_below_minimum", "floor_below_minimum"]
    # A tie short-circuits neither: the floor gate is still reported.
    result = evaluate(
        [_alt("SYN-X", 10.0, 1.0, 5.0), _alt("SYN-Y", 10.0, 1.0, 6.0)], _params(0.0, 2.0)
    )
    assert codes(result)[:2] == ["projection_tie", "floor_below_minimum"]


# --- 8. the inputs schema is closed ---------------------------------------------------------------


def test_outcome_fields_cannot_enter_the_inputs_contract() -> None:
    leaked = _inputs([{**A, "actual": 25.0}, B], _params(0.3, None))
    with pytest.raises(contracts.ContractError, match="actual"):
        contracts.check_decision_inputs(leaked)
    # The pure function ignores what it does not know (documented) but the contract stops it at
    # the boundary, so receipts can never carry it.
    result = policy.evaluate(leaked)
    assert result["status"] == "recommend" and result["recommended_player_id"] == "SYN-A"
    assert result == policy.evaluate(_inputs([A, B], _params(0.3, None))) | {
        "decision_id": result["decision_id"]
    }


@pytest.mark.parametrize(
    "field", ["abs_error", "regret", "hit", "prediction_rank", "best_eligible_points"]
)
def test_every_hindsight_field_is_rejected_at_the_boundary(field: str) -> None:
    with pytest.raises(contracts.ContractError, match=field):
        contracts.check_decision_inputs(_inputs([{**A, field: 1.0}, B], _params(0.3, None)))


def test_golden_inputs_pass_the_closed_contract() -> None:
    for case in GOLDEN["policy"]:
        if case["name"].startswith("parameters_invalid"):
            continue
        contracts.check_decision_inputs(case["inputs"])


# --- 9. mode limitations --------------------------------------------------------------------------


def test_published_weekly_adds_the_game_day_limitation() -> None:
    snap = _snapshot(mode="published_weekly", family="weekly", publication_status="published")
    result = evaluate([A, B], _params(0.3, None), snapshot=snap)
    assert result["limitations"] == [
        "availability_user_assumed",
        "snapshot_not_game_day_verified",
        "baseline_unreconciled",
        "floors_not_probabilities",
    ]


def test_synthetic_mode_adds_the_synthetic_limitation() -> None:
    result = evaluate([A, B], _params(0.3, None), snapshot=_snapshot(mode="synthetic"))
    assert "synthetic_evidence" in result["limitations"]
    historical = _snapshot(
        mode="historical_replay", family="frozen_test", publication_status="recorded"
    )
    assert (
        "synthetic_evidence"
        not in evaluate([A, B], _params(0.3, None), snapshot=historical)["limitations"]
    )


@pytest.mark.parametrize(("reconciled", "present"), [(False, True), (True, False), (None, True)])
def test_baseline_unreconciled_limitation(reconciled: bool | None, present: bool) -> None:
    snap = _snapshot(
        mode="historical_replay",
        family="weekly",
        publication_status="published",
        baseline_reconciled=reconciled,
    )
    result = evaluate([A, B], _params(0.3, None), snapshot=snap)
    assert ("baseline_unreconciled" in result["limitations"]) is present


def test_every_result_ends_with_floors_not_probabilities() -> None:
    for case in GOLDEN["policy"]:
        limitations = case["expected"]["limitations"]
        assert limitations[-1] == "floors_not_probabilities"
        assert limitations.count("floors_not_probabilities") == 1
        assert set(limitations) <= set(SPEC["limitations"])


# --- 10. explanations -----------------------------------------------------------------------------


def test_render_reason_is_deterministic_and_matches_the_explanation() -> None:
    for case in GOLDEN["policy"]:
        result = case["expected"]
        rendered = [policy.render_reason(r["code"], r["detail"]) for r in result["reasons"]]
        assert rendered == result["explanation"]
        assert rendered == [policy.render_reason(r["code"], r["detail"]) for r in result["reasons"]]
        assert len(result["explanation"]) == len(result["reasons"])


def test_render_reason_uses_canonical_number_formatting() -> None:
    text = policy.render_reason(
        "recommend_highest_projection",
        {"player": "A", "value": 14.0, "gap": 0.1 + 0.2, "second": "B"},
    )
    assert text == (
        "A has the highest PPR projection (14), 0.3 ahead of B, conditional on the recorded "
        "eligibility assumptions."
    )
    assert "14.0" not in text
    assert policy.render_reason("gap_meets_minimum", {"gap": 2, "threshold": 0.5}) == (
        "The projection gap 2 meets the minimum 0.5."
    )
    assert policy.render_reason("floor_guardrail_disabled", {}) == "No floor guardrail is enabled."


def test_render_reason_formats_lists_bools_and_nulls() -> None:
    assert policy.render_reason("projection_missing", {"players": ["A", "B"]}).startswith(
        "A, B have"
    )
    assert policy._fmt(True) == "yes" and policy._fmt(False) == "no"
    assert policy._fmt(None) == "—"
    assert policy._fmt(6.0) == format_number(6.0) == "6"


def test_render_reason_is_the_spec_template() -> None:
    for code, entry in SPEC["reasons"].items():
        assert policy.render_reason(code, {}) == entry["template"]
    with pytest.raises(KeyError):
        policy.render_reason("not_a_reason", {})


# --- identities ------------------------------------------------------------------------------------


def test_decision_id_covers_parameters_and_assumptions_but_not_order() -> None:
    base = _inputs([A, B], _params(0.3, 5.0))
    assert policy.decision_id(base) == content_id(policy.normalize_inputs(base))
    assert policy.decision_id(_inputs([B, A], _params(0.3, 5.0))) == policy.decision_id(base)
    assert policy.decision_id(_inputs([A, B], _params(0.3, 6.0))) != policy.decision_id(base)
    assumed = _alt("SYN-B", 13.6, 7.0, 13.0, availability="unavailable", basis="user_excluded")
    assert policy.decision_id(_inputs([A, assumed], _params(0.3, 5.0))) != policy.decision_id(base)
    assert policy.decision_id(
        _inputs([A, B], _params(0.3, 5.0), slot="FLEX")
    ) != policy.decision_id(base)


def test_normalize_inputs_pins_versions_and_sorts_without_mutating() -> None:
    raw = {"snapshot": _snapshot(), "slot": "RB", "alternatives": [B, A], "parameters": _params()}
    before = copy.deepcopy(raw)
    norm = policy.normalize_inputs(raw)
    assert raw == before
    assert norm["schema_version"] == SCHEMA_VERSIONS["decision_inputs"]
    assert norm["policy_version"] == POLICY_VERSION
    assert [a["player_id"] for a in norm["alternatives"]] == ["SYN-A", "SYN-B"]


def test_default_parameters_match_the_spec() -> None:
    assert policy.default_parameters() == {
        "min_projection_gap": 0.5,
        "min_floor": None,
        "require_independent_baseline": True,
        "model_only_exploration": False,
    }
    assert math.isclose(policy.TOL, 1e-6)
    assert policy.CHOICE_MIN == 2 and policy.CHOICE_MAX == 8


def test_result_identity_is_the_content_id() -> None:
    result = evaluate([A, B], _params(0.3, 5.0))
    assert policy.result_identity(result) == content_id(result)
    assert policy.result_identity(result) != result["decision_id"]
