"""Decision receipts (``ffai.decision_lab.receipts``): immutable inputs and result, appended events.

Pins the two identities (decision id over the inputs, event ids over the event list), the
"recommendation is never the user's action" rule, the single-action rule, outcome compatibility,
tamper detection on replay, and idempotent merge.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from ffai.decision_lab import contracts, golden, metrics, policy, receipts
from ffai.decision_lab.canonical import content_id
from ffai.decision_lab.golden import A, B, _inputs, _params

GOLDEN = json.loads(golden.GOLDEN_PATH.read_text(encoding="utf-8"))
LIFECYCLE = GOLDEN["receipt"]
T0, T1, T2 = "2099-01-01T00:00:00+00:00", "2099-01-01T00:01:00+00:00", "2099-01-01T00:02:00+00:00"
MODEL = "synthetic-model-v1"


# --- a small synthetic inputs snapshot (the bundle file shape, see contracts.INPUTS_SNAPSHOT) -------


def _row(
    pid: str, pos: str, proj: float | None, floor: float | None, base: float | None, **kw: Any
) -> dict[str, Any]:
    return {
        "player_id": pid,
        "display_name": f"Player {pid[-1]}",
        "team": "SYN",
        "position": pos,
        "model_version": MODEL,
        "candidate": "rf",
        "projection": proj,
        "floor": floor,
        "ceiling": None if proj is None else proj + 8.0,
        "baseline": base,
        "baseline_provenance": kw.get("provenance", "player_history"),
        "availability_default": kw.get("availability", "assumed_available"),
        "display_source": "synthetic",
        "source_row_ref": None,
    }


ROWS = [
    _row("SYN-A", "RB", 14.0, 6.0, 12.0),
    _row("SYN-B", "RB", 13.6, 7.0, 13.0),
    _row("SYN-C", "WR", 12.0, 5.0, 10.0, availability="realized_stats_row"),
    _row("SYN-D", "TE", 9.0, 2.0, 8.0, availability="unknown"),
    _row("SYN-E", "RB", 8.0, 1.0, 7.0, availability="unavailable"),
    _row("SYN-Q", "QB", 20.0, 10.0, 18.0),
]


def make_snapshot() -> dict[str, Any]:
    return {
        "schema_version": "inputs_snapshot/1.0",
        "snapshot_id": "syn-test",
        "mode": "synthetic",
        "source_family": "synthetic",
        "season": 2099,
        "week": 1,
        "model_version": MODEL,
        "feature_version": None,
        "candidate_by_position": {"QB": "rf", "RB": "rf", "WR": "rf", "TE": "rf"},
        "scoring_format": "ppr",
        "data_cutoff": {"season": 2098, "week": 18},
        "generated_at_utc": None,
        "publication": {"status": "synthetic", "run_id": None, "action": None, "note": None},
        "population": {
            "conditioning": "synthetic",
            "description": "synthetic rows",
            "n_rows": len(ROWS),
            "exclusions": [],
        },
        "baseline": {
            "name": "trailing_mean",
            "provenance_basis": "synthetic",
            "reconciled": True,
            "reconciled_to": None,
            "note": None,
        },
        "champion_at_source": None,
        "source": {"mart": None, "files": [], "mart_export": None},
        "rows": copy.deepcopy(ROWS),
    }


SNAPSHOT = make_snapshot()
contracts.check_inputs_snapshot(SNAPSHOT)


def make_outcomes(actuals: dict[str, float | None] | None = None) -> dict[str, Any]:
    actuals = actuals or {
        "SYN-A": 8.0,
        "SYN-B": 16.0,
        "SYN-C": 10.0,
        "SYN-D": 5.0,
        "SYN-E": 2.0,
        "SYN-Q": 25.0,
    }
    rows = [
        {
            "player_id": r["player_id"],
            "actual": actuals.get(r["player_id"]),
            "actual_source": "synthetic" if actuals.get(r["player_id"]) is not None else None,
        }
        for r in ROWS
    ]
    return {
        "schema_version": "outcome_snapshot/1.0",
        "snapshot_id": "syn-test",
        "inputs_content_sha256": content_id(SNAPSHOT),
        "season": 2099,
        "week": 1,
        "model_version": MODEL,
        "scoring_format": "ppr",
        "observed_at_utc": None,
        "source": {"mart": None, "files": []},
        "coverage": {
            "scored": len(rows),
            "observed": sum(1 for r in rows if r["actual"] is not None),
        },
        "rows": rows,
    }


OUTCOMES = make_outcomes()
contracts.check_outcome_snapshot(OUTCOMES, SNAPSHOT)


def inputs_lookup(snapshot_id: str) -> dict[str, Any] | None:
    return SNAPSHOT if snapshot_id == "syn-test" else None


def outcome_lookup(snapshot_id: str) -> dict[str, Any] | None:
    return OUTCOMES if snapshot_id == "syn-test" else None


def bundle_receipt(parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    di = receipts.build_decision_inputs(
        SNAPSHOT,
        slot="RB",
        player_ids=["SYN-A", "SYN-B"],
        parameters=parameters or {"min_projection_gap": 0.3},
    )
    return receipts.new_receipt(di, created_at_utc=T0, case_id="syn-test-case")


GOLDEN_INPUTS = _inputs([A, B], _params(0.3, 5.0))


# --- build_decision_inputs -------------------------------------------------------------------------


def test_build_decision_inputs_resolves_defaults_and_is_normalized() -> None:
    di = receipts.build_decision_inputs(
        SNAPSHOT, slot="FLEX", player_ids=["SYN-D", "SYN-C", "SYN-A", "SYN-E"]
    )
    contracts.check_decision_inputs(di)
    assert di == policy.normalize_inputs(di)
    assert [a["player_id"] for a in di["alternatives"]] == ["SYN-A", "SYN-C", "SYN-D", "SYN-E"]
    by_id = {a["player_id"]: a for a in di["alternatives"]}
    assert (by_id["SYN-A"]["availability"], by_id["SYN-A"]["availability_basis"]) == (
        "assumed_available",
        "user_assumed",
    )
    assert (by_id["SYN-C"]["availability"], by_id["SYN-C"]["availability_basis"]) == (
        "realized_stats_row",
        "hindsight_stats_row",
    )
    assert (by_id["SYN-D"]["availability"], by_id["SYN-D"]["availability_basis"]) == (
        "unknown",
        "not_verified",
    )
    assert (by_id["SYN-E"]["availability"], by_id["SYN-E"]["availability_basis"]) == (
        "unavailable",
        "user_excluded",
    )
    assert di["parameters"] == policy.default_parameters()
    assert di["slot"] == "FLEX"
    assert (
        di["schema_version"] == "decision_inputs/1.0"
        and di["policy_version"] == policy.POLICY_VERSION
    )
    alt = by_id["SYN-A"]
    assert (alt["season"], alt["week"], alt["model_version"], alt["candidate"]) == (
        2099,
        1,
        MODEL,
        "rf",
    )
    assert (alt["projection"], alt["floor"], alt["ceiling"], alt["baseline"]) == (
        14.0,
        6.0,
        22.0,
        12.0,
    )
    assert alt["display_name"] == "Player A" and alt["team"] == "SYN" and alt["position"] == "RB"


def test_build_decision_inputs_snapshot_ref_carries_the_content_digest() -> None:
    di = receipts.build_decision_inputs(SNAPSHOT, slot="RB", player_ids=["SYN-A", "SYN-B"])
    ref = di["snapshot"]
    assert ref["inputs_sha256"] == content_id(SNAPSHOT)
    assert ref["snapshot_id"] == "syn-test" and ref["mode"] == "synthetic"
    assert ref["evidence_status"] == "verified" and ref["evidence_detail"] is None
    assert ref["baseline_reconciled"] is True
    assert (
        ref["scoring_format"] == "ppr"
        and ref["candidate_by_position"] == SNAPSHOT["candidate_by_position"]
    )
    corrupt = receipts.build_decision_inputs(
        SNAPSHOT,
        slot="RB",
        player_ids=["SYN-A", "SYN-B"],
        evidence_status="corrupt",
        evidence_detail="bad bytes",
    )
    assert corrupt["snapshot"]["evidence_status"] == "corrupt"
    assert policy.evaluate(corrupt)["status"] == "hold"


def test_build_decision_inputs_applies_overrides_and_parameters() -> None:
    di = receipts.build_decision_inputs(
        SNAPSHOT,
        slot="RB",
        player_ids=["SYN-A", "SYN-B", "SYN-E"],
        overrides={
            "SYN-E": {"availability": "assumed_available", "availability_basis": "user_assumed"},
            "SYN-B": {"availability": "unavailable"},
        },
        parameters={"min_floor": 5.0, "min_projection_gap": 0.3},
    )
    by_id = {a["player_id"]: a for a in di["alternatives"]}
    assert (by_id["SYN-E"]["availability"], by_id["SYN-E"]["availability_basis"]) == (
        "assumed_available",
        "user_assumed",
    )
    # A partial override keeps the default basis for the untouched field.
    assert (by_id["SYN-B"]["availability"], by_id["SYN-B"]["availability_basis"]) == (
        "unavailable",
        "user_assumed",
    )
    assert di["parameters"] == {
        "min_projection_gap": 0.3,
        "min_floor": 5.0,
        "require_independent_baseline": True,
        "model_only_exploration": False,
    }
    contracts.check_decision_inputs(di)


def test_build_decision_inputs_raises_on_unknown_player() -> None:
    with pytest.raises(receipts.ReceiptError, match="SYN-Z"):
        receipts.build_decision_inputs(SNAPSHOT, slot="RB", player_ids=["SYN-A", "SYN-Z"])


def test_build_decision_inputs_does_not_mutate_the_snapshot() -> None:
    snap = make_snapshot()
    receipts.build_decision_inputs(
        snap, slot="RB", player_ids=["SYN-A"], overrides={"SYN-A": {"availability": "unavailable"}}
    )
    assert snap == SNAPSHOT


# --- new_receipt -------------------------------------------------------------------------------------


def test_new_receipt_initial_state() -> None:
    r = receipts.new_receipt(
        GOLDEN_INPUTS, created_at_utc=T0, case_id="c", prediction_note="I expect A"
    )
    assert r["schema_version"] == "decision_receipt/1.0"
    assert r["decision_id"] == policy.decision_id(GOLDEN_INPUTS) == r["result"]["decision_id"]
    assert r["parent_decision_id"] is None
    assert (
        r["created_at_utc"] == T0 and r["case_id"] == "c" and r["prediction_note"] == "I expect A"
    )
    assert r["inputs"] == policy.normalize_inputs(GOLDEN_INPUTS)
    assert r["result"] == policy.evaluate(GOLDEN_INPUTS)
    assert r["result_sha256"] == content_id(r["result"])
    assert r["events"] == []
    assert r["action"] == {
        "state": "not_recorded",
        "action_id": None,
        "at_utc": None,
        "chosen_player_id": None,
        "kind": None,
        "note": None,
    }
    assert r["outcome"] == {
        "state": "not_attached",
        "outcome_snapshot_id": None,
        "outcome_id": None,
        "attached_at_utc": None,
        "metrics": None,
    }
    assert r["trust"] == {"note": receipts.TRUST_NOTE}
    # The recommendation is never copied into the action block.
    assert r["result"]["recommended_player_id"] == "SYN-A"
    assert r["action"]["chosen_player_id"] is None
    receipts.validate_receipt(r)
    contracts.validate("receipt", r)


def test_new_receipt_rejects_inputs_that_fail_the_contract() -> None:
    with pytest.raises(contracts.ContractError):
        receipts.new_receipt(
            _inputs([{**A, "actual": 9.0}, B], _params(0.3, None)), created_at_utc=T0
        )
    with pytest.raises(contracts.ContractError):
        receipts.new_receipt(_inputs([A, B], _params(float("nan"), None)), created_at_utc=T0)


def test_new_receipt_normalizes_alternative_order() -> None:
    r = receipts.new_receipt(_inputs([B, A], _params(0.3, 5.0)), created_at_utc=T0)
    assert r["decision_id"] == receipts.new_receipt(GOLDEN_INPUTS, created_at_utc=T1)["decision_id"]
    assert [a["player_id"] for a in r["inputs"]["alternatives"]] == ["SYN-A", "SYN-B"]


# --- actions -----------------------------------------------------------------------------------------


def test_record_action_appends_a_recomputable_event() -> None:
    r0 = receipts.new_receipt(GOLDEN_INPUTS, created_at_utc=T0)
    r1 = receipts.record_action(
        r0, chosen_player_id="SYN-B", kind="self_reported_real", at_utc=T1, note="gut call"
    )
    assert r0["events"] == [] and r0["action"]["state"] == "not_recorded"  # immutable input
    assert len(r1["events"]) == 1
    ev = r1["events"][0]
    assert ev["seq"] == 1 and ev["event_type"] == "action_recorded" and ev["at_utc"] == T1
    assert ev["payload"] == {
        "chosen_player_id": "SYN-B",
        "kind": "self_reported_real",
        "note": "gut call",
    }
    assert ev["event_id"] == receipts.event_id(
        r1["decision_id"], 1, "action_recorded", T1, ev["payload"]
    )
    assert ev["event_id"] == content_id(
        {
            "decision_id": r1["decision_id"],
            "seq": 1,
            "event_type": "action_recorded",
            "at_utc": T1,
            "payload": ev["payload"],
        }
    )
    assert r1["action"] == {
        "state": "recorded",
        "action_id": ev["event_id"],
        "at_utc": T1,
        "chosen_player_id": "SYN-B",
        "kind": "self_reported_real",
        "note": "gut call",
    }
    assert r1["outcome"]["state"] == "not_attached"
    assert r1["decision_id"] == r0["decision_id"] and r1["result_sha256"] == r0["result_sha256"]
    assert r1["inputs"] == r0["inputs"] and r1["result"] == r0["result"]
    receipts.validate_receipt(r1)


def test_second_action_is_refused() -> None:
    r0 = receipts.new_receipt(GOLDEN_INPUTS, created_at_utc=T0)
    r1 = receipts.record_action(r0, chosen_player_id="SYN-A", kind="hypothetical_replay", at_utc=T1)
    with pytest.raises(receipts.ReceiptError, match="already recorded"):
        receipts.record_action(r1, chosen_player_id="SYN-B", kind="hypothetical_replay", at_utc=T2)
    with pytest.raises(receipts.ReceiptError, match="already recorded"):
        receipts.decline_action(r1, at_utc=T2)
    declined = receipts.decline_action(r0, at_utc=T1)
    with pytest.raises(receipts.ReceiptError, match="already recorded"):
        receipts.record_action(
            declined, chosen_player_id="SYN-A", kind="hypothetical_replay", at_utc=T2
        )


def test_record_action_rejects_unknown_player_and_kind() -> None:
    r0 = receipts.new_receipt(GOLDEN_INPUTS, created_at_utc=T0)
    with pytest.raises(receipts.ReceiptError, match="not among the nominated"):
        receipts.record_action(r0, chosen_player_id="SYN-Z", kind="hypothetical_replay", at_utc=T1)
    with pytest.raises(receipts.ReceiptError, match="unknown action kind"):
        receipts.record_action(r0, chosen_player_id="SYN-A", kind="real", at_utc=T1)
    with pytest.raises(receipts.ReceiptError, match="unknown event type"):
        receipts.append_event(r0, "action_edited", {}, T1)


def test_excluded_but_nominated_player_can_be_chosen() -> None:
    di = receipts.build_decision_inputs(
        SNAPSHOT,
        slot="RB",
        player_ids=["SYN-A", "SYN-B", "SYN-E"],
        parameters={"min_projection_gap": 0.3},
    )
    r0 = receipts.new_receipt(di, created_at_utc=T0)
    assert (
        next(row for row in r0["result"]["ranking"] if row["player_id"] == "SYN-E")["exclusion"]
        == "excluded_unavailable"
    )
    r1 = receipts.record_action(r0, chosen_player_id="SYN-E", kind="hypothetical_replay", at_utc=T1)
    receipts.validate_receipt(r1)


def test_decline_action() -> None:
    r0 = receipts.new_receipt(
        GOLDEN_INPUTS,
        created_at_utc=T0,
        case_id="syn-ambiguity-floor-relaxation",
        prediction_note="I expect A",
    )
    r = receipts.decline_action(r0, at_utc=T1, note="skip")
    ev = r["events"][0]
    assert ev["event_type"] == "action_declined" and ev["payload"] == {"note": "skip"}
    assert r["action"] == {
        "state": "declined",
        "action_id": ev["event_id"],
        "at_utc": T1,
        "chosen_player_id": None,
        "kind": None,
        "note": "skip",
    }
    assert r["decision_id"] == r0["decision_id"]
    receipts.validate_receipt(r)
    assert r == LIFECYCLE["receipt_declined"]


# --- outcomes ------------------------------------------------------------------------------------------


def test_attach_outcome_computes_metrics_and_keeps_identities() -> None:
    r0 = receipts.new_receipt(
        GOLDEN_INPUTS,
        created_at_utc=T0,
        case_id="syn-ambiguity-floor-relaxation",
        prediction_note="I expect A",
    )
    r1 = receipts.record_action(
        r0, chosen_player_id="SYN-A", kind="hypothetical_replay", at_utc=T1, note=None
    )
    r2 = receipts.attach_outcome(r1, LIFECYCLE["outcome_snapshot"], at_utc=T2)
    assert r0["decision_id"] == r1["decision_id"] == r2["decision_id"]
    assert r0["result_sha256"] == r1["result_sha256"] == r2["result_sha256"]
    assert r2["inputs"] == r0["inputs"] and r2["result"] == r0["result"]
    assert [e["seq"] for e in r2["events"]] == [1, 2]
    ev = r2["events"][1]
    oid = content_id(LIFECYCLE["outcome_snapshot"])
    assert oid == LIFECYCLE["outcome_id"]
    assert oid != r2["decision_id"] and oid != r2["result_sha256"]
    assert ev["event_type"] == "outcome_attached"
    assert (
        ev["payload"]["outcome_snapshot_id"] == "syn-golden" and ev["payload"]["outcome_id"] == oid
    )
    assert ev["payload"]["metrics"] == metrics.compute_metrics(
        r1["result"], r1["action"], {**LIFECYCLE["outcome_snapshot"], "outcome_id": oid}
    )
    assert ev["payload"]["metrics"]["chosen_actual_points"] == 8.0
    assert ev["payload"]["metrics"]["choice_set_regret"] == 8.0
    assert r2["outcome"] == {
        "state": "attached",
        "outcome_snapshot_id": "syn-golden",
        "outcome_id": oid,
        "attached_at_utc": T2,
        "metrics": ev["payload"]["metrics"],
    }
    assert r2["action"] == r1["action"]
    receipts.validate_receipt(r2)


def test_golden_receipt_lifecycle_reproduces_exactly() -> None:
    r0 = receipts.new_receipt(
        LIFECYCLE["inputs"],
        created_at_utc=T0,
        case_id="syn-ambiguity-floor-relaxation",
        prediction_note="I expect A",
    )
    r1 = receipts.record_action(
        r0, chosen_player_id="SYN-A", kind="hypothetical_replay", at_utc=T1, note=None
    )
    r2 = receipts.attach_outcome(r1, LIFECYCLE["outcome_snapshot"], at_utc=T2)
    assert r0 == LIFECYCLE["receipt_new"]
    assert r1 == LIFECYCLE["receipt_after_action"]
    assert r2 == LIFECYCLE["receipt_after_outcome"]
    assert LIFECYCLE["invariants"] == {
        "decision_id_unchanged": True,
        "result_sha_unchanged": True,
        "event_ids": [e["event_id"] for e in r2["events"]],
    }
    for key in ("receipt_new", "receipt_after_action", "receipt_after_outcome", "receipt_declined"):
        receipts.validate_receipt(LIFECYCLE[key])


def test_attach_outcome_without_an_action_uses_not_recorded_metrics() -> None:
    r0 = receipts.new_receipt(GOLDEN_INPUTS, created_at_utc=T0)
    r = receipts.attach_outcome(r0, LIFECYCLE["outcome_snapshot"], at_utc=T1)
    m = r["outcome"]["metrics"]
    assert m["action_state"] == "not_recorded" and m["chosen_actual_reason"] == "no_action_recorded"
    assert m["model_policy_vs_baseline_choice"] == -8.0
    assert r["action"]["state"] == "not_recorded"
    receipts.validate_receipt(r)
    # An action can still be recorded after outcomes were attached (append-only, one action).
    r_after = receipts.record_action(
        r, chosen_player_id="SYN-B", kind="hypothetical_replay", at_utc=T2
    )
    assert [e["event_type"] for e in r_after["events"]] == ["outcome_attached", "action_recorded"]
    receipts.validate_receipt(r_after)


@pytest.mark.parametrize(
    ("patch", "fragment"),
    [
        ({"snapshot_id": "other-snapshot"}, "is not syn-golden"),
        ({"season": 2098}, "different season/week"),
        ({"week": 2}, "different season/week"),
        ({"model_version": "other-model"}, "different model version"),
        ({"inputs_content_sha256": "f" * 64}, "different inputs snapshot content"),
    ],
)
def test_attach_outcome_from_an_incompatible_snapshot_raises(
    patch: dict[str, Any], fragment: str
) -> None:
    r1 = LIFECYCLE["receipt_after_action"]
    bad = {**LIFECYCLE["outcome_snapshot"], **patch}
    assert receipts.outcome_compatible(r1, bad) is not None
    with pytest.raises(receipts.ReceiptError, match=fragment):
        receipts.attach_outcome(r1, bad, at_utc=T2)
    assert receipts.outcome_compatible(r1, LIFECYCLE["outcome_snapshot"]) is None


def test_attach_outcome_requires_a_valid_outcome_snapshot() -> None:
    r1 = LIFECYCLE["receipt_after_action"]
    bad = copy.deepcopy(LIFECYCLE["outcome_snapshot"])
    bad["coverage"]["observed"] = 1
    with pytest.raises(contracts.ContractError, match="coverage"):
        receipts.attach_outcome(r1, bad, at_utc=T2)
    half = {**LIFECYCLE["outcome_snapshot"], "scoring_format": "half"}
    with pytest.raises(contracts.ContractError, match="ppr"):
        receipts.attach_outcome(r1, half, at_utc=T2)


# --- validate_receipt: tamper detection ---------------------------------------------------------------


def _tampered(key: str, mutate) -> dict[str, Any]:  # noqa: ANN001
    r = copy.deepcopy(LIFECYCLE[key])
    mutate(r)
    return r


def _set_status(r: dict[str, Any]) -> None:
    r["result"]["status"] = "recommend" if r["result"]["status"] != "recommend" else "review"


def _set_recommended(r: dict[str, Any]) -> None:
    r["result"]["recommended_player_id"] = "SYN-B"


def _drop_reason(r: dict[str, Any]) -> None:
    del r["result"]["reasons"][-1]
    del r["result"]["explanation"][-1]


def _edit_result_sha(r: dict[str, Any]) -> None:
    r["result_sha256"] = "0" * 64


def _edit_decision_id(r: dict[str, Any]) -> None:
    r["decision_id"] = "1" * 64


def _edit_projection(r: dict[str, Any]) -> None:
    r["inputs"]["alternatives"][1]["projection"] = 15.0  # B now leads: the replay differs


def _edit_event_id(r: dict[str, Any]) -> None:
    r["events"][0]["event_id"] = "2" * 64


def _reorder_events(r: dict[str, Any]) -> None:
    r["events"].reverse()


def _edit_action_block(r: dict[str, Any]) -> None:
    r["action"]["chosen_player_id"] = "SYN-B"


def _edit_metrics(r: dict[str, Any]) -> None:
    r["events"][1]["payload"]["metrics"]["choice_set_regret"] = 0.0
    r["outcome"]["metrics"]["choice_set_regret"] = 0.0


def _edit_parameters(r: dict[str, Any]) -> None:
    r["inputs"]["parameters"]["min_floor"] = 8.0


def _unsort_alternatives(r: dict[str, Any]) -> None:
    r["inputs"]["alternatives"].reverse()


@pytest.mark.parametrize(
    ("key", "mutate", "check"),
    [
        # The stored digest still matches a fresh evaluation; the stored *result* is what differs.
        ("receipt_after_outcome", _set_status, "result_replay"),
        ("receipt_after_outcome", _set_recommended, "result_replay"),
        ("receipt_after_outcome", _drop_reason, "result_replay"),
        ("receipt_after_outcome", _edit_result_sha, "result_sha256"),
        ("receipt_after_outcome", _edit_decision_id, "decision_id"),
        ("receipt_after_outcome", _edit_projection, "decision_id"),
        ("receipt_after_outcome", _edit_parameters, "decision_id"),
        ("receipt_after_outcome", _unsort_alternatives, "inputs_normalized"),
        ("receipt_after_outcome", _edit_event_id, "event_id"),
        ("receipt_after_outcome", _reorder_events, "event_sequence"),
        ("receipt_after_action", _edit_action_block, "action_projection"),
        ("receipt_after_outcome", _edit_metrics, "event_id"),
        ("receipt_declined", _edit_action_block, "action_projection"),
    ],
)
def test_validate_receipt_fails_on_tampering(key: str, mutate, check: str) -> None:  # noqa: ANN001
    with pytest.raises(receipts.ReceiptError) as exc:
        receipts.validate_receipt(_tampered(key, mutate))
    assert str(exc.value).startswith(f"{check}:")


def test_validate_receipt_passes_on_golden_and_reports_every_check() -> None:
    checks = receipts.validate_receipt(LIFECYCLE["receipt_after_outcome"])
    assert [c["check"] for c in checks] == [
        "schema",
        "decision_id",
        "result_replay",
        "events",
        "projections",
    ]
    assert all(c["ok"] for c in checks)
    assert checks[2]["status"] == "recommend" and checks[2]["recommended_player_id"] == "SYN-A"
    assert checks[3]["n"] == 2
    assert checks[4] == {
        "check": "projections",
        "ok": True,
        "action_state": "recorded",
        "outcome_state": "attached",
    }


def test_edited_projection_with_recomputed_ids_still_fails_on_result_replay() -> None:
    # An attacker who edits an alternative AND recomputes the decision id is caught by the result
    # replay: the stored result no longer matches a fresh evaluation.
    r = copy.deepcopy(LIFECYCLE["receipt_new"])
    r["inputs"]["alternatives"][1]["projection"] = 15.0
    r["decision_id"] = policy.decision_id(r["inputs"])
    with pytest.raises(receipts.ReceiptError, match="^result_sha256"):
        receipts.validate_receipt(r)
    r["result_sha256"] = content_id(r["result"])
    with pytest.raises(receipts.ReceiptError, match="^result_sha256"):
        receipts.validate_receipt(r)


def test_result_edited_with_a_matching_digest_fails_on_replay() -> None:
    r = copy.deepcopy(LIFECYCLE["receipt_new"])
    r["result"]["explanation"][0] = "edited"
    r["result_sha256"] = content_id(r["result"])
    with pytest.raises(receipts.ReceiptError, match="^result_sha256"):
        receipts.validate_receipt(r)


def test_chosen_player_not_nominated_fails_membership() -> None:
    r = copy.deepcopy(LIFECYCLE["receipt_new"])
    payload = {"chosen_player_id": "SYN-Z", "kind": "hypothetical_replay", "note": None}
    eid = receipts.event_id(r["decision_id"], 1, "action_recorded", T1, payload)
    r["events"] = [
        {
            "event_id": eid,
            "seq": 1,
            "event_type": "action_recorded",
            "at_utc": T1,
            "payload": payload,
        }
    ]
    r["action"], r["outcome"] = receipts.derive_state(r)
    with pytest.raises(receipts.ReceiptError, match="^chosen_membership"):
        receipts.validate_receipt(r)


def test_two_action_events_fail_single_action() -> None:
    r = copy.deepcopy(LIFECYCLE["receipt_after_action"])
    payload = {"note": None}
    eid = receipts.event_id(r["decision_id"], 2, "action_declined", T2, payload)
    r["events"].append(
        {
            "event_id": eid,
            "seq": 2,
            "event_type": "action_declined",
            "at_utc": T2,
            "payload": payload,
        }
    )
    r["action"], r["outcome"] = receipts.derive_state(r)
    with pytest.raises(receipts.ReceiptError, match="^single_action"):
        receipts.validate_receipt(r)


def test_schema_violations_fail_first() -> None:
    r = copy.deepcopy(LIFECYCLE["receipt_new"])
    r["extra"] = 1
    with pytest.raises(receipts.ReceiptError, match="^schema"):
        receipts.validate_receipt(r)
    r = copy.deepcopy(LIFECYCLE["receipt_new"])
    del r["action"]
    with pytest.raises(receipts.ReceiptError, match="^schema"):
        receipts.validate_receipt(r)


# --- validate_receipt with bundle lookups ---------------------------------------------------------------


def test_validate_with_matching_snapshot_and_outcome_passes() -> None:
    r0 = bundle_receipt()
    r1 = receipts.record_action(r0, chosen_player_id="SYN-A", kind="hypothetical_replay", at_utc=T1)
    r2 = receipts.attach_outcome(r1, OUTCOMES, at_utc=T2)
    checks = receipts.validate_receipt(
        r2, inputs_lookup=inputs_lookup, outcome_lookup=outcome_lookup
    )
    names = [c["check"] for c in checks]
    assert names == [
        "schema",
        "decision_id",
        "result_replay",
        "events",
        "projections",
        "snapshot_digest",
        "metrics_replay",
    ]
    assert checks[-2]["snapshot_id"] == "syn-test"
    assert checks[-1]["outcome_id"] == content_id(OUTCOMES) == r2["outcome"]["outcome_id"]


def test_validate_with_a_missing_snapshot_fails() -> None:
    r0 = bundle_receipt()
    with pytest.raises(receipts.ReceiptError, match="^snapshot_available"):
        receipts.validate_receipt(r0, inputs_lookup=lambda _sid: None)


def test_validate_with_a_different_snapshot_digest_fails() -> None:
    r0 = bundle_receipt()
    altered = make_snapshot()
    altered["rows"][2]["projection"] = 13.0  # a player that is not even nominated
    with pytest.raises(receipts.ReceiptError, match="^snapshot_digest"):
        receipts.validate_receipt(r0, inputs_lookup=lambda _sid: altered)


def test_validate_with_altered_alternative_values_fails() -> None:
    # Same digest claimed by the receipt but the alternative row was edited *in the receipt* and the
    # ids recomputed: the bundle row wins.
    r0 = bundle_receipt()
    r = copy.deepcopy(r0)
    r["inputs"]["alternatives"][0]["floor"] = 6.5
    r["decision_id"] = policy.decision_id(r["inputs"])
    r["result"] = policy.evaluate(r["inputs"])
    r["result_sha256"] = content_id(r["result"])
    receipts.validate_receipt(r)  # internally consistent
    with pytest.raises(receipts.ReceiptError, match="^alternative_values: SYN-A.floor"):
        receipts.validate_receipt(r, inputs_lookup=inputs_lookup)


def test_validate_with_an_alternative_missing_from_the_snapshot_fails() -> None:
    r0 = bundle_receipt()
    smaller = make_snapshot()
    smaller["rows"] = [row for row in smaller["rows"] if row["player_id"] != "SYN-B"]
    smaller["population"]["n_rows"] = len(smaller["rows"])
    r = copy.deepcopy(r0)
    r["inputs"]["snapshot"]["inputs_sha256"] = content_id(smaller)
    r["decision_id"] = policy.decision_id(r["inputs"])
    r["result"] = policy.evaluate(r["inputs"])
    r["result_sha256"] = content_id(r["result"])
    with pytest.raises(receipts.ReceiptError, match="^alternative_membership: SYN-B"):
        receipts.validate_receipt(r, inputs_lookup=lambda _sid: smaller)


def test_validate_with_outcome_lookup_recomputes_metrics_and_catches_tampering() -> None:
    r0 = bundle_receipt()
    r1 = receipts.record_action(r0, chosen_player_id="SYN-A", kind="hypothetical_replay", at_utc=T1)
    r2 = receipts.attach_outcome(r1, OUTCOMES, at_utc=T2)
    # Tamper with the metrics and recompute the event id so the digest chain is consistent.
    r = copy.deepcopy(r2)
    ev = r["events"][1]
    ev["payload"]["metrics"]["choice_set_regret"] = 0.0
    ev["event_id"] = receipts.event_id(
        r["decision_id"], ev["seq"], ev["event_type"], ev["at_utc"], ev["payload"]
    )
    r["action"], r["outcome"] = receipts.derive_state(r)
    receipts.validate_receipt(r)  # internally consistent without the bundle
    with pytest.raises(receipts.ReceiptError, match="^metrics_replay"):
        receipts.validate_receipt(r, outcome_lookup=outcome_lookup)


def test_validate_with_outcome_lookup_detects_a_different_outcome_file() -> None:
    r0 = bundle_receipt()
    r2 = receipts.attach_outcome(r0, OUTCOMES, at_utc=T1)
    other = make_outcomes({"SYN-A": 9.0, "SYN-B": 16.0})
    with pytest.raises(receipts.ReceiptError, match="^outcome_digest"):
        receipts.validate_receipt(r2, outcome_lookup=lambda _sid: other)
    with pytest.raises(receipts.ReceiptError, match="^outcome_available"):
        receipts.validate_receipt(r2, outcome_lookup=lambda _sid: None)


def test_outcome_event_from_another_snapshot_fails() -> None:
    r0 = bundle_receipt()
    r2 = receipts.attach_outcome(r0, OUTCOMES, at_utc=T1)
    r = copy.deepcopy(r2)
    ev = r["events"][0]
    ev["payload"]["outcome_snapshot_id"] = "other"
    ev["event_id"] = receipts.event_id(
        r["decision_id"], 1, ev["event_type"], ev["at_utc"], ev["payload"]
    )
    r["action"], r["outcome"] = receipts.derive_state(r)
    with pytest.raises(receipts.ReceiptError, match="^outcome_snapshot"):
        receipts.validate_receipt(r)


# --- merge ---------------------------------------------------------------------------------------------


def test_merge_identical_receipts_is_idempotent() -> None:
    r2 = LIFECYCLE["receipt_after_outcome"]
    assert receipts.merge(r2, copy.deepcopy(r2)) == r2
    r0 = LIFECYCLE["receipt_new"]
    assert receipts.merge(r0, copy.deepcopy(r0)) == r0


def test_merge_accepts_a_superset_of_events_in_either_direction() -> None:
    r0, r1, r2 = (
        LIFECYCLE["receipt_new"],
        LIFECYCLE["receipt_after_action"],
        LIFECYCLE["receipt_after_outcome"],
    )
    assert receipts.merge(r0, r2) == r2
    assert receipts.merge(r2, r0) == r2
    assert receipts.merge(r1, r2) == r2
    merged = receipts.merge(r2, r1)
    assert merged == r2
    assert merged["action"]["state"] == "recorded" and merged["outcome"]["state"] == "attached"


def test_merge_with_a_different_result_is_a_conflict() -> None:
    r0 = LIFECYCLE["receipt_new"]
    other = copy.deepcopy(r0)
    other["result_sha256"] = "3" * 64
    with pytest.raises(receipts.ConflictError, match="different inputs or result"):
        receipts.merge(r0, other)
    other = copy.deepcopy(r0)
    other["inputs"]["parameters"]["min_floor"] = 8.0  # same decision_id claimed, different inputs
    with pytest.raises(receipts.ConflictError, match="different inputs or result"):
        receipts.merge(r0, other)


def test_merge_with_a_diverging_event_is_a_conflict() -> None:
    r0 = LIFECYCLE["receipt_new"]
    recorded = LIFECYCLE["receipt_after_action"]
    declined = LIFECYCLE["receipt_declined"]
    assert recorded["decision_id"] == declined["decision_id"]
    with pytest.raises(receipts.ConflictError, match="event 1 differs"):
        receipts.merge(recorded, declined)
    later = receipts.record_action(
        r0, chosen_player_id="SYN-A", kind="hypothetical_replay", at_utc=T2
    )
    with pytest.raises(receipts.ConflictError, match="event 1 differs"):
        receipts.merge(recorded, later)  # same choice, different timestamp: a different event


def test_merge_with_different_decision_ids_is_a_conflict() -> None:
    r0 = LIFECYCLE["receipt_new"]
    other = receipts.new_receipt(_inputs([A, B], _params(0.3, 6.0)), created_at_utc=T0)
    assert other["decision_id"] != r0["decision_id"]
    with pytest.raises(receipts.ConflictError, match="different decision ids"):
        receipts.merge(r0, other)
    assert issubclass(receipts.ConflictError, receipts.ReceiptError)


# --- parent / child decisions -----------------------------------------------------------------------------


def test_changing_a_parameter_starts_a_new_decision_linked_to_its_parent() -> None:
    parent = receipts.new_receipt(GOLDEN_INPUTS, created_at_utc=T0)
    child_inputs = copy.deepcopy(GOLDEN_INPUTS)
    child_inputs["parameters"]["min_floor"] = 8.0
    child = receipts.new_receipt(
        child_inputs, created_at_utc=T1, parent_decision_id=parent["decision_id"]
    )
    assert child["decision_id"] != parent["decision_id"]
    assert child["parent_decision_id"] == parent["decision_id"]
    assert parent["parent_decision_id"] is None
    assert parent["result"]["status"] == "recommend" and child["result"]["status"] == "review"
    receipts.validate_receipt(child)
    contracts.validate("receipt", child)


def test_changing_an_availability_assumption_starts_a_new_decision() -> None:
    parent = bundle_receipt()
    di = receipts.build_decision_inputs(
        SNAPSHOT,
        slot="RB",
        player_ids=["SYN-A", "SYN-B"],
        overrides={"SYN-A": {"availability": "unavailable", "availability_basis": "user_excluded"}},
        parameters={"min_projection_gap": 0.3},
    )
    child = receipts.new_receipt(di, created_at_utc=T1, parent_decision_id=parent["decision_id"])
    assert child["decision_id"] != parent["decision_id"]
    assert child["parent_decision_id"] == parent["decision_id"]
    assert child["result"]["status"] == "review"
    assert "excluded_unavailable" in [r["code"] for r in child["result"]["reasons"]]


def test_timestamps_and_notes_never_enter_the_decision_id() -> None:
    a = receipts.new_receipt(GOLDEN_INPUTS, created_at_utc=T0, prediction_note="x", case_id="one")
    b = receipts.new_receipt(GOLDEN_INPUTS, created_at_utc=T2, prediction_note="y", case_id="two")
    assert a["decision_id"] == b["decision_id"]
    assert a["result_sha256"] == b["result_sha256"]
