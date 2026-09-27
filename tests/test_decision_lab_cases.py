"""Curated case library: every expected block is reproducible from the committed bundle."""

from __future__ import annotations

import pytest

from ffai.config import REPO_ROOT
from ffai.decision_lab import bundle, cases, metrics, policy, receipts

COMMITTED = REPO_ROOT / "artifacts" / "decision_lab"
AMBIGUITY = "syn-ambiguity-floor-relaxation"

pytestmark = pytest.mark.skipif(
    not (COMMITTED / "manifest.json").exists(), reason="run scripts/export_decision_lab.py"
)


@pytest.fixture(scope="module")
def lab() -> bundle.Bundle:
    return bundle.load_bundle(COMMITTED)


@pytest.fixture(scope="module")
def case_list(lab: bundle.Bundle) -> list[dict]:
    return lab.cases()["cases"]


def _case(case_list: list[dict], case_id: str) -> dict:
    return next(c for c in case_list if c["case_id"] == case_id)


def test_every_case_expected_matches_fresh_evaluation(lab, case_list):
    assert len(case_list) >= 17
    for c in case_list:
        snap = lab.inputs_snapshot(c["snapshot_id"])
        result = policy.evaluate(cases.case_inputs(c, snap))
        assert result["status"] == c["expected"]["status"], c["case_id"]
        assert result["recommended_player_id"] == c["expected"]["recommended_player_id"], c[
            "case_id"
        ]
        assert [r["code"] for r in result["reasons"]] == c["expected"]["reason_codes"], c["case_id"]
        assert c["mode"] == snap["mode"]
        expected_limits = list(result["limitations"])
        if "outcomes_inspectable" not in expected_limits:
            expected_limits.append("outcomes_inspectable")
        assert c["limitations"] == expected_limits, c["case_id"]
        assert c["outcomes_available"] == (lab.outcome_snapshot(c["snapshot_id"]) is not None)
        assert c["outcome_snapshot_id"] == (c["snapshot_id"] if c["outcomes_available"] else None)


def test_every_experiment_expected_matches_fresh_evaluation(lab, case_list):
    n = 0
    for c in case_list:
        snap = lab.inputs_snapshot(c["snapshot_id"])
        for ex in c["experiments"]:
            n += 1
            result = policy.evaluate(cases.experiment_inputs(c, ex, snap))
            assert result["status"] == ex["expected"]["status"], (c["case_id"], ex["label"])
            assert result["recommended_player_id"] == ex["expected"]["recommended_player_id"]
    assert n >= 7


def test_ambiguity_case_numbers(lab, case_list):
    c = _case(case_list, AMBIGUITY)
    snap = lab.inputs_snapshot(AMBIGUITY)
    assert c["parameters"]["min_projection_gap"] == 0.5 and c["parameters"]["min_floor"] == 8.0
    base = policy.evaluate(cases.case_inputs(c, snap))
    assert base["status"] == "review"
    codes = {r["code"] for r in base["reasons"]}
    assert {"gap_below_minimum", "floor_below_minimum", "baseline_disagrees"} <= codes
    assert base["leading"] == {"first": "SYN-A", "second": "SYN-B", "gap": 0.4}
    relaxed = policy.evaluate(cases.experiment_inputs(c, c["experiments"][0], snap))
    assert relaxed["status"] == "review"
    assert relaxed["gates"]["floor_ok"] is True and relaxed["gates"]["gap_ok"] is False
    resolved_inputs = cases.experiment_inputs(c, c["experiments"][1], snap)
    resolved = policy.evaluate(resolved_inputs)
    assert resolved["status"] == "recommend" and resolved["recommended_player_id"] == "SYN-A"
    assert resolved["baseline"]["preferred_player_id"] == "SYN-B"
    assert resolved["baseline"]["agrees_with_model_leader"] is False

    outcomes = lab.outcome_snapshot(AMBIGUITY)
    assert outcomes is not None
    actual = {r["player_id"]: r["actual"] for r in outcomes["rows"]}
    assert actual == {"SYN-A": 8.0, "SYN-B": 16.0}

    r0 = receipts.new_receipt(
        resolved_inputs, created_at_utc="2099-01-01T00:00:00+00:00", case_id=AMBIGUITY
    )
    r1 = receipts.record_action(
        r0, chosen_player_id="SYN-A", kind="hypothetical_replay", at_utc="2099-01-01T00:01:00+00:00"
    )
    r2 = receipts.attach_outcome(r1, outcomes, at_utc="2099-01-01T00:02:00+00:00")
    assert r2["decision_id"] == r0["decision_id"]
    m = r2["outcome"]["metrics"]
    assert m["choice_set_regret"] == 8.0
    assert m["points_vs_baseline_choice"] == -8.0
    assert m["model_policy_vs_baseline_choice"] == -8.0
    assert m["outcome_id"] == lab.outcome_id(AMBIGUITY)

    no_action = metrics.compute_metrics(
        resolved, r0["action"], {**outcomes, "outcome_id": lab.outcome_id(AMBIGUITY)}
    )
    assert no_action["chosen_actual_points"] is None
    assert no_action["chosen_actual_reason"] == "no_action_recorded"
    assert no_action["choice_set_regret"] is None
    assert no_action["choice_set_regret_reason"] == "no_action_recorded"
    assert no_action["points_vs_baseline_choice"] is None
    assert no_action["model_policy_vs_baseline_choice"] == -8.0


def test_titles_do_not_reveal_outcomes(case_list):
    for c in case_list:
        lowered = c["title"].lower()
        for w in ("wins", "loses", "regret", "beat"):
            assert w not in lowered, c["title"]
        assert c["selection"] is not None
        if c["case_id"].startswith("real-"):
            assert c["selection"]["method"] == "deterministic_discovery"
            assert c["title"].split(" · ")[0].split()[0].isdigit()
        else:
            assert c["selection"]["method"] == "curated_synthetic"
            assert c["selection"]["sealed_pattern"] is None


def test_real_cases_reference_snapshot_players(lab, case_list):
    real = [c for c in case_list if c["case_id"].startswith("real-")]
    assert len(real) >= 6
    patterns = {c["selection"]["sealed_pattern"] for c in real}
    assert {"model_wins", "baseline_wins", "agreement", "ambiguity", None} <= patterns
    for c in real:
        snap = lab.inputs_snapshot(c["snapshot_id"])
        rows = {r["player_id"]: r for r in snap["rows"]}
        assert set(c["alternatives"]) <= set(rows)
        assert c["source_row_refs"] == [rows[p]["source_row_ref"] for p in c["alternatives"]]
        assert {rows[p]["position"] for p in c["alternatives"]} == {c["slot"]}
        projections = [rows[p]["projection"] for p in c["alternatives"]]
        assert projections == sorted(projections, reverse=True)
        top = sorted(
            (r for r in snap["rows"] if r["position"] == c["slot"]),
            key=lambda r: (-r["projection"], r["player_id"]),
        )[: len(c["alternatives"])]
        assert [r["player_id"] for r in top] == c["alternatives"]
        if snap["source_family"] == "weekly" and snap["mode"] == "historical_replay":
            assert all(v["availability"] == "assumed_available" for v in c["overrides"].values())
        pattern = c["selection"]["sealed_pattern"]
        if pattern in ("model_wins", "baseline_wins", "tie"):
            result = policy.evaluate(cases.case_inputs(c, snap))
            outcomes = lab.outcome_snapshot(c["snapshot_id"])
            actual = {
                r["player_id"]: r["actual"] for r in outcomes["rows"] if r["actual"] is not None
            }
            rec, base = result["recommended_player_id"], result["baseline"]["preferred_player_id"]
            assert rec != base
            sign = (actual[rec] > actual[base]) - (actual[rec] < actual[base])
            assert {1: "model_wins", -1: "baseline_wins", 0: "tie"}[sign] == pattern
    published = [c for c in real if c["snapshot_id"] == lab.manifest["latest_weekly_snapshot_id"]]
    assert len(published) == 1
    assert published[0]["expected"]["status"] == "review"
    assert "availability_unresolved" in published[0]["expected"]["reason_codes"]
    assert published[0]["outcomes_available"] is False
    assert published[0]["experiments"][0]["expected"]["status"] in ("recommend", "review")


def test_notes_mention_sealed_patterns_and_bias(lab):
    notes = " ".join(lab.cases()["notes"]).lower()
    assert "sealed" in notes and "not secure blinding" in notes
    assert "not an unbiased sample" in notes
    assert "half-ppr" in notes


def test_synthetic_snapshots_are_labelled_synthetic(lab, case_list):
    for c in case_list:
        if not c["case_id"].startswith("syn-"):
            continue
        snap = lab.inputs_snapshot(c["snapshot_id"])
        assert snap["mode"] == "synthetic" and snap["source_family"] == "synthetic"
        assert snap["model_version"] == "synthetic-model-v1" and snap["season"] == 2099
        assert snap["source"]["files"] == []
        assert {r["team"] for r in snap["rows"]} == {"SYN"}
        assert all(r["display_name"].startswith("Synthetic ") for r in snap["rows"])
        assert "synthetic_evidence" in c["limitations"]
    blocked = _case(case_list, "syn-blocked-evidence")
    assert lab.inputs_snapshot(blocked["snapshot_id"])["publication"]["status"] == "blocked"
    assert blocked["expected"]["status"] == "hold"
    assert blocked["experiments"][0]["expected"]["status"] == "hold"
