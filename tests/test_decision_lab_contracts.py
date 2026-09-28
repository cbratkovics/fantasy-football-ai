"""JSON Schemas and semantic checks for every Decision Lab document (``ffai.decision_lab.contracts``).

Schemas are closed (``additionalProperties: false``) and required-but-nullable fields must be
present; the semantic ``check_*`` functions cover what a schema cannot say. The frontend mirror
of ``policy_spec.json`` must be byte-identical to the Python one (like tests/test_project_config.py).
"""

from __future__ import annotations

import copy
import json
from typing import Any

import jsonschema
import pytest

from ffai.config import REPO_ROOT
from ffai.decision_lab import POLICY_VERSION, SCHEMA_VERSIONS, SPEC_PATH, contracts, golden
from ffai.decision_lab.canonical import content_id, file_sha256
from ffai.decision_lab.contracts import ContractError

GOLDEN = json.loads(golden.GOLDEN_PATH.read_text(encoding="utf-8"))
LIFECYCLE = GOLDEN["receipt"]
FRONTEND_SPEC = REPO_ROOT / "frontend-next" / "src" / "lib" / "decision-lab" / "policy_spec.json"
MODEL = "synthetic-model-v1"
SHA0 = "0" * 64


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


def make_inputs_snapshot() -> dict[str, Any]:
    rows = [
        _row("SYN-A", "RB", 14.0, 6.0, 12.0),
        _row("SYN-B", "RB", 13.6, 7.0, 13.0),
        _row("SYN-C", "WR", None, None, 10.0, availability="unknown"),
    ]
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
            "n_rows": len(rows),
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
        "rows": rows,
    }


def make_outcome_snapshot(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "outcome_snapshot/1.0",
        "snapshot_id": inputs["snapshot_id"],
        "inputs_content_sha256": content_id(inputs),
        "season": inputs["season"],
        "week": inputs["week"],
        "model_version": inputs["model_version"],
        "scoring_format": "ppr",
        "observed_at_utc": "2099-01-08T00:00:00+00:00",
        "source": {"mart": None, "files": []},
        "coverage": {"scored": 3, "observed": 2},
        "rows": [
            {"player_id": "SYN-A", "actual": 8.0, "actual_source": "synthetic"},
            {"player_id": "SYN-B", "actual": 0.0, "actual_source": "synthetic"},
            {"player_id": "SYN-C", "actual": None, "actual_source": None},
        ],
    }


def make_case(case_id: str = "syn-case-one", **kw: Any) -> dict[str, Any]:
    case = {
        "case_id": case_id,
        "title": "Synthetic case",
        "mode": "synthetic",
        "snapshot_id": "syn-test",
        "slot": "RB",
        "alternatives": ["SYN-A", "SYN-B"],
        "overrides": {},
        "parameters": {
            "min_projection_gap": 0.3,
            "min_floor": None,
            "require_independent_baseline": True,
            "model_only_exploration": False,
        },
        "setup": "A leads B by 0.4.",
        "experiments": [
            {
                "label": "raise the floor",
                "parameters": {"min_floor": 8.0},
                "overrides": {},
                "expected": {"status": "review", "recommended_player_id": None},
            },
            {
                "label": "no change",
                "expected": {"status": "recommend", "recommended_player_id": "SYN-A"},
            },
        ],
        "expected": {
            "status": "recommend",
            "recommended_player_id": "SYN-A",
            "reason_codes": ["recommend_highest_projection"],
        },
        "outcomes_available": True,
        "outcome_snapshot_id": "syn-test",
        "limitations": ["synthetic_evidence"],
        "selection": {
            "method": "curated_synthetic",
            "criteria": "boundary",
            "sealed_pattern": None,
        },
        "source_row_refs": [],
    }
    case.update(kw)
    return case


def make_cases(*cases: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "decision_lab_cases/1.0",
        "policy_version": POLICY_VERSION,
        "cases": list(cases) or [make_case()],
        "notes": ["synthetic"],
    }


def _file_ref(**kw: Any) -> dict[str, Any]:
    ref = {"path": "inputs/syn-test.json", "file_sha256": SHA0, "content_sha256": SHA0, "n_rows": 3}
    ref.update(kw)
    return ref


def make_manifest_snapshot(
    snapshot_id: str = "syn-test", mode: str = "synthetic"
) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot_id,
        "mode": mode,
        "source_family": "synthetic" if mode == "synthetic" else "weekly",
        "season": 2099,
        "week": 1,
        "model_version": MODEL,
        "candidate_by_position": {"RB": "rf"},
        "inputs": _file_ref(path=f"inputs/{snapshot_id}.json"),
        "outcomes": {
            "path": f"outcomes/{snapshot_id}.json",
            "file_sha256": SHA0,
            "content_sha256": SHA0,
            "n_rows": 3,
            "n_observed": 2,
        },
        "population": {
            "conditioning": "synthetic",
            "description": "synthetic rows",
            "n_rows": 3,
            "exclusions": [],
        },
        "baseline": {
            "name": "trailing_mean",
            "provenance_basis": "synthetic",
            "reconciled": None,
            "reconciled_to": None,
            "note": None,
        },
        "reconciliation": None,
    }


def make_manifest(**kw: Any) -> dict[str, Any]:
    doc = {
        "schema_version": "decision_lab_manifest/1.0",
        "exporter_version": "test",
        "policy_version": POLICY_VERSION,
        "schema_versions": dict(SCHEMA_VERSIONS),
        "code_revision": {"produced_at": None, "note": "test"},
        "lineage": {},
        "sources": [{"role": "mart", "path": "x.parquet", "sha256": None, "rows": None}],
        "snapshots": [
            make_manifest_snapshot("syn-test"),
            make_manifest_snapshot("weekly-2099-w1", "published_weekly"),
        ],
        "cases": _file_ref(path="cases.json", n_rows=1),
        "latest_weekly_snapshot_id": "weekly-2099-w1",
        "policy_spec": {"path": "policy_spec.json", "sha256": file_sha256(SPEC_PATH)},
        "digest_coverage": {"file_sha256": "bytes", "content_sha256": "canonical json"},
    }
    doc.update(kw)
    return doc


VALID_DOCS: dict[str, Any] = {
    "decision_inputs": LIFECYCLE["inputs"],
    "decision_result": LIFECYCLE["receipt_new"]["result"],
    "receipt": LIFECYCLE["receipt_after_outcome"],
    "inputs_snapshot": make_inputs_snapshot(),
    "outcome_snapshot": make_outcome_snapshot(make_inputs_snapshot()),
    "cases": make_cases(),
    "manifest": make_manifest(),
}

CHECKERS = {
    "decision_inputs": contracts.check_decision_inputs,
    "decision_result": contracts.check_decision_result,
    "receipt": lambda d: contracts.validate("receipt", d),
    "inputs_snapshot": contracts.check_inputs_snapshot,
    "outcome_snapshot": contracts.check_outcome_snapshot,
    "cases": contracts.check_cases,
    "manifest": contracts.check_manifest,
}


# --- schemas -----------------------------------------------------------------------------------------


def test_every_schema_is_registered_once_per_schema_version() -> None:
    # outcome_metrics has no standalone schema: a metrics block is verified by replay
    # (receipts.validate_receipt recomputes it from the outcome snapshot).
    assert set(contracts.SCHEMAS) == set(SCHEMA_VERSIONS) - {"outcome_metrics"}


@pytest.mark.parametrize("name", sorted(contracts.SCHEMAS))
def test_schema_is_a_valid_draft_2020_12_schema(name: str) -> None:
    jsonschema.Draft202012Validator.check_schema(contracts.SCHEMAS[name])


@pytest.mark.parametrize("name", sorted(contracts.SCHEMAS))
def test_top_level_schema_is_closed_with_every_property_required(name: str) -> None:
    schema = contracts.SCHEMAS[name]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == sorted(schema["properties"])
    assert schema["properties"]["schema_version"] == {"const": SCHEMA_VERSIONS[name]}


@pytest.mark.parametrize("name", sorted(VALID_DOCS))
def test_valid_documents_pass_schema_and_semantic_checks(name: str) -> None:
    contracts.validate(name, VALID_DOCS[name])
    CHECKERS[name](copy.deepcopy(VALID_DOCS[name]))


@pytest.mark.parametrize("name", sorted(VALID_DOCS))
def test_additional_top_level_properties_are_rejected(name: str) -> None:
    doc = copy.deepcopy(VALID_DOCS[name])
    doc["unexpected_field"] = 1
    with pytest.raises(ContractError, match="unexpected_field"):
        contracts.validate(name, doc)
    with pytest.raises(ContractError):
        CHECKERS[name](doc)


@pytest.mark.parametrize("name", sorted(VALID_DOCS))
def test_wrong_schema_version_is_rejected(name: str) -> None:
    doc = copy.deepcopy(VALID_DOCS[name])
    doc["schema_version"] = doc["schema_version"].replace("1.0", "9.9")
    with pytest.raises(ContractError, match="schema_version"):
        contracts.validate(name, doc)


def test_validate_names_the_document_and_the_field() -> None:
    doc = copy.deepcopy(VALID_DOCS["decision_result"])
    doc["gates"]["gap_ok"] = "yes"
    with pytest.raises(ContractError, match=r"^decision_result: gates/gap_ok:"):
        contracts.validate("decision_result", doc)
    with pytest.raises(KeyError):
        contracts.validate("not_a_schema", {})


# --- golden documents ------------------------------------------------------------------------------------


def test_golden_policy_inputs_and_results_validate() -> None:
    for case in GOLDEN["policy"]:
        contracts.check_decision_result(case["expected"])
        if case["name"] != "parameters_invalid_floor_type":
            contracts.check_decision_inputs(case["inputs"])


def test_golden_parameters_invalid_inputs_split_between_contract_and_policy() -> None:
    by_name = {c["name"]: c for c in GOLDEN["policy"]}
    # A wrongly typed parameter is a structural violation: the contract rejects it.
    with pytest.raises(ContractError, match="parameters/min_floor"):
        contracts.check_decision_inputs(by_name["parameters_invalid_floor_type"]["inputs"])
    # An out-of-range parameter is well-formed: the contract accepts it and the policy HOLDs.
    negative = by_name["parameters_invalid_negative_gap"]
    contracts.check_decision_inputs(negative["inputs"])
    assert negative["expected"]["status"] == "hold"
    assert negative["expected"]["reasons"][0]["code"] == "parameters_invalid"


def test_golden_receipts_validate() -> None:
    for key in ("receipt_new", "receipt_after_action", "receipt_after_outcome", "receipt_declined"):
        contracts.validate("receipt", LIFECYCLE[key])
    contracts.check_outcome_snapshot(LIFECYCLE["outcome_snapshot"])


# --- required nullable fields ----------------------------------------------------------------------------


def test_recommended_player_id_must_be_present_even_when_null() -> None:
    review = copy.deepcopy(
        next(c["expected"] for c in GOLDEN["policy"] if c["expected"]["status"] == "review")
    )
    assert review["recommended_player_id"] is None
    contracts.check_decision_result(review)
    del review["recommended_player_id"]
    with pytest.raises(ContractError, match="recommended_player_id"):
        contracts.validate("decision_result", review)


def test_review_or_hold_may_not_carry_a_recommendation() -> None:
    review = copy.deepcopy(
        next(c["expected"] for c in GOLDEN["policy"] if c["expected"]["status"] == "review")
    )
    review["recommended_player_id"] = review["ranking"][0]["player_id"]
    with pytest.raises(ContractError, match="carries a recommended_player_id"):
        contracts.check_decision_result(review)
    recommend = copy.deepcopy(VALID_DOCS["decision_result"])
    assert recommend["status"] == "recommend"
    recommend["recommended_player_id"] = None
    with pytest.raises(ContractError, match="recommend without recommended_player_id"):
        contracts.check_decision_result(recommend)
    recommend["recommended_player_id"] = "SYN-Z"
    with pytest.raises(ContractError, match="not among the alternatives"):
        contracts.check_decision_result(recommend)


def test_leading_and_baseline_ids_must_be_alternatives() -> None:
    doc = copy.deepcopy(VALID_DOCS["decision_result"])
    doc["leading"]["second"] = "SYN-Z"
    with pytest.raises(ContractError, match="leading.second"):
        contracts.check_decision_result(doc)
    doc = copy.deepcopy(VALID_DOCS["decision_result"])
    doc["baseline"]["preferred_player_id"] = "SYN-Z"
    with pytest.raises(ContractError, match="baseline.preferred_player_id"):
        contracts.check_decision_result(doc)


def test_unknown_status_and_reason_codes_are_rejected() -> None:
    doc = copy.deepcopy(VALID_DOCS["decision_result"])
    doc["status"] = "maybe"
    with pytest.raises(ContractError, match="status"):
        contracts.validate("decision_result", doc)
    doc = copy.deepcopy(VALID_DOCS["decision_result"])
    doc["reasons"][0]["code"] = "vibes"
    with pytest.raises(ContractError, match="reasons/0/code"):
        contracts.validate("decision_result", doc)
    doc = copy.deepcopy(VALID_DOCS["decision_result"])
    doc["limitations"].append("not_a_limitation")
    with pytest.raises(ContractError, match="limitations"):
        contracts.validate("decision_result", doc)


def test_action_block_is_required_with_nulls_before_anything_is_recorded() -> None:
    receipt = copy.deepcopy(LIFECYCLE["receipt_new"])
    assert receipt["action"]["state"] == "not_recorded" and receipt["action"]["action_id"] is None
    contracts.validate("receipt", receipt)
    del receipt["action"]["action_id"]
    with pytest.raises(ContractError, match="action"):
        contracts.validate("receipt", receipt)
    receipt = copy.deepcopy(LIFECYCLE["receipt_new"])
    receipt["action"]["kind"] = "guess"
    with pytest.raises(ContractError, match="action/kind"):
        contracts.validate("receipt", receipt)


def test_nested_objects_are_closed_too() -> None:
    inputs = copy.deepcopy(VALID_DOCS["decision_inputs"])
    inputs["parameters"]["min_ceiling"] = 1.0
    with pytest.raises(ContractError, match="parameters"):
        contracts.validate("decision_inputs", inputs)
    inputs = copy.deepcopy(VALID_DOCS["decision_inputs"])
    inputs["snapshot"]["extra"] = 1
    with pytest.raises(ContractError, match="snapshot"):
        contracts.validate("decision_inputs", inputs)
    result = copy.deepcopy(VALID_DOCS["decision_result"])
    result["ranking"][0]["actual"] = 9.0
    with pytest.raises(ContractError, match="ranking/0"):
        contracts.validate("decision_result", result)


# --- decision inputs ---------------------------------------------------------------------------------------


def test_decision_inputs_reject_outcome_fields_and_non_finite_numbers() -> None:
    doc = copy.deepcopy(VALID_DOCS["decision_inputs"])
    doc["alternatives"][0]["actual"] = 20.0
    with pytest.raises(ContractError, match="actual"):
        contracts.check_decision_inputs(doc)
    doc = copy.deepcopy(VALID_DOCS["decision_inputs"])
    doc["alternatives"][0]["projection"] = float("nan")
    with pytest.raises(ContractError, match="SYN-A.projection"):
        contracts.check_decision_inputs(doc)
    doc = copy.deepcopy(VALID_DOCS["decision_inputs"])
    doc["parameters"]["min_floor"] = float("inf")
    with pytest.raises(ContractError, match="parameters.min_floor"):
        contracts.check_decision_inputs(doc)
    doc = copy.deepcopy(VALID_DOCS["decision_inputs"])
    doc["alternatives"][0]["availability"] = "probably"
    with pytest.raises(ContractError, match="availability"):
        contracts.check_decision_inputs(doc)


def test_decision_inputs_pin_the_policy_version() -> None:
    doc = copy.deepcopy(VALID_DOCS["decision_inputs"])
    doc["policy_version"] = "0.9.0"
    with pytest.raises(ContractError, match="policy_version"):
        contracts.validate("decision_inputs", doc)


# --- inputs snapshot ------------------------------------------------------------------------------------------


def test_inputs_snapshot_semantic_checks() -> None:
    snap = make_inputs_snapshot()
    snap["rows"].append(dict(snap["rows"][0]))
    snap["population"]["n_rows"] = 4
    with pytest.raises(ContractError, match="duplicate player_id SYN-A"):
        contracts.check_inputs_snapshot(snap)

    snap = make_inputs_snapshot()
    snap["population"]["n_rows"] = 99
    with pytest.raises(ContractError, match="n_rows"):
        contracts.check_inputs_snapshot(snap)

    snap = make_inputs_snapshot()
    snap["rows"][0]["floor"] = 14.5
    with pytest.raises(ContractError, match="floor above projection"):
        contracts.check_inputs_snapshot(snap)

    snap = make_inputs_snapshot()
    snap["rows"][0]["ceiling"] = 13.0
    with pytest.raises(ContractError, match="ceiling below projection"):
        contracts.check_inputs_snapshot(snap)

    snap = make_inputs_snapshot()
    snap["rows"][0]["actual"] = 20.0
    with pytest.raises(ContractError, match="actual"):
        contracts.check_inputs_snapshot(snap)

    snap = make_inputs_snapshot()
    snap["scoring_format"] = "half"
    with pytest.raises(ContractError, match="scoring_format must be ppr"):
        contracts.check_inputs_snapshot(snap)

    snap = make_inputs_snapshot()
    snap["rows"][0]["model_version"] = "other"
    with pytest.raises(ContractError, match="model_version differs"):
        contracts.check_inputs_snapshot(snap)

    snap = make_inputs_snapshot()
    snap["rows"][0]["candidate"] = "xgb"
    with pytest.raises(ContractError, match="candidate xgb != rf"):
        contracts.check_inputs_snapshot(snap)

    snap = make_inputs_snapshot()
    snap["rows"][0]["projection"] = float("nan")
    with pytest.raises(ContractError, match="finite"):
        contracts.check_inputs_snapshot(snap)


def test_inputs_snapshot_floor_equal_to_projection_is_allowed() -> None:
    snap = make_inputs_snapshot()
    snap["rows"][0]["floor"] = snap["rows"][0]["projection"]
    snap["rows"][0]["ceiling"] = snap["rows"][0]["projection"]
    contracts.check_inputs_snapshot(snap)


# --- outcome snapshot ---------------------------------------------------------------------------------------------


def test_outcome_snapshot_semantic_checks() -> None:
    inputs = make_inputs_snapshot()

    out = make_outcome_snapshot(inputs)
    out["coverage"]["observed"] = 3
    with pytest.raises(ContractError, match="coverage counts disagree"):
        contracts.check_outcome_snapshot(out)

    out = make_outcome_snapshot(inputs)
    out["coverage"]["scored"] = 2
    with pytest.raises(ContractError, match="coverage counts disagree"):
        contracts.check_outcome_snapshot(out)

    out = make_outcome_snapshot(inputs)
    out["rows"][0]["actual_source"] = None
    with pytest.raises(ContractError, match="observed actual without a source"):
        contracts.check_outcome_snapshot(out)

    out = make_outcome_snapshot(inputs)
    out["rows"].append({"player_id": "SYN-Z", "actual": 1.0, "actual_source": "synthetic"})
    out["coverage"] = {"scored": 4, "observed": 3}
    contracts.check_outcome_snapshot(out)  # standalone it is fine
    with pytest.raises(ContractError, match="players not in the inputs snapshot"):
        contracts.check_outcome_snapshot(out, inputs)

    out = make_outcome_snapshot(inputs)
    out["inputs_content_sha256"] = "f" * 64
    contracts.check_outcome_snapshot(out)
    with pytest.raises(ContractError, match="inputs_content_sha256 does not match"):
        contracts.check_outcome_snapshot(out, inputs)

    out = make_outcome_snapshot(inputs)
    out["snapshot_id"] = "other"
    with pytest.raises(ContractError, match="different snapshot_id"):
        contracts.check_outcome_snapshot(out, inputs)

    out = make_outcome_snapshot(inputs)
    out["week"] = 2
    with pytest.raises(ContractError, match="period or model differs"):
        contracts.check_outcome_snapshot(out, inputs)

    out = make_outcome_snapshot(inputs)
    out["rows"].append(dict(out["rows"][0]))
    out["coverage"] = {"scored": 4, "observed": 3}
    with pytest.raises(ContractError, match="duplicate player_id"):
        contracts.check_outcome_snapshot(out)

    out = make_outcome_snapshot(inputs)
    out["scoring_format"] = "half"
    with pytest.raises(ContractError, match="scoring_format must be ppr"):
        contracts.check_outcome_snapshot(out)


def test_outcome_snapshot_zero_actual_counts_as_observed() -> None:
    inputs = make_inputs_snapshot()
    out = make_outcome_snapshot(inputs)
    assert out["rows"][1]["actual"] == 0.0 and out["coverage"]["observed"] == 2
    contracts.check_outcome_snapshot(out, inputs)


# --- cases ---------------------------------------------------------------------------------------------------------


def test_cases_semantic_checks() -> None:
    contracts.check_cases(make_cases(make_case("syn-case-one"), make_case("syn-case-two")))
    with pytest.raises(ContractError, match="duplicate case_id syn-case-one"):
        contracts.check_cases(make_cases(make_case("syn-case-one"), make_case("syn-case-one")))
    with pytest.raises(ContractError, match="override for a player not nominated: SYN-Z"):
        contracts.check_cases(
            make_cases(
                make_case(
                    overrides={
                        "SYN-Z": {
                            "availability": "unavailable",
                            "availability_basis": "user_excluded",
                        }
                    }
                )
            )
        )
    with pytest.raises(ContractError, match="duplicate player_id"):
        contracts.check_cases(make_cases(make_case(alternatives=["SYN-A", "SYN-A"])))
    with pytest.raises(ContractError, match="case_id"):
        contracts.check_cases(make_cases(make_case("Bad Case Id")))
    with pytest.raises(ContractError, match="overrides"):
        contracts.check_cases(
            make_cases(make_case(overrides={"SYN-A": {"availability": "unavailable"}}))
        )
    with pytest.raises(ContractError, match="parameters.min_floor"):
        contracts.check_cases(
            make_cases(
                make_case(parameters={**make_case()["parameters"], "min_floor": float("nan")})
            )
        )


def test_cases_pin_the_policy_version_and_status_vocabulary() -> None:
    doc = make_cases()
    doc["policy_version"] = "0.1.0"
    with pytest.raises(ContractError, match="policy_version"):
        contracts.check_cases(doc)
    doc = make_cases(
        make_case(expected={"status": "publish", "recommended_player_id": None, "reason_codes": []})
    )
    with pytest.raises(ContractError, match="status"):
        contracts.check_cases(doc)


# --- manifest -------------------------------------------------------------------------------------------------------


def test_manifest_semantic_checks() -> None:
    contracts.check_manifest(make_manifest())
    with pytest.raises(ContractError, match="duplicate snapshot_id"):
        contracts.check_manifest(
            make_manifest(snapshots=[make_manifest_snapshot("dup"), make_manifest_snapshot("dup")])
        )
    with pytest.raises(ContractError, match="latest_weekly_snapshot_id is not a listed snapshot"):
        contracts.check_manifest(make_manifest(latest_weekly_snapshot_id="weekly-2099-w9"))
    contracts.check_manifest(make_manifest(latest_weekly_snapshot_id=None))
    with pytest.raises(ContractError, match="schema_versions differ"):
        contracts.check_manifest(
            make_manifest(schema_versions={**SCHEMA_VERSIONS, "receipt": "decision_receipt/2.0"})
        )
    with pytest.raises(ContractError, match="schema_versions differ"):
        contracts.check_manifest(
            make_manifest(
                schema_versions={k: v for k, v in SCHEMA_VERSIONS.items() if k != "cases"}
            )
        )


def test_manifest_schema_rejects_bad_digests_and_missing_blocks() -> None:
    doc = make_manifest()
    doc["policy_spec"]["sha256"] = "not-a-digest"
    with pytest.raises(ContractError, match="policy_spec/sha256"):
        contracts.check_manifest(doc)
    doc = make_manifest()
    del doc["snapshots"][0]["outcomes"]
    with pytest.raises(ContractError, match="snapshots/0"):
        contracts.check_manifest(doc)
    doc = make_manifest()
    doc["snapshots"][0]["outcomes"] = None
    contracts.check_manifest(doc)


# --- frontend mirror ---------------------------------------------------------------------------------------------------


def test_frontend_policy_spec_mirror_is_byte_identical() -> None:
    assert FRONTEND_SPEC.exists(), f"missing mirror {FRONTEND_SPEC}"
    assert FRONTEND_SPEC.read_bytes() == SPEC_PATH.read_bytes(), (
        "frontend-next/src/lib/decision-lab/policy_spec.json differs from ffai/decision_lab/policy_spec.json; "
        "copy the Python spec over the mirror"
    )
    assert json.loads(FRONTEND_SPEC.read_text(encoding="utf-8"))["policy_version"] == POLICY_VERSION


def test_frontend_spec_digest_matches_policy_spec_bytes() -> None:
    """The browser cannot hash its imported spec byte-exactly, so it carries the digest in
    ``spec_digest.json``; it must equal the digest of the shared ``policy_spec.json`` bytes."""
    from ffai.config import REPO_ROOT
    from ffai.decision_lab import SPEC_PATH
    from ffai.decision_lab.canonical import file_sha256

    digest_file = REPO_ROOT / "frontend-next" / "src" / "lib" / "decision-lab" / "spec_digest.json"
    recorded = json.loads(digest_file.read_text(encoding="utf-8"))["sha256"]
    assert recorded == file_sha256(SPEC_PATH)
