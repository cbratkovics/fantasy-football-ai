"""Golden fixtures shared by the Python reference and the TypeScript mirror.

``python -m ffai.decision_lab.golden --write`` regenerates ``tests/fixtures/decision_lab/golden.json``;
``--check`` exits 1 when the committed file differs (the pytest suite runs the check). The
TypeScript tests read the same file and must reproduce every ``expected`` block exactly: canonical
JSON text, SHA-256 digests, policy results (status, recommended id, ordered reason codes,
gap/gate values, baseline preference, limitations, explanation text), outcome metrics, and a
receipt lifecycle (decision id unchanged after an action and an outcome are appended).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ffai.config import REPO_ROOT
from ffai.decision_lab import POLICY_VERSION, metrics, policy, receipts
from ffai.decision_lab.canonical import canonical_json, content_id, sha256_hex

GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "decision_lab" / "golden.json"
SYN_MODEL = "synthetic-model-v1"


def _snapshot(
    *,
    mode: str = "synthetic",
    family: str = "synthetic",
    scoring: str = "ppr",
    publication_status: str = "synthetic",
    evidence_status: str = "verified",
    note: str | None = None,
    baseline_reconciled: bool | None = None,
) -> dict[str, Any]:
    return {
        "snapshot_id": "syn-golden",
        "mode": mode,
        "source_family": family,
        "season": 2099,
        "week": 1,
        "model_version": SYN_MODEL,
        "feature_version": None,
        "candidate_by_position": {"QB": "rf", "RB": "rf", "WR": "rf", "TE": "rf"},
        "scoring_format": scoring,
        "inputs_sha256": "0" * 64,
        "data_cutoff": {"season": 2098, "week": 18},
        "generated_at_utc": None,
        "publication": {"status": publication_status, "run_id": None, "action": None, "note": note},
        "evidence_status": evidence_status,
        "evidence_detail": "byte digest mismatch" if evidence_status == "corrupt" else None,
        "baseline_reconciled": baseline_reconciled,
    }


def _alt(
    pid: str,
    projection: float | None,
    floor: float | None,
    baseline: float | None,
    *,
    position: str = "RB",
    availability: str = "assumed_available",
    basis: str = "user_assumed",
    provenance: str = "player_history",
    candidate: str = "rf",
    model_version: str = SYN_MODEL,
    season: int = 2099,
    week: int = 1,
) -> dict[str, Any]:
    return {
        "player_id": pid,
        "display_name": f"Synthetic {pid[-1]}",
        "team": "SYN",
        "position": position,
        "model_version": model_version,
        "candidate": candidate,
        "season": season,
        "week": week,
        "projection": projection,
        "floor": floor,
        "ceiling": None if projection is None else projection + 8.0,
        "baseline": baseline,
        "baseline_provenance": provenance,
        "availability": availability,
        "availability_basis": basis,
        "source_row_ref": None,
    }


def _params(
    gap: float = 0.5, floor: float | None = None, require: bool = True, model_only: bool = False
) -> dict[str, Any]:
    return {
        "min_projection_gap": gap,
        "min_floor": floor,
        "require_independent_baseline": require,
        "model_only_exploration": model_only,
    }


def _inputs(
    alts: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    slot: str = "RB",
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return policy.normalize_inputs(
        {
            "schema_version": "decision_inputs/1.0",
            "policy_version": POLICY_VERSION,
            "snapshot": snapshot or _snapshot(),
            "slot": slot,
            "alternatives": alts,
            "parameters": params,
        }
    )


A = _alt("SYN-A", 14.0, 6.0, 12.0)
B = _alt("SYN-B", 13.6, 7.0, 13.0)


def policy_cases() -> list[dict[str, Any]]:
    cases: list[tuple[str, dict[str, Any]]] = [
        ("ambiguity_floor_8", _inputs([A, B], _params(0.5, 8.0))),
        ("ambiguity_floor_5_gap_unchanged", _inputs([A, B], _params(0.5, 5.0))),
        ("ambiguity_resolved_gap_0_3", _inputs([A, B], _params(0.3, 5.0))),
        ("order_shuffled_same_result", _inputs([B, A], _params(0.3, 5.0))),
        ("gap_equality_qualifies", _inputs([A, B], _params(0.4, None))),
        ("floor_equality_qualifies", _inputs([A, B], _params(0.3, 6.0))),
        ("floor_just_below_fails", _inputs([A, B], _params(0.3, 6.000001 + 1e-6))),
        (
            "exact_tie_gap_zero",
            _inputs(
                [_alt("SYN-X", 10.0, 1.0, 5.0), _alt("SYN-Y", 10.0, 1.0, 6.0)], _params(0.0, None)
            ),
        ),
        (
            "float_boundary_0_1_plus_0_2",
            _inputs(
                [_alt("SYN-P", 0.1 + 0.2, 0.0, 1.0), _alt("SYN-Q", 0.3, 0.0, 0.5)],
                _params(0.0, None),
            ),
        ),
        (
            "floor_missing_with_guardrail",
            _inputs([_alt("SYN-A", 14.0, None, 12.0), B], _params(0.3, 5.0)),
        ),
        (
            "floor_missing_without_guardrail",
            _inputs([_alt("SYN-A", 14.0, None, 12.0), B], _params(0.3, None)),
        ),
        ("projection_missing", _inputs([_alt("SYN-A", None, 6.0, 12.0), B], _params(0.3, None))),
        (
            "one_option_after_exclusion",
            _inputs(
                [
                    A,
                    _alt(
                        "SYN-B", 13.6, 7.0, 13.0, availability="unavailable", basis="user_excluded"
                    ),
                ],
                _params(0.3, None),
            ),
        ),
        (
            "slot_ineligible_flex_qb",
            _inputs(
                [A, B, _alt("SYN-C", 20.0, 10.0, 18.0, position="QB")],
                _params(0.3, None),
                slot="FLEX",
            ),
        ),
        ("slot_unsupported", _inputs([A, B], _params(0.3, None), slot="K")),
        (
            "unknown_availability_review",
            _inputs(
                [_alt("SYN-A", 14.0, 6.0, 12.0, availability="unknown", basis="not_verified"), B],
                _params(0.3, None),
            ),
        ),
        (
            "hindsight_population_limitation",
            _inputs(
                [
                    _alt(
                        "SYN-A",
                        14.0,
                        6.0,
                        12.0,
                        availability="realized_stats_row",
                        basis="hindsight_stats_row",
                    ),
                    _alt(
                        "SYN-B",
                        13.6,
                        7.0,
                        13.0,
                        availability="realized_stats_row",
                        basis="hindsight_stats_row",
                    ),
                ],
                _params(0.3, None),
            ),
        ),
        (
            "baseline_fallback_required",
            _inputs(
                [_alt("SYN-A", 14.0, 6.0, 14.0, provenance="model_prediction_fallback"), B],
                _params(0.3, None),
            ),
        ),
        (
            "baseline_fallback_model_only",
            _inputs(
                [_alt("SYN-A", 14.0, 6.0, 14.0, provenance="model_prediction_fallback"), B],
                _params(0.3, None, True, True),
            ),
        ),
        (
            "baseline_missing_not_required",
            _inputs(
                [_alt("SYN-A", 14.0, 6.0, None, provenance="unknown"), B],
                _params(0.3, None, False, False),
            ),
        ),
        (
            "baseline_tie",
            _inputs(
                [_alt("SYN-A", 14.0, 6.0, 12.0), _alt("SYN-B", 13.0, 7.0, 12.0)], _params(0.3, None)
            ),
        ),
        ("baseline_agrees", _inputs([_alt("SYN-A", 14.0, 6.0, 15.0), B], _params(0.3, None))),
        (
            "scoring_unsupported_hold",
            _inputs([A, B], _params(0.1, None), snapshot=_snapshot(scoring="half")),
        ),
        (
            "evidence_corrupt_hold_despite_low_gap",
            _inputs([A, B], _params(0.0, None), snapshot=_snapshot(evidence_status="corrupt")),
        ),
        (
            "evidence_blocked_hold",
            _inputs(
                [A, B],
                _params(0.1, None),
                snapshot=_snapshot(
                    publication_status="blocked", note="lab export failed after gold export"
                ),
            ),
        ),
        (
            "mixed_periods_hold",
            _inputs([A, _alt("SYN-B", 13.6, 7.0, 13.0, week=2)], _params(0.1, None)),
        ),
        (
            "mixed_model_versions_hold",
            _inputs(
                [A, _alt("SYN-B", 13.6, 7.0, 13.0, model_version="other-model")], _params(0.1, None)
            ),
        ),
        (
            "mixed_candidates_hold",
            _inputs([A, _alt("SYN-B", 13.6, 7.0, 13.0, candidate="xgb")], _params(0.1, None)),
        ),
        ("parameters_invalid_negative_gap", _inputs([A, B], _params(-1.0, None))),
        (
            "parameters_invalid_floor_type",
            _inputs([A, B], {**_params(0.3, None), "min_floor": "8"}),
        ),
        (
            "choice_set_too_large",
            _inputs([_alt(f"SYN-{i}", 10.0 + i, 1.0, 5.0) for i in range(9)], _params(0.3, None)),
        ),
        ("duplicate_alternative", _inputs([A, dict(A), B], _params(0.3, None))),
        ("empty_choice_set", _inputs([], _params(0.3, None))),
        (
            "published_weekly_limitation",
            _inputs(
                [A, B],
                _params(0.3, None),
                snapshot=_snapshot(
                    mode="published_weekly",
                    family="weekly",
                    publication_status="published",
                    baseline_reconciled=None,
                ),
            ),
        ),
        (
            "baseline_unreconciled_limitation",
            _inputs(
                [A, B],
                _params(0.3, None),
                snapshot=_snapshot(
                    mode="historical_replay",
                    family="weekly",
                    publication_status="published",
                    baseline_reconciled=False,
                ),
            ),
        ),
        (
            "three_way_ranking_display_tie",
            _inputs(
                [
                    _alt("SYN-C", 12.0, 3.0, 9.0),
                    _alt("SYN-A", 12.0, 3.0, 9.5),
                    _alt("SYN-B", 15.0, 3.0, 9.0),
                ],
                _params(0.3, None),
            ),
        ),
    ]
    out = []
    for name, inputs in cases:
        result = policy.evaluate(inputs)
        out.append({"name": name, "inputs": inputs, "expected": result})
    return out


def metrics_cases() -> list[dict[str, Any]]:
    inputs = _inputs([A, B], _params(0.3, 5.0))
    result = policy.evaluate(inputs)
    full = {
        "schema_version": "outcome_snapshot/1.0",
        "snapshot_id": "syn-golden",
        "outcome_id": "1" * 64,
        "rows": [{"player_id": "SYN-A", "actual": 8.0}, {"player_id": "SYN-B", "actual": 16.0}],
    }
    partial = {
        **full,
        "outcome_id": "2" * 64,
        "rows": [{"player_id": "SYN-A", "actual": 8.0}, {"player_id": "SYN-B", "actual": None}],
    }
    zero_neg = {
        **full,
        "outcome_id": "3" * 64,
        "rows": [{"player_id": "SYN-A", "actual": 0.0}, {"player_id": "SYN-B", "actual": -1.2}],
    }
    extreme = {
        **full,
        "outcome_id": "4" * 64,
        "rows": [{"player_id": "SYN-A", "actual": 999.5}, {"player_id": "SYN-B", "actual": -50.0}],
    }
    recorded_a = {
        "state": "recorded",
        "action_id": "a" * 64,
        "at_utc": "2099-01-01T00:00:00+00:00",
        "chosen_player_id": "SYN-A",
        "kind": "hypothetical_replay",
        "note": None,
    }
    recorded_b = {**recorded_a, "chosen_player_id": "SYN-B"}
    none = {
        "state": "not_recorded",
        "action_id": None,
        "at_utc": None,
        "chosen_player_id": None,
        "kind": None,
        "note": None,
    }
    declined = {
        "state": "declined",
        "action_id": "b" * 64,
        "at_utc": "2099-01-01T00:00:00+00:00",
        "chosen_player_id": None,
        "kind": None,
        "note": "no choice",
    }
    cases = [
        ("chosen_a_full_outcomes", recorded_a, full),
        ("chosen_b_full_outcomes", recorded_b, full),
        ("no_action_full_outcomes", none, full),
        ("declined_full_outcomes", declined, full),
        ("chosen_a_partial_outcomes", recorded_a, partial),
        ("chosen_b_missing_own_outcome", recorded_b, partial),
        ("zero_and_negative_actuals", recorded_b, zero_neg),
        ("extreme_outcomes_do_not_touch_result", recorded_a, extreme),
    ]
    return [
        {
            "name": n,
            "result": result,
            "action": a,
            "outcome_snapshot": o,
            "expected": metrics.compute_metrics(result, a, o),
        }
        for n, a, o in cases
    ]


def receipt_lifecycle() -> dict[str, Any]:
    inputs = _inputs([A, B], _params(0.3, 5.0))
    r0 = receipts.new_receipt(
        inputs,
        created_at_utc="2099-01-01T00:00:00+00:00",
        case_id="syn-ambiguity-floor-relaxation",
        prediction_note="I expect A",
    )
    r1 = receipts.record_action(
        r0,
        chosen_player_id="SYN-A",
        kind="hypothetical_replay",
        at_utc="2099-01-01T00:01:00+00:00",
        note=None,
    )
    outcome_snapshot = {
        "schema_version": "outcome_snapshot/1.0",
        "snapshot_id": "syn-golden",
        "inputs_content_sha256": "0" * 64,
        "season": 2099,
        "week": 1,
        "model_version": SYN_MODEL,
        "scoring_format": "ppr",
        "observed_at_utc": None,
        "source": {"mart": None, "files": []},
        "coverage": {"scored": 2, "observed": 2},
        "rows": [
            {"player_id": "SYN-A", "actual": 8.0, "actual_source": "synthetic"},
            {"player_id": "SYN-B", "actual": 16.0, "actual_source": "synthetic"},
        ],
    }
    r2 = receipts.attach_outcome(r1, outcome_snapshot, at_utc="2099-01-01T00:02:00+00:00")
    r_declined = receipts.decline_action(r0, at_utc="2099-01-01T00:01:00+00:00", note="skip")
    return {
        "inputs": inputs,
        "outcome_snapshot": outcome_snapshot,
        "outcome_id": content_id(outcome_snapshot),
        "receipt_new": r0,
        "receipt_after_action": r1,
        "receipt_after_outcome": r2,
        "receipt_declined": r_declined,
        "invariants": {
            "decision_id_unchanged": r0["decision_id"] == r1["decision_id"] == r2["decision_id"],
            "result_sha_unchanged": r0["result_sha256"] == r2["result_sha256"],
            "event_ids": [e["event_id"] for e in r2["events"]],
        },
    }


def canonical_vectors() -> list[dict[str, Any]]:
    values: list[Any] = [
        {"b": [1, 2.5, None, True, False], "a": 'x"y\\z\n'},
        {"n": 14.0, "m": 13.6, "z": -0.0, "s": 0.1 + 0.2, "t": 1 / 128, "big": 123456789.123456789},
        [],
        {},
        "Ja'Marr Chase — été",
        {"nested": {"k": [{"y": 1}, {"x": 2}]}},
        0.000001,
        -13.6,
        1e-7,
    ]
    return [
        {"value": v, "canonical": canonical_json(v), "sha256": content_id(v)} for v in values
    ] + [
        {"text": "", "sha256": sha256_hex("")},
        {"text": "abc", "sha256": sha256_hex("abc")},
    ]


def build() -> dict[str, Any]:
    return {
        "golden_version": "1",
        "policy_version": POLICY_VERSION,
        "canonical": canonical_vectors(),
        "policy": policy_cases(),
        "metrics": metrics_cases(),
        "receipt": receipt_lifecycle(),
    }


def write(path: Path = GOLDEN_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def check(path: Path = GOLDEN_PATH) -> bool:
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8")) == build()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        print(f"wrote {write()}")
        return 0
    if args.check:
        ok = check()
        print(
            "golden fixtures are current"
            if ok
            else "golden fixtures are stale: run python -m ffai.decision_lab.golden --write"
        )
        return 0 if ok else 1
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
