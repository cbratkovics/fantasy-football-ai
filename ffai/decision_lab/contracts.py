"""JSON Schemas and semantic validation for every Decision Lab document (ADR-0032).

Structural checks use ``jsonschema`` (Draft 2020-12, ``additionalProperties: false`` throughout
so a tampered or misspelled field is rejected, not ignored). Semantic checks — grain uniqueness,
finite numbers, supported units, cross-file compatibility — live in the ``check_*`` functions
below because a schema cannot express them. Every failure raises :class:`ContractError` with a
message that names the document and the field; nothing is coerced or defaulted.

Required-but-nullable fields are deliberate: ``recommended_player_id`` must be *present* and
``null`` for HOLD / REVIEW, and an action block must be present with ``state: not_recorded`` and
null ids before anything is recorded.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import jsonschema

from ffai.decision_lab import POLICY_VERSION, SCHEMA_VERSIONS, spec
from ffai.decision_lab.canonical import content_id

_SPEC = spec()
SHA256 = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
NULLABLE_STR = {"type": ["string", "null"]}
NULLABLE_NUM = {"type": ["number", "null"]}
NULLABLE_INT = {"type": ["integer", "null"]}
PLAYER_ID = {"type": "string", "minLength": 1, "maxLength": 64}
POSITION = {"type": "string", "enum": list(_SPEC["slots"]["FLEX"]) + ["QB"]}
SLOT = {"type": "string", "enum": sorted(_SPEC["slots"])}
MODE = {"type": "string", "enum": _SPEC["modes"]}
SOURCE_FAMILY = {"type": "string", "enum": _SPEC["source_families"]}
AVAILABILITY = {"type": "string", "enum": _SPEC["availability"]["values"]}
AVAILABILITY_BASIS = {"type": "string", "enum": _SPEC["availability"]["bases"]}
BASELINE_PROVENANCE = {"type": "string", "enum": _SPEC["baseline_provenance"]}
STATUS = {"type": "string", "enum": _SPEC["statuses"]}
UTC = {"type": ["string", "null"], "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}"}


class ContractError(ValueError):
    """A document violates its schema or a semantic rule. Always fail closed."""


def _obj(props: dict[str, Any], required: Iterable[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": sorted(required) if required is not None else sorted(props),
        "additionalProperties": False,
    }


PERIOD = _obj({"season": {"type": "integer"}, "week": {"type": "integer", "minimum": 1}})

PUBLICATION = _obj(
    {
        "status": {"type": "string", "enum": ["recorded", "published", "synthetic", "blocked"]},
        "run_id": NULLABLE_STR,
        "action": NULLABLE_STR,
        "note": NULLABLE_STR,
    }
)

SNAPSHOT_REF = _obj(
    {
        "snapshot_id": {"type": "string", "minLength": 1},
        "mode": MODE,
        "source_family": SOURCE_FAMILY,
        "season": {"type": "integer"},
        "week": {"type": "integer", "minimum": 1},
        "model_version": {"type": "string"},
        "feature_version": NULLABLE_STR,
        "candidate_by_position": {"type": "object", "additionalProperties": {"type": "string"}},
        "scoring_format": {"type": "string"},
        "inputs_sha256": SHA256,
        "data_cutoff": {"oneOf": [PERIOD, {"type": "null"}]},
        "generated_at_utc": UTC,
        "publication": PUBLICATION,
        "evidence_status": {"type": "string", "enum": ["verified", "unverified", "corrupt"]},
        "evidence_detail": NULLABLE_STR,
        "baseline_reconciled": {"type": ["boolean", "null"]},
    }
)

ALTERNATIVE = _obj(
    {
        "player_id": PLAYER_ID,
        "display_name": NULLABLE_STR,
        "team": NULLABLE_STR,
        "position": {"type": "string"},
        "model_version": {"type": "string"},
        "candidate": {"type": "string"},
        "season": {"type": "integer"},
        "week": {"type": "integer"},
        "projection": NULLABLE_NUM,
        "floor": NULLABLE_NUM,
        "ceiling": NULLABLE_NUM,
        "baseline": NULLABLE_NUM,
        "baseline_provenance": BASELINE_PROVENANCE,
        "availability": AVAILABILITY,
        "availability_basis": AVAILABILITY_BASIS,
        "source_row_ref": NULLABLE_STR,
    }
)

PARAMETERS = _obj(
    {
        "min_projection_gap": {"type": "number"},
        "min_floor": NULLABLE_NUM,
        "require_independent_baseline": {"type": "boolean"},
        "model_only_exploration": {"type": "boolean"},
    }
)

DECISION_INPUTS = _obj(
    {
        "schema_version": {"const": SCHEMA_VERSIONS["decision_inputs"]},
        "policy_version": {"const": POLICY_VERSION},
        "snapshot": SNAPSHOT_REF,
        "slot": {"type": "string"},
        "alternatives": {"type": "array", "items": ALTERNATIVE, "maxItems": 64},
        "parameters": PARAMETERS,
    }
)

REASON = _obj(
    {
        "code": {"type": "string", "enum": sorted(_SPEC["reasons"])},
        "severity": {"type": "string", "enum": ["hold", "review", "exclusion", "info"]},
        "detail": {"type": "object"},
    }
)

RANKING_ROW = _obj(
    {
        "player_id": PLAYER_ID,
        "display_name": NULLABLE_STR,
        "position": NULLABLE_STR,
        "projection": NULLABLE_NUM,
        "floor": NULLABLE_NUM,
        "ceiling": NULLABLE_NUM,
        "baseline": NULLABLE_NUM,
        "baseline_provenance": BASELINE_PROVENANCE,
        "availability": NULLABLE_STR,
        "availability_basis": NULLABLE_STR,
        "comparable": {"type": "boolean"},
        "exclusion": NULLABLE_STR,
        "rank": NULLABLE_INT,
        "tied_with_next": {"type": "boolean"},
    }
)

DECISION_RESULT = _obj(
    {
        "schema_version": {"const": SCHEMA_VERSIONS["decision_result"]},
        "policy_version": {"const": POLICY_VERSION},
        "decision_id": SHA256,
        "status": STATUS,
        "recommended_player_id": {"type": ["string", "null"]},
        "reasons": {"type": "array", "items": REASON},
        "ranking": {"type": "array", "items": RANKING_ROW},
        "leading": _obj({"first": NULLABLE_STR, "second": NULLABLE_STR, "gap": NULLABLE_NUM}),
        "gates": _obj(
            {
                "min_projection_gap": NULLABLE_NUM,
                "gap_ok": {"type": ["boolean", "null"]},
                "min_floor": NULLABLE_NUM,
                "floor_checked": {"type": "boolean"},
                "leading_floor": NULLABLE_NUM,
                "floor_ok": {"type": ["boolean", "null"]},
            }
        ),
        "baseline": _obj(
            {
                "status": {"type": "string", "enum": ["available", "unavailable", "tie"]},
                "preferred_player_id": NULLABLE_STR,
                "preferred_value": NULLABLE_NUM,
                "agrees_with_model_leader": {"type": ["boolean", "null"]},
                "missing_for": {"type": "array", "items": {"type": "string"}},
                "provenance": {"type": "object", "additionalProperties": BASELINE_PROVENANCE},
            }
        ),
        "counts": _obj(
            {
                "nominated": {"type": "integer"},
                "excluded": {"type": "integer"},
                "comparable": {"type": "integer"},
            }
        ),
        "limitations": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(_SPEC["limitations"])},
        },
        "explanation": {"type": "array", "items": {"type": "string"}},
    }
)

SNAPSHOT_ROW = _obj(
    {
        "player_id": PLAYER_ID,
        "display_name": NULLABLE_STR,
        "team": NULLABLE_STR,
        "position": {"type": "string"},
        "model_version": {"type": "string"},
        "candidate": {"type": "string"},
        "projection": NULLABLE_NUM,
        "floor": NULLABLE_NUM,
        "ceiling": NULLABLE_NUM,
        "baseline": NULLABLE_NUM,
        "baseline_provenance": BASELINE_PROVENANCE,
        "availability_default": AVAILABILITY,
        "display_source": {
            "type": ["string", "null"],
            "enum": ["prediction_source", "dim_player_current", "synthetic", None],
        },
        "source_row_ref": NULLABLE_STR,
    }
)

SOURCE_FILE = _obj(
    {
        "role": {"type": "string"},
        "path": {"type": "string"},
        "sha256": {"type": ["string", "null"]},
        "rows": NULLABLE_INT,
    },
    required=("role", "path", "sha256"),
)

POPULATION = _obj(
    {
        "conditioning": {
            "type": "string",
            "enum": ["realized_stats_rows", "scored_eligible_players", "synthetic"],
        },
        "description": {"type": "string"},
        "n_rows": {"type": "integer"},
        "exclusions": {"type": "array", "items": {"type": "string"}},
    }
)

BASELINE_META = _obj(
    {
        "name": {"type": "string"},
        "provenance_basis": {
            "type": "string",
            "enum": [
                "mart_column",
                "population_rule_and_reconciliation",
                "population_rule_unreconciled",
                "synthetic",
                "unknown",
            ],
        },
        "reconciled": {"type": ["boolean", "null"]},
        "reconciled_to": NULLABLE_STR,
        "note": NULLABLE_STR,
    }
)

MART_EXPORT = _obj(
    {
        "exported_at_utc": NULLABLE_STR,
        "target": NULLABLE_STR,
        "code_commit": NULLABLE_STR,
        "invocation_id": NULLABLE_STR,
    }
)

INPUTS_SNAPSHOT = _obj(
    {
        "schema_version": {"const": SCHEMA_VERSIONS["inputs_snapshot"]},
        "snapshot_id": {"type": "string", "minLength": 1},
        "mode": MODE,
        "source_family": SOURCE_FAMILY,
        "season": {"type": "integer"},
        "week": {"type": "integer", "minimum": 1},
        "model_version": {"type": "string"},
        "feature_version": NULLABLE_STR,
        "candidate_by_position": {"type": "object", "additionalProperties": {"type": "string"}},
        "scoring_format": {"type": "string"},
        "data_cutoff": {"oneOf": [PERIOD, {"type": "null"}]},
        "generated_at_utc": UTC,
        "publication": PUBLICATION,
        "population": POPULATION,
        "baseline": BASELINE_META,
        "champion_at_source": {
            "oneOf": [
                _obj(
                    {
                        "model_version": {"type": "string"},
                        "candidate_by_position": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                        "basis": {
                            "type": "string",
                            "enum": ["eval_artifact", "predictions_file", "synthetic"],
                        },
                    }
                ),
                {"type": "null"},
            ]
        },
        "source": _obj(
            {
                "mart": NULLABLE_STR,
                "files": {"type": "array", "items": SOURCE_FILE},
                "mart_export": {"oneOf": [MART_EXPORT, {"type": "null"}]},
            }
        ),
        "rows": {"type": "array", "items": SNAPSHOT_ROW},
    }
)

OUTCOME_ROW = _obj(
    {
        "player_id": PLAYER_ID,
        "actual": NULLABLE_NUM,
        "actual_source": {
            "type": ["string", "null"],
            "enum": ["stats", "artifact", "synthetic", None],
        },
    }
)

OUTCOME_SNAPSHOT = _obj(
    {
        "schema_version": {"const": SCHEMA_VERSIONS["outcome_snapshot"]},
        "snapshot_id": {"type": "string", "minLength": 1},
        "inputs_content_sha256": SHA256,
        "season": {"type": "integer"},
        "week": {"type": "integer", "minimum": 1},
        "model_version": {"type": "string"},
        "scoring_format": {"type": "string"},
        "observed_at_utc": UTC,
        "source": _obj({"mart": NULLABLE_STR, "files": {"type": "array", "items": SOURCE_FILE}}),
        "coverage": _obj({"scored": {"type": "integer"}, "observed": {"type": "integer"}}),
        "rows": {"type": "array", "items": OUTCOME_ROW},
    }
)

EVENT = _obj(
    {
        "event_id": SHA256,
        "seq": {"type": "integer", "minimum": 1},
        "event_type": {"type": "string", "enum": _SPEC["event_types"]},
        "at_utc": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}"},
        "payload": {"type": "object"},
    }
)

ACTION = _obj(
    {
        "state": {"type": "string", "enum": ["not_recorded", "recorded", "declined"]},
        "action_id": {"oneOf": [SHA256, {"type": "null"}]},
        "at_utc": UTC,
        "chosen_player_id": NULLABLE_STR,
        "kind": {"type": ["string", "null"], "enum": _SPEC["action_kinds"] + [None]},
        "note": NULLABLE_STR,
    }
)

OUTCOME_STATE = _obj(
    {
        "state": {"type": "string", "enum": ["not_attached", "attached"]},
        "outcome_snapshot_id": NULLABLE_STR,
        "outcome_id": {"oneOf": [SHA256, {"type": "null"}]},
        "attached_at_utc": UTC,
        "metrics": {"type": ["object", "null"]},
    }
)

RECEIPT = _obj(
    {
        "schema_version": {"const": SCHEMA_VERSIONS["receipt"]},
        "decision_id": SHA256,
        "parent_decision_id": {"oneOf": [SHA256, {"type": "null"}]},
        "created_at_utc": {
            "type": "string",
            "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}",
        },
        "case_id": NULLABLE_STR,
        "prediction_note": NULLABLE_STR,
        "inputs": DECISION_INPUTS,
        "result": DECISION_RESULT,
        "result_sha256": SHA256,
        "events": {"type": "array", "items": EVENT, "maxItems": 256},
        "action": ACTION,
        "outcome": OUTCOME_STATE,
        "trust": _obj({"note": {"type": "string"}}),
    }
)

EXPERIMENT = _obj(
    {
        "label": {"type": "string"},
        "parameters": {"type": "object"},
        "overrides": {"type": "object"},
        "expected": _obj({"status": STATUS, "recommended_player_id": NULLABLE_STR}),
    },
    required=("label", "expected"),
)

CASE = _obj(
    {
        "case_id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]{2,80}$"},
        "title": {"type": "string"},
        "mode": MODE,
        "snapshot_id": {"type": "string"},
        "slot": {"type": "string"},
        "alternatives": {"type": "array", "items": PLAYER_ID, "minItems": 1, "maxItems": 12},
        "overrides": {
            "type": "object",
            "additionalProperties": _obj(
                {"availability": AVAILABILITY, "availability_basis": AVAILABILITY_BASIS}
            ),
        },
        "parameters": PARAMETERS,
        "setup": {"type": "string"},
        "experiments": {"type": "array", "items": EXPERIMENT},
        "expected": _obj(
            {
                "status": STATUS,
                "recommended_player_id": NULLABLE_STR,
                "reason_codes": {"type": "array", "items": {"type": "string"}},
            }
        ),
        "outcomes_available": {"type": "boolean"},
        "outcome_snapshot_id": NULLABLE_STR,
        "limitations": {"type": "array", "items": {"type": "string"}},
        "selection": {
            "oneOf": [
                _obj(
                    {
                        "method": {
                            "type": "string",
                            "enum": ["curated_synthetic", "deterministic_discovery"],
                        },
                        "criteria": {"type": "string"},
                        "sealed_pattern": NULLABLE_STR,
                    }
                ),
                {"type": "null"},
            ]
        },
        "source_row_refs": {"type": "array", "items": {"type": "string"}},
    }
)

CASES = _obj(
    {
        "schema_version": {"const": SCHEMA_VERSIONS["cases"]},
        "policy_version": {"const": POLICY_VERSION},
        "cases": {"type": "array", "items": CASE},
        "notes": {"type": "array", "items": {"type": "string"}},
    }
)

FILE_REF = _obj(
    {
        "path": {"type": "string"},
        "file_sha256": SHA256,
        "content_sha256": SHA256,
        "n_rows": {"type": "integer"},
    }
)

OUTCOME_FILE_REF = _obj(
    {
        "path": {"type": "string"},
        "file_sha256": SHA256,
        "content_sha256": SHA256,
        "n_rows": {"type": "integer"},
        "n_observed": {"type": "integer"},
    }
)

RECONCILIATION = _obj(
    {
        "status": {"type": "string", "enum": ["matched", "mismatch", "unavailable"]},
        "reference": NULLABLE_STR,
        "expected": {"type": ["object", "null"]},
        "observed": {"type": ["object", "null"]},
        "tolerance": NULLABLE_NUM,
    }
)

MANIFEST_SNAPSHOT = _obj(
    {
        "snapshot_id": {"type": "string"},
        "mode": MODE,
        "source_family": SOURCE_FAMILY,
        "season": {"type": "integer"},
        "week": {"type": "integer"},
        "model_version": {"type": "string"},
        "candidate_by_position": {"type": "object", "additionalProperties": {"type": "string"}},
        "inputs": FILE_REF,
        "outcomes": {"oneOf": [OUTCOME_FILE_REF, {"type": "null"}]},
        "population": POPULATION,
        "baseline": BASELINE_META,
        "reconciliation": {"oneOf": [RECONCILIATION, {"type": "null"}]},
    }
)

MANIFEST = _obj(
    {
        "schema_version": {"const": SCHEMA_VERSIONS["manifest"]},
        "exporter_version": {"type": "string"},
        "policy_version": {"const": POLICY_VERSION},
        "schema_versions": {"type": "object", "additionalProperties": {"type": "string"}},
        "code_revision": _obj({"produced_at": NULLABLE_STR, "note": {"type": "string"}}),
        "lineage": {"type": "object"},
        "sources": {"type": "array", "items": SOURCE_FILE},
        "snapshots": {"type": "array", "items": MANIFEST_SNAPSHOT},
        "cases": FILE_REF.copy()
        | {"required": ["path", "file_sha256", "content_sha256", "n_rows"]},
        "latest_weekly_snapshot_id": NULLABLE_STR,
        "policy_spec": _obj({"path": {"type": "string"}, "sha256": SHA256}),
        "digest_coverage": {"type": "object", "additionalProperties": {"type": "string"}},
    }
)

SCHEMAS: dict[str, dict[str, Any]] = {
    "decision_inputs": DECISION_INPUTS,
    "decision_result": DECISION_RESULT,
    "inputs_snapshot": INPUTS_SNAPSHOT,
    "outcome_snapshot": OUTCOME_SNAPSHOT,
    "receipt": RECEIPT,
    "cases": CASES,
    "manifest": MANIFEST,
}


def validate(name: str, document: Any) -> None:
    """Structural validation against one of :data:`SCHEMAS`; raises :class:`ContractError`."""
    schema = SCHEMAS[name]
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        where = "/".join(str(p) for p in first.absolute_path) or "<root>"
        raise ContractError(f"{name}: {where}: {first.message}")


def _finite_or_null(value: Any, where: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ContractError(f"{where}: expected a finite number or null, got {value!r}")


def _unique(ids: list[str], where: str) -> None:
    seen: set[str] = set()
    for pid in ids:
        if pid in seen:
            raise ContractError(f"{where}: duplicate player_id {pid}")
        seen.add(pid)


def check_inputs_snapshot(doc: dict[str, Any]) -> None:
    validate("inputs_snapshot", doc)
    if doc["scoring_format"] != _SPEC["scoring_format"]:
        raise ContractError(f"inputs_snapshot {doc['snapshot_id']}: scoring_format must be ppr")
    _unique([r["player_id"] for r in doc["rows"]], f"inputs_snapshot {doc['snapshot_id']}")
    if doc["population"]["n_rows"] != len(doc["rows"]):
        raise ContractError(f"inputs_snapshot {doc['snapshot_id']}: population.n_rows != len(rows)")
    for r in doc["rows"]:
        where = f"inputs_snapshot {doc['snapshot_id']} row {r['player_id']}"
        for k in ("projection", "floor", "ceiling", "baseline"):
            _finite_or_null(r.get(k), f"{where}.{k}")
        if r["model_version"] != doc["model_version"]:
            raise ContractError(f"{where}: model_version differs from the snapshot")
        expected_cand = doc["candidate_by_position"].get(r["position"])
        if expected_cand is not None and r["candidate"] != expected_cand:
            raise ContractError(f"{where}: candidate {r['candidate']} != {expected_cand}")
        if r["position"] not in _SPEC["slots"]:
            raise ContractError(f"{where}: unsupported position {r['position']}")
        if (
            r["floor"] is not None
            and r["projection"] is not None
            and r["floor"] > r["projection"] + 1e-9
        ):
            raise ContractError(f"{where}: floor above projection")
        if (
            r["ceiling"] is not None
            and r["projection"] is not None
            and r["ceiling"] < r["projection"] - 1e-9
        ):
            raise ContractError(f"{where}: ceiling below projection")
    forbidden = {
        "actual",
        "abs_error",
        "regret",
        "hit",
        "downside",
        "prediction_rank",
        "best_eligible_points",
    }
    for r in doc["rows"]:
        leak = forbidden & set(r)
        if leak:
            raise ContractError(
                f"inputs_snapshot {doc['snapshot_id']}: outcome field(s) in inputs: {sorted(leak)}"
            )


def check_outcome_snapshot(doc: dict[str, Any], inputs: dict[str, Any] | None = None) -> None:
    validate("outcome_snapshot", doc)
    if doc["scoring_format"] != _SPEC["scoring_format"]:
        raise ContractError(f"outcome_snapshot {doc['snapshot_id']}: scoring_format must be ppr")
    _unique([r["player_id"] for r in doc["rows"]], f"outcome_snapshot {doc['snapshot_id']}")
    observed = 0
    for r in doc["rows"]:
        _finite_or_null(
            r.get("actual"), f"outcome_snapshot {doc['snapshot_id']} row {r['player_id']}.actual"
        )
        if r["actual"] is not None:
            observed += 1
            if r.get("actual_source") is None:
                raise ContractError(
                    f"outcome_snapshot {doc['snapshot_id']} row {r['player_id']}: observed actual without a source"
                )
    if doc["coverage"]["observed"] != observed or doc["coverage"]["scored"] != len(doc["rows"]):
        raise ContractError(
            f"outcome_snapshot {doc['snapshot_id']}: coverage counts disagree with rows"
        )
    if inputs is not None:
        if doc["snapshot_id"] != inputs["snapshot_id"]:
            raise ContractError("outcome_snapshot pairs with a different snapshot_id")
        if doc["inputs_content_sha256"] != content_id(inputs):
            raise ContractError(
                f"outcome_snapshot {doc['snapshot_id']}: inputs_content_sha256 does not match the inputs snapshot"
            )
        if (doc["season"], doc["week"], doc["model_version"]) != (
            inputs["season"],
            inputs["week"],
            inputs["model_version"],
        ):
            raise ContractError(
                f"outcome_snapshot {doc['snapshot_id']}: period or model differs from inputs"
            )
        input_ids = {r["player_id"] for r in inputs["rows"]}
        extra = {r["player_id"] for r in doc["rows"]} - input_ids
        if extra:
            raise ContractError(
                f"outcome_snapshot {doc['snapshot_id']}: players not in the inputs snapshot: {sorted(extra)[:5]}"
            )


def check_decision_inputs(doc: dict[str, Any]) -> None:
    validate("decision_inputs", doc)
    for a in doc["alternatives"]:
        for k in ("projection", "floor", "ceiling", "baseline"):
            _finite_or_null(a.get(k), f"decision_inputs alternative {a['player_id']}.{k}")
    for k in ("min_projection_gap", "min_floor"):
        _finite_or_null(doc["parameters"].get(k), f"decision_inputs parameters.{k}")


def check_decision_result(doc: dict[str, Any]) -> None:
    validate("decision_result", doc)
    if doc["status"] == "recommend" and not doc["recommended_player_id"]:
        raise ContractError("decision_result: recommend without recommended_player_id")
    if doc["status"] != "recommend" and doc["recommended_player_id"] is not None:
        raise ContractError(f"decision_result: {doc['status']} carries a recommended_player_id")
    ids = {r["player_id"] for r in doc["ranking"]}
    if doc["recommended_player_id"] is not None and doc["recommended_player_id"] not in ids:
        raise ContractError("decision_result: recommended_player_id not among the alternatives")
    for k in ("first", "second"):
        v = doc["leading"][k]
        if v is not None and v not in ids:
            raise ContractError(f"decision_result: leading.{k} not among the alternatives")
    bp = doc["baseline"]["preferred_player_id"]
    if bp is not None and bp not in ids:
        raise ContractError(
            "decision_result: baseline.preferred_player_id not among the alternatives"
        )


def check_cases(doc: dict[str, Any]) -> None:
    validate("cases", doc)
    seen: set[str] = set()
    for c in doc["cases"]:
        if c["case_id"] in seen:
            raise ContractError(f"cases: duplicate case_id {c['case_id']}")
        seen.add(c["case_id"])
        _unique(c["alternatives"], f"case {c['case_id']}")
        for pid in c["overrides"]:
            if pid not in c["alternatives"]:
                raise ContractError(
                    f"case {c['case_id']}: override for a player not nominated: {pid}"
                )
        for k in ("min_projection_gap", "min_floor"):
            _finite_or_null(c["parameters"].get(k), f"case {c['case_id']} parameters.{k}")


def check_manifest(doc: dict[str, Any]) -> None:
    validate("manifest", doc)
    ids = [s["snapshot_id"] for s in doc["snapshots"]]
    if len(set(ids)) != len(ids):
        raise ContractError("manifest: duplicate snapshot_id")
    if doc["latest_weekly_snapshot_id"] is not None and doc["latest_weekly_snapshot_id"] not in ids:
        raise ContractError("manifest: latest_weekly_snapshot_id is not a listed snapshot")
    if doc["schema_versions"] != SCHEMA_VERSIONS:
        raise ContractError("manifest: schema_versions differ from this code's policy_spec")
