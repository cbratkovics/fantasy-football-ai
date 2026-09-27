"""The reference decision policy (``policy_spec.json`` ``policy_version`` 1.0.0).

``evaluate(inputs)`` maps one :data:`decision_inputs/1.0` document to one
:data:`decision_result/1.0` document. It is a pure function of its argument: no clock, no file,
no randomness, and it accepts *only* the input schema — target-week actuals, error metrics,
hindsight ranks and regret cannot reach it because the input schema has no field for them.

Precedence (each step keeps computing so the result carries the full evidence; the status is the
most severe reason found):

1. **HOLD** — invalid or incompatible evidence: corrupt or blocked snapshot, unsupported scoring
   format or slot, alternatives from mixed periods or model versions, invalid parameters, more
   than the maximum number of alternatives, duplicates. Invalid input never becomes a weaker
   recommendation.
2. **Exclusions** — alternatives marked unavailable by the user, or whose position cannot fill
   the slot, are excluded with a visible reason. Then **REVIEW** when a remaining alternative has
   no finite projection, when availability is unresolved (``unknown``) for a remaining
   alternative, when fewer than two comparable alternatives remain, or when an independent
   baseline is required and missing (unless model-only exploration is on, which records a
   limitation instead).
3. **Ranking** — comparable alternatives ordered by projection descending; ties broken by
   player id *for display only* and flagged.
4. **Gap** — ``gap = q(first) - q(second)`` before any floor gate; ``|gap| <= tolerance`` is a
   tie and produces REVIEW even when ``min_projection_gap`` is zero.
5. ``gap < min_projection_gap - tolerance`` produces REVIEW; equality qualifies.
6. **Floor guardrail** (only when ``min_floor`` is not null): a missing floor on the leading
   option, or ``floor < min_floor - tolerance``, produces REVIEW; equality qualifies. The policy
   never nominates a lower-ranked player instead — that would be a different objective.
7. Otherwise **RECOMMEND** the leading option, conditional on the recorded assumptions.

The baseline (causal trailing mean) is a separately labelled comparator: its preferred player
is reported next to the model's, never substituted for it.
"""

from __future__ import annotations

from typing import Any

from ffai.decision_lab import POLICY_VERSION, SCHEMA_VERSIONS, spec
from ffai.decision_lab.canonical import CanonicalError, content_id, format_number, q

_SPEC = spec()
TOL: float = float(_SPEC["tolerance"])
SLOTS: dict[str, list[str]] = _SPEC["slots"]
CHOICE_MIN: int = int(_SPEC["choice_set"]["min"])
CHOICE_MAX: int = int(_SPEC["choice_set"]["max"])
COMPARABLE_AVAILABILITY = set(_SPEC["availability"]["comparable"])
INDEPENDENT_BASELINE = set(_SPEC["independent_baseline_provenance"])
REASONS: dict[str, dict[str, str]] = _SPEC["reasons"]
LIMITATIONS: dict[str, str] = _SPEC["limitations"]
PARAM_SPEC = _SPEC["parameters"]

INPUTS_SCHEMA = SCHEMA_VERSIONS["decision_inputs"]
RESULT_SCHEMA = SCHEMA_VERSIONS["decision_result"]


def default_parameters() -> dict[str, Any]:
    return {k: v["default"] for k, v in PARAM_SPEC.items()}


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float):
        return format_number(value)
    if isinstance(value, list | tuple):
        return ", ".join(_fmt(v) for v in value)
    if value is None:
        return "—"
    return str(value)


def render_reason(code: str, detail: dict[str, Any]) -> str:
    """Deterministic explanation text from a reason code and its measured detail values."""
    template = REASONS[code]["template"]
    text = template
    for key, value in detail.items():
        text = text.replace("{" + key + "}", _fmt(value))
    return text


def _reason(code: str, **detail: Any) -> dict[str, Any]:
    return {"code": code, "severity": REASONS[code]["severity"], "detail": detail}


def _is_num(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def _finite(x: Any) -> float | None:
    if not _is_num(x):
        return None
    try:
        return q(float(x))
    except CanonicalError:
        return None


def _label(alt: dict[str, Any]) -> str:
    name = alt.get("display_name")
    return f"{name} ({alt['player_id']})" if name else str(alt["player_id"])


def decision_id(inputs: dict[str, Any]) -> str:
    """The decision identity: sha256 of the canonical inputs (alternatives sorted by player id).

    Covers schema and policy versions, the snapshot reference (including the inputs snapshot's
    content digest), slot, every alternative row with its availability fields, and the
    parameters. It does not cover outcomes, events, timestamps, or the computed result.
    """
    return content_id(normalize_inputs(inputs))


def normalize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Alternatives sorted by ``player_id`` and versions pinned; the form that is hashed."""
    alts = sorted(
        (dict(a) for a in inputs.get("alternatives", [])),
        key=lambda a: str(a.get("player_id")),
    )
    out = dict(inputs)
    out["schema_version"] = INPUTS_SCHEMA
    out["policy_version"] = POLICY_VERSION
    out["alternatives"] = alts
    return out


def _validate_parameters(params: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    gap = params.get("min_projection_gap")
    if _finite(gap) is None:
        problems.append("min_projection_gap must be a finite number")
    else:
        lo, hi = PARAM_SPEC["min_projection_gap"]["min"], PARAM_SPEC["min_projection_gap"]["max"]
        if not lo <= float(gap) <= hi:
            problems.append(f"min_projection_gap must be between {lo} and {hi}")
    floor = params.get("min_floor")
    if floor is not None:
        if _finite(floor) is None:
            problems.append("min_floor must be null or a finite number")
        else:
            lo, hi = PARAM_SPEC["min_floor"]["min"], PARAM_SPEC["min_floor"]["max"]
            if not lo <= float(floor) <= hi:
                problems.append(f"min_floor must be between {lo} and {hi}")
    for flag in ("require_independent_baseline", "model_only_exploration"):
        if not isinstance(params.get(flag), bool):
            problems.append(f"{flag} must be a boolean")
    unknown = sorted(set(params) - set(PARAM_SPEC))
    if unknown:
        problems.append(f"unknown parameter(s): {', '.join(unknown)}")
    return problems


def evaluate(inputs: dict[str, Any]) -> dict[str, Any]:
    """Apply the policy to one ``decision_inputs/1.0`` document.

    Pure. Bad domain values (unsupported slot, wrong scoring, out-of-range or wrongly typed
    parameters) become HOLD reasons. The two things that do raise are a non-dict argument
    (``TypeError``) and a non-finite number anywhere in the inputs (``CanonicalError``): JSON
    cannot carry NaN/Infinity, ``contracts.check_decision_inputs`` rejects them at the boundary,
    and the policy will not invent an identity for an input it cannot serialize.
    """
    if not isinstance(inputs, dict):
        raise TypeError("inputs must be a dict")
    norm = normalize_inputs(inputs)
    snapshot: dict[str, Any] = dict(norm.get("snapshot") or {})
    slot = norm.get("slot")
    alts: list[dict[str, Any]] = list(norm["alternatives"])
    params: dict[str, Any] = dict(norm.get("parameters") or {})
    reasons: list[dict[str, Any]] = []
    limitations: list[str] = []

    # --- 1. evidence validity → HOLD -----------------------------------------------------
    if snapshot.get("evidence_status") == "corrupt":
        reasons.append(_reason("evidence_corrupt", detail=snapshot.get("evidence_detail") or "?"))
    pub = snapshot.get("publication") or {}
    if pub.get("status") == "blocked":
        reasons.append(_reason("evidence_blocked", detail=pub.get("note") or "blocked"))
    fmt = snapshot.get("scoring_format")
    if fmt != _SPEC["scoring_format"]:
        reasons.append(_reason("scoring_format_unsupported", value=fmt))
    if slot not in SLOTS:
        reasons.append(_reason("slot_unsupported", value=slot))
    periods = sorted({f"{a.get('season')}-w{a.get('week')}" for a in alts if "season" in a})
    snap_period = f"{snapshot.get('season')}-w{snapshot.get('week')}"
    if periods and (len(periods) > 1 or periods != [snap_period]):
        reasons.append(_reason("mixed_periods", detail=periods + [f"snapshot {snap_period}"]))
    versions = sorted({f"{a.get('model_version')}/{a.get('candidate')}" for a in alts})
    if versions:
        expected_versions = {
            f"{snapshot.get('model_version')}/{c}"
            for c in (snapshot.get("candidate_by_position") or {}).values()
        }
        if len({v.split("/")[0] for v in versions}) > 1 or (
            expected_versions and not set(versions) <= expected_versions
        ):
            reasons.append(_reason("mixed_model_versions", detail=versions))
    param_problems = _validate_parameters(params)
    if param_problems:
        reasons.append(_reason("parameters_invalid", detail="; ".join(param_problems)))
    if len(alts) > CHOICE_MAX:
        reasons.append(_reason("choice_set_too_large", count=len(alts), limit=CHOICE_MAX))
    seen: set[str] = set()
    for a in alts:
        pid = str(a.get("player_id"))
        if pid in seen:
            reasons.append(_reason("duplicate_alternative", player_id=pid))
        seen.add(pid)

    # --- 2. exclusions and choice-set checks → REVIEW -------------------------------------
    eligible_positions = set(SLOTS.get(slot, []))
    ranking_rows: list[dict[str, Any]] = []
    comparable: list[dict[str, Any]] = []
    unresolved: list[str] = []
    missing_projection: list[str] = []
    for a in alts:
        row = {
            "player_id": a["player_id"],
            "display_name": a.get("display_name"),
            "position": a.get("position"),
            "projection": _finite(a.get("projection")),
            "floor": _finite(a.get("floor")),
            "ceiling": _finite(a.get("ceiling")),
            "baseline": _finite(a.get("baseline")),
            "baseline_provenance": a.get("baseline_provenance") or "unknown",
            "availability": a.get("availability"),
            "availability_basis": a.get("availability_basis"),
            "comparable": False,
            "exclusion": None,
            "rank": None,
            "tied_with_next": False,
        }
        if a.get("availability") == "unavailable":
            row["exclusion"] = "excluded_unavailable"
            reasons.append(_reason("excluded_unavailable", player=_label(a)))
        elif a.get("position") not in eligible_positions:
            row["exclusion"] = "excluded_slot_ineligible"
            reasons.append(
                _reason(
                    "excluded_slot_ineligible",
                    player=_label(a),
                    position=a.get("position"),
                    slot=slot,
                )
            )
        else:
            if row["projection"] is None:
                missing_projection.append(_label(a))
            if a.get("availability") not in COMPARABLE_AVAILABILITY:
                unresolved.append(_label(a))
            if row["projection"] is not None and a.get("availability") in COMPARABLE_AVAILABILITY:
                row["comparable"] = True
                comparable.append(row)
            if a.get("availability") == "assumed_available":
                if "availability_user_assumed" not in limitations:
                    limitations.append("availability_user_assumed")
            if a.get("availability") == "realized_stats_row":
                if "population_hindsight_conditioned" not in limitations:
                    limitations.append("population_hindsight_conditioned")
        ranking_rows.append(row)
    if missing_projection:
        reasons.append(_reason("projection_missing", players=missing_projection))
    if unresolved:
        reasons.append(_reason("availability_unresolved", players=unresolved))
    if len(comparable) < CHOICE_MIN:
        reasons.append(
            _reason("insufficient_choice_set", count=len(comparable), minimum=CHOICE_MIN)
        )

    # baseline availability (independent comparator) over the comparable set
    no_baseline = [
        r
        for r in comparable
        if r["baseline"] is None or r["baseline_provenance"] not in INDEPENDENT_BASELINE
    ]
    require_baseline = params.get("require_independent_baseline") is True
    model_only = params.get("model_only_exploration") is True
    if no_baseline and require_baseline and not model_only:
        reasons.append(
            _reason(
                "baseline_unavailable",
                players=[_label(r) for r in no_baseline],
            )
        )
    if no_baseline and model_only:
        reasons.append(
            _reason("baseline_not_independent", players=[_label(r) for r in no_baseline])
        )
        limitations.append("baseline_not_independent")

    # --- 3. ranking (display order by player id on ties) ----------------------------------
    ordered = sorted(comparable, key=lambda r: (-r["projection"], str(r["player_id"])))
    for i, r in enumerate(ordered):
        r["rank"] = i + 1
        if i + 1 < len(ordered) and abs(r["projection"] - ordered[i + 1]["projection"]) <= TOL:
            r["tied_with_next"] = True

    # --- 4./5. gap before the floor gate ---------------------------------------------------
    leading = {"first": None, "second": None, "gap": None}
    gates: dict[str, Any] = {
        "min_projection_gap": _finite(params.get("min_projection_gap")),
        "gap_ok": None,
        "min_floor": (
            _finite(params.get("min_floor")) if params.get("min_floor") is not None else None
        ),
        "floor_checked": False,
        "leading_floor": None,
        "floor_ok": None,
    }
    if len(ordered) >= 2 and not param_problems:
        first, second = ordered[0], ordered[1]
        gap = q(first["projection"] - second["projection"])
        leading = {"first": first["player_id"], "second": second["player_id"], "gap": gap}
        min_gap = gates["min_projection_gap"]
        if abs(gap) <= TOL:
            gates["gap_ok"] = False
            reasons.append(_reason("projection_tie", value=first["projection"]))
        elif gap < min_gap - TOL:
            gates["gap_ok"] = False
            reasons.append(
                _reason(
                    "gap_below_minimum",
                    gap=gap,
                    threshold=min_gap,
                    first=_label(first),
                    second=_label(second),
                )
            )
        else:
            gates["gap_ok"] = True
            reasons.append(_reason("gap_meets_minimum", gap=gap, threshold=min_gap))
        # --- 6. floor guardrail ---------------------------------------------------------
        min_floor = gates["min_floor"]
        gates["leading_floor"] = first["floor"]
        if min_floor is None:
            reasons.append(_reason("floor_guardrail_disabled"))
        else:
            gates["floor_checked"] = True
            if first["floor"] is None:
                gates["floor_ok"] = False
                reasons.append(_reason("floor_missing", player=_label(first), threshold=min_floor))
            elif first["floor"] < min_floor - TOL:
                gates["floor_ok"] = False
                reasons.append(
                    _reason(
                        "floor_below_minimum",
                        player=_label(first),
                        floor=first["floor"],
                        threshold=min_floor,
                    )
                )
            else:
                gates["floor_ok"] = True
                reasons.append(
                    _reason("floor_meets_minimum", floor=first["floor"], threshold=min_floor)
                )

    # --- baseline preference (separate comparator) ----------------------------------------
    baseline_block: dict[str, Any] = {
        "status": "unavailable",
        "preferred_player_id": None,
        "preferred_value": None,
        "agrees_with_model_leader": None,
        "missing_for": [r["player_id"] for r in no_baseline],
        "provenance": {r["player_id"]: r["baseline_provenance"] for r in comparable},
    }
    if comparable and not no_baseline:
        by_base = sorted(comparable, key=lambda r: (-r["baseline"], str(r["player_id"])))
        top = by_base[0]
        if len(by_base) > 1 and abs(top["baseline"] - by_base[1]["baseline"]) <= TOL:
            tied = [r for r in by_base if abs(r["baseline"] - top["baseline"]) <= TOL]
            baseline_block.update({"status": "tie", "preferred_value": top["baseline"]})
            reasons.append(_reason("baseline_tie", players=[_label(r) for r in tied]))
        else:
            baseline_block.update(
                {
                    "status": "available",
                    "preferred_player_id": top["player_id"],
                    "preferred_value": top["baseline"],
                }
            )
            if leading["first"] is not None:
                agrees = top["player_id"] == leading["first"]
                baseline_block["agrees_with_model_leader"] = agrees
                reasons.append(
                    _reason(
                        "baseline_agrees" if agrees else "baseline_disagrees",
                        player=_label(top),
                        value=top["baseline"],
                    )
                )

    # --- status ------------------------------------------------------------------------------
    severities = {r["severity"] for r in reasons}
    if "hold" in severities:
        status = "hold"
    elif "review" in severities:
        status = "review"
    else:
        status = "recommend"
    recommended = None
    if status == "recommend":
        first = ordered[0]
        recommended = first["player_id"]
        reasons.insert(
            0,
            _reason(
                "recommend_highest_projection",
                player=_label(first),
                value=first["projection"],
                gap=leading["gap"],
                second=_label(ordered[1]),
            ),
        )
    mode = snapshot.get("mode")
    if mode == "published_weekly":
        limitations.append("snapshot_not_game_day_verified")
    if mode == "synthetic":
        limitations.append("synthetic_evidence")
    # Anything short of a positive reconciliation is a limitation on real evidence (synthetic
    # snapshots already carry synthetic_evidence and have nothing to reconcile).
    if (
        mode != "synthetic"
        and snapshot.get("baseline_reconciled") is not True
        and "baseline_unreconciled" not in limitations
    ):
        limitations.append("baseline_unreconciled")
    limitations.append("floors_not_probabilities")

    result = {
        "schema_version": RESULT_SCHEMA,
        "policy_version": POLICY_VERSION,
        "decision_id": content_id(norm),
        "status": status,
        "recommended_player_id": recommended,
        "reasons": reasons,
        "ranking": ranking_rows,
        "leading": leading,
        "gates": gates,
        "baseline": baseline_block,
        "counts": {
            "nominated": len(alts),
            "excluded": sum(1 for r in ranking_rows if r["exclusion"]),
            "comparable": len(comparable),
        },
        "limitations": limitations,
    }
    result["explanation"] = [render_reason(r["code"], r["detail"]) for r in reasons]
    return result


def result_identity(result: dict[str, Any]) -> str:
    """Content identity of a result (used to detect tampered receipts)."""
    return content_id(result)
