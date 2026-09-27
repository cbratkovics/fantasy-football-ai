"""Curated case library (``decision_lab_cases/1.0``): synthetic regression cases and real cases
discovered deterministically from the exported evidence.

Synthetic cases each own a small, unmistakably synthetic inputs snapshot (season 2099, player ids
``SYN-*``); their expected results are never hand-typed: every ``expected`` block is the output of
:func:`ffai.decision_lab.policy.evaluate` on decision inputs built with
:func:`ffai.decision_lab.receipts.build_decision_inputs`, so the file cannot drift from the policy.

Real cases are found by one fixed procedure over the historical snapshots (top-four projections
per position and week, default parameters) and selected by pattern (model/baseline disagreement
with each side ahead where the evidence contains it, an agreement case, an ambiguity case, the
published weekly snapshot). Titles never reveal outcomes; the pattern is sealed in
``selection.sealed_pattern`` and is inspectable in the static file — an educational reveal, not
blinding. Curated examples are not an unbiased performance sample.

Experiment semantics (shared with the TypeScript ``Experiment`` type): ``parameters`` is a partial
override merged onto the case parameters; ``overrides`` replaces the case overrides when present.
"""

from __future__ import annotations

from typing import Any

from ffai.decision_lab import POLICY_VERSION, SCHEMA_VERSIONS, policy, receipts
from ffai.decision_lab.canonical import content_id

SYN_MODEL = "synthetic-model-v1"
SYN_SEASON = 2099
SYN_WEEK = 1
POSITIONS = ("QB", "RB", "WR", "TE")
CASES_SCHEMA = SCHEMA_VERSIONS["cases"]
FAMILY_SHORT = {"out_of_sample_season": "oos", "frozen_test": "frozen", "weekly": "weekly"}
TITLE_FORBIDDEN = ("wins", "loses", "regret", "beat")

CASES_NOTES = [
    "Sealed patterns (selection.sealed_pattern) are stored in this static file and can be read by "
    "anyone; the reveal sequence in the lab is educational, not secure blinding.",
    "Curated cases are chosen to illustrate specific policy behaviours and patterns; they are not "
    "an unbiased sample of model performance and support no aggregate claim.",
    "Every expected block was computed by ffai.decision_lab.policy.evaluate on decision inputs "
    "built from the case's snapshot with receipts.build_decision_inputs; nothing is hand-typed.",
    "Experiment semantics: experiment.parameters is merged onto the case parameters; "
    "experiment.overrides, when present, replaces the case overrides.",
    "Unsupported scoring formats (for example half-PPR) cannot appear in the bundle because "
    "contracts.check_inputs_snapshot refuses non-PPR snapshots; that HOLD path is covered by the "
    "policy unit tests (tests/test_decision_lab_policy.py) and the golden fixtures instead.",
]


# --- synthetic snapshots -----------------------------------------------------------------------


def _row(
    pid: str,
    position: str,
    projection: float | None,
    floor: float | None,
    baseline: float | None,
    *,
    provenance: str = "player_history",
    availability_default: str = "assumed_available",
) -> dict[str, Any]:
    ceiling = None if projection is None else projection + 8.0
    return {
        "player_id": pid,
        "display_name": f"Synthetic {pid.split('-')[-1]}",
        "team": "SYN",
        "position": position,
        "model_version": SYN_MODEL,
        "candidate": "rf",
        "projection": projection,
        "floor": floor,
        "ceiling": ceiling,
        "baseline": baseline,
        "baseline_provenance": provenance,
        "availability_default": availability_default,
        "display_source": "synthetic",
        "source_row_ref": None,
    }


def synthetic_snapshot(
    snapshot_id: str,
    rows: list[dict[str, Any]],
    *,
    publication_status: str = "synthetic",
    publication_note: str | None = None,
    baseline_note: str | None = None,
) -> dict[str, Any]:
    """An ``inputs_snapshot/1.0`` document for a synthetic case (season 2099, no source files)."""
    return {
        "schema_version": SCHEMA_VERSIONS["inputs_snapshot"],
        "snapshot_id": snapshot_id,
        "mode": "synthetic",
        "source_family": "synthetic",
        "season": SYN_SEASON,
        "week": SYN_WEEK,
        "model_version": SYN_MODEL,
        "feature_version": None,
        "candidate_by_position": {p: "rf" for p in POSITIONS},
        "scoring_format": "ppr",
        "data_cutoff": {"season": SYN_SEASON - 1, "week": 18},
        "generated_at_utc": None,
        "publication": {
            "status": publication_status,
            "run_id": None,
            "action": None,
            "note": publication_note or "synthetic inputs for boundary and regression checks",
        },
        "population": {
            "conditioning": "synthetic",
            "description": "hand-built synthetic rows; never evidence about any real player",
            "n_rows": len(rows),
            "exclusions": [],
        },
        "baseline": {
            "name": "synthetic",
            "provenance_basis": "synthetic",
            "reconciled": None,
            "reconciled_to": None,
            "note": baseline_note or "baseline values and provenance are set per case",
        },
        "champion_at_source": {
            "model_version": SYN_MODEL,
            "candidate_by_position": {p: "rf" for p in POSITIONS},
            "basis": "synthetic",
        },
        "source": {"mart": None, "files": [], "mart_export": None},
        "rows": sorted(rows, key=lambda r: r["player_id"]),
    }


def synthetic_outcomes(inputs: dict[str, Any], actuals: dict[str, float | None]) -> dict[str, Any]:
    """An ``outcome_snapshot/1.0`` document for a synthetic inputs snapshot."""
    rows = []
    for r in inputs["rows"]:
        a = actuals.get(r["player_id"])
        rows.append(
            {
                "player_id": r["player_id"],
                "actual": a,
                "actual_source": "synthetic" if a is not None else None,
            }
        )
    return {
        "schema_version": SCHEMA_VERSIONS["outcome_snapshot"],
        "snapshot_id": inputs["snapshot_id"],
        "inputs_content_sha256": content_id(inputs),
        "season": inputs["season"],
        "week": inputs["week"],
        "model_version": inputs["model_version"],
        "scoring_format": inputs["scoring_format"],
        "observed_at_utc": None,
        "source": {"mart": None, "files": []},
        "coverage": {
            "scored": len(rows),
            "observed": sum(1 for r in rows if r["actual"] is not None),
        },
        "rows": rows,
    }


def _p(**kw: Any) -> dict[str, Any]:
    params = policy.default_parameters()
    params.update(kw)
    return params


def _assume(*pids: str) -> dict[str, dict[str, str]]:
    return {
        p: {"availability": "assumed_available", "availability_basis": "user_assumed"} for p in pids
    }


SYNTHETIC_SPECS: list[dict[str, Any]] = [
    {
        "case_id": "syn-ambiguity-floor-relaxation",
        "title": "Ambiguity: relaxing the floor guardrail does not resolve a small gap",
        "setup": (
            "Two running backs 0.4 points apart (14.0 vs 13.6) with floors 6.0 and 7.0. The minimum "
            "gap is 0.5 and a floor guardrail of 8 is on, so the comparison stays in review. Relaxing "
            "the floor to 5 changes nothing because the gap still fails; only lowering the minimum gap "
            "to 0.3 produces a recommendation, and the baseline prefers the other player."
        ),
        "slot": "RB",
        "rows": [_row("SYN-A", "RB", 14.0, 6.0, 12.0), _row("SYN-B", "RB", 13.6, 7.0, 13.0)],
        "alternatives": ["SYN-A", "SYN-B"],
        "overrides": {},
        "parameters": _p(min_projection_gap=0.5, min_floor=8.0),
        "outcomes": {"SYN-A": 8.0, "SYN-B": 16.0},
        "experiments": [
            {"label": "relax floor to 5", "parameters": {"min_floor": 5.0}},
            {
                "label": "gap 0.3, floor 5",
                "parameters": {"min_projection_gap": 0.3, "min_floor": 5.0},
            },
        ],
        "criteria": "floor relaxation cannot bypass the ambiguity margin; outcomes A=8, B=16",
    },
    {
        "case_id": "syn-exact-tie",
        "title": "Exact projection tie with a zero minimum gap",
        "setup": (
            "Two players project exactly 12.0. Even with the minimum gap set to 0 the policy answers "
            "review: a tie is not evidence, and the displayed order is by player id only."
        ),
        "slot": "WR",
        "rows": [_row("SYN-A", "WR", 12.0, 4.0, 10.0), _row("SYN-B", "WR", 12.0, 4.0, 11.0)],
        "alternatives": ["SYN-A", "SYN-B"],
        "overrides": {},
        "parameters": _p(min_projection_gap=0.0),
        "outcomes": None,
        "experiments": [],
        "criteria": "|gap| <= tolerance produces review even at min gap 0",
    },
    {
        "case_id": "syn-threshold-equality",
        "title": "Gap and floor exactly at their thresholds",
        "setup": (
            "The gap (0.4) equals the minimum gap and the leading floor (6.0) equals the floor "
            "guardrail; equality qualifies on both gates, so the leader is recommended."
        ),
        "slot": "RB",
        "rows": [_row("SYN-A", "RB", 14.0, 6.0, 12.0), _row("SYN-B", "RB", 13.6, 7.0, 11.0)],
        "alternatives": ["SYN-A", "SYN-B"],
        "overrides": {},
        "parameters": _p(min_projection_gap=0.4, min_floor=6.0),
        "outcomes": None,
        "experiments": [],
        "criteria": "equality qualifies on the gap gate and the floor gate",
    },
    {
        "case_id": "syn-missing-and-negative-outcomes",
        "title": "Missing, zero and negative outcomes",
        "setup": (
            "Three tight ends with a clear leader. The outcome snapshot has one unobserved player, one "
            "zero and one negative score; zero and negative are valid observations, the missing one "
            "leaves the choice-set regret unavailable instead of imputing it."
        ),
        "slot": "TE",
        "rows": [
            _row("SYN-A", "TE", 14.0, 5.0, 11.0),
            _row("SYN-B", "TE", 12.0, 4.0, 10.0),
            _row("SYN-C", "TE", 9.0, 2.0, 8.0),
        ],
        "alternatives": ["SYN-A", "SYN-B", "SYN-C"],
        "overrides": {},
        "parameters": _p(),
        "outcomes": {"SYN-A": None, "SYN-B": 0.0, "SYN-C": -1.5},
        "experiments": [],
        "criteria": "metrics eligibility with null, zero and negative actuals",
    },
    {
        "case_id": "syn-slot-and-exclusion",
        "title": "FLEX slot with an ineligible position and a user exclusion",
        "setup": (
            "A quarterback is nominated for FLEX (slot-ineligible) and one running back is marked "
            "unavailable, leaving a single comparable alternative: review. Removing the exclusion "
            "restores a two-player comparison and a recommendation."
        ),
        "slot": "FLEX",
        "rows": [
            _row("SYN-A", "RB", 14.0, 6.0, 12.0),
            _row("SYN-B", "RB", 13.0, 5.0, 11.0),
            _row("SYN-C", "QB", 20.0, 10.0, 18.0),
        ],
        "alternatives": ["SYN-A", "SYN-B", "SYN-C"],
        "overrides": {
            "SYN-B": {"availability": "unavailable", "availability_basis": "user_excluded"}
        },
        "parameters": _p(),
        "outcomes": None,
        "experiments": [{"label": "remove the exclusion", "overrides": {}}],
        "criteria": "slot eligibility and explicit exclusion shrink the comparable set",
    },
    {
        "case_id": "syn-unknown-availability",
        "title": "Unknown availability must be resolved first",
        "setup": (
            "Both players default to unknown availability (as published weekly rows do). The policy "
            "answers review until the user records an explicit assumption; assuming both available "
            "yields a recommendation with the user-assumed availability limitation."
        ),
        "slot": "WR",
        "rows": [
            _row("SYN-A", "WR", 14.0, 6.0, 12.0, availability_default="unknown"),
            _row("SYN-B", "WR", 12.0, 5.0, 11.0, availability_default="unknown"),
        ],
        "alternatives": ["SYN-A", "SYN-B"],
        "overrides": {},
        "parameters": _p(),
        "outcomes": None,
        "experiments": [{"label": "assume both available", "overrides": _assume("SYN-A", "SYN-B")}],
        "criteria": "unresolved availability is a review condition, not a default",
    },
    {
        "case_id": "syn-baseline-fallback",
        "title": "Leader without an independent baseline",
        "setup": (
            "The leading player's baseline is the model's own prediction (fallback), so no independent "
            "comparator exists and the policy answers review. Model-only exploration proceeds and "
            "records that the baseline is not independent."
        ),
        "slot": "RB",
        "rows": [
            _row("SYN-A", "RB", 14.0, 6.0, 14.0, provenance="model_prediction_fallback"),
            _row("SYN-B", "RB", 12.0, 5.0, 11.0),
        ],
        "alternatives": ["SYN-A", "SYN-B"],
        "overrides": {},
        "parameters": _p(),
        "outcomes": None,
        "experiments": [
            {"label": "model-only exploration", "parameters": {"model_only_exploration": True}}
        ],
        "criteria": "a model-derived baseline is never presented as a baseline comparison",
    },
    {
        "case_id": "syn-blocked-evidence",
        "title": "Blocked evidence cannot be unblocked by lowering thresholds",
        "setup": (
            "The snapshot carries a publication block. The policy holds, and setting the minimum gap "
            "to zero still holds: invalid evidence never becomes a weaker recommendation."
        ),
        "slot": "RB",
        "rows": [_row("SYN-A", "RB", 14.0, 6.0, 12.0), _row("SYN-B", "RB", 12.0, 5.0, 11.0)],
        "alternatives": ["SYN-A", "SYN-B"],
        "overrides": {},
        "parameters": _p(),
        "publication_status": "blocked",
        "publication_note": "synthetic block: the lab export failed after the gold export",
        "outcomes": None,
        "experiments": [{"label": "min gap 0", "parameters": {"min_projection_gap": 0.0}}],
        "criteria": "HOLD precedence over every threshold",
    },
    {
        "case_id": "syn-one-option",
        "title": "A single alternative is not a comparison",
        "setup": "Only one player is nominated; a comparison needs at least two comparable alternatives.",
        "slot": "TE",
        "rows": [_row("SYN-A", "TE", 10.0, 3.0, 9.0)],
        "alternatives": ["SYN-A"],
        "overrides": {},
        "parameters": _p(),
        "outcomes": None,
        "experiments": [],
        "criteria": "choice set below the minimum",
    },
    {
        "case_id": "syn-floor-missing",
        "title": "Leader without a floor while the guardrail is on",
        "setup": (
            "The leading player has no model floor while a floor guardrail of 5 is enabled: review. "
            "Disabling the guardrail lets the projection order stand."
        ),
        "slot": "WR",
        "rows": [_row("SYN-A", "WR", 14.0, None, 12.0), _row("SYN-B", "WR", 13.0, 7.0, 11.0)],
        "alternatives": ["SYN-A", "SYN-B"],
        "overrides": {},
        "parameters": _p(min_floor=5.0),
        "outcomes": None,
        "experiments": [
            {"label": "disable the floor guardrail", "parameters": {"min_floor": None}}
        ],
        "criteria": "a missing floor is a review condition only when the guardrail is on",
    },
]


def synthetic_snapshots() -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """``(inputs_snapshot, outcome_snapshot | None)`` for every synthetic case."""
    out = []
    for spec in SYNTHETIC_SPECS:
        inputs = synthetic_snapshot(
            spec["case_id"],
            spec["rows"],
            publication_status=spec.get("publication_status", "synthetic"),
            publication_note=spec.get("publication_note"),
        )
        outcomes = synthetic_outcomes(inputs, spec["outcomes"]) if spec["outcomes"] else None
        out.append((inputs, outcomes))
    return out


# --- evaluation helpers ----------------------------------------------------------------------


def case_inputs(case: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Decision inputs for a case as stored (slot, alternatives, overrides, parameters)."""
    return receipts.build_decision_inputs(
        snapshot,
        slot=case["slot"],
        player_ids=list(case["alternatives"]),
        overrides=dict(case["overrides"]),
        parameters=dict(case["parameters"]),
    )


def experiment_inputs(
    case: dict[str, Any], experiment: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, Any]:
    """Decision inputs for one experiment: parameters merged, overrides replaced when present."""
    params = dict(case["parameters"])
    params.update(experiment.get("parameters") or {})
    overrides = (
        dict(experiment["overrides"]) if "overrides" in experiment else dict(case["overrides"])
    )
    return receipts.build_decision_inputs(
        snapshot,
        slot=case["slot"],
        player_ids=list(case["alternatives"]),
        overrides=overrides,
        parameters=params,
    )


def _expected(result: dict[str, Any], *, with_codes: bool = True) -> dict[str, Any]:
    exp: dict[str, Any] = {
        "status": result["status"],
        "recommended_player_id": result["recommended_player_id"],
    }
    if with_codes:
        exp["reason_codes"] = [r["code"] for r in result["reasons"]]
    return exp


def _finish_case(
    case: dict[str, Any],
    snapshot: dict[str, Any],
    outcomes: dict[str, Any] | None,
    experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the case and its experiments; fill the derived fields."""
    result = policy.evaluate(case_inputs(case, snapshot))
    rows = {r["player_id"]: r for r in snapshot["rows"]}
    finished = dict(case)
    finished["mode"] = snapshot["mode"]
    finished["snapshot_id"] = snapshot["snapshot_id"]
    finished["expected"] = _expected(result)
    finished["experiments"] = []
    for ex in experiments:
        r = policy.evaluate(experiment_inputs(case, ex, snapshot))
        entry = {"label": ex["label"], "expected": _expected(r, with_codes=False)}
        if "parameters" in ex:
            entry["parameters"] = dict(ex["parameters"])
        if "overrides" in ex:
            entry["overrides"] = dict(ex["overrides"])
        finished["experiments"].append(entry)
    finished["outcomes_available"] = outcomes is not None
    finished["outcome_snapshot_id"] = snapshot["snapshot_id"] if outcomes is not None else None
    limitations = list(result["limitations"])
    if "outcomes_inspectable" not in limitations:
        limitations.append("outcomes_inspectable")
    finished["limitations"] = limitations
    finished["source_row_refs"] = [
        rows[pid]["source_row_ref"]
        for pid in case["alternatives"]
        if rows[pid].get("source_row_ref")
    ]
    return finished


def synthetic_cases(
    snapshots: dict[str, dict[str, Any]], outcomes: dict[str, dict[str, Any] | None]
) -> list[dict[str, Any]]:
    cases = []
    for spec in SYNTHETIC_SPECS:
        snap = snapshots[spec["case_id"]]
        base = {
            "case_id": spec["case_id"],
            "title": spec["title"],
            "slot": spec["slot"],
            "alternatives": list(spec["alternatives"]),
            "overrides": dict(spec["overrides"]),
            "parameters": dict(spec["parameters"]),
            "setup": spec["setup"],
            "selection": {
                "method": "curated_synthetic",
                "criteria": spec["criteria"],
                "sealed_pattern": None,
            },
        }
        cases.append(_finish_case(base, snap, outcomes.get(spec["case_id"]), spec["experiments"]))
    return cases


# --- real case discovery ---------------------------------------------------------------------

_WORDS = {2: "two", 3: "three", 4: "four"}


def _top_n(snapshot: dict[str, Any], position: str, n: int) -> list[str]:
    rows = [
        r
        for r in snapshot["rows"]
        if r["position"] == position and isinstance(r.get("projection"), int | float)
    ]
    rows.sort(key=lambda r: (-float(r["projection"]), r["player_id"]))
    return [r["player_id"] for r in rows[:n]]


def _real_case_id(snapshot: dict[str, Any], position: str, alternatives: list[str]) -> str:
    fam = FAMILY_SHORT[snapshot["source_family"]]
    digest = content_id(sorted(alternatives))[:6]
    return f"real-{fam}-{snapshot['season']}-w{snapshot['week']:02d}-{position.lower()}-{digest}"


def _real_title(snapshot: dict[str, Any], position: str, n: int, *, published: bool = False) -> str:
    tail = " (published snapshot)" if published else ""
    return f"{snapshot['season']} week {snapshot['week']} · {position} · {_WORDS[n]} highest projections{tail}"


def _weekly_overrides(snapshot: dict[str, Any], pids: list[str]) -> dict[str, dict[str, str]]:
    if snapshot["source_family"] == "weekly":
        return _assume(*pids)
    return {}


def _screening_view(snapshot: dict[str, Any], pids: list[str]) -> dict[str, Any]:
    """The snapshot restricted to the nominated rows. Screening only needs the policy status,
    which does not depend on the snapshot digest, and hashing a 600-row snapshot for each of the
    ~150 candidate choice sets dominated the build time; selected cases are re-evaluated against
    the full snapshot in :func:`_finish_case`."""
    wanted = set(pids)
    return {**snapshot, "rows": [r for r in snapshot["rows"] if r["player_id"] in wanted]}


def _candidate_case(
    snapshot: dict[str, Any], position: str, n: int = 4
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    pids = _top_n(snapshot, position, n)
    if len(pids) < 2:
        return None
    case = {
        "case_id": _real_case_id(snapshot, position, pids),
        "title": _real_title(snapshot, position, len(pids)),
        "slot": position,
        "alternatives": pids,
        "overrides": _weekly_overrides(snapshot, pids),
        "parameters": policy.default_parameters(),
        "setup": (
            f"The {_WORDS[len(pids)]} {position} rows with the highest projections in this snapshot "
            f"(ties by player id), default parameters"
            + (
                "; availability user-assumed for every alternative (weekly rows default to unknown)."
                if snapshot["source_family"] == "weekly"
                else "; availability is the realized stats row (hindsight-conditioned population)."
            )
        ),
    }
    result = policy.evaluate(case_inputs(case, _screening_view(snapshot, pids)))
    return case, result


def _pattern(result: dict[str, Any], outcomes: dict[str, Any] | None) -> str | None:
    """model_wins / baseline_wins / tie for a recommend with baseline disagreement and both
    outcomes observed; agreement when the baseline agrees; ambiguity for a gap review."""
    codes = {r["code"] for r in result["reasons"]}
    if result["status"] == "review" and "gap_below_minimum" in codes:
        return "ambiguity"
    if result["status"] != "recommend":
        return None
    rec = result["recommended_player_id"]
    base = result["baseline"]["preferred_player_id"]
    if base is None:
        return None
    if base == rec:
        return "agreement"
    if outcomes is None:
        return None
    actual = {r["player_id"]: r["actual"] for r in outcomes["rows"] if r.get("actual") is not None}
    if rec not in actual or base not in actual:
        return None
    if actual[rec] > actual[base]:
        return "model_wins"
    if actual[rec] < actual[base]:
        return "baseline_wins"
    return "tie"


def discovery_order(snapshots: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Historical snapshots in discovery order: oos 2025 weeks ascending, frozen 2024, weekly
    historical weeks ascending."""
    hist = [s for s in snapshots.values() if s["mode"] == "historical_replay"]
    rank = {"out_of_sample_season": 0, "frozen_test": 1, "weekly": 2}
    return sorted(hist, key=lambda s: (rank[s["source_family"]], s["season"], s["week"]))


def _criteria(pattern: str, snapshot: dict[str, Any]) -> str:
    fam = snapshot["source_family"]
    how = "top-four projections per position and week, default parameters"
    what = {
        "model_wins": "first recommend with baseline disagreement where the recommended player scored more",
        "baseline_wins": "first recommend with baseline disagreement where the baseline's player scored more",
        "tie": "first recommend with baseline disagreement where both scored the same",
        "agreement": "first recommend where the baseline prefers the same player",
        "ambiguity": "first review caused by a projection gap below the minimum",
        "disagreement": "first recommend with baseline disagreement and both outcomes observed",
    }[pattern]
    return f"{fam}: {what} ({how})"


def real_cases(
    snapshots: dict[str, dict[str, Any]],
    outcomes: dict[str, dict[str, Any] | None],
    *,
    published_snapshot_id: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Deterministic discovery. Returns the cases and notes about patterns absent in the evidence."""
    wanted: list[tuple[str, str]] = [
        ("out_of_sample_season", "model_wins"),
        ("out_of_sample_season", "baseline_wins"),
        ("frozen_test", "model_wins"),
        ("frozen_test", "baseline_wins"),
        ("weekly", "disagreement"),
        ("out_of_sample_season", "agreement"),
        ("out_of_sample_season", "ambiguity"),
    ]
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for snap in discovery_order(snapshots):
        fam = snap["source_family"]
        if fam == "weekly" and snap["week"] != 2:
            continue  # the weekly disagreement case is drawn from 2026 week 2 by design
        for pos in POSITIONS:
            cand = _candidate_case(snap, pos)
            if cand is None:
                continue
            case, result = cand
            pattern = _pattern(result, outcomes.get(snap["snapshot_id"]))
            if pattern is None:
                continue
            keys = [(fam, pattern)]
            if fam == "weekly" and pattern in ("model_wins", "baseline_wins", "tie"):
                keys.append((fam, "disagreement"))
            for key in keys:
                if key in wanted and key not in found:
                    case = dict(case)
                    case["selection"] = {
                        "method": "deterministic_discovery",
                        "criteria": _criteria(key[1], snap),
                        "sealed_pattern": pattern,
                    }
                    found[key] = _finish_case(case, snap, outcomes.get(snap["snapshot_id"]), [])
    cases = [found[k] for k in wanted if k in found]
    notes = [
        f"No {pattern} case exists in the {fam} evidence under the discovery rule; it is omitted."
        for fam, pattern in wanted
        if (fam, pattern) not in found
    ]
    if published_snapshot_id is not None and published_snapshot_id in snapshots:
        snap = snapshots[published_snapshot_id]
        pids = _top_n(snap, "RB", 3)
        if len(pids) >= 2:
            case = {
                "case_id": _real_case_id(snap, "RB", pids),
                "title": _real_title(snap, "RB", len(pids), published=True),
                "slot": "RB",
                "alternatives": pids,
                "overrides": {},
                "parameters": policy.default_parameters(),
                "setup": (
                    "The three RB rows with the highest projections in the published weekly snapshot, "
                    "no availability overrides: the policy asks for an explicit assumption first. The "
                    "experiment assumes every alternative available. No outcomes exist yet."
                ),
                "selection": {
                    "method": "deterministic_discovery",
                    "criteria": "published weekly snapshot: RB top-three projections, no overrides",
                    "sealed_pattern": None,
                },
            }
            experiments = [{"label": "assume all available", "overrides": _assume(*pids)}]
            cases.append(_finish_case(case, snap, None, experiments))
    return cases, notes


# --- the file ----------------------------------------------------------------------------------


def build_cases(
    snapshots: dict[str, dict[str, Any]],
    outcomes: dict[str, dict[str, Any] | None],
    *,
    published_snapshot_id: str | None,
) -> dict[str, Any]:
    """The ``decision_lab_cases/1.0`` document for a set of loaded snapshots."""
    syn = synthetic_cases(snapshots, outcomes)
    real, notes = real_cases(snapshots, outcomes, published_snapshot_id=published_snapshot_id)
    for c in syn + real:
        lowered = c["title"].lower()
        for w in TITLE_FORBIDDEN:
            if w in lowered.split() or w in lowered:
                raise ValueError(f"case {c['case_id']} title reveals outcomes: {c['title']!r}")
    return {
        "schema_version": CASES_SCHEMA,
        "policy_version": POLICY_VERSION,
        "cases": syn + real,
        "notes": CASES_NOTES + notes,
    }
