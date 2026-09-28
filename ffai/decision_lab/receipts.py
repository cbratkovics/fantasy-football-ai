"""Decision receipts: immutable inputs and result, plus appended events (ADR-0033).

A receipt is what the browser stores locally and what ``lab-replay`` re-validates. Two kinds of
identity, documented separately:

* **decision identity** — ``decision_id = sha256(canonical(decision_inputs))``: schema and policy
  versions, the snapshot reference (including the inputs snapshot's content digest), slot,
  every alternative with its availability fields, and the parameters. Replaying the same semantic
  inputs reproduces it; attaching outcomes, recording an action, or a new timestamp never changes
  it. Changing an assumption or a parameter produces a *new* decision whose
  ``parent_decision_id`` links back.
* **event identity** — ``event_id = sha256(canonical({decision_id, seq, event_type, at_utc,
  payload}))``. Events are appended, never rewritten; ``action`` and ``outcome`` on the receipt
  are projections of the event list and are recomputed on validation.

The initial action state is ``not_recorded`` with null ids and time; the recommendation is never
auto-filled as the user's choice. At most one action event (``action_recorded`` or
``action_declined``) may exist. Outcome events must reference the decision's own snapshot.

Trust boundary: digests let a reader check a receipt against known evidence; they are not
signatures and prove nothing about who acted when. Local storage is editable and erasable;
"append-only" is application behaviour, not a tamper-proof audit service.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ffai.decision_lab import POLICY_VERSION, SCHEMA_VERSIONS, contracts, metrics, policy, spec
from ffai.decision_lab.canonical import content_id

RECEIPT_SCHEMA = SCHEMA_VERSIONS["receipt"]
TRUST_NOTE = (
    "Digests support integrity checks against known evidence; they are not signatures and do not "
    "prove who acted or when. Local storage is editable and erasable; append-only is application "
    "behaviour, not a tamper-proof audit service."
)
ACTION_EVENTS = {"action_recorded", "action_declined"}
_SPEC = spec()


class ReceiptError(ValueError):
    """A receipt is malformed, tampered, or an operation on it is not allowed."""


class ConflictError(ReceiptError):
    """Two receipts share a decision_id but disagree."""


SnapshotLookup = Callable[[str], dict[str, Any] | None]


# --- building decision inputs from a snapshot ------------------------------------------------


def snapshot_ref(
    inputs_snapshot: dict[str, Any],
    *,
    evidence_status: str = "verified",
    evidence_detail: str | None = None,
) -> dict[str, Any]:
    """The ``snapshot`` block of ``decision_inputs`` for a loaded inputs snapshot."""
    return {
        "snapshot_id": inputs_snapshot["snapshot_id"],
        "mode": inputs_snapshot["mode"],
        "source_family": inputs_snapshot["source_family"],
        "season": inputs_snapshot["season"],
        "week": inputs_snapshot["week"],
        "model_version": inputs_snapshot["model_version"],
        "feature_version": inputs_snapshot.get("feature_version"),
        "candidate_by_position": dict(inputs_snapshot["candidate_by_position"]),
        "scoring_format": inputs_snapshot["scoring_format"],
        "inputs_sha256": content_id(inputs_snapshot),
        "data_cutoff": inputs_snapshot.get("data_cutoff"),
        "generated_at_utc": inputs_snapshot.get("generated_at_utc"),
        "publication": dict(inputs_snapshot["publication"]),
        "evidence_status": evidence_status,
        "evidence_detail": evidence_detail,
        "baseline_reconciled": (inputs_snapshot.get("baseline") or {}).get("reconciled"),
    }


def build_decision_inputs(
    inputs_snapshot: dict[str, Any],
    *,
    slot: str,
    player_ids: list[str],
    overrides: dict[str, dict[str, str]] | None = None,
    parameters: dict[str, Any] | None = None,
    evidence_status: str = "verified",
    evidence_detail: str | None = None,
) -> dict[str, Any]:
    """Resolve nominated players against a snapshot into a ``decision_inputs/1.0`` document.

    Unknown player ids raise; the caller decides whether that is a UI error or a HOLD. Availability
    starts from each row's ``availability_default`` and user overrides replace it.
    """
    overrides = overrides or {}
    rows = {r["player_id"]: r for r in inputs_snapshot["rows"]}
    alternatives = []
    for pid in player_ids:
        if pid not in rows:
            raise ReceiptError(f"player {pid} is not in snapshot {inputs_snapshot['snapshot_id']}")
        r = rows[pid]
        default_basis = {
            "realized_stats_row": "hindsight_stats_row",
            "unknown": "not_verified",
            "assumed_available": "user_assumed",
            "unavailable": "user_excluded",
        }[r["availability_default"]]
        ov = overrides.get(pid) or {}
        alternatives.append(
            {
                "player_id": pid,
                "display_name": r.get("display_name"),
                "team": r.get("team"),
                "position": r["position"],
                "model_version": r["model_version"],
                "candidate": r["candidate"],
                "season": inputs_snapshot["season"],
                "week": inputs_snapshot["week"],
                "projection": r.get("projection"),
                "floor": r.get("floor"),
                "ceiling": r.get("ceiling"),
                "baseline": r.get("baseline"),
                "baseline_provenance": r.get("baseline_provenance") or "unknown",
                "availability": ov.get("availability", r["availability_default"]),
                "availability_basis": ov.get("availability_basis", default_basis),
                "source_row_ref": r.get("source_row_ref"),
            }
        )
    params = policy.default_parameters()
    params.update(parameters or {})
    doc = {
        "schema_version": SCHEMA_VERSIONS["decision_inputs"],
        "policy_version": POLICY_VERSION,
        "snapshot": snapshot_ref(
            inputs_snapshot, evidence_status=evidence_status, evidence_detail=evidence_detail
        ),
        "slot": slot,
        "alternatives": alternatives,
        "parameters": params,
    }
    return policy.normalize_inputs(doc)


# --- receipts --------------------------------------------------------------------------------


def _empty_action() -> dict[str, Any]:
    return {
        "state": "not_recorded",
        "action_id": None,
        "at_utc": None,
        "chosen_player_id": None,
        "kind": None,
        "note": None,
    }


def _empty_outcome() -> dict[str, Any]:
    return {
        "state": "not_attached",
        "outcome_snapshot_id": None,
        "outcome_id": None,
        "attached_at_utc": None,
        "metrics": None,
    }


def new_receipt(
    inputs: dict[str, Any],
    *,
    created_at_utc: str,
    case_id: str | None = None,
    parent_decision_id: str | None = None,
    prediction_note: str | None = None,
) -> dict[str, Any]:
    """Evaluate the policy and wrap inputs + result into a fresh receipt (no events yet)."""
    contracts.check_decision_inputs(policy.normalize_inputs(inputs))
    norm = policy.normalize_inputs(inputs)
    result = policy.evaluate(norm)
    contracts.check_decision_result(result)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "decision_id": result["decision_id"],
        "parent_decision_id": parent_decision_id,
        "created_at_utc": created_at_utc,
        "case_id": case_id,
        "prediction_note": prediction_note,
        "inputs": norm,
        "result": result,
        "result_sha256": content_id(result),
        "events": [],
        "action": _empty_action(),
        "outcome": _empty_outcome(),
        "trust": {"note": TRUST_NOTE},
    }
    return receipt


def event_id(
    decision_id: str, seq: int, event_type: str, at_utc: str, payload: dict[str, Any]
) -> str:
    return content_id(
        {
            "decision_id": decision_id,
            "seq": seq,
            "event_type": event_type,
            "at_utc": at_utc,
            "payload": payload,
        }
    )


def derive_state(receipt: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project the event list onto the ``action`` and ``outcome`` blocks."""
    action = _empty_action()
    outcome = _empty_outcome()
    for ev in receipt["events"]:
        t, p = ev["event_type"], ev["payload"]
        if t == "action_recorded":
            action = {
                "state": "recorded",
                "action_id": ev["event_id"],
                "at_utc": ev["at_utc"],
                "chosen_player_id": p["chosen_player_id"],
                "kind": p["kind"],
                "note": p.get("note"),
            }
        elif t == "action_declined":
            action = {
                "state": "declined",
                "action_id": ev["event_id"],
                "at_utc": ev["at_utc"],
                "chosen_player_id": None,
                "kind": None,
                "note": p.get("note"),
            }
        elif t == "outcome_attached":
            outcome = {
                "state": "attached",
                "outcome_snapshot_id": p["outcome_snapshot_id"],
                "outcome_id": p["outcome_id"],
                "attached_at_utc": ev["at_utc"],
                "metrics": p["metrics"],
            }
    return action, outcome


def append_event(
    receipt: dict[str, Any], event_type: str, payload: dict[str, Any], at_utc: str
) -> dict[str, Any]:
    """Return a new receipt with the event appended (inputs and result untouched)."""
    if event_type not in _SPEC["event_types"]:
        raise ReceiptError(f"unknown event type {event_type}")
    if event_type in ACTION_EVENTS and any(
        e["event_type"] in ACTION_EVENTS for e in receipt["events"]
    ):
        raise ReceiptError(
            "an action is already recorded for this decision; change an assumption to start a new decision"
        )
    seq = len(receipt["events"]) + 1
    ev = {
        "event_id": event_id(receipt["decision_id"], seq, event_type, at_utc, payload),
        "seq": seq,
        "event_type": event_type,
        "at_utc": at_utc,
        "payload": payload,
    }
    out = dict(receipt)
    out["events"] = [*receipt["events"], ev]
    out["action"], out["outcome"] = derive_state(out)
    return out


def record_action(
    receipt: dict[str, Any],
    *,
    chosen_player_id: str,
    kind: str,
    at_utc: str,
    note: str | None = None,
) -> dict[str, Any]:
    nominated = {r["player_id"] for r in receipt["result"]["ranking"]}
    if chosen_player_id not in nominated:
        raise ReceiptError(
            f"chosen player {chosen_player_id} was not among the nominated alternatives"
        )
    if kind not in _SPEC["action_kinds"]:
        raise ReceiptError(f"unknown action kind {kind}")
    payload = {"chosen_player_id": chosen_player_id, "kind": kind, "note": note}
    return append_event(receipt, "action_recorded", payload, at_utc)


def decline_action(
    receipt: dict[str, Any], *, at_utc: str, note: str | None = None
) -> dict[str, Any]:
    return append_event(receipt, "action_declined", {"note": note}, at_utc)


def outcome_compatible(receipt: dict[str, Any], outcome_snapshot: dict[str, Any]) -> str | None:
    """``None`` when the outcome snapshot belongs to this decision's snapshot, else the reason."""
    snap = receipt["inputs"]["snapshot"]
    if outcome_snapshot.get("snapshot_id") != snap["snapshot_id"]:
        return (
            f"outcome snapshot {outcome_snapshot.get('snapshot_id')} is not {snap['snapshot_id']}"
        )
    if outcome_snapshot.get("inputs_content_sha256") != snap["inputs_sha256"]:
        return "outcome snapshot pairs with a different inputs snapshot content"
    if (outcome_snapshot.get("season"), outcome_snapshot.get("week")) != (
        snap["season"],
        snap["week"],
    ):
        return "outcome snapshot is for a different season/week"
    if outcome_snapshot.get("model_version") != snap["model_version"]:
        return "outcome snapshot is for a different model version"
    if outcome_snapshot.get("scoring_format") != snap["scoring_format"]:
        return "outcome snapshot uses a different scoring format"
    return None


def attach_outcome(
    receipt: dict[str, Any], outcome_snapshot: dict[str, Any], *, at_utc: str
) -> dict[str, Any]:
    """Append an ``outcome_attached`` event with metrics computed from the outcome snapshot."""
    contracts.check_outcome_snapshot(outcome_snapshot)
    why = outcome_compatible(receipt, outcome_snapshot)
    if why:
        raise ReceiptError(f"cannot attach outcomes: {why}")
    oid = content_id(outcome_snapshot)
    snapshot_with_id = {**outcome_snapshot, "outcome_id": oid}
    m = metrics.compute_metrics(receipt["result"], receipt["action"], snapshot_with_id)
    payload = {
        "outcome_snapshot_id": outcome_snapshot["snapshot_id"],
        "outcome_id": oid,
        "metrics": m,
    }
    return append_event(receipt, "outcome_attached", payload, at_utc)


# --- validation, replay, merge ----------------------------------------------------------------


def validate_receipt(
    receipt: dict[str, Any],
    *,
    inputs_lookup: SnapshotLookup | None = None,
    outcome_lookup: SnapshotLookup | None = None,
) -> list[dict[str, Any]]:
    """Every check a replay performs. Returns the check list; raises on the first failure.

    With ``inputs_lookup`` the decision's snapshot digest is verified against the bundle; with
    ``outcome_lookup`` the attached metrics are recomputed from the bundle's outcome snapshot.
    """
    checks: list[dict[str, Any]] = []

    def ok(name: str, **detail: Any) -> None:
        checks.append({"check": name, "ok": True, **detail})

    def fail(name: str, message: str) -> None:
        checks.append({"check": name, "ok": False, "message": message})
        raise ReceiptError(f"{name}: {message}")

    try:
        contracts.validate("receipt", receipt)
    except contracts.ContractError as exc:
        fail("schema", str(exc))
    ok("schema", schema_version=receipt["schema_version"])

    norm = policy.normalize_inputs(receipt["inputs"])
    if norm != receipt["inputs"]:
        fail("inputs_normalized", "alternatives are not in canonical order or versions differ")
    try:
        contracts.check_decision_inputs(norm)
    except contracts.ContractError as exc:
        fail("inputs_contract", str(exc))
    did = content_id(norm)
    if did != receipt["decision_id"]:
        fail("decision_id", f"recomputed {did[:12]} != stored {receipt['decision_id'][:12]}")
    ok("decision_id", decision_id=did)

    recomputed = policy.evaluate(norm)
    if content_id(recomputed) != receipt["result_sha256"]:
        fail("result_sha256", "stored result digest does not match a fresh evaluation")
    if recomputed != receipt["result"]:
        fail("result_replay", "stored result differs from a fresh evaluation (tampered or stale)")
    ok(
        "result_replay",
        status=recomputed["status"],
        recommended_player_id=recomputed["recommended_player_id"],
    )

    for i, ev in enumerate(receipt["events"], start=1):
        if ev["seq"] != i:
            fail("event_sequence", f"event {i} has seq {ev['seq']}")
        eid = event_id(
            receipt["decision_id"], ev["seq"], ev["event_type"], ev["at_utc"], ev["payload"]
        )
        if eid != ev["event_id"]:
            fail("event_id", f"event {i} digest mismatch")
    if sum(1 for e in receipt["events"] if e["event_type"] in ACTION_EVENTS) > 1:
        fail("single_action", "more than one action event")
    ok("events", n=len(receipt["events"]))

    action, outcome = derive_state(receipt)
    if action != receipt["action"]:
        fail("action_projection", "action block does not match the events")
    if outcome != receipt["outcome"]:
        fail("outcome_projection", "outcome block does not match the events")
    nominated = {r["player_id"] for r in recomputed["ranking"]}
    if action["state"] == "recorded" and action["chosen_player_id"] not in nominated:
        fail("chosen_membership", "chosen player is not a nominated alternative")
    ok("projections", action_state=action["state"], outcome_state=outcome["state"])

    if inputs_lookup is not None:
        snap = inputs_lookup(receipt["inputs"]["snapshot"]["snapshot_id"])
        if snap is None:
            fail(
                "snapshot_available",
                f"snapshot {receipt['inputs']['snapshot']['snapshot_id']} is not in the bundle",
            )
        if content_id(snap) != receipt["inputs"]["snapshot"]["inputs_sha256"]:
            fail("snapshot_digest", "the bundle's inputs snapshot has a different content digest")
        rows = {r["player_id"]: r for r in snap["rows"]}
        for a in norm["alternatives"]:
            row = rows.get(a["player_id"])
            if row is None:
                fail("alternative_membership", f"{a['player_id']} is not in the snapshot")
            for k in ("projection", "floor", "ceiling", "baseline", "position"):
                if row.get(k) != a.get(k):
                    fail(
                        "alternative_values", f"{a['player_id']}.{k} differs from the snapshot row"
                    )
        ok("snapshot_digest", snapshot_id=snap["snapshot_id"])

    if outcome["state"] == "attached":
        ev = [e for e in receipt["events"] if e["event_type"] == "outcome_attached"][-1]
        if ev["payload"]["outcome_snapshot_id"] != receipt["inputs"]["snapshot"]["snapshot_id"]:
            fail("outcome_snapshot", "outcome attached from a different snapshot")
        if outcome_lookup is not None:
            osnap = outcome_lookup(ev["payload"]["outcome_snapshot_id"])
            if osnap is None:
                fail("outcome_available", "outcome snapshot is not in the bundle")
            oid = content_id(osnap)
            if oid != ev["payload"]["outcome_id"]:
                fail(
                    "outcome_digest", "the bundle's outcome snapshot has a different content digest"
                )
            why = outcome_compatible(receipt, osnap)
            if why:
                fail("outcome_compatible", why)
            # Metrics are recomputed with the action state as of the attach event.
            partial = dict(receipt)
            partial["events"] = [e for e in receipt["events"] if e["seq"] < ev["seq"]]
            act, _ = derive_state(partial)
            m = metrics.compute_metrics(recomputed, act, {**osnap, "outcome_id": oid})
            if m != ev["payload"]["metrics"]:
                fail("metrics_replay", "stored outcome metrics differ from a fresh computation")
            ok("metrics_replay", outcome_id=oid)
    return checks


IMMUTABLE_FIELDS: tuple[str, ...] = (
    "schema_version",
    "decision_id",
    "parent_decision_id",
    "created_at_utc",
    "case_id",
    "prediction_note",
    "inputs",
    "result",
    "result_sha256",
    "trust",
)
"""Everything in a receipt that is fixed when the receipt is created. Only ``events`` (and the
``action`` / ``outcome`` projections derived from them) may grow afterwards."""


def immutable_content(receipt: dict[str, Any]) -> dict[str, Any]:
    """The receipt without its event history: the part two records must share to be merged."""
    return {k: receipt.get(k) for k in IMMUTABLE_FIELDS}


def merge(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Idempotent import: identical receipts merge to one; the same decision with a superset of
    events is accepted; anything else with the same ``decision_id`` is a conflict.

    Two receipts merge only when their complete immutable content is equal
    (:data:`IMMUTABLE_FIELDS`, including ``created_at_utc``, ``prediction_note``, ``case_id`` and
    ``parent_decision_id``) and one event history is a prefix of the other. Because the decision
    id is a digest of the inputs alone, two receipts can share it while recording a different
    creation time, note, case or parent; that is a conflict, reported by field name, never
    resolved by preferring the longer history. On success the merged receipt keeps ``existing``'s
    immutable content (identical to ``incoming``'s) and the longer event history.
    """
    if existing["decision_id"] != incoming["decision_id"]:
        raise ConflictError("receipts have different decision ids")
    differing = [k for k in IMMUTABLE_FIELDS if existing.get(k) != incoming.get(k)]
    if differing:
        raise ConflictError(
            "same decision_id but different immutable content: " + ", ".join(differing)
        )
    a, b = existing["events"], incoming["events"]
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    for x, y in zip(shorter, longer, strict=False):
        if x["event_id"] != y["event_id"]:
            raise ConflictError(f"event {x['seq']} differs between the two receipts")
    out = dict(existing)
    out["events"] = list(longer)
    out["action"], out["outcome"] = derive_state(out)
    return out
