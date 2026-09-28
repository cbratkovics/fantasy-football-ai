"""Receipt replay CLI: re-validate, recompute, and extend a decision receipt against a bundle.

    python -m ffai.decision_lab.replay RECEIPT.json [--bundle DIR] [--json]
    python -m ffai.decision_lab.replay --new --case CASE_ID --bundle DIR --out NEW.json [--at-utc ISO]
    python -m ffai.decision_lab.replay RECEIPT.json --record-action PLAYER_ID --kind KIND --out NEW.json
    python -m ffai.decision_lab.replay RECEIPT.json --decline --out NEW.json
    python -m ffai.decision_lab.replay RECEIPT.json --bundle DIR --attach-outcomes --out NEW.json

Every invocation validates the input receipt first (schema, decision id, fresh policy evaluation,
event chain, projections; and against the bundle: snapshot digest, alternative values, outcome
digest, metrics replay). The explanation printed is derived from the *replayed* result, never
from the stored one. Exit status is 0 when every check passes and 1 otherwise.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from ffai.config import REPO_ROOT
from ffai.decision_lab import bundle, cases, contracts, policy, receipts

DEFAULT_BUNDLE = REPO_ROOT / "artifacts" / "decision_lab"


class ReplayError(RuntimeError):
    """The command could not be carried out; nothing was written."""


def _now_utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReplayError(f"receipt not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReplayError(f"{path} is not valid JSON: {exc}") from exc


def _write(path: Path, doc: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _lookups(b: bundle.Bundle | None) -> tuple[Any, Any]:
    """Snapshot lookups that answer ``None`` for unknown ids instead of raising."""
    if b is None:
        return None, None

    def inputs(sid: str) -> dict[str, Any] | None:
        try:
            return b.inputs_snapshot(sid)
        except bundle.BundleError:
            return None

    def outcomes(sid: str) -> dict[str, Any] | None:
        try:
            return b.outcome_snapshot(sid)
        except bundle.BundleError:
            return None

    return inputs, outcomes


def validate(
    receipt: dict[str, Any], b: bundle.Bundle | None
) -> tuple[list[dict[str, Any]], str | None]:
    """Run every check; returns ``(checks, error)`` where ``error`` is the first failure."""
    inputs_lookup, outcome_lookup = _lookups(b)
    try:
        checks = receipts.validate_receipt(
            receipt, inputs_lookup=inputs_lookup, outcome_lookup=outcome_lookup
        )
        return checks, None
    except receipts.ReceiptError as exc:
        return [], str(exc)
    except (KeyError, TypeError, contracts.ContractError) as exc:
        return [], f"malformed receipt: {type(exc).__name__}: {exc}"


def replayed_result(receipt: dict[str, Any]) -> dict[str, Any]:
    return policy.evaluate(policy.normalize_inputs(receipt["inputs"]))


class Report:
    """Collects everything one invocation reports; printed as a table as it goes, or as one JSON
    document at the end with ``--json``."""

    def __init__(self, as_json: bool):
        self.as_json = as_json
        self.doc: dict[str, Any] = {"ok": True}

    def say(self, text: str) -> None:
        if not self.as_json:
            print(text)

    def checks(self, checks: list[dict[str, Any]], error: str | None) -> None:
        self.doc["checks"] = checks
        self.doc["error"] = error
        self.doc["ok"] = error is None
        if self.as_json:
            return
        width = max([len(c["check"]) for c in checks] + [12])
        for c in checks:
            detail = ", ".join(f"{k}={v}" for k, v in c.items() if k not in ("check", "ok"))
            print(f"  {'ok  ' if c['ok'] else 'FAIL'}  {c['check']:<{width}}  {detail}")
        if error:
            print(f"  FAIL  {error}")

    def explanation(self, receipt: dict[str, Any]) -> dict[str, Any]:
        result = replayed_result(receipt)
        self.doc["replayed"] = {
            "status": result["status"],
            "recommended_player_id": result["recommended_player_id"],
            "explanation": result["explanation"],
            "limitations": result["limitations"],
        }
        if not self.as_json:
            print(
                f"replayed status: {result['status']}  recommended: {result['recommended_player_id']}"
            )
            for line in result["explanation"]:
                print(f"  - {line}")
            if result["limitations"]:
                print("  limitations: " + ", ".join(result["limitations"]))
        return result

    def attached(self, old: dict[str, Any], new: dict[str, Any]) -> None:
        m = new["outcome"]["metrics"]
        summary = {
            "decision_id_unchanged": new["decision_id"] == old["decision_id"],
            "result_sha256_unchanged": new["result_sha256"] == old["result_sha256"],
            "outcome_id": new["outcome"]["outcome_id"],
            "metrics": m,
        }
        self.doc["attached"] = summary
        if not self.as_json:
            print(f"decision_id unchanged: {summary['decision_id_unchanged']}")
            print(f"result_sha256 unchanged: {summary['result_sha256_unchanged']}")
            print(f"outcome_id: {summary['outcome_id']}")
            print("metrics:")
            print(json.dumps(m, indent=1, sort_keys=True))

    def error(self, message: str) -> None:
        self.doc["ok"] = False
        self.doc["error"] = message
        if not self.as_json:
            print(f"ERROR: {message}")

    def flush(self) -> None:
        if self.as_json:
            print(json.dumps(self.doc, indent=1, sort_keys=True))


def new_receipt_for_case(b: bundle.Bundle, case_id: str, *, at_utc: str) -> dict[str, Any]:
    doc = b.cases()
    matches = [c for c in doc["cases"] if c["case_id"] == case_id]
    if not matches:
        raise ReplayError(f"case {case_id} is not in the bundle")
    case = matches[0]
    snapshot = b.inputs_snapshot(case["snapshot_id"])
    try:
        inputs = cases.case_inputs(case, snapshot)
        return receipts.new_receipt(inputs, created_at_utc=at_utc, case_id=case_id)
    except (receipts.ReceiptError, contracts.ContractError) as exc:
        raise ReplayError(str(exc)) from exc


def attach(receipt: dict[str, Any], b: bundle.Bundle, *, at_utc: str) -> dict[str, Any]:
    sid = receipt["inputs"]["snapshot"]["snapshot_id"]
    try:
        osnap = b.outcome_snapshot(sid)
    except bundle.BundleError as exc:
        raise ReplayError(str(exc)) from exc
    if osnap is None:
        raise ReplayError(f"the bundle has no outcome snapshot for {sid}")
    try:
        return receipts.attach_outcome(receipt, osnap, at_utc=at_utc)
    except (receipts.ReceiptError, contracts.ContractError) as exc:
        raise ReplayError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("receipt", nargs="?", type=Path, help="receipt JSON to validate / extend")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--json", action="store_true", help="one JSON document on stdout")
    parser.add_argument("--out", type=Path, help="where to write the new receipt")
    parser.add_argument("--at-utc", default=None, help="event time (default: now, UTC, seconds)")
    parser.add_argument("--new", action="store_true", help="create a receipt for --case")
    parser.add_argument("--case", help="case id from the bundle's cases.json (with --new)")
    parser.add_argument(
        "--record-action", metavar="PLAYER_ID", help="append an action_recorded event"
    )
    parser.add_argument("--kind", choices=["hypothetical_replay", "self_reported_real"])
    parser.add_argument("--note", default=None)
    parser.add_argument("--decline", action="store_true", help="append an action_declined event")
    parser.add_argument("--attach-outcomes", action="store_true")
    args = parser.parse_args(argv)
    at_utc = args.at_utc or _now_utc()
    report = Report(args.json)
    try:
        code = _run(args, at_utc, report)
    except ReplayError as exc:
        report.error(str(exc))
        code = 1
    report.flush()
    return code


def _run(args: argparse.Namespace, at_utc: str, report: Report) -> int:
    b: bundle.Bundle | None = None
    if args.new or args.attach_outcomes or Path(args.bundle).exists():
        try:
            b = bundle.load_bundle(args.bundle)
        except bundle.BundleError as exc:
            if args.new or args.attach_outcomes:
                raise ReplayError(f"cannot load bundle {args.bundle}: {exc}") from exc
            report.say(f"note: bundle at {args.bundle} not loaded ({exc}); bundle checks skipped")

    if args.new:
        if not args.case or not args.out:
            raise ReplayError("--new requires --case and --out")
        assert b is not None
        receipt = new_receipt_for_case(b, args.case, at_utc=at_utc)
        checks, error = validate(receipt, b)
        if error:
            raise ReplayError(f"new receipt failed validation: {error}")
        _write(args.out, receipt)
        report.doc["wrote"] = str(args.out)
        report.doc["decision_id"] = receipt["decision_id"]
        report.say(f"wrote {args.out}  decision_id={receipt['decision_id']}  case={args.case}")
        report.checks(checks, None)
        report.explanation(receipt)
        return 0

    if args.receipt is None:
        raise ReplayError("a receipt path is required unless --new is given")
    receipt = _read(args.receipt)
    checks, error = validate(receipt, b)
    report.checks(checks, error)
    if error:
        return 1
    report.doc["decision_id"] = receipt["decision_id"]
    if b is None:
        report.say("  note: no bundle: snapshot and outcome digests were not checked")

    if not (args.record_action or args.decline or args.attach_outcomes):
        report.explanation(receipt)
        return 0

    if not args.out:
        raise ReplayError("--out is required when writing a new receipt")
    try:
        if args.record_action:
            if not args.kind:
                raise ReplayError("--record-action requires --kind")
            new = receipts.record_action(
                receipt,
                chosen_player_id=args.record_action,
                kind=args.kind,
                at_utc=at_utc,
                note=args.note,
            )
            report.say(f"recorded action: {args.record_action} ({args.kind}) at {at_utc}")
        elif args.decline:
            new = receipts.decline_action(receipt, at_utc=at_utc, note=args.note)
            report.say(f"recorded declined action at {at_utc}")
        else:
            assert b is not None
            new = attach(receipt, b, at_utc=at_utc)
            report.attached(receipt, new)
    except receipts.ReceiptError as exc:
        raise ReplayError(str(exc)) from exc
    _, error = validate(new, b)
    if error:
        raise ReplayError(f"the new receipt failed validation: {error}")
    _write(args.out, new)
    report.doc["wrote"] = str(args.out)
    report.say(f"wrote {args.out}  decision_id={new['decision_id']}")
    report.explanation(new)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
