"""Replay CLI round trip (subprocess) on the ambiguity case plus receipt merge semantics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ffai.config import REPO_ROOT
from ffai.decision_lab import bundle, receipts

COMMITTED = REPO_ROOT / "artifacts" / "decision_lab"
CASE = "syn-ambiguity-floor-relaxation"
T0, T1, T2 = (
    "2099-01-01T00:00:00+00:00",
    "2099-01-01T00:01:00+00:00",
    "2099-01-01T00:02:00+00:00",
)

pytestmark = pytest.mark.skipif(
    not (COMMITTED / "manifest.json").exists(), reason="run scripts/export_decision_lab.py"
)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ffai.decision_lab.replay", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def test_cli_round_trip(tmp_path):
    r0, r1, r2 = tmp_path / "r0.json", tmp_path / "r1.json", tmp_path / "r2.json"
    new = run("--new", "--case", CASE, "--bundle", str(COMMITTED), "--out", str(r0), "--at-utc", T0)
    assert new.returncode == 0, new.stdout + new.stderr
    assert "snapshot_digest" in new.stdout and "replayed status: review" in new.stdout
    d0 = _load(r0)
    assert d0["case_id"] == CASE and d0["action"]["state"] == "not_recorded"

    rec = run(
        str(r0),
        "--bundle",
        str(COMMITTED),
        "--record-action",
        "SYN-A",
        "--kind",
        "hypothetical_replay",
        "--out",
        str(r1),
        "--at-utc",
        T1,
    )
    assert rec.returncode == 0, rec.stdout + rec.stderr
    d1 = _load(r1)
    assert d1["decision_id"] == d0["decision_id"]
    assert d1["action"] == {
        "state": "recorded",
        "action_id": d1["events"][0]["event_id"],
        "at_utc": T1,
        "chosen_player_id": "SYN-A",
        "kind": "hypothetical_replay",
        "note": None,
    }

    att = run(
        str(r1),
        "--bundle",
        str(COMMITTED),
        "--attach-outcomes",
        "--out",
        str(r2),
        "--at-utc",
        T2,
        "--json",
    )
    assert att.returncode == 0, att.stdout + att.stderr
    payload = json.loads(att.stdout)
    assert payload["attached"]["decision_id_unchanged"] is True
    assert payload["attached"]["result_sha256_unchanged"] is True
    assert payload["attached"]["metrics"]["choice_set_regret"] == 8.0
    assert payload["attached"]["metrics"]["points_vs_baseline_choice"] == -8.0
    d2 = _load(r2)
    assert d2["decision_id"] == d0["decision_id"] and d2["result_sha256"] == d0["result_sha256"]
    assert d2["outcome"]["state"] == "attached"
    assert d2["outcome"]["outcome_id"] == bundle.load_bundle(COMMITTED).outcome_id(CASE)

    val = run(str(r2), "--bundle", str(COMMITTED))
    assert val.returncode == 0, val.stdout + val.stderr
    assert "metrics_replay" in val.stdout and "FAIL" not in val.stdout

    val_json = run(str(r2), "--bundle", str(COMMITTED), "--json")
    assert val_json.returncode == 0
    checks = json.loads(val_json.stdout)
    assert checks["ok"] is True
    assert [c["check"] for c in checks["checks"]] == [
        "schema",
        "decision_id",
        "result_replay",
        "events",
        "projections",
        "snapshot_digest",
        "metrics_replay",
    ]

    # tampering the stored result is detected
    tampered = tmp_path / "tampered.json"
    doc = _load(r2)
    doc["result"]["status"] = "recommend"
    doc["result"]["recommended_player_id"] = "SYN-A"
    _dump(tampered, doc)
    bad = run(str(tampered), "--bundle", str(COMMITTED))
    assert bad.returncode == 1
    assert "result_replay" in bad.stdout or "result_sha256" in bad.stdout

    # tampering an input value against the bundle is detected too
    tampered2 = tmp_path / "tampered2.json"
    doc = _load(r1)
    doc["inputs"]["alternatives"][0]["projection"] = 99.0
    _dump(tampered2, doc)
    bad2 = run(str(tampered2), "--bundle", str(COMMITTED))
    assert bad2.returncode == 1 and "decision_id" in bad2.stdout

    # a second action is refused
    dup = run(str(r1), "--bundle", str(COMMITTED), "--decline", "--out", str(tmp_path / "x.json"))
    assert dup.returncode == 1 and "already recorded" in dup.stdout


def test_attach_from_a_different_snapshot_is_refused(tmp_path):
    b = bundle.load_bundle(COMMITTED)
    r0 = tmp_path / "r0.json"
    assert (
        run(
            "--new", "--case", CASE, "--bundle", str(COMMITTED), "--out", str(r0), "--at-utc", T0
        ).returncode
        == 0
    )
    receipt = _load(r0)
    other = b.outcome_snapshot("syn-missing-and-negative-outcomes")
    assert other is not None
    with pytest.raises(receipts.ReceiptError, match="is not syn-ambiguity-floor-relaxation"):
        receipts.attach_outcome(receipt, other, at_utc=T2)
    # via the CLI: a case whose snapshot has no outcomes cannot attach
    tie = tmp_path / "tie.json"
    assert (
        run(
            "--new",
            "--case",
            "syn-exact-tie",
            "--bundle",
            str(COMMITTED),
            "--out",
            str(tie),
            "--at-utc",
            T0,
        ).returncode
        == 0
    )
    res = run(
        str(tie),
        "--bundle",
        str(COMMITTED),
        "--attach-outcomes",
        "--out",
        str(tmp_path / "tie2.json"),
    )
    assert res.returncode == 1 and "no outcome snapshot" in res.stdout
    assert not (tmp_path / "tie2.json").exists()
    # a receipt whose outcome event names another snapshot fails validation
    forged = receipts.attach_outcome(receipt, b.outcome_snapshot(CASE), at_utc=T2)
    forged["events"][0]["payload"]["outcome_snapshot_id"] = "syn-missing-and-negative-outcomes"
    forged["events"][0]["event_id"] = receipts.event_id(
        forged["decision_id"], 1, "outcome_attached", T2, forged["events"][0]["payload"]
    )
    forged["action"], forged["outcome"] = receipts.derive_state(forged)
    path = tmp_path / "forged.json"
    _dump(path, forged)
    res = run(str(path), "--bundle", str(COMMITTED))
    assert res.returncode == 1 and "outcome_snapshot" in res.stdout


def test_duplicate_import_merge_is_idempotent(tmp_path):
    r0, r1 = tmp_path / "r0.json", tmp_path / "r1.json"
    assert (
        run(
            "--new", "--case", CASE, "--bundle", str(COMMITTED), "--out", str(r0), "--at-utc", T0
        ).returncode
        == 0
    )
    assert (
        run(
            str(r0),
            "--bundle",
            str(COMMITTED),
            "--record-action",
            "SYN-B",
            "--kind",
            "self_reported_real",
            "--out",
            str(r1),
            "--at-utc",
            T1,
        ).returncode
        == 0
    )
    d0, d1 = _load(r0), _load(r1)
    assert receipts.merge(d1, d1) == d1
    assert receipts.merge(d0, d1) == d1
    assert receipts.merge(d1, d0) == d1
    other = receipts.record_action(
        d0, chosen_player_id="SYN-A", kind="hypothetical_replay", at_utc=T1
    )
    with pytest.raises(receipts.ConflictError):
        receipts.merge(d1, other)


def test_missing_bundle_and_unknown_case(tmp_path):
    res = run(
        "--new", "--case", "nope", "--bundle", str(COMMITTED), "--out", str(tmp_path / "x.json")
    )
    assert res.returncode == 1 and "not in the bundle" in res.stdout
    res = run(
        "--new",
        "--case",
        CASE,
        "--bundle",
        str(tmp_path / "missing"),
        "--out",
        str(tmp_path / "x.json"),
    )
    assert res.returncode == 1 and "cannot load bundle" in res.stdout
