"""Fail-closed loader: every inconsistency between the manifest and the files is an error."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ffai.config import REPO_ROOT
from ffai.decision_lab import bundle
from ffai.decision_lab.canonical import content_id, file_sha256

COMMITTED = REPO_ROOT / "artifacts" / "decision_lab"
SYN = "syn-ambiguity-floor-relaxation"

pytestmark = pytest.mark.skipif(
    not (COMMITTED / "manifest.json").exists(), reason="run scripts/export_decision_lab.py"
)


def _dumps(doc) -> str:
    return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


@pytest.fixture
def copy(tmp_path: Path) -> Path:
    dest = tmp_path / "bundle"
    shutil.copytree(COMMITTED, dest)
    return dest


def _manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(root: Path, doc: dict) -> None:
    (root / "manifest.json").write_text(_dumps(doc), encoding="utf-8")


def _entry(doc: dict, sid: str) -> dict:
    return next(s for s in doc["snapshots"] if s["snapshot_id"] == sid)


def test_committed_bundle_loads(copy):
    b = bundle.load_bundle(copy)
    assert SYN in b.snapshot_ids
    snap = b.inputs_snapshot(SYN)
    assert content_id(snap) == b.snapshot_entry(SYN)["inputs"]["content_sha256"]
    assert b.outcome_id(SYN) == content_id(b.outcome_snapshot(SYN))
    assert b.build_info() is not None and b.last_attempt() is None


def test_missing_bundle_and_missing_file(copy, tmp_path):
    with pytest.raises(bundle.BundleError, match="no bundle"):
        bundle.load_bundle(tmp_path / "nowhere")
    (copy / f"inputs/{SYN}.json").unlink()
    with pytest.raises(bundle.BundleError, match="missing"):
        bundle.load_bundle(copy).inputs_snapshot(SYN)


def test_byte_digest_mismatch(copy):
    path = copy / f"inputs/{SYN}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["rows"][0]["display_name"] = "Edited"
    path.write_text(_dumps(doc), encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="byte digest mismatch"):
        bundle.load_bundle(copy).inputs_snapshot(SYN)


def test_content_digest_mismatch_with_matching_bytes(copy):
    # same bytes digest claimed, but the manifest's content digest is wrong
    m = _manifest(copy)
    _entry(m, SYN)["inputs"]["content_sha256"] = "0" * 64
    _write_manifest(copy, m)
    with pytest.raises(bundle.BundleError, match="content digest mismatch"):
        bundle.load_bundle(copy).inputs_snapshot(SYN)


def test_row_count_mismatch(copy):
    m = _manifest(copy)
    _entry(m, SYN)["inputs"]["n_rows"] += 1
    _write_manifest(copy, m)
    with pytest.raises(bundle.BundleError, match="row count"):
        bundle.load_bundle(copy).inputs_snapshot(SYN)


def test_outcome_pointing_at_different_inputs_digest(copy):
    path = copy / f"outcomes/{SYN}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["inputs_content_sha256"] = "f" * 64
    path.write_text(_dumps(doc), encoding="utf-8")
    m = _manifest(copy)
    ref = _entry(m, SYN)["outcomes"]
    ref["file_sha256"] = file_sha256(path)
    ref["content_sha256"] = content_id(doc)
    _write_manifest(copy, m)
    with pytest.raises(bundle.BundleError, match="inputs_content_sha256"):
        bundle.load_bundle(copy).outcome_snapshot(SYN)


def test_observed_count_mismatch(copy):
    m = _manifest(copy)
    _entry(m, SYN)["outcomes"]["n_observed"] = 0
    _write_manifest(copy, m)
    with pytest.raises(bundle.BundleError, match="observed count"):
        bundle.load_bundle(copy).outcome_snapshot(SYN)


def test_wrong_policy_spec_sha(copy):
    m = _manifest(copy)
    m["policy_spec"]["sha256"] = "1" * 64
    _write_manifest(copy, m)
    with pytest.raises(bundle.BundleError, match="different policy_spec"):
        bundle.load_bundle(copy)
    assert bundle.load_bundle(copy, verify_spec=False).snapshot_ids


def test_manifest_contract_violations(copy):
    m = _manifest(copy)
    m["latest_weekly_snapshot_id"] = "weekly-1999-w01-nothing-rf"
    _write_manifest(copy, m)
    with pytest.raises(bundle.BundleError, match="latest_weekly_snapshot_id"):
        bundle.load_bundle(copy)
    m = _manifest(copy)
    m["latest_weekly_snapshot_id"] = None
    m["snapshots"].append(dict(m["snapshots"][0]))
    _write_manifest(copy, m)
    with pytest.raises(bundle.BundleError, match="duplicate snapshot_id"):
        bundle.load_bundle(copy)


def test_snapshot_identity_must_match_manifest(copy):
    m = _manifest(copy)
    entry = _entry(m, SYN)
    entry["mode"] = "historical_replay"
    _write_manifest(copy, m)
    with pytest.raises(bundle.BundleError, match="snapshot identity"):
        bundle.load_bundle(copy).inputs_snapshot(SYN)


def test_cases_referencing_unknown_snapshot(copy):
    path = copy / "cases.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["cases"][0]["snapshot_id"] = "syn-does-not-exist"
    path.write_text(_dumps(doc), encoding="utf-8")
    m = _manifest(copy)
    m["cases"]["file_sha256"] = file_sha256(path)
    m["cases"]["content_sha256"] = content_id(doc)
    _write_manifest(copy, m)
    with pytest.raises(bundle.BundleError, match="unknown snapshot"):
        bundle.load_bundle(copy).cases()
