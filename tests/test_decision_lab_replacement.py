"""Failure-safe replacement of the canonical bundle and its public copy (``exporter.export_bundle``).

Every test injects a failure at one point of the sequence — staging build, staged copy,
replacement after the first swap (each rename in turn), final verification, and the rollback
itself — through the exporter's filesystem seams (``_move``, ``_copytree``,
``_verify_published``, ``build_into``) and then asserts, on real directories under ``tmp_path``:

* the previously published canonical bundle still loads fail-closed and the previous public copy
  still equals it (old evidence remains usable), or, on a first build, both destinations are
  absent again;
* no file of the *new* build sits at either destination and no ``.build-*`` / ``.previous-*``
  sibling remains, so a partial copy can never be staged by the weekly commit;
* ``_last_attempt.json`` records the stage, the restoration report and a truthful
  ``published_state``;
* when the rollback cannot be verified the error says so (``restored=False``) and the CLI exits 2.

The bundle is built once from the committed artifacts; the other tests stage copies of it (the
replacement logic, not the build, is under test).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

import scripts.export_decision_lab as cli
from ffai.config import REPO_ROOT
from ffai.decision_lab import bundle, exporter

ARTIFACTS = REPO_ROOT / "artifacts"
MART = ARTIFACTS / "marts" / "fct_player_week.parquet"
pytestmark = pytest.mark.skipif(not MART.exists(), reason="committed marts are required")

NEW_MARKER = {"marker": "new build"}
OLD_MARKER = {"marker": "previously published"}


@pytest.fixture(scope="module")
def real(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """One real export with a public copy: the first build without prior outputs."""
    root = tmp_path_factory.mktemp("real")
    out, public = root / "decision_lab", root / "site" / "decision-lab"
    exporter.export_bundle(ARTIFACTS, out, public_copy=public)
    return out, public


def _marker(path: Path) -> dict[str, Any]:
    return json.loads((path / bundle.BUILD_NAME).read_text(encoding="utf-8"))


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _siblings(path: Path) -> list[str]:
    return sorted(p.name for p in path.parent.glob(f"{path.name}.*"))


def _previous_published(real: tuple[Path, Path], tmp_path: Path) -> tuple[Path, Path]:
    """A previously published canonical bundle + identical public copy, marked as such."""
    src, _ = real
    out, public = tmp_path / "decision_lab", tmp_path / "site" / "decision-lab"
    shutil.copytree(src, out)
    (out / bundle.BUILD_NAME).write_text(json.dumps(OLD_MARKER), encoding="utf-8")
    shutil.copytree(out, public)
    return out, public


def _fake_build(monkeypatch: pytest.MonkeyPatch, src: Path, *, fail: bool = False) -> None:
    """Stage a copy of the real bundle (marked as the new build) instead of rebuilding."""

    def build(artifacts: Path, out: Path, *, code_revision: str | None) -> dict[str, Any]:
        if fail:
            raise exporter.ExportError("injected staging failure")
        shutil.copytree(src, out)
        (out / bundle.BUILD_NAME).write_text(json.dumps(NEW_MARKER), encoding="utf-8")
        return json.loads((out / bundle.MANIFEST_NAME).read_text(encoding="utf-8"))

    monkeypatch.setattr(exporter, "build_into", build)


def _failing_move(
    monkeypatch: pytest.MonkeyPatch, *, fail_calls: set[int]
) -> list[tuple[str, str]]:
    """``_move`` that raises on the given 1-based call numbers; returns the call log."""
    calls: list[tuple[str, str]] = []
    original = exporter._move

    def move(src: Path, dst: Path) -> None:
        calls.append((src.name, dst.name))
        if len(calls) in fail_calls:
            raise OSError(f"injected rename failure on call {len(calls)}")
        original(src, dst)

    monkeypatch.setattr(exporter, "_move", move)
    return calls


def _assert_previous_intact(out: Path, public: Path, before_out: dict, before_pub: dict) -> None:
    after_out, after_pub = _files(out), _files(public)
    attempt = after_out.pop(bundle.LAST_ATTEMPT_NAME, None)
    assert attempt is not None, "failure diagnostics were not recorded"
    assert after_out == before_out, "the previously published canonical bundle changed"
    assert after_pub == before_pub, "the previously published public copy changed"
    assert _marker(out) == OLD_MARKER and _marker(public) == OLD_MARKER
    bundle.load_bundle(out).verify_all()
    assert exporter.compare_dirs(out, public, ignore={bundle.LAST_ATTEMPT_NAME}) == []
    assert _siblings(out) == [] and _siblings(public) == []


def _attempt(out: Path) -> dict[str, Any]:
    return json.loads((out / bundle.LAST_ATTEMPT_NAME).read_text(encoding="utf-8"))


# --- success paths ------------------------------------------------------------------------------


def test_first_build_without_prior_outputs(real: tuple[Path, Path]) -> None:
    out, public = real
    bundle.load_bundle(out).verify_all()
    assert exporter.compare_dirs(out, public) == []
    assert not (out / bundle.LAST_ATTEMPT_NAME).exists()
    assert _siblings(out) == [] and _siblings(public) == []


def test_successful_replacement_replaces_both_and_removes_parked_versions(
    real: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, public = _previous_published(real, tmp_path)
    (out / bundle.LAST_ATTEMPT_NAME).write_text("{}", encoding="utf-8")  # from an older failure
    _fake_build(monkeypatch, real[0])
    manifest = exporter.export_bundle(ARTIFACTS, out, public_copy=public)
    assert "_verify_report" in manifest
    assert _marker(out) == NEW_MARKER and _marker(public) == NEW_MARKER
    assert not (out / bundle.LAST_ATTEMPT_NAME).exists()
    assert exporter.compare_dirs(out, public) == []
    bundle.load_bundle(out).verify_all()
    assert _siblings(out) == [] and _siblings(public) == []


def test_first_build_failure_leaves_no_destination_behind(
    real: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, public = tmp_path / "decision_lab", tmp_path / "site" / "decision-lab"
    _fake_build(monkeypatch, real[0])
    _failing_move(monkeypatch, fail_calls={2})  # placing the staged public copy fails
    with pytest.raises(exporter.ExportError) as info:
        exporter.export_bundle(ARTIFACTS, out, public_copy=public)
    assert info.value.restored is True and info.value.stage == "replacement"
    assert not public.exists()
    assert sorted(p.name for p in out.iterdir()) == [bundle.LAST_ATTEMPT_NAME]
    assert _attempt(out)["published_state"] == "no_previous_bundle"
    assert _siblings(out) == [] and _siblings(public) == []


# --- injected failures with a previously published bundle ------------------------------------------


@pytest.mark.parametrize("point", ["build", "copy"])
def test_staging_failure_touches_neither_destination(
    real: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    out, public = _previous_published(real, tmp_path)
    before_out, before_pub = _files(out), _files(public)
    _fake_build(monkeypatch, real[0], fail=point == "build")
    if point == "copy":

        def copytree(src: Path, dst: Path) -> None:
            raise OSError("injected copy failure")

        monkeypatch.setattr(exporter, "_copytree", copytree)
    moves = _failing_move(monkeypatch, fail_calls=set())
    with pytest.raises(exporter.ExportError, match="injected") as info:
        exporter.export_bundle(ARTIFACTS, out, public_copy=public)
    assert info.value.restored is True and info.value.stage == "staging"
    assert moves == [], "nothing published may move before staging is complete"
    _assert_previous_intact(out, public, before_out, before_pub)
    note = _attempt(out)
    assert note["stage"] == "staging" and note["restoration"]["status"] == "not_needed"
    assert note["published_state"] == "previous_bundle_preserved"


@pytest.mark.parametrize("failing_call", [1, 2, 3, 4])
def test_replacement_failure_at_each_rename_restores_both(
    real: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_call: int
) -> None:
    """Renames in order: park canonical, place canonical, park public, place public."""
    out, public = _previous_published(real, tmp_path)
    before_out, before_pub = _files(out), _files(public)
    _fake_build(monkeypatch, real[0])
    _failing_move(monkeypatch, fail_calls={failing_call})
    with pytest.raises(exporter.ExportError, match="injected rename failure") as info:
        exporter.export_bundle(ARTIFACTS, out, public_copy=public)
    assert info.value.restored is True and info.value.stage == "replacement"
    _assert_previous_intact(out, public, before_out, before_pub)
    note = _attempt(out)
    assert note["stage"] == "replacement" and note["restoration"]["status"] == "verified"
    assert note["published_state"] == "previous_bundle_preserved"
    assert note["restoration"]["errors"] == []


def test_final_verification_failure_restores_both(
    real: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, public = _previous_published(real, tmp_path)
    before_out, before_pub = _files(out), _files(public)
    _fake_build(monkeypatch, real[0])
    seen: list[tuple[Path, Path | None]] = []

    def verify(o: Path, p: Path | None) -> dict[str, Any]:
        seen.append((o, p))
        assert _marker(o) == NEW_MARKER and p is not None and _marker(p) == NEW_MARKER
        raise exporter.ExportError("injected final verification failure")

    monkeypatch.setattr(exporter, "_verify_published", verify)
    with pytest.raises(exporter.ExportError, match="injected final verification") as info:
        exporter.export_bundle(ARTIFACTS, out, public_copy=public)
    assert seen == [(out, public)], "final verification runs on the published paths"
    assert info.value.restored is True and info.value.stage == "verification"
    _assert_previous_intact(out, public, before_out, before_pub)
    assert _attempt(out)["stage"] == "verification"


def test_unverifiable_rollback_is_reported_and_blocks_publication(
    real: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parking the public copy fails and the rollback renames fail too: the exporter must not
    claim the old bundle is preserved, and the CLI must exit 2."""
    out, public = _previous_published(real, tmp_path)
    _fake_build(monkeypatch, real[0])
    _failing_move(monkeypatch, fail_calls={3, 4, 5})
    with pytest.raises(exporter.ExportError, match="restoration could not be established") as info:
        exporter.export_bundle(ARTIFACTS, out, public_copy=public)
    assert info.value.restored is False
    note = _attempt(out)
    assert note["published_state"] == "unknown"
    assert note["restoration"]["status"] == "failed" and note["restoration"]["errors"]
    # The truthful state: a parked previous version still exists and must be dealt with.
    assert any(name.startswith("decision_lab.previous-") for name in _siblings(out))

    # CLI: exit 2 (publish nothing), not 1 (previous preserved).
    out2, public2 = _previous_published(real, tmp_path / "cli")
    _failing_move(monkeypatch, fail_calls={3, 4, 5})
    code = cli.main(
        ["--artifacts", str(ARTIFACTS), "--out", str(out2), "--public-copy", str(public2)]
    )
    assert code == 2


def test_cli_exit_1_when_previous_is_preserved(
    real: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, public = _previous_published(real, tmp_path)
    before_out, before_pub = _files(out), _files(public)
    _fake_build(monkeypatch, real[0])
    _failing_move(monkeypatch, fail_calls={3})
    code = cli.main(
        ["--artifacts", str(ARTIFACTS), "--out", str(out), "--public-copy", str(public)]
    )
    assert code == 1
    _assert_previous_intact(out, public, before_out, before_pub)


def test_leftover_sibling_from_an_interrupted_export_is_refused(
    real: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, public = _previous_published(real, tmp_path)
    before_out, before_pub = _files(out), _files(public)
    leftover = out.parent / f"{out.name}.previous-{os.getpid()}"
    leftover.mkdir()
    _fake_build(monkeypatch, real[0])
    with pytest.raises(exporter.ExportError, match="leftover directory"):
        exporter.export_bundle(ARTIFACTS, out, public_copy=public)
    assert _files(out) == before_out and _files(public) == before_pub


# --- mirror() on its own ---------------------------------------------------------------------------


def test_mirror_restores_the_previous_copy_when_the_swap_fails(
    real: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src, _ = real
    dest = tmp_path / "copy"
    shutil.copytree(src, dest)
    (dest / bundle.BUILD_NAME).write_text(json.dumps(OLD_MARKER), encoding="utf-8")
    before = _files(dest)
    _failing_move(monkeypatch, fail_calls={2})  # parking succeeds, placing the staged copy fails
    with pytest.raises(exporter.ExportError, match="injected rename failure") as info:
        exporter.mirror(src, dest)
    assert info.value.restored is True
    assert _files(dest) == before and _siblings(dest) == []
    exporter.mirror(src, dest)
    assert exporter.compare_dirs(src, dest) == [] and _siblings(dest) == []
