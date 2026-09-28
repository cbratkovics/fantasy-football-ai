"""scripts/lab_review_notes.py writes private review notes only outside the repository."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import types

import pytest

from ffai.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "lab_review_notes.py"
FIXED = dt.datetime(2026, 9, 27, 12, 34, 56)
FORBIDDEN_WORDS = ("interview", "résumé", "resume", "career", "coaching")


def _load():
    spec = importlib.util.spec_from_file_location("lab_review_notes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


notes = _load()


@pytest.fixture
def frozen_clock(monkeypatch):
    class FakeDatetime:
        @staticmethod
        def now():
            return FIXED

    monkeypatch.setattr(notes, "dt", types.SimpleNamespace(datetime=FakeDatetime))


def test_refuses_a_directory_inside_the_repository(tmp_path, capsys) -> None:
    inside = REPO_ROOT / "tmp-private-notes-should-not-exist"
    assert not inside.exists()
    code = notes.main(["--case-id", "case-a", "--out-dir", str(inside)])
    assert code == 2
    assert not inside.exists()
    assert "refusing" in capsys.readouterr().err
    assert notes.refusal_reason(REPO_ROOT / "docs", notes.protected_roots()) is not None
    assert notes.refusal_reason(tmp_path, notes.protected_roots()) is None


def test_writes_the_template_outside_with_blank_prompts(tmp_path, frozen_clock) -> None:
    out_dir = tmp_path / "notes"
    code = notes.main(["--case-id", "case-a", "--out-dir", str(out_dir), "--revision", "abc1234"])
    assert code == 0
    written = list(out_dir.glob("*.md"))
    assert [p.name for p in written] == ["20260927-123456-case-a.md"]
    text = written[0].read_text(encoding="utf-8")
    for section in notes.SECTIONS:
        assert f"## {section}" in text
    assert "Revision: `abc1234`" in text
    assert text.splitlines().count(notes.BLANK_PROMPT) == len(notes.SECTIONS)
    lower = text.lower()
    for word in FORBIDDEN_WORDS:
        assert word not in lower
    assert str(REPO_ROOT) not in text


def test_embeds_the_log_verbatim_and_the_command(tmp_path, frozen_clock) -> None:
    log = tmp_path / "run.log"
    log_text = "line one\n  indented ``` with ticks\nline three\n"
    log.write_text(log_text, encoding="utf-8")
    out_dir = tmp_path / "notes"
    code = notes.main(
        [
            "--case-id",
            "case-b",
            "--out-dir",
            str(out_dir),
            "--revision",
            "abc1234",
            "--command",
            "make lab-replay RECEIPT=r.json",
            "--log",
            str(log),
        ]
    )
    assert code == 0
    text = next(out_dir.glob("*.md")).read_text(encoding="utf-8")
    assert log_text.rstrip() in text
    assert "make lab-replay RECEIPT=r.json" in text
    assert "````" in text  # a longer fence because the log itself contains ```
    assert str(REPO_ROOT) not in text


def test_includes_the_receipt_summary_without_replaying(tmp_path, frozen_clock) -> None:
    receipt = {
        "decision_id": "f" * 64,
        "result": {"status": "recommend", "recommended_player_id": "00-0012345"},
        "action": {"state": "recorded"},
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    out_dir = tmp_path / "notes"
    code = notes.main(
        [
            "--case-id",
            "case-c",
            "--out-dir",
            str(out_dir),
            "--revision",
            "abc",
            "--receipt",
            str(path),
        ]
    )
    assert code == 0
    text = next(out_dir.glob("*.md")).read_text(encoding="utf-8")
    assert f"decision_id: `{'f' * 64}`" in text
    assert "status: recommend" in text
    assert "recommended id: 00-0012345" in text
    assert "action state: recorded" in text
    assert "not replayed" in text


def test_never_overwrites_an_existing_file(tmp_path, frozen_clock) -> None:
    out_dir = tmp_path / "notes"
    args = ["--case-id", "case-d", "--out-dir", str(out_dir), "--revision", "abc"]
    assert notes.main(args) == 0
    first = out_dir / "20260927-123456-case-d.md"
    first.write_text("keep me\n", encoding="utf-8")
    assert notes.main(args) == 0
    assert notes.main(args) == 0
    assert first.read_text(encoding="utf-8") == "keep me\n"
    assert sorted(p.name for p in out_dir.glob("*.md")) == [
        "20260927-123456-case-d-2.md",
        "20260927-123456-case-d-3.md",
        "20260927-123456-case-d.md",
    ]


def test_case_id_is_sanitised_for_the_file_name(tmp_path) -> None:
    path = notes.unique_path(tmp_path, FIXED, "../weird case/id")
    assert path.parent == tmp_path
    assert path.name == "20260927-123456-weird-case-id.md"
