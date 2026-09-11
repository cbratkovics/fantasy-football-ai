"""Manifest I/O and the deterministic promotion rule."""

from __future__ import annotations

import pytest

from ffai.models import registry


def test_manifest_roundtrip(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    registry.write_manifest({"champion": {"model_version": "v1", "candidate": {"QB": "rf"}}}, path)
    m = registry.read_manifest(path)
    assert m["manifest_version"] == registry.MANIFEST_VERSION
    assert "updated_at_utc" in m
    assert registry.slot(m, "champion") == ("v1", {"QB": "rf"})
    with pytest.raises(KeyError):
        registry.slot(m, "challenger")


def test_missing_manifest_is_a_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        registry.read_manifest(tmp_path / "nope.json")


def test_promote_requires_four_recent_wins_and_frozen_test_win() -> None:
    ok, why = registry.should_promote([5, 5, 5, 5], [4, 4, 4, 4], 5.0, 4.5)
    assert ok and "won all 4" in why
    ok, why = registry.should_promote([5, 5, 5, 5], [4, 4, 6, 4], 5.0, 4.5)
    assert not ok and "3/4" in why
    ok, why = registry.should_promote([5, 5, 5, 5], [4, 4, 4, 4], 4.0, 4.5)
    assert not ok and "frozen-test" in why
    ok, why = registry.should_promote([5, 5], [4, 4], 5.0, 4.5)
    assert not ok and "fewer than 4" in why
    # Only the last four weeks count.
    ok, _ = registry.should_promote([1, 5, 5, 5, 5], [9, 4, 4, 4, 4], 5.0, 4.5)
    assert ok


def test_evaluations_helper_falls_back_to_legacy_eval_id() -> None:
    legacy = {"eval_id": "eval-a"}
    evs = registry.evaluations(legacy)
    assert evs == [
        {"eval_id": "eval-a", "kind": "frozen_test", "season": None, "path": "eval/eval-a.json"}
    ]
    assert registry.evaluations({}) == []


def test_register_evaluation_keeps_frozen_as_legacy_eval_id_and_replaces_by_id() -> None:
    m = {"eval_id": "eval-a"}
    registry.register_evaluation(
        m,
        {
            "eval_id": "eval-a-oos2025",
            "kind": "out_of_sample_season",
            "season": 2025,
            "path": "eval/x.json",
        },
    )
    registry.register_evaluation(
        m, {"eval_id": "eval-a", "kind": "frozen_test", "season": 2024, "path": "eval/eval-a.json"}
    )
    ids = [e["eval_id"] for e in m["evaluations"]]
    assert ids == ["eval-a", "eval-a-oos2025"]  # sorted by season
    assert m["eval_id"] == "eval-a"
    registry.register_evaluation(
        m,
        {
            "eval_id": "eval-a-oos2025",
            "kind": "out_of_sample_season",
            "season": 2025,
            "path": "eval/y.json",
        },
    )
    assert len(m["evaluations"]) == 2 and m["evaluations"][1]["path"] == "eval/y.json"
