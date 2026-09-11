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
