"""Optional artifact routing must never fetch providers or break forecast serving."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from ffai.context import build_snapshot
from ffai.serve.app import app


def test_missing_optional_context_is_named_404(tmp_path, monkeypatch):
    monkeypatch.setenv("FFAI_ARTIFACTS_DIR", str(tmp_path))
    response = TestClient(app).get("/context")
    assert response.status_code == 404
    assert response.json() == {"detail": {"reason": "context_unavailable"}}


def test_invalid_optional_context_is_named_503(tmp_path, monkeypatch):
    monkeypatch.setenv("FFAI_ARTIFACTS_DIR", str(tmp_path))
    dest = tmp_path / "context"
    dest.mkdir()
    (dest / "snapshot.json").write_text('{"schema_version":"invalid"}')
    response = TestClient(app).get("/context")
    assert response.status_code == 503
    assert response.json() == {"detail": {"reason": "context_invalid"}}


def test_verified_integrity_does_not_promote_fixture_to_live(tmp_path, monkeypatch):
    monkeypatch.setenv("FFAI_ARTIFACTS_DIR", str(tmp_path))
    sources = json.loads((Path(__file__).parent / "fixtures/context/sources.json").read_text())
    snapshot = build_snapshot(
        sources["sources"],
        season=2026,
        week=9,
        as_of="2026-10-31T13:00:00Z",
        observed_at="2026-10-31T13:00:00Z",
        mode="fixture",
    )
    dest = tmp_path / "context"
    dest.mkdir()
    (dest / "snapshot.json").write_text(json.dumps(snapshot))
    response = TestClient(app).get("/context")
    assert response.status_code == 200
    assert response.json()["context_snapshot_id"] == snapshot["context_snapshot_id"]
    assert response.json()["sources"]["rosters"]["state"] == "fixture"
    assert all(p["forecast"]["state"] != "present" for p in response.json()["players"])
