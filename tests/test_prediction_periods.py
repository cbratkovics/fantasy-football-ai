"""The period index is derived only from loaded artifacts, not the current clock."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from ffai.serve import app as api


def test_loaded_period_index_has_no_speculative_weeks(monkeypatch):
    def payload(version, n):
        return {
            "model_version": version,
            "feature_version": "f",
            "generated_at_utc": "2026-10-27T12:00:00Z",
            "predictions": [{}] * n,
        }

    monkeypatch.setattr(
        api,
        "_state",
        lambda: SimpleNamespace(
            predictions={
                (2024, 17): payload("historical", 2),
                (2026, 3): payload("older", 3),
                (2026, 8): payload("latest", 4),
            }
        ),
    )
    body = TestClient(api.app).get("/prediction-periods").json()
    assert [(p["season"], p["week"]) for p in body["periods"]] == [(2026, 8), (2026, 3), (2024, 17)]
    assert body["latest"]["model_version"] == "latest"
    assert body["latest"]["n"] == 4


def test_no_loaded_predictions_returns_explicit_empty_index(monkeypatch):
    monkeypatch.setattr(api, "_state", lambda: SimpleNamespace(predictions={}))
    assert TestClient(api.app).get("/prediction-periods").json() == {"latest": None, "periods": []}
