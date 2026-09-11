"""API contract tests against the committed artifacts (no network, no database)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ffai.config import ARTIFACTS_DIR, POSITIONS
from ffai.serve import app as app_module
from ffai.serve import schemas

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "manifest.json").exists(), reason="no committed manifest"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as c:
        yield c


def test_health_reports_versions(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = schemas.Health.model_validate(r.json())
    assert body.status == "ok"
    assert body.model_version and body.feature_version == "asof_v1"
    assert sorted(body.positions_loaded) == sorted(POSITIONS)
    assert set(body.data_through) == {"season", "week"}


def test_manifest_has_champion_and_eval(client) -> None:
    m = client.get("/manifest").json()["manifest"]
    assert m["champion"]["model_version"]
    assert m["eval_id"]
    assert m["manifest_version"]


def test_predictions_latest_week_all_formats(client) -> None:
    m = client.get("/manifest").json()["manifest"]
    latest = m["predictions"]["latest"]
    assert latest, "manifest has no latest predictions file"
    season, week = latest.split("/")[1], int(latest.rsplit("_", 1)[1].split(".")[0])
    ppr = client.get(f"/predictions/{season}/{week}")
    assert ppr.status_code == 200
    body = schemas.PredictionsResponse.model_validate(ppr.json())
    assert body.n > 100 and body.scoring == "ppr"
    assert all(p.floor <= p.prediction <= p.ceiling for p in body.predictions)
    assert all(p.model_version == body.model_version for p in body.predictions)
    std = schemas.PredictionsResponse.model_validate(
        client.get(f"/predictions/{season}/{week}?scoring=standard").json()
    )
    half = schemas.PredictionsResponse.model_validate(
        client.get(f"/predictions/{season}/{week}?scoring=half").json()
    )
    by_id = {p.player_id: p for p in body.predictions}
    half_by_id = {p.player_id: p for p in half.predictions}
    assert set(by_id) == set(half_by_id) == {p.player_id for p in std.predictions}
    for s in std.predictions:
        p, h = by_id[s.player_id], half_by_id[s.player_id]
        assert s.prediction == pytest.approx(p.prediction - p.receptions_estimate, abs=0.011)
        assert h.prediction == pytest.approx(p.prediction - 0.5 * p.receptions_estimate, abs=0.011)
    # Each list is sorted by the prediction under its own scoring format.
    for lst in (body, std, half):
        vals = [p.prediction for p in lst.predictions]
        assert vals == sorted(vals, reverse=True)
    qb = client.get(f"/predictions/{season}/{week}?position=QB").json()
    assert qb["n"] > 0 and all(p["position"] == "QB" for p in qb["predictions"])
    assert client.get(f"/predictions/{season}/{week}?position=K").status_code == 422
    assert client.get("/predictions/1999/1").status_code == 404


def test_player_history(client) -> None:
    m = client.get("/manifest").json()["manifest"]
    latest = m["predictions"]["latest"]
    season, week = latest.split("/")[1], int(latest.rsplit("_", 1)[1].split(".")[0])
    first = client.get(f"/predictions/{season}/{week}").json()["predictions"][0]
    r = client.get(f"/players/{first['player_id']}")
    assert r.status_code == 200
    body = schemas.PlayerResponse.model_validate(r.json())
    assert body.history and body.history == sorted(body.history, key=lambda h: (h.season, h.week))
    assert any(h.source == "weekly" for h in body.history)
    assert client.get("/players/does-not-exist").status_code == 404


def test_tiers_per_position(client) -> None:
    for pos in POSITIONS:
        r = client.get(f"/tiers/{pos}")
        assert r.status_code == 200
        body = schemas.TiersResponse.model_validate(r.json())
        assert body.position == pos and body.tiers and body.tiers[0].tier == 1
        assert body.evaluation is None or "spearman" in body.evaluation
    assert client.get("/tiers/K").status_code == 404


def test_performance_is_the_eval_artifact(client) -> None:
    body = client.get("/performance").json()
    assert body["artifact_version"] and body["metrics"]["mae"] > 0
    assert body["baseline"]["name"] == "causal_trailing_mean"
    assert set(body["cohorts"]) == set(POSITIONS)
    assert "within_3_rate" in body["metric_definitions"]
    assert body["metrics"]["mae"] > 3.0, "MAE below 3 would suggest leakage"
