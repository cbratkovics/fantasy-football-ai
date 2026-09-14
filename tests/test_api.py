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


def test_root_lists_entry_points(client) -> None:
    body = client.get("/").json()
    assert body["docs"] == "/docs" and body["health"] == "/health"
    assert body["performance"] == "/performance" and body["name"]


def test_performance_lists_every_registered_evaluation(client) -> None:
    body = client.get("/performance").json()
    assert body["model_version"] and body["feature_version"] == "asof_v1"
    evs = body["evaluations"]
    assert evs, "no evaluations"
    kinds = {e["kind"] for e in evs}
    assert "frozen_test" in kinds
    for art in evs:
        assert art["artifact_version"] and art["eval_id"] and art["season"]
        assert art["baseline"]["name"] == "causal_trailing_mean"
        assert set(art["cohorts"]) == set(POSITIONS)
        assert "within_3_rate" in art["metric_definitions"]
        assert art["metrics"]["mae"] > 3.0, "MAE below 3 would suggest leakage"
        assert not art["input"]["path"].startswith("/"), "public artifact carries a local path"
    health = client.get("/health").json()
    assert set(health["evaluations"]) == {e["eval_id"] for e in evs}
    if len(evs) > 1:
        assert "out_of_sample_season" in kinds


def test_performance_by_eval_id_keeps_the_single_artifact_shape(client) -> None:
    evs = client.get("/performance").json()["evaluations"]
    for art in evs:
        one = client.get(f"/performance/{art['eval_id']}").json()
        assert one["eval_id"] == art["eval_id"] and one["metrics"] == art["metrics"]
    assert client.get("/performance/does-not-exist").status_code == 404


# --- gold marts (artifacts/marts/*.parquet exported by the dbt build; ADR-0018) ---

marts_present = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "marts" / "_export_manifest.json").exists(), reason="no exported marts"
)


@marts_present
def test_root_and_health_advertise_the_marts(client) -> None:
    assert client.get("/").json()["marts"] == "/marts/weekly_eval"
    assert client.get("/health").json()["marts_exported_at_utc"]


@marts_present
def test_marts_weekly_eval_reconciles_with_the_evaluation_artifacts(client) -> None:
    """The API-level restatement of the dbt reconciliation test: the n-weighted MAE and
    within-3 over a window's weeks equal the published artifact to 1e-4."""
    evs = client.get("/performance").json()["evaluations"]
    for art in evs:
        cand = art["model"]["candidate"]["QB"]
        body = client.get(
            f"/marts/weekly_eval?season={art['season']}&candidate={cand}&model_version={art['model']['version']}"
        ).json()
        assert body["source"] == "gold marts exported by the weekly build"
        assert (
            body["mart"] == "fct_weekly_eval"
            and body["export"]["row_counts"]["fct_weekly_eval"] > 0
        )
        rows = [r for r in body["rows"] if r["eval_window"] == f"{art['kind']}:{art['season']}"]
        assert rows, "no mart rows for the evaluation window"
        n = sum(r["n"] for r in rows)
        mae = sum(r["mae"] * r["n"] for r in rows) / n
        w3 = sum(r["within_3_rate"] * r["n"] for r in rows) / n
        assert n == art["metrics"]["n"]
        assert mae == pytest.approx(art["metrics"]["mae"], abs=1e-4)
        assert w3 == pytest.approx(art["metrics"]["within_3_rate"], abs=1e-4)
    assert client.get("/marts/weekly_eval?cohort=K").status_code == 422


@marts_present
def test_marts_player_week_history(client) -> None:
    art = client.get("/performance").json()["evaluations"][0]
    # any player from the frozen test: take the first row of fct_weekly_eval's season via /players
    m = client.get("/manifest").json()["manifest"]
    latest = m["predictions"]["latest"]
    season, week = latest.split("/")[1], int(latest.rsplit("_", 1)[1].split(".")[0])
    first = client.get(f"/predictions/{season}/{week}").json()["predictions"][0]
    r = client.get(f"/marts/player_week/{first['player_id']}")
    assert r.status_code == 200
    body = schemas.PlayerWeekResponse.model_validate(r.json())
    assert body.mart == "fct_player_week" and body.n == len(body.rows) > 0
    keys = [(x.season, x.week) for x in body.rows]
    assert keys == sorted(keys)
    for x in body.rows:
        if x.actual is not None:
            assert x.abs_error == pytest.approx(abs(x.actual - x.prediction), abs=1e-9)
            assert x.within_3 == (x.abs_error <= 3)
            assert x.actual_source in ("stats", "artifact")
    both = client.get(f"/marts/player_week/{first['player_id']}?candidate=all").json()
    assert both["n"] >= body.n and both["candidate"] == "all"
    assert client.get("/marts/player_week/does-not-exist").status_code == 404
    assert art["eval_id"]  # the artifact is still what /performance serves


@marts_present
def test_marts_decisions_sweep(client) -> None:
    r = client.get("/marts/decisions")
    assert r.status_code == 200
    body = schemas.DecisionsResponse.model_validate(r.json())
    assert body.min_floor == 6.0 and 6.0 in body.available_min_floors
    assert body.n == len(body.rows) > 0 and body.rows[0].min_floor == 6.0
    cohorts = [s.cohort for s in body.summary]
    assert cohorts[0] == "ALL" and set(cohorts[1:]) == set(POSITIONS)
    overall = body.summary[0]
    assert overall.eligible_decisions == sum(r.eligible_decisions for r in body.rows)
    assert overall.recommendations == sum(r.recommendations for r in body.rows)
    for s in body.summary:
        if s.hit_rate is not None:
            assert 0 <= s.hit_rate <= 1 and 0 <= (s.downside_rate or 0) <= 1
    lowest = client.get(f"/marts/decisions?min_floor={body.available_min_floors[0]}").json()
    assert lowest["summary"][0]["recommendation_rate"] == pytest.approx(1.0)
    qb = client.get("/marts/decisions?min_floor=6&position=QB&candidate=rf").json()
    assert qb["rows"] and all(r["position"] == "QB" for r in qb["rows"])
    assert client.get("/marts/decisions?min_floor=6.37").status_code == 422
    assert client.get("/marts/decisions?position=K").status_code == 422
