"""FastAPI application serving committed artifacts.

Startup reads ``artifacts/manifest.json`` and loads: the champion pipelines per position (to
validate the artifact set and expose versions), the tiers artifact, every predictions file, the
frozen test predictions of the champion, and the evaluation artifact. It fails fast with a clear
error when the manifest is missing. There is no database, cache, auth, payment, or LLM code.

Run locally:  ``uvicorn ffai.serve.app:app --port 7860``
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ffai import __version__
from ffai.config import ARTIFACTS_DIR, POSITIONS
from ffai.models import registry
from ffai.serve import schemas

log = logging.getLogger("ffai.serve")

DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://winmyleague.ai",
    "https://www.winmyleague.ai",
]


class State:
    """Everything loaded at startup."""

    manifest: dict[str, Any]
    model_version: str
    candidate: str | dict[str, str]
    metadata: dict[str, Any]
    pipelines: dict[str, Any]
    tiers: dict[str, Any] | None
    tiers_meta: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    predictions: dict[tuple[int, int], dict[str, Any]]
    frozen_test: pd.DataFrame | None
    loaded_at_utc: str


def load_state(artifacts: Path = ARTIFACTS_DIR) -> State:
    manifest = registry.read_manifest(artifacts / "manifest.json")
    s = State()
    s.manifest = manifest
    s.model_version, s.candidate = registry.slot(manifest, "champion")
    s.metadata = registry.read_model_metadata(s.model_version, artifacts)
    s.pipelines = registry.load_pipelines(s.model_version, s.candidate, artifacts)

    s.tiers, s.tiers_meta = None, None
    tv = manifest.get("tiers_version")
    if tv:
        tdir = artifacts / "tiers" / tv
        s.tiers = json.loads((tdir / "tiers.json").read_text(encoding="utf-8"))
        s.tiers_meta = json.loads((tdir / "metadata.json").read_text(encoding="utf-8"))

    s.evaluation = None
    ev = manifest.get("eval_id")
    if ev:
        s.evaluation = json.loads((artifacts / "eval" / f"{ev}.json").read_text(encoding="utf-8"))

    s.predictions = {}
    pdir = artifacts / "predictions"
    if pdir.exists():
        for f in sorted(pdir.glob("*/week_*.json")):
            payload = json.loads(f.read_text(encoding="utf-8"))
            s.predictions[(int(payload["season"]), int(payload["week"]))] = payload

    s.frozen_test = None
    tp = registry.model_dir(s.model_version, artifacts) / s.metadata.get(
        "test_predictions_path", "test_predictions.csv"
    )
    if tp.exists():
        df = pd.read_csv(tp)
        champ = df.apply(
            lambda r: r["candidate"] == registry.candidate_for(s.candidate, r["position"]), axis=1
        )
        s.frozen_test = df[champ].reset_index(drop=True)

    s.loaded_at_utc = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    return s


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts = Path(os.environ.get("FFAI_ARTIFACTS_DIR", ARTIFACTS_DIR))
    try:
        app.state.s = load_state(artifacts)
    except FileNotFoundError as exc:  # fail fast, clearly
        raise RuntimeError(f"cannot start: {exc}") from exc
    st: State = app.state.s
    log.info(
        "loaded model %s (%s) tiers=%s eval=%s predictions=%d",
        st.model_version,
        st.candidate,
        st.manifest.get("tiers_version"),
        st.manifest.get("eval_id"),
        len(st.predictions),
    )
    yield


app = FastAPI(
    title="ffai — fantasy football projections",
    version=__version__,
    description="Artifact-backed weekly projections. Every response names its model and feature version.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FFAI_CORS_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _state() -> State:
    return app.state.s


def _reception_weight(fmt: str) -> float:
    return {"ppr": 0.0, "half": 0.5, "standard": 1.0}[fmt]


@app.get("/health", response_model=schemas.Health)
def health() -> schemas.Health:
    s = _state()
    return schemas.Health(
        status="ok",
        model_version=s.model_version,
        feature_version=s.metadata["feature_version"],
        tiers_version=s.manifest.get("tiers_version"),
        eval_id=s.manifest.get("eval_id"),
        data_through=s.manifest.get("data_through", s.metadata["data_through"]),
        loaded_at_utc=s.loaded_at_utc,
        positions_loaded=sorted(s.pipelines),
    )


@app.get("/manifest", response_model=schemas.ManifestResponse)
def manifest() -> schemas.ManifestResponse:
    return schemas.ManifestResponse(manifest=_state().manifest)


@app.get("/predictions/{season}/{week}", response_model=schemas.PredictionsResponse)
def predictions(
    season: int,
    week: int,
    scoring: Annotated[schemas.ScoringFormat, Query()] = "ppr",
    position: Annotated[str | None, Query()] = None,
) -> schemas.PredictionsResponse:
    s = _state()
    payload = s.predictions.get((season, week))
    if payload is None:
        available = sorted(s.predictions)
        raise HTTPException(404, f"no predictions for {season} week {week}; available: {available}")
    if position is not None and position not in POSITIONS:
        raise HTTPException(422, f"position must be one of {POSITIONS}")
    w = _reception_weight(scoring)
    out = []
    for r in payload["predictions"]:
        if position and r["position"] != position:
            continue
        adj = w * r["receptions_estimate"]
        out.append(
            schemas.PredictionRecord(
                player_id=r["player_id"],
                name=r.get("name"),
                team=r.get("team"),
                position=r["position"],
                prediction=round(r["prediction"] - adj, 2),
                floor=round(max(0.0, r["floor"] - adj), 2),
                ceiling=round(r["ceiling"] - adj, 2),
                receptions_estimate=r["receptions_estimate"],
                model_version=payload["model_version"],
                candidate=r["candidate"],
                actual=r.get("actual"),
            )
        )
    out.sort(key=lambda r: (-r.prediction, r.player_id))
    return schemas.PredictionsResponse(
        season=season,
        week=week,
        scoring=scoring,
        scoring_derivation=payload["scoring"]["derivation"],
        interval_method=payload["scoring"]["interval"],
        model_version=payload["model_version"],
        feature_version=payload["feature_version"],
        generated_at_utc=payload["generated_at_utc"],
        data_through=payload["data_through"],
        n=len(out),
        predictions=out,
    )


@app.get("/players/{player_id}", response_model=schemas.PlayerResponse)
def player(player_id: str) -> schemas.PlayerResponse:
    s = _state()
    rows: list[schemas.PlayerHistoryRow] = []
    name = team = position = None
    if s.frozen_test is not None:
        ft = s.frozen_test[s.frozen_test["player_id"] == player_id]
        for r in ft.itertuples(index=False):
            name, team, position = r.player_display_name, r.team, r.position
            rows.append(
                schemas.PlayerHistoryRow(
                    season=int(r.season),
                    week=int(r.week),
                    prediction=float(r.prediction),
                    floor=float(r.prediction_floor),
                    ceiling=float(r.prediction_ceiling),
                    actual=float(r.actual),
                    model_version=s.model_version,
                    source="frozen_test",
                )
            )
    for (season, week), payload in sorted(s.predictions.items()):
        for r in payload["predictions"]:
            if r["player_id"] == player_id:
                name, team, position = r.get("name"), r.get("team"), r["position"]
                rows.append(
                    schemas.PlayerHistoryRow(
                        season=season,
                        week=week,
                        prediction=r["prediction"],
                        floor=r["floor"],
                        ceiling=r["ceiling"],
                        actual=r.get("actual"),
                        model_version=payload["model_version"],
                        source="weekly",
                    )
                )
    if not rows:
        raise HTTPException(404, f"no predictions on record for player {player_id}")
    rows.sort(key=lambda r: (r.season, r.week))
    return schemas.PlayerResponse(
        player_id=player_id,
        name=name,
        team=team,
        position=position,
        model_version=s.model_version,
        feature_version=s.metadata["feature_version"],
        generated_at_utc=s.loaded_at_utc,
        history=rows,
    )


@app.get("/tiers/{position}", response_model=schemas.TiersResponse)
def tiers(position: str) -> schemas.TiersResponse:
    s = _state()
    if s.tiers is None or s.tiers_meta is None:
        raise HTTPException(404, "no tiers artifact in manifest")
    if position not in s.tiers["positions"]:
        raise HTTPException(404, f"no tiers for position {position!r}")
    ev = (s.tiers_meta.get("evaluation") or {}).get("positions", {}).get(position)
    return schemas.TiersResponse(
        position=position,
        season=s.tiers["season"],
        tier_version=s.tiers["tier_version"],
        method=s.tiers_meta["method"],
        inputs=s.tiers_meta["inputs"],
        evaluation=ev,
        generated_at_utc=s.tiers_meta["generated_at_utc"],
        tiers=[schemas.Tier(**t) for t in s.tiers["positions"][position]],
    )


@app.get("/performance")
def performance() -> dict[str, Any]:
    s = _state()
    if s.evaluation is None:
        raise HTTPException(404, "no evaluation artifact in manifest")
    return s.evaluation
