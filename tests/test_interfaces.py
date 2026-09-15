"""The seams (ffai.interfaces) are satisfied by this repository's implementations, and every
committed artifact validates against its JSON Schema under artifacts/schemas/ (ADR-0028)."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest

from ffai import interfaces, scoring
from ffai.config import ARTIFACTS_DIR, PROJECT, SCHEMAS_DIR
from ffai.data import nflverse
from ffai.eval import drift
from ffai.features import asof

SCHEMAS = {p.stem.removesuffix(".schema"): p for p in SCHEMAS_DIR.glob("*.schema.json")}


def _schema(name: str) -> dict:
    return json.loads(SCHEMAS[name].read_text(encoding="utf-8"))


def test_every_schema_is_itself_valid() -> None:
    assert set(SCHEMAS) == {
        "eval_artifact",
        "manifest",
        "model_metadata",
        "predictions_file",
        "drift_report",
    }
    for name in SCHEMAS:
        jsonschema.Draft202012Validator.check_schema(_schema(name))


def test_nflverse_loader_satisfies_source_loader() -> None:
    loader = nflverse.LOADER
    assert isinstance(loader, interfaces.SourceLoader)
    assert loader.LIBRARY.startswith(("nflreadpy==", "nfl_data_py=="))
    assert PROJECT.entity_key in loader.ID_COLUMNS
    assert set(PROJECT.period_columns) <= set(loader.ID_COLUMNS)
    assert PROJECT.target_column in loader.STAT_COLUMNS
    assert loader.periods_in_season(2019) == 17 and loader.periods_in_season(2024) == 18
    assert loader.cache_path_for("player_stats", [2019, 2024]).name.startswith(
        "player_stats_2019-2024_"
    )


def test_target_spec_derives_and_reconciles(stats: pd.DataFrame) -> None:
    spec = scoring.TARGET_SPEC
    assert isinstance(spec, interfaces.TargetSpec)
    assert spec.column == PROJECT.target_column and spec.format in spec.formats
    derived = spec.derive(stats, spec.format)
    assert len(derived) == len(stats)
    np.testing.assert_allclose(derived.to_numpy(), stats[spec.column].to_numpy(), atol=0.01)
    assert spec.reconcile(stats).empty


def test_asof_satisfies_feature_module() -> None:
    assert isinstance(asof, interfaces.FeatureModule)
    assert asof.FEATURE_VERSION == "asof_v1"
    assert tuple(asof.KEY_COLUMNS) == (PROJECT.entity_key, *PROJECT.period_columns)
    assert set(asof.features_for_position(PROJECT.cohorts[0])) <= set(asof.all_feature_names())


@pytest.mark.parametrize("path", sorted((ARTIFACTS_DIR / "eval").glob("eval-*.json")))
def test_committed_eval_artifacts_validate(path: Path) -> None:
    art = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.validate(art, _schema("eval_artifact"))
    assert set(art["cohorts"]) == set(PROJECT.cohorts)
    assert not art["input"]["path"].startswith("/")


def test_committed_manifest_validates() -> None:
    jsonschema.validate(
        json.loads((ARTIFACTS_DIR / "manifest.json").read_text()), _schema("manifest")
    )


@pytest.mark.parametrize("path", sorted((ARTIFACTS_DIR / "models").glob("*/metadata.json")))
def test_committed_model_metadata_validates(path: Path) -> None:
    meta = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.validate(meta, _schema("model_metadata"))
    assert set(meta["positions"]) == set(PROJECT.cohorts)
    assert set(meta["candidates"]) == set(PROJECT.candidates)


@pytest.mark.parametrize("path", sorted((ARTIFACTS_DIR / "predictions").glob("*/week_*.json")))
def test_committed_prediction_files_validate(path: Path) -> None:
    jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), _schema("predictions_file"))


def test_drift_run_report_matches_its_schema() -> None:
    rng = np.random.default_rng(0)
    ref = {"f1": [float(x) for x in np.quantile(rng.normal(size=500), np.linspace(0, 1, 11))]}
    report = drift.drift_report(pd.DataFrame({"f1": rng.normal(size=200)}), ref, ["f1"])
    schema = _schema("drift_report")
    jsonschema.validate(report, {"$ref": "#/$defs/position_report", "$defs": schema["$defs"]})
    run = drift.run_report(
        run_id="run-20260101T000000Z",
        at_utc="2026-01-01T00:00:00+00:00",
        season=2026,
        week=2,
        model_version="m",
        status=report["status"],
        positions={"QB": report},
    )
    jsonschema.validate(run, schema)
    assert run["drift_report_version"] == drift.REPORT_VERSION


@pytest.mark.parametrize("path", sorted((ARTIFACTS_DIR / "drift").glob("run-*.json")))
def test_committed_drift_reports_validate(path: Path) -> None:
    jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), _schema("drift_report"))
