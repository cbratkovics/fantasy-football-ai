"""Train per-position champion / challenger candidates and write a versioned model artifact.

Design (frozen so results stay comparable with the legacy 2025-07-31 metadata):

* seasons: train 2019-2022, validation 2023, test 2024 (``ffai.config``)
* candidates per position: RandomForest with the legacy hyper-parameters (``rf``) and
  XGBoost (``xgb``); each is an sklearn ``Pipeline(StandardScaler, model)`` fit on a DataFrame so
  the pipeline carries ``feature_names_in_``
* the candidate with the lower validation MAE is the champion for that position, the other is
  the challenger; both are persisted
* intervals: validation residual quantiles (10th / 90th) per position and candidate, applied as
  ``prediction + q10`` / ``prediction + q90`` at serving time (floor clipped at 0)
* drift reference: decile edges of every feature on the training rows, plus top-10 feature
  importances per position (used by ``ffai.eval.drift``)

Everything the evaluator and the model card need is recorded in ``metadata.json``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ffai import __version__ as ffai_version
from ffai.config import (
    ARTIFACTS_DIR,
    POSITIONS,
    RANDOM_STATE,
    TEST_SEASON,
    TRAIN_SEASONS,
    VAL_SEASON,
)
from ffai.data import nflverse
from ffai.features import asof

CANDIDATES: tuple[str, ...] = ("rf", "xgb")
INTERVAL_QUANTILES: tuple[float, float] = (0.10, 0.90)
N_DECILES = 10
TOP_IMPORTANCE = 10

RF_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
# Legacy challenger hyper-parameters, unchanged.
XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def input_sha256(stats: pd.DataFrame) -> str:
    """Order-independent content hash of the weekly stats frame."""
    keyed = stats.sort_values(list(asof.KEY_COLUMNS)).reset_index(drop=True)
    cols = list(asof.KEY_COLUMNS) + [c for c in nflverse.STAT_COLUMNS if c in keyed.columns]
    h = pd.util.hash_pandas_object(keyed[cols], index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()


def make_candidate(name: str) -> Pipeline:
    if name == "rf":
        model = RandomForestRegressor(**RF_PARAMS)
    elif name == "xgb":
        import xgboost as xgb

        model = xgb.XGBRegressor(**XGB_PARAMS)
    else:
        raise ValueError(f"unknown candidate {name!r}")
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def split_by_season(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": frame[frame["season"].isin(TRAIN_SEASONS)],
        "val": frame[frame["season"] == VAL_SEASON],
        "test": frame[frame["season"] == TEST_SEASON],
    }


def _importances(pipe: Pipeline, features: list[str]) -> dict[str, float]:
    model = pipe.named_steps["model"]
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    order = np.argsort(imp)[::-1][:TOP_IMPORTANCE]
    return {features[i]: float(imp[i]) for i in order}


def _deciles(frame: pd.DataFrame, features: list[str]) -> dict[str, list[float]]:
    qs = np.linspace(0, 1, N_DECILES + 1)
    return {f: [float(v) for v in np.quantile(frame[f].to_numpy(), qs)] for f in features}


def _model_version(feature_version: str, sha: str, now: dt.datetime) -> str:
    return f"{now:%Y%m%d}-{feature_version}-{sha[:8]}"


def fit_position(
    position: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    candidates: tuple[str, ...] = CANDIDATES,
) -> dict[str, Any]:
    """Fit every candidate for one position. Returns pipelines and validation diagnostics."""
    feats = asof.features_for_position(position)
    x_tr, y_tr = train[feats], train["target"]
    x_va, y_va = val[feats], val["target"]
    out: dict[str, Any] = {"features": feats, "pipelines": {}, "val_mae": {}, "residual_q": {}}
    for name in candidates:
        pipe = make_candidate(name).fit(x_tr, y_tr)
        pred_va = pipe.predict(x_va)
        resid = y_va.to_numpy() - pred_va
        out["pipelines"][name] = pipe
        out["val_mae"][name] = float(mean_absolute_error(y_va, pred_va))
        out["residual_q"][name] = [float(np.quantile(resid, q)) for q in INTERVAL_QUANTILES]
    return out


def train_all(
    stats: pd.DataFrame,
    *,
    artifacts: Path = ARTIFACTS_DIR,
    candidates: tuple[str, ...] = CANDIDATES,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Train all positions, persist ``artifacts/models/<model_version>/``, return the metadata."""
    now = now or dt.datetime.now(dt.UTC)
    sha = input_sha256(stats)
    model_version = _model_version(asof.FEATURE_VERSION, sha, now)
    out_dir = artifacts / "models" / model_version
    out_dir.mkdir(parents=True, exist_ok=True)

    features = asof.training_frame(asof.build_features(stats))
    parts = split_by_season(features)
    data_through = stats.sort_values(["season", "week"]).iloc[-1]

    positions_meta: dict[str, Any] = {}
    test_predictions: list[pd.DataFrame] = []
    for pos in POSITIONS:
        tr = parts["train"][parts["train"]["position"] == pos]
        va = parts["val"][parts["val"]["position"] == pos]
        te = parts["test"][parts["test"]["position"] == pos]
        fitted = fit_position(pos, tr, va, candidates)
        feats = fitted["features"]
        champion = min(fitted["val_mae"], key=fitted["val_mae"].get)
        challenger = next((c for c in candidates if c != champion), None)

        baseline_mean = float(tr["target"].mean())
        cand_meta: dict[str, Any] = {}
        for name, pipe in fitted["pipelines"].items():
            joblib.dump(pipe, out_dir / f"{pos}_{name}.pkl", compress=3)
            pred_te = pipe.predict(te[feats])
            cand_meta[name] = {
                "val_mae": fitted["val_mae"][name],
                "test_mae": float(mean_absolute_error(te["target"], pred_te)),
                "residual_quantiles": {
                    "q": list(INTERVAL_QUANTILES),
                    "values": fitted["residual_q"][name],
                },
                "top_feature_importance": _importances(pipe, feats),
                "hyperparameters": RF_PARAMS if name == "rf" else XGB_PARAMS,
            }
            frame = te[list(asof.KEY_COLUMNS) + ["position", "player_display_name", "team"]].copy()
            frame["candidate"] = name
            frame["prediction"] = np.round(pred_te, 3)
            q10, q90 = fitted["residual_q"][name]
            frame["prediction_floor"] = np.round(np.clip(pred_te + q10, 0, None), 3)
            frame["prediction_ceiling"] = np.round(pred_te + q90, 3)
            frame["actual"] = te["target"].to_numpy()
            test_predictions.append(frame)

        positions_meta[pos] = {
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_test": int(len(te)),
            "n_features": len(feats),
            "features": feats,
            "champion": champion,
            "challenger": challenger,
            "baseline_position_mean": {
                "value": baseline_mean,
                "test_mae": float(
                    mean_absolute_error(te["target"], np.full(len(te), baseline_mean))
                ),
            },
            "candidates": cand_meta,
            "drift_reference_deciles": _deciles(tr, feats),
        }

    preds = pd.concat(test_predictions, ignore_index=True)
    preds.to_csv(out_dir / "test_predictions.csv", index=False)

    metadata = {
        "model_version": model_version,
        "feature_version": asof.FEATURE_VERSION,
        "ffai_version": ffai_version,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": _xgb_version() if "xgb" in candidates else None,
        "data_library": nflverse.LIBRARY,
        "trained_at_utc": now.isoformat(timespec="seconds"),
        "data_through": {"season": int(data_through["season"]), "week": int(data_through["week"])},
        "seasons": {"train": list(TRAIN_SEASONS), "val": VAL_SEASON, "test": TEST_SEASON},
        "target": "fantasy_points_ppr (nflverse, regular season)",
        "input_sha256": sha,
        "input_rows": int(len(stats)),
        "code_commit": git_commit(),
        "interval_method": (
            "validation-season residual quantiles (10th/90th) per position and candidate, "
            "added to the point prediction; floor clipped at 0"
        ),
        "candidates": list(candidates),
        "positions": positions_meta,
        "test_predictions_path": "test_predictions.csv",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _xgb_version() -> str:
    import xgboost

    return xgboost.__version__


def fit_predict_for_week(
    features: pd.DataFrame,
    position: str,
    season: int,
    week: int,
    candidate: str = "rf",
) -> pd.DataFrame:
    """Rolling-origin helper: fit ``candidate`` on rows strictly before (season, week) and
    predict that week. Returns the week's rows with a ``prediction`` column."""
    pos = features[features["position"] == position]
    before = pos[(pos["season"] < season) | ((pos["season"] == season) & (pos["week"] < week))]
    target_rows = pos[(pos["season"] == season) & (pos["week"] == week)]
    feats = asof.features_for_position(position)
    if before.empty or target_rows.empty:
        return target_rows.assign(prediction=np.nan)
    pipe = make_candidate(candidate).fit(before[feats], before["target"])
    return target_rows.assign(prediction=pipe.predict(target_rows[feats]))
