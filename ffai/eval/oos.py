"""Score a complete season with a frozen model artifact (out-of-sample evaluation input).

Every row of ``season`` that has at least one prior game is predicted with the persisted
pipelines exactly as the frozen test season was: features come from ``ffai.features.asof`` over
the full stats frame, so week 1 of the season uses the previous seasons' history and later weeks
use the season's own earlier weeks. Nothing is refit. The output has the same columns as
``test_predictions.csv`` (both candidates, floors/ceilings from the training-time residual
quantiles) so ``scripts/evaluate.py`` can consume it unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ffai.config import ARTIFACTS_DIR, POSITIONS
from ffai.features import asof
from ffai.models import registry


def score_season(
    stats: pd.DataFrame,
    season: int,
    model_version: str,
    *,
    artifacts: Path = ARTIFACTS_DIR,
) -> pd.DataFrame:
    """Predictions for every played row of ``season`` from the frozen ``model_version``."""
    meta = registry.read_model_metadata(model_version, artifacts)
    if season <= max(meta["seasons"]["train"] + [meta["seasons"]["val"], meta["seasons"]["test"]]):
        raise ValueError(
            f"season {season} was used to train, validate, or test {model_version}; "
            "out-of-sample scoring needs a later season"
        )
    if stats["season"].max() < season:
        raise ValueError(f"stats frame has no rows for season {season}")
    feats = asof.training_frame(asof.build_features(stats))
    rows = feats[feats["season"] == season]
    out: list[pd.DataFrame] = []
    for pos in POSITIONS:
        pr = rows[rows["position"] == pos]
        if pr.empty:
            continue
        cols = asof.features_for_position(pos)
        for cand in meta["candidates"]:
            pipe = registry.load_pipelines(model_version, cand, artifacts)[pos]
            pred = pipe.predict(pr[cols])
            q10, q90 = meta["positions"][pos]["candidates"][cand]["residual_quantiles"]["values"]
            frame = pr[list(asof.KEY_COLUMNS) + ["position", "player_display_name", "team"]].copy()
            frame["candidate"] = cand
            frame["prediction"] = np.round(pred, 3)
            frame["prediction_floor"] = np.round(np.clip(pred + q10, 0, None), 3)
            frame["prediction_ceiling"] = np.round(pred + q90, 3)
            frame["actual"] = pr["target"].to_numpy()
            out.append(frame)
    result = pd.concat(out, ignore_index=True)
    return result.sort_values(["candidate", "position", "season", "week", "player_id"]).reset_index(
        drop=True
    )


def oos_predictions_path(model_version: str, season: int, artifacts: Path = ARTIFACTS_DIR) -> Path:
    return registry.model_dir(model_version, artifacts) / f"oos_predictions_{season}.csv"
