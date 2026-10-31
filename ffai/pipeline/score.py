"""Score an upcoming week with the champion (or challenger) models.

``score_week`` builds as-of features for every eligible player as of the end of the last
completed week, predicts PPR points per position, and returns a JSON-serialisable record that the
weekly job writes to ``artifacts/predictions/<season>/week_<ww>.json`` and the API serves.

Eligible players: anyone with at least one regular-season stat row in ``season`` or ``season-1``
(so week 1 of a new season uses last season's players) and at least one prior game overall.

Scoring formats: the model predicts PPR points. Standard and half-PPR are **derived by the same
rules** (the formats differ only by the reception weight, proven by
``tests/test_scoring_reconciliation.py``) using the as-of expected receptions
(``receptions_L3_avg``): ``standard = ppr - 1.0 * receptions_est``, ``half = ppr - 0.5 * receptions_est``.
The reception estimate is included in every record so the derivation is transparent.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ffai.config import ARTIFACTS_DIR, POSITIONS
from ffai.features import asof
from ffai.models import registry
from ffai.scoring import FORMATS, REQUIRED_COLUMNS, RULES_ID, score

PREDICTIONS_VERSION = "1.0"


def predictions_path(season: int, week: int, artifacts: Path = ARTIFACTS_DIR) -> Path:
    return artifacts / "predictions" / str(season) / f"week_{week:02d}.json"


def eligible_targets(stats: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    recent = stats[stats["season"].isin([season - 1, season]) & stats["position"].isin(POSITIONS)]
    players = recent.sort_values(["season", "week"]).groupby("player_id", sort=False).last()
    return pd.DataFrame(
        {
            "player_id": players.index,
            "season": season,
            "week": week,
            "position": players["position"].to_numpy(),
            "team": players["team"].to_numpy(),
            "player_display_name": players["player_display_name"].to_numpy(),
        }
    )


def score_week(
    stats: pd.DataFrame,
    season: int,
    week: int,
    *,
    model_version: str,
    candidate: str | dict[str, str],
    artifacts: Path = ARTIFACTS_DIR,
    pipelines: dict[str, Any] | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Predict every eligible player for (season, week) from stats strictly before it."""
    now = now or dt.datetime.now(dt.UTC)
    prior = stats[
        (stats["season"] < season) | ((stats["season"] == season) & (stats["week"] < week))
    ]
    if prior.empty:
        raise ValueError("no stat rows before the requested week")
    meta = registry.read_model_metadata(model_version, artifacts)
    pipelines = pipelines or registry.load_pipelines(model_version, candidate, artifacts)
    targets = eligible_targets(prior, season, week)
    feats = asof.build_features(prior, targets=targets)
    stubs = feats[feats[asof.TARGET_FLAG] & feats[asof.HISTORY_FLAG]]

    records: list[dict[str, Any]] = []
    for pos in POSITIONS:
        rows = stubs[stubs["position"] == pos]
        if rows.empty:
            continue
        cand = registry.candidate_for(candidate, pos)
        cols = asof.features_for_position(pos)
        pred = pipelines[pos].predict(rows[cols])
        q10, q90 = meta["positions"][pos]["candidates"][cand]["residual_quantiles"]["values"]
        rec_est = rows["receptions_L3_avg"].to_numpy(dtype="float64")
        for i, r in enumerate(rows.itertuples(index=False)):
            p = float(pred[i])
            records.append(
                {
                    "player_id": r.player_id,
                    "name": r.player_display_name,
                    "team": r.team,
                    "position": pos,
                    "prediction": round(p, 2),
                    "floor": round(max(0.0, p + q10), 2),
                    "ceiling": round(p + q90, 2),
                    "receptions_estimate": round(float(rec_est[i]), 2),
                    "candidate": cand,
                }
            )
    records.sort(key=lambda r: (-r["prediction"], r["player_id"]))
    data_through = prior.sort_values(["season", "week"]).iloc[-1]
    return {
        "predictions_version": PREDICTIONS_VERSION,
        "season": int(season),
        "week": int(week),
        "model_version": model_version,
        "feature_version": meta["feature_version"],
        "candidate": candidate,
        "generated_at_utc": now.isoformat(timespec="seconds"),
        "data_through": {"season": int(data_through["season"]), "week": int(data_through["week"])},
        "scoring": {
            "model_target": "ppr",
            "derivation": (
                "standard = ppr - receptions_estimate; half = ppr - 0.5 * receptions_estimate "
                "(receptions_estimate = as-of L3 average receptions)"
            ),
            "interval": meta["interval_method"],
        },
        "n": len(records),
        "predictions": records,
    }


def write_predictions(payload: dict[str, Any], artifacts: Path = ARTIFACTS_DIR) -> Path:
    path = predictions_path(payload["season"], payload["week"], artifacts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return path


def attach_actuals(payload: dict[str, Any], stats: pd.DataFrame) -> tuple[dict[str, Any], int]:
    """Add ``actual`` (realised PPR points) to each record once the week has been played.

    Returns the updated payload and the number of records that received an actual. Players with
    no stat row for that week (inactive, bye, injured) keep ``actual: null``.
    """
    wk = stats[(stats["season"] == payload["season"]) & (stats["week"] == payload["week"])]
    # Validate grain before indexing: repeated IDs may not choose an outcome by row order.
    relevant = ["fantasy_points_ppr", *[c for c in REQUIRED_COLUMNS if c in wk.columns]]
    for _, group in wk.groupby("player_id"):
        if len(group[relevant].drop_duplicates()) > 1:
            raise ValueError("conflicting actual stat rows for one player-period")
    wk = wk.drop_duplicates("player_id").set_index("player_id")
    n = 0
    for rec in payload["predictions"]:
        # A refresh replaces one outcome revision atomically. Old conversion fields must not
        # survive a corrected, missing or malformed new row.
        for key in (
            "actuals_by_format",
            "actual_rules_id",
            "actual_source_revision",
            "actual_receptions",
            "actual_receptions_revision",
            "actual_stats",
            "actual_stats_revision",
        ):
            rec.pop(key, None)
        rec["actual"] = None
        if rec["player_id"] not in wk.index:
            continue
        row = wk.loc[rec["player_id"]]
        v = row["fantasy_points_ppr"]
        if pd.isna(v) or not math.isfinite(float(v)):
            continue
        rec["actual"] = round(float(v), 2)
        n += 1
        if not all(c in row.index for c in REQUIRED_COLUMNS):
            continue
        components = {c: None if pd.isna(row[c]) else float(row[c]) for c in REQUIRED_COLUMNS}
        if any(v is not None and not math.isfinite(v) for v in components.values()):
            continue
        frame = pd.DataFrame([components])
        values = {fmt: round(float(score(frame, fmt).iloc[0]), 2) for fmt in FORMATS}
        if abs(values["ppr"] - rec["actual"]) > 0.011:
            # Keep the legacy observation but never attach inconsistent conversion evidence.
            continue
        revision = hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()
        rec.update(
            actuals_by_format=values,
            actual_rules_id=RULES_ID,
            actual_source_revision=revision,
            actual_receptions=components["receptions"] or 0,
            actual_receptions_revision=revision,
        )
    payload["actuals_attached_at_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    return payload, n
