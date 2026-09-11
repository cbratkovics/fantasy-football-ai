"""Create an auditable model-and-policy evaluation artifact from point-in-time predictions.

Ported from the branch ``backend/evaluation/decision_evaluator.py`` and extended. The evaluator
deliberately consumes predictions rather than training a model, so it can be applied to any
candidate and so leakage review stays separate from the trainers.

Inputs: rows with ``player_id, season, week, position, prediction, actual`` and optionally
``prediction_floor``, ``decision_score``. An optional ``history`` frame (``player_id, season,
week, position, actual``) seeds the causal baseline with outcomes from before ``test_start``.

Metrics (all computed on the same rows):

* ``mae``, ``median_ae``, ``rmse``
* ``within_3_rate`` = mean(|actual − prediction| ≤ 3), ``within_5_rate`` likewise

Baseline: *causal trailing mean* — for each row, the mean of the player's realised outcomes from
strictly earlier periods, falling back to the position's earlier outcomes, falling back to the
row's own prediction when nothing earlier exists. Outcomes of a period are appended only after
every row of that period has been scored.

Artifact schema (``artifact_version`` 2.0):

    artifact_version, eval_id, generated_at_utc, code_commit,
    input {path, sha256, n_rows}, model {version, feature_version, candidate},
    split {strategy, test_start_season, test_start_week},
    metrics {n, mae, median_ae, rmse, within_3_rate, within_5_rate},
    baseline {name, ...same metrics},
    cohorts {position: {metrics..., baseline: {...}}},
    rolling_origin {candidate, folds: [{season, week, n, mae, baseline_mae}], mean_mae, mean_baseline_mae}
    policy_sweep [ {threshold, eligible, recommended, recommendation_rate, selected_mae, downside_rate} ]
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

ARTIFACT_VERSION = "2.0"
REQUIRED_COLUMNS = {"player_id", "season", "week", "position", "prediction", "actual"}
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.5, 0.7, 0.9)
BASELINE_NAME = "causal_trailing_mean"


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _metrics(rows: Sequence[Mapping[str, Any]], prediction_key: str) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "mae": None,
            "median_ae": None,
            "rmse": None,
            "within_3_rate": None,
            "within_5_rate": None,
        }
    errors = [
        abs(_finite(r[prediction_key], prediction_key) - _finite(r["actual"], "actual"))
        for r in rows
    ]
    return {
        "n": len(rows),
        "mae": round(mean(errors), 4),
        "median_ae": round(median(errors), 4),
        "rmse": round(math.sqrt(mean(e * e for e in errors)), 4),
        "within_3_rate": round(mean(e <= 3 for e in errors), 4),
        "within_5_rate": round(mean(e <= 5 for e in errors), 4),
    }


def _period(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["season"]), int(row["week"])


def add_causal_baseline(
    rows: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Add ``trailing_mean_baseline`` using only outcomes from earlier (season, week) periods."""
    player_history: dict[str, list[float]] = defaultdict(list)
    position_history: dict[str, list[float]] = defaultdict(list)
    # Seed with history strictly before the earliest evaluated period.
    if rows:
        first_period = min(_period(r) for r in rows)
        for h in sorted(history, key=_period):
            if _period(h) < first_period:
                actual = _finite(h["actual"], "actual")
                player_history[str(h["player_id"])].append(actual)
                position_history[str(h["position"])].append(actual)

    ordered = sorted(rows, key=lambda r: (_period(r), str(r["player_id"])))
    result: list[dict[str, Any]] = []
    by_period: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for r in ordered:
        by_period[_period(r)].append(r)
    for period in sorted(by_period):
        period_rows = by_period[period]
        for source in period_rows:
            row = dict(source)
            hist = player_history[str(row["player_id"])] or position_history[str(row["position"])]
            row["trailing_mean_baseline"] = (
                mean(hist) if hist else _finite(row["prediction"], "prediction")
            )
            result.append(row)
        # Outcomes from the same week become available only after every row is scored.
        for row in period_rows:
            actual = _finite(row["actual"], "actual")
            player_history[str(row["player_id"])].append(actual)
            position_history[str(row["position"])].append(actual)
    return result


def _policy_metrics(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    selected = [r for r in rows if _finite(r["decision_score"], "decision_score") >= threshold]
    with_floor = [r for r in selected if r.get("prediction_floor") not in (None, "")]
    return {
        "threshold": threshold,
        "eligible": len(rows),
        "recommended": len(selected),
        "recommendation_rate": round(len(selected) / len(rows), 4) if rows else None,
        "selected_mae": _metrics(selected, "prediction")["mae"],
        "downside_rate": (
            round(
                mean(
                    _finite(r["actual"], "actual")
                    < _finite(r["prediction_floor"], "prediction_floor")
                    for r in with_floor
                ),
                4,
            )
            if with_floor
            else None
        ),
    }


def evaluate_rows(
    rows: Iterable[Mapping[str, Any]],
    test_start: tuple[int, int],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    history: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Evaluate prediction rows with a forward time holdout starting at ``test_start``."""
    materialized = list(rows)
    if not materialized:
        raise ValueError("evaluation input is empty")
    has_scores = all(
        "decision_score" in r and r["decision_score"] not in (None, "") for r in materialized
    )
    for index, row in enumerate(materialized, start=1):
        missing = REQUIRED_COLUMNS - set(row)
        if missing:
            raise ValueError(f"row {index} missing required columns: {', '.join(sorted(missing))}")
        if has_scores:
            score = _finite(row["decision_score"], "decision_score")
            if not 0 <= score <= 1:
                raise ValueError(f"row {index} decision_score must be between 0 and 1")
    enriched = add_causal_baseline(materialized, list(history))
    test = [r for r in enriched if _period(r) >= tuple(test_start)]
    if not test:
        raise ValueError("test_start produces an empty test set")
    positions = sorted({str(r["position"]) for r in test})
    cohorts = {}
    for p in positions:
        sub = [r for r in test if r["position"] == p]
        cohorts[p] = {
            **_metrics(sub, "prediction"),
            "baseline": _metrics(sub, "trailing_mean_baseline"),
        }
    return {
        "split": {
            "strategy": "forward_time_holdout",
            "test_start_season": int(test_start[0]),
            "test_start_week": int(test_start[1]),
        },
        "metrics": _metrics(test, "prediction"),
        "baseline": {"name": BASELINE_NAME, **_metrics(test, "trailing_mean_baseline")},
        "cohorts": cohorts,
        "policy_sweep": [_policy_metrics(test, t) for t in thresholds] if has_scores else [],
    }


def evaluate_frame(
    predictions: pd.DataFrame,
    test_start: tuple[int, int],
    *,
    history: pd.DataFrame | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """DataFrame front-end for :func:`evaluate_rows`."""
    rows = predictions.to_dict("records")
    hist = history.to_dict("records") if history is not None else ()
    return evaluate_rows(rows, test_start, thresholds=thresholds, history=hist)


FitPredict = Callable[[pd.DataFrame, str, int, int, str], pd.DataFrame]


def rolling_origin(
    features: pd.DataFrame,
    season: int,
    fit_predict: FitPredict,
    *,
    candidate: str = "rf",
    weeks: Iterable[int] | None = None,
    history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Rolling-origin evaluation inside ``season``.

    For each week ``w`` (default 2..max week in the season) every position's model is refit on
    all rows strictly before (season, w) via ``fit_predict`` and scored on week ``w``. Per-fold
    MAE is reported next to the causal trailing-mean baseline MAE on the same rows.
    """
    positions = sorted(features["position"].unique())
    in_season = features[features["season"] == season]
    if weeks is None:
        weeks = range(2, int(in_season["week"].max()) + 1)
    folds = []
    for w in weeks:
        parts = [fit_predict(features, p, season, w, candidate) for p in positions]
        fold = pd.concat(parts, ignore_index=True)
        fold = fold[fold["prediction"].notna()]
        if fold.empty:
            continue
        fold = fold.rename(columns={"target": "actual"})
        rows = fold[["player_id", "season", "week", "position", "prediction", "actual"]].to_dict(
            "records"
        )
        hist_rows = history.to_dict("records") if history is not None else ()
        enriched = add_causal_baseline(rows, list(hist_rows))
        folds.append(
            {
                "season": int(season),
                "week": int(w),
                "n": len(enriched),
                "mae": _metrics(enriched, "prediction")["mae"],
                "baseline_mae": _metrics(enriched, "trailing_mean_baseline")["mae"],
            }
        )
    return {
        "candidate": candidate,
        "strategy": "refit on all rows strictly before (season, week); score that week",
        "folds": folds,
        "mean_mae": round(mean(f["mae"] for f in folds), 4) if folds else None,
        "mean_baseline_mae": round(mean(f["baseline_mae"] for f in folds), 4) if folds else None,
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_artifact(
    report: dict[str, Any],
    *,
    input_path: Path | None,
    input_rows: int,
    model: dict[str, Any],
    rolling: dict[str, Any] | None = None,
    eval_id: str | None = None,
) -> dict[str, Any]:
    """Wrap an evaluation report with provenance fields."""
    now = dt.datetime.now(dt.UTC)
    cand = model.get("candidate", "")
    if isinstance(cand, dict):
        cand = cand[next(iter(cand))] if len(set(cand.values())) == 1 else "mixed"
    eval_id = eval_id or f"eval-{now:%Y%m%d}-{model.get('version', 'model')}-{cand}".rstrip("-")
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "eval_id": eval_id,
        "generated_at_utc": now.isoformat(timespec="seconds"),
        "code_commit": git_commit(),
        "input": {
            "path": str(input_path) if input_path else None,
            "sha256": sha256_of_file(input_path) if input_path else None,
            "n_rows": int(input_rows),
        },
        "model": model,
        "metric_definitions": {
            "mae": "mean(|actual - prediction|)",
            "median_ae": "median(|actual - prediction|)",
            "rmse": "sqrt(mean((actual - prediction)^2))",
            "within_3_rate": "mean(|actual - prediction| <= 3), same rows as mae",
            "within_5_rate": "mean(|actual - prediction| <= 5), same rows as mae",
            "baseline": (
                "causal trailing mean of the player's earlier realised points (position fallback); "
                "outcomes of a week are added only after that week is scored"
            ),
        },
        **report,
    }
    if rolling is not None:
        artifact["rolling_origin"] = rolling
    return artifact


def write_artifact(artifact: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV with point-in-time predictions and outcomes")
    parser.add_argument(
        "--test-start", required=True, help="First test period as SEASON-WEEK, e.g. 2024-1"
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON artifact")
    parser.add_argument(
        "--history", type=Path, help="CSV with earlier actuals to seed the baseline"
    )
    args = parser.parse_args()
    season, week = (int(part) for part in args.test_start.split("-", maxsplit=1))
    preds = pd.read_csv(args.input)
    hist = pd.read_csv(args.history) if args.history else None
    report = evaluate_frame(preds, (season, week), history=hist)
    artifact = build_artifact(report, input_path=args.input, input_rows=len(preds), model={})
    write_artifact(artifact, args.output)
    print(f"Wrote evaluation artifact to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
