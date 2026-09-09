"""Create an auditable model-and-policy evaluation artifact from a prediction CSV.

The evaluator deliberately consumes predictions rather than training a model. This keeps
evaluation independent from the many prototype trainers and makes leakage review easier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

REQUIRED_COLUMNS = {
    "player_id", "season", "week", "position", "prediction", "actual", "decision_score"
}


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _metrics(rows: Sequence[Mapping[str, Any]], prediction_key: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mae": None, "median_absolute_error": None}
    errors = [abs(_finite(row[prediction_key], prediction_key) - _finite(row["actual"], "actual")) for row in rows]
    return {"n": len(rows), "mae": round(mean(errors), 4), "median_absolute_error": round(median(errors), 4)}


def _add_causal_baseline(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add a trailing mean using only outcomes from earlier season/week periods."""
    ordered = sorted(rows, key=lambda row: (int(row["season"]), int(row["week"]), str(row["player_id"])))
    player_history: dict[str, list[float]] = defaultdict(list)
    position_history: dict[str, list[float]] = defaultdict(list)
    result: list[dict[str, Any]] = []

    for period in sorted({(int(row["season"]), int(row["week"])) for row in ordered}):
        period_rows = [row for row in ordered if (int(row["season"]), int(row["week"])) == period]
        for source in period_rows:
            row = dict(source)
            player_values = player_history[str(row["player_id"])]
            position_values = position_history[str(row["position"])]
            history = player_values or position_values
            row["trailing_mean_baseline"] = mean(history) if history else _finite(row["prediction"], "prediction")
            result.append(row)
        # Outcomes from the same week become available only after every row is scored.
        for row in period_rows:
            actual = _finite(row["actual"], "actual")
            player_history[str(row["player_id"])].append(actual)
            position_history[str(row["position"])].append(actual)
    return result


def _policy_metrics(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    selected = [row for row in rows if _finite(row["decision_score"], "decision_score") >= threshold]
    with_floor = [row for row in selected if row.get("prediction_floor") not in (None, "")]
    return {
        "threshold": threshold,
        "eligible": len(rows),
        "recommended": len(selected),
        "recommendation_rate": round(len(selected) / len(rows), 4) if rows else None,
        "selected_mae": _metrics(selected, "prediction")["mae"],
        "downside_rate": round(mean(
            _finite(row["actual"], "actual") < _finite(row["prediction_floor"], "prediction_floor")
            for row in with_floor
        ), 4) if with_floor else None,
    }


def evaluate_rows(
    rows: Iterable[Mapping[str, Any]], test_start: tuple[int, int], thresholds: Sequence[float] = (0.5, 0.7, 0.9)
) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("evaluation input is empty")
    for index, row in enumerate(materialized, start=1):
        missing = REQUIRED_COLUMNS - set(row)
        if missing:
            raise ValueError(f"row {index} missing required columns: {', '.join(sorted(missing))}")
        score = _finite(row["decision_score"], "decision_score")
        if not 0 <= score <= 1:
            raise ValueError(f"row {index} decision_score must be between 0 and 1")
    enriched = _add_causal_baseline(materialized)
    test = [row for row in enriched if (int(row["season"]), int(row["week"])) >= test_start]
    if not test:
        raise ValueError("test_start produces an empty test set")
    positions = sorted({str(row["position"]) for row in test})
    return {
        "split": {"strategy": "forward_time_holdout", "test_start_season": test_start[0], "test_start_week": test_start[1]},
        "model": _metrics(test, "prediction"),
        "baseline": {"name": "causal_trailing_mean", **_metrics(test, "trailing_mean_baseline")},
        "cohorts": {position: _metrics([row for row in test if row["position"] == position], "prediction") for position in positions},
        "policy_sweep": [_policy_metrics(test, threshold) for threshold in thresholds],
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV with point-in-time predictions and realized outcomes")
    parser.add_argument("--test-start", required=True, help="First test period as SEASON-WEEK, for example 2024-10")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON artifact")
    args = parser.parse_args()
    season, week = (int(part) for part in args.test_start.split("-", maxsplit=1))
    raw = args.input.read_bytes()
    with args.input.open(newline="", encoding="utf-8") as handle:
        report = evaluate_rows(csv.DictReader(handle), (season, week))
    artifact = {
        "artifact_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(args.input), "sha256": hashlib.sha256(raw).hexdigest()},
        "code_commit": _git_commit(),
        **report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote evaluation artifact to {args.output}")


if __name__ == "__main__":
    main()
