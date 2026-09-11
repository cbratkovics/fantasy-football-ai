"""Evaluator: causal baseline, forward holdout, metric definitions, rolling origin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffai.eval import evaluator

ROWS = [
    {
        "player_id": "a",
        "season": 2024,
        "week": 1,
        "position": "QB",
        "prediction": 12,
        "actual": 10,
        "decision_score": 0.8,
        "prediction_floor": 8,
    },
    {
        "player_id": "b",
        "season": 2024,
        "week": 1,
        "position": "QB",
        "prediction": 20,
        "actual": 30,
        "decision_score": 0.4,
        "prediction_floor": 14,
    },
    {
        "player_id": "a",
        "season": 2024,
        "week": 2,
        "position": "QB",
        "prediction": 14,
        "actual": 12,
        "decision_score": 0.9,
        "prediction_floor": 11,
    },
    {
        "player_id": "b",
        "season": 2024,
        "week": 2,
        "position": "QB",
        "prediction": 25,
        "actual": 20,
        "decision_score": 0.6,
        "prediction_floor": 18,
    },
]


def test_report_compares_model_and_causal_baseline() -> None:
    report = evaluator.evaluate_rows(ROWS, (2024, 2), thresholds=(0.5, 0.7))
    assert report["metrics"]["n"] == 2
    assert report["metrics"]["mae"] == 3.5
    assert report["metrics"]["median_ae"] == 3.5
    # Week-two baselines are week-one player outcomes (10 and 30), never week-two outcomes.
    assert report["baseline"]["mae"] == 6
    assert report["baseline"]["name"] == "causal_trailing_mean"
    assert report["policy_sweep"][1]["recommended"] == 1
    assert report["policy_sweep"][1]["downside_rate"] == 0
    assert report["split"] == {
        "strategy": "forward_time_holdout",
        "test_start_season": 2024,
        "test_start_week": 2,
    }


def test_within_rates_are_computed_on_the_same_rows_as_mae() -> None:
    report = evaluator.evaluate_rows(ROWS, (2024, 1))
    errors = np.array([2, 10, 2, 5])
    assert report["metrics"]["mae"] == pytest.approx(errors.mean())
    assert report["metrics"]["within_3_rate"] == pytest.approx((errors <= 3).mean())
    assert report["metrics"]["within_5_rate"] == pytest.approx((errors <= 5).mean())
    assert report["metrics"]["rmse"] == pytest.approx(np.sqrt((errors**2).mean()), abs=1e-4)


def test_history_seeds_the_baseline_before_the_test_window() -> None:
    history = [
        {"player_id": "a", "season": 2023, "week": 17, "position": "QB", "actual": 40},
        {"player_id": "b", "season": 2023, "week": 17, "position": "QB", "actual": 0},
    ]
    report = evaluator.evaluate_rows(ROWS, (2024, 1), history=history)
    # Week 1 baselines are the seeded 2023 values: |10-40| = 30 and |30-0| = 30.
    assert report["cohorts"]["QB"]["baseline"]["n"] == 4
    enriched = evaluator.add_causal_baseline(ROWS, history)
    wk1 = {r["player_id"]: r["trailing_mean_baseline"] for r in enriched if r["week"] == 1}
    assert wk1 == {"a": 40, "b": 0}


def test_policy_sweep_is_empty_without_decision_scores() -> None:
    rows = [{k: v for k, v in r.items() if k != "decision_score"} for r in ROWS]
    report = evaluator.evaluate_rows(rows, (2024, 1))
    assert report["policy_sweep"] == []


def test_rejects_missing_contract_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        evaluator.evaluate_rows([{"player_id": "a"}], (2024, 1))


def test_rejects_empty_test_window() -> None:
    with pytest.raises(ValueError, match="empty test set"):
        evaluator.evaluate_rows(ROWS, (2025, 1))


def test_rolling_origin_refits_strictly_before_each_week() -> None:
    seen: list[tuple[int, int]] = []

    def fit_predict(features: pd.DataFrame, position: str, season: int, week: int, cand: str):
        seen.append((season, week))
        rows = features[
            (features["position"] == position)
            & (features["season"] == season)
            & (features["week"] == week)
        ]
        before = features[
            (features["season"] < season)
            | ((features["season"] == season) & (features["week"] < week))
        ]
        assert not before.empty
        return rows.assign(prediction=before["target"].mean())

    feats = pd.DataFrame(
        {
            "player_id": ["p"] * 4 + ["q"] * 4,
            "season": [2024] * 8,
            "week": [1, 2, 3, 4] * 2,
            "position": ["RB"] * 8,
            "target": [10, 12, 14, 16, 5, 6, 7, 8],
        }
    )
    out = evaluator.rolling_origin(feats, 2024, fit_predict, candidate="rf", weeks=[2, 3, 4])
    assert [f["week"] for f in out["folds"]] == [2, 3, 4]
    assert sorted(set(seen)) == [(2024, 2), (2024, 3), (2024, 4)]
    assert out["mean_mae"] is not None and out["mean_baseline_mae"] is not None


def test_build_artifact_records_provenance(tmp_path) -> None:
    csv = tmp_path / "preds.csv"
    pd.DataFrame(ROWS).to_csv(csv, index=False)
    report = evaluator.evaluate_rows(ROWS, (2024, 1))
    art = evaluator.build_artifact(
        report, input_path=csv, input_rows=4, model={"version": "v", "candidate": "rf"}
    )
    assert art["artifact_version"] == evaluator.ARTIFACT_VERSION
    assert art["input"]["sha256"] == evaluator.sha256_of_file(csv)
    assert art["input"]["n_rows"] == 4
    assert "code_commit" in art and "generated_at_utc" in art
    assert art["metric_definitions"]["within_3_rate"].startswith("mean(|actual - prediction| <= 3)")
    path = evaluator.write_artifact(art, tmp_path / "out" / "e.json")
    assert path.exists()


def test_input_path_is_repo_relative_and_kind_is_recorded(tmp_path) -> None:
    from ffai.config import REPO_ROOT

    inside = REPO_ROOT / "artifacts" / "models" / "v" / "test_predictions.csv"
    assert evaluator.repo_relative(inside) == "artifacts/models/v/test_predictions.csv"
    assert evaluator.repo_relative(None) is None
    outside = tmp_path / "preds.csv"
    pd.DataFrame(ROWS).to_csv(outside, index=False)
    assert "/" not in evaluator.repo_relative(outside)  # never a machine-specific absolute path
    report = evaluator.evaluate_rows(ROWS, (2024, 1))
    art = evaluator.build_artifact(
        report,
        input_path=outside,
        input_rows=4,
        model={"version": "v", "candidate": {"QB": "rf"}},
        kind="out_of_sample_season",
        season=2025,
        id_suffix="-oos2025",
    )
    assert art["kind"] == "out_of_sample_season" and art["season"] == 2025
    assert art["eval_id"].endswith("-rf-oos2025")
    assert art["input"]["path"] == "preds.csv"
    with pytest.raises(ValueError, match="kind"):
        evaluator.build_artifact(report, input_path=None, input_rows=4, model={}, kind="nope")
