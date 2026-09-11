#!/usr/bin/env python
"""Evaluate a frozen model artifact, write the evaluation artifact, register it in the
manifest, and regenerate docs/MODEL_CARD.md.

    python scripts/evaluate.py                                  # frozen 2024 test season
    python scripts/evaluate.py --kind out_of_sample_season --season 2025

Frozen test: reads models/<version>/test_predictions.csv (written at training time).
Out-of-sample season: scores every played row of the season with the persisted pipelines
(ffai/eval/oos.py), writes models/<version>/oos_predictions_<season>.csv, then evaluates it.

Every number in the model card traces to artifacts/eval/<eval_id>.json or the model metadata.
``--assert-identical`` refuses to overwrite an existing artifact whose metrics differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

import argparse
import json
from collections import Counter

import pandas as pd

from ffai.config import ARTIFACTS_DIR, MIN_SEASON, POSITIONS, TEST_SEASON
from ffai.data import nflverse
from ffai.eval import evaluator, model_card, oos
from ffai.features import asof
from ffai.models import registry, train

METRIC_KEYS = ("n", "mae", "median_ae", "rmse", "within_3_rate", "within_5_rate")


def newest(dir_: Path) -> str:
    versions = sorted(p.name for p in dir_.iterdir() if p.is_dir())
    if not versions:
        raise FileNotFoundError(f"no versions under {dir_}")
    return versions[-1]


def metrics_view(artifact: dict) -> dict:
    """The comparable part of an artifact (everything except provenance and timestamps)."""
    view = {
        "metrics": {k: artifact["metrics"][k] for k in METRIC_KEYS},
        "baseline": {k: artifact["baseline"][k] for k in METRIC_KEYS},
        "cohorts": {
            p: {k: c[k] for k in METRIC_KEYS}
            | {"baseline": {k: c["baseline"][k] for k in METRIC_KEYS}}
            for p, c in artifact["cohorts"].items()
        },
    }
    if artifact.get("rolling_origin"):
        view["rolling_origin"] = {
            "mean_mae": artifact["rolling_origin"]["mean_mae"],
            "mean_baseline_mae": artifact["rolling_origin"]["mean_baseline_mae"],
            "folds": artifact["rolling_origin"]["folds"],
        }
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--tiers-version", default=None)
    parser.add_argument("--kind", choices=evaluator.EVAL_KINDS, default="frozen_test")
    parser.add_argument("--season", type=int, default=None, help="for out_of_sample_season")
    parser.add_argument("--no-rolling", action="store_true", help="skip rolling-origin folds")
    parser.add_argument(
        "--assert-identical",
        action="store_true",
        help="fail if an artifact with the same eval_id exists and its metrics differ",
    )
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()

    artifacts = args.artifacts
    model_version = args.model_version or newest(artifacts / "models")
    tiers_dir = artifacts / "tiers"
    manifest = (
        registry.read_manifest(artifacts / "manifest.json")
        if (artifacts / "manifest.json").exists()
        else {}
    )
    tiers_version = (
        args.tiers_version
        or manifest.get("tiers_version")
        or (newest(tiers_dir) if tiers_dir.exists() else None)
    )
    meta = registry.read_model_metadata(model_version, artifacts)
    champions = {p: meta["positions"][p]["champion"] for p in POSITIONS}
    challengers = {p: meta["positions"][p]["challenger"] for p in POSITIONS}

    if args.kind == "frozen_test":
        season = TEST_SEASON
        preds_path = registry.model_dir(model_version, artifacts) / meta["test_predictions_path"]
        preds = pd.read_csv(preds_path)
        stats = nflverse.load_weekly_stats(range(MIN_SEASON, season + 1))
        id_suffix = ""
    else:
        if args.season is None:
            parser.error("--season is required for out_of_sample_season")
        season = args.season
        stats = nflverse.load_weekly_stats(range(MIN_SEASON, season + 1))
        preds = oos.score_season(stats, season, model_version, artifacts=artifacts)
        preds_path = oos.oos_predictions_path(model_version, season, artifacts)
        preds.to_csv(preds_path, index=False)
        print(f"wrote {evaluator.repo_relative(preds_path)} ({len(preds)} rows, both candidates)")
        id_suffix = f"-oos{season}"

    champ_rows = preds[preds.apply(lambda r: r["candidate"] == champions[r["position"]], axis=1)]
    history = stats[stats["season"] < season][
        ["player_id", "season", "week", "position", "fantasy_points_ppr"]
    ].rename(columns={"fantasy_points_ppr": "actual"})

    report = evaluator.evaluate_frame(champ_rows, (season, 1), history=history)

    rolling = None
    if not args.no_rolling:
        features = asof.training_frame(asof.build_features(stats))
        rolling_candidate = Counter(champions.values()).most_common(1)[0][0]
        rolling = evaluator.rolling_origin(
            features,
            season,
            train.fit_predict_for_week,
            candidate=rolling_candidate,
            history=history,
        )

    artifact = evaluator.build_artifact(
        report,
        input_path=preds_path,
        input_rows=len(champ_rows),
        model={
            "version": model_version,
            "feature_version": meta["feature_version"],
            "candidate": champions,
            "challenger": challengers,
            "trained_at_utc": meta["trained_at_utc"],
            "input_sha256": meta["input_sha256"],
        },
        rolling=rolling,
        kind=args.kind,
        season=season,
        id_suffix=id_suffix,
    )
    out_path = artifacts / "eval" / f"{artifact['eval_id']}.json"
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        identical = metrics_view(existing) == metrics_view(artifact)
        print(f"existing artifact {artifact['eval_id']}: metrics identical={identical}")
        if args.assert_identical and not identical:
            raise SystemExit("refusing to overwrite: metrics differ from the committed artifact")
    out = evaluator.write_artifact(artifact, out_path)
    print(f"wrote {evaluator.repo_relative(out)}")
    m = artifact["metrics"]
    print(
        f"{args.kind} {season}: n={m['n']} mae={m['mae']} within3={m['within_3_rate']} "
        f"baseline_mae={artifact['baseline']['mae']}"
    )
    if rolling:
        print(
            f"rolling-origin mean mae={rolling['mean_mae']} baseline={rolling['mean_baseline_mae']}"
        )

    manifest.update(
        {
            "feature_version": meta["feature_version"],
            "champion": {"model_version": model_version, "candidate": champions},
            "challenger": {"model_version": model_version, "candidate": challengers},
            "tiers_version": tiers_version,
        }
    )
    manifest.setdefault("data_through", meta["data_through"])
    registry.register_evaluation(
        manifest,
        {
            "eval_id": artifact["eval_id"],
            "kind": args.kind,
            "season": season,
            "path": f"eval/{artifact['eval_id']}.json",
            "generated_at_utc": artifact["generated_at_utc"],
        },
    )
    manifest.setdefault("predictions", {"latest": None})
    manifest.setdefault(
        "last_run",
        {
            "run_id": "bootstrap",
            "at_utc": artifact["generated_at_utc"],
            "action": "PUBLISH",
            "reasons": ["initial evaluation"],
        },
    )
    registry.write_manifest(manifest, artifacts / "manifest.json")
    card = model_card.write_model_card(manifest, artifacts=artifacts)
    print(f"manifest updated; model card written to {card}")


if __name__ == "__main__":
    main()
