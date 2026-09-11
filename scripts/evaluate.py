#!/usr/bin/env python
"""Evaluate a model artifact on the frozen 2024 test season, write the evaluation artifact,
update the manifest, and regenerate docs/MODEL_CARD.md.

    python scripts/evaluate.py [--model-version V] [--tiers-version T] [--no-rolling]

Every number in the model card traces to artifacts/eval/<eval_id>.json or the model metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from ffai.config import ARTIFACTS_DIR, HISTORICAL_SEASONS, POSITIONS, TEST_SEASON
from ffai.data import nflverse
from ffai.eval import evaluator, model_card
from ffai.features import asof
from ffai.models import registry, train


def newest(dir_: Path) -> str:
    versions = sorted(p.name for p in dir_.iterdir() if p.is_dir())
    if not versions:
        raise FileNotFoundError(f"no versions under {dir_}")
    return versions[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--tiers-version", default=None)
    parser.add_argument("--no-rolling", action="store_true", help="skip rolling-origin folds")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()

    artifacts = args.artifacts
    model_version = args.model_version or newest(artifacts / "models")
    tiers_dir = artifacts / "tiers"
    tiers_version = args.tiers_version or (newest(tiers_dir) if tiers_dir.exists() else None)
    meta = registry.read_model_metadata(model_version, artifacts)
    champions = {p: meta["positions"][p]["champion"] for p in POSITIONS}
    challengers = {p: meta["positions"][p]["challenger"] for p in POSITIONS}

    preds_path = registry.model_dir(model_version, artifacts) / meta["test_predictions_path"]
    preds = pd.read_csv(preds_path)
    champ_rows = preds[preds.apply(lambda r: r["candidate"] == champions[r["position"]], axis=1)]

    stats = nflverse.load_weekly_stats(HISTORICAL_SEASONS)
    history = stats[stats["season"] < TEST_SEASON][
        ["player_id", "season", "week", "position", "fantasy_points_ppr"]
    ].rename(columns={"fantasy_points_ppr": "actual"})

    report = evaluator.evaluate_frame(champ_rows, (TEST_SEASON, 1), history=history)

    rolling = None
    if not args.no_rolling:
        features = asof.training_frame(asof.build_features(stats))
        rolling_candidate = Counter(champions.values()).most_common(1)[0][0]
        rolling = evaluator.rolling_origin(
            features,
            TEST_SEASON,
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
    )
    out = evaluator.write_artifact(artifact, artifacts / "eval" / f"{artifact['eval_id']}.json")
    print(f"wrote {out}")
    m = artifact["metrics"]
    print(
        f"test {TEST_SEASON}: n={m['n']} mae={m['mae']} within3={m['within_3_rate']} "
        f"baseline_mae={artifact['baseline']['mae']}"
    )
    if rolling:
        print(
            f"rolling-origin mean mae={rolling['mean_mae']} baseline={rolling['mean_baseline_mae']}"
        )

    manifest = (
        registry.read_manifest(artifacts / "manifest.json")
        if (artifacts / "manifest.json").exists()
        else {}
    )
    manifest.update(
        {
            "feature_version": meta["feature_version"],
            "champion": {"model_version": model_version, "candidate": champions},
            "challenger": {"model_version": model_version, "candidate": challengers},
            "tiers_version": tiers_version,
            "eval_id": artifact["eval_id"],
            "data_through": meta["data_through"],
        }
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
    card = model_card.write_model_card(
        model_version, artifact["eval_id"], tiers_version=tiers_version, artifacts=artifacts
    )
    print(f"manifest updated; model card written to {card}")


if __name__ == "__main__":
    main()
