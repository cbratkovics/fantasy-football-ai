#!/usr/bin/env python
"""Score one upcoming week with the manifest champion and write the predictions artifact.

    python scripts/score_week.py --season 2025 --week 1 [--through-season 2024]

Features come only from stat rows strictly before (season, week). ``--through-season`` limits
the loaded seasons (default: through the requested season).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

import argparse

from ffai.config import ARTIFACTS_DIR, MIN_SEASON
from ffai.data import nflverse
from ffai.models import registry
from ffai.pipeline import score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--through-season", type=int, default=None)
    parser.add_argument("--slot", choices=["champion", "challenger"], default="champion")
    args = parser.parse_args()
    manifest = registry.read_manifest(ARTIFACTS_DIR / "manifest.json")
    model_version, candidate = registry.slot(manifest, args.slot)
    last = args.through_season or args.season
    stats = nflverse.load_weekly_stats(range(MIN_SEASON, last + 1))
    payload = score.score_week(
        stats, args.season, args.week, model_version=model_version, candidate=candidate
    )
    path = score.write_predictions(payload)
    manifest["predictions"] = {"latest": str(path.relative_to(ARTIFACTS_DIR))}
    registry.write_manifest(manifest)
    print(f"wrote {path} ({payload['n']} players, data through {payload['data_through']})")


if __name__ == "__main__":
    main()
