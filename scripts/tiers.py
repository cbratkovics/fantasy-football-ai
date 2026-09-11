#!/usr/bin/env python
"""Build preseason GMM draft tiers for a season from prior-season aggregates.

    python scripts/tiers.py --season 2024

Writes artifacts/tiers/<tier_version>/{model.pkl, tiers.json, metadata.json}. When the season's
own stats exist, the tiers are evaluated against realised PPR/game (Spearman, within-band rate).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

import argparse

from ffai.config import ARTIFACTS_DIR, MIN_SEASON, TEST_SEASON
from ffai.data import nflverse
from ffai.eval import model_card
from ffai.models import registry, tiers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=TEST_SEASON)
    parser.add_argument("--no-rosters", action="store_true", help="skip roster ages")
    parser.add_argument(
        "--previous",
        default=None,
        help="tier_version to compare against (default: the manifest's tiers_version)",
    )
    parser.add_argument("--update-manifest", action="store_true", help="point the manifest at it")
    args = parser.parse_args()
    stats = nflverse.load_weekly_stats(range(MIN_SEASON, args.season + 1))
    rosters = None if args.no_rosters else nflverse.load_rosters([args.season])
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    manifest = registry.read_manifest(manifest_path) if manifest_path.exists() else None
    previous = args.previous or (manifest or {}).get("tiers_version")
    meta = tiers.build_tiers(stats, args.season, rosters=rosters, previous_tier_version=previous)
    print(f"tier_version={meta['tier_version']}")
    if meta.get("comparison_to_previous"):
        for pos, c in meta["comparison_to_previous"]["positions"].items():
            b, a = c["before"], c["after"]
            print(
                f"  {pos}: components {b['n_components']}->{a['n_components']} "
                f"spearman {b.get('spearman')}->{a.get('spearman')} "
                f"within_band {b.get('within_band_rate')}->{a.get('within_band_rate')} kept={c['kept']}"
            )
    if args.update_manifest and manifest is not None:
        manifest["tiers_version"] = meta["tier_version"]
        registry.write_manifest(manifest, manifest_path)
        model_card.write_model_card(manifest)
        print("manifest tiers_version updated; model card regenerated")
    for pos, p in meta["positions"].items():
        ev = (meta.get("evaluation") or {}).get("positions", {}).get(pos, {})
        print(
            f"{pos}: players={p['n_players']} components={p['n_components']} pca={p['pca_components']} "
            f"spearman={ev.get('spearman')} within_band={ev.get('within_band_rate')}"
        )


if __name__ == "__main__":
    main()
