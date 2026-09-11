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

from ffai.config import MIN_SEASON, TEST_SEASON
from ffai.data import nflverse
from ffai.models import tiers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=TEST_SEASON)
    parser.add_argument("--no-rosters", action="store_true", help="skip roster ages")
    args = parser.parse_args()
    stats = nflverse.load_weekly_stats(range(MIN_SEASON, args.season + 1))
    rosters = None if args.no_rosters else nflverse.load_rosters([args.season])
    meta = tiers.build_tiers(stats, args.season, rosters=rosters)
    print(f"tier_version={meta['tier_version']}")
    for pos, p in meta["positions"].items():
        ev = (meta.get("evaluation") or {}).get("positions", {}).get(pos, {})
        print(
            f"{pos}: players={p['n_players']} components={p['n_components']} pca={p['pca_components']} "
            f"spearman={ev.get('spearman')} within_band={ev.get('within_band_rate')}"
        )


if __name__ == "__main__":
    main()
