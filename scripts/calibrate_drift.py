#!/usr/bin/env python
"""Backtest the drift rule over every target week of past seasons, under the training-time
bucket reference alone and under the hybrid reference (ADR-0031), and print the held windows.

    python scripts/calibrate_drift.py [--seasons 2021-2024] [--weeks 2-18]

For each (season, target week) the window is the last WINDOW_WEEKS played weeks before the
target (exactly as the weekly job builds it), scored per position. "bucket" is the ADR-0010
construction; "hybrid" switches to a reference built from the training seasons' rows at the
window's own week positions when the window crosses a season boundary. Needs the full nflverse
history (network on a cache miss). ADR-0010 scanned target weeks 5-18 only; ADR-0031 records the
weeks 2-18 table this script prints.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

import argparse

import pandas as pd

from ffai.config import MIN_SEASON, POSITIONS
from ffai.data import nflverse
from ffai.eval import drift
from ffai.features import asof
from ffai.models import registry
from ffai.pipeline import weekly


def _range(text: str) -> list[int]:
    first, last = (int(p) for p in text.split("-", 1))
    return list(range(first, last + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default="2021-2024")
    ap.add_argument("--weeks", default="2-18")
    a = ap.parse_args()
    seasons, weeks = _range(a.seasons), _range(a.weeks)

    stats = nflverse.load_weekly_stats(range(MIN_SEASON, max(seasons) + 1))
    manifest = registry.read_manifest()
    version, cand_spec = registry.slot(manifest, "champion")
    meta = registry.read_model_metadata(version)
    played = asof.training_frame(asof.build_features(stats))
    training = played[played["season"].isin(meta["seasons"]["train"])]

    def monitored(pos: str) -> list[str]:
        pm = meta["positions"][pos]
        cand = registry.candidate_for(cand_spec, pos)
        return [
            f
            for f in pm["candidates"][cand]["top_feature_importance"]
            if f not in weekly.DRIFT_EXCLUDED_FEATURES and not f.endswith("_season_avg")
        ]

    rows = []
    for season in seasons:
        for week in weeks:
            prior = played[
                (played["season"] < season)
                | ((played["season"] == season) & (played["week"] < week))
            ]
            periods = prior[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
            window = prior.merge(periods.tail(drift.WINDOW_WEEKS), on=["season", "week"])
            for pos in POSITIONS:
                sub = window[window["position"] == pos]
                pm = meta["positions"][pos]
                feats = monitored(pos)
                bucket = str(drift.week_bucket(drift.window_weeks(sub)[-1]))
                bucket_ref = (
                    pm["drift_reference"]["buckets"].get(bucket) or pm["drift_reference"]["global"]
                )
                before = drift.drift_report(sub, bucket_ref, feats)
                ref, ref_meta = drift.reference_for_window(
                    sub, pm["drift_reference"], training[training["position"] == pos], feats
                )
                after = drift.drift_report(sub, ref, feats, reference=ref_meta)
                rows.append(
                    {
                        "season": season,
                        "target_week": week,
                        "window_weeks": drift.window_weeks(sub),
                        "position": pos,
                        "n": before["n"],
                        "mode": ref_meta["mode"],
                        "bucket_status": before["status"],
                        "bucket_median": before["median_monitored"],
                        "hybrid_status": after["status"],
                        "hybrid_median": after["median_monitored"],
                    }
                )
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    for label, sub in (
        ("target weeks 2-4", df[df.target_week <= 4]),
        ("target weeks 5-18", df[df.target_week >= 5]),
        ("all", df),
    ):
        if sub.empty:
            continue
        b = sub[sub.bucket_status == "hold"]
        h = sub[sub.hybrid_status == "hold"]
        print(
            f"{label}: {sub[['season', 'target_week']].drop_duplicates().shape[0]} windows x {len(POSITIONS)} positions; "
            f"bucket holds {len(b)} {b.groupby('position').size().to_dict()}, hybrid holds {len(h)} {h.groupby('position').size().to_dict()}"
        )
    held = df[(df.bucket_status == "hold") | (df.hybrid_status == "hold")]
    print("\nheld position-windows under either rule:")
    print(held.to_string(index=False))


if __name__ == "__main__":
    main()
