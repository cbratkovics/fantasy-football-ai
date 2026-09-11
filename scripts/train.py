#!/usr/bin/env python
"""Train champion/challenger candidates on nflverse 2019-2024 and write a model artifact.

python scripts/train.py [--seasons 2019-2024] [--refresh]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

import argparse

from ffai.config import HISTORICAL_SEASONS
from ffai.data import nflverse
from ffai.models import train


def _parse_seasons(text: str) -> list[int]:
    first, last = (int(p) for p in text.split("-", 1))
    return list(range(first, last + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        default=f"{HISTORICAL_SEASONS[0]}-{HISTORICAL_SEASONS[-1]}",
        help="e.g. 2019-2024",
    )
    parser.add_argument("--refresh", action="store_true", help="bypass the parquet cache")
    args = parser.parse_args()
    stats = nflverse.load_weekly_stats(_parse_seasons(args.seasons), refresh=args.refresh)
    meta = train.train_all(stats)
    print(f"model_version={meta['model_version']}")
    for pos, p in meta["positions"].items():
        c = p["candidates"]
        print(
            f"{pos}: champion={p['champion']} "
            + " ".join(
                f"{k}(val {v['val_mae']:.3f}, test {v['test_mae']:.3f})" for k, v in c.items()
            )
            + f" baseline_test={p['baseline_position_mean']['test_mae']:.3f}"
        )


if __name__ == "__main__":
    main()
