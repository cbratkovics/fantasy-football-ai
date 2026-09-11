#!/usr/bin/env python
"""Run the autonomous weekly job (see ffai/pipeline/weekly.py).

    python scripts/run_weekly.py [--season S --week W] [--dry-run]

Without --season/--week the current season and next week are read from nflverse schedules.
--dry-run computes everything but writes no artifact, manifest, or run log. Exit code is 0 for
PUBLISH/PROMOTE and 2 for HOLD so the workflow can branch on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

import argparse
import json

from ffai.pipeline import weekly


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.season is None) != (args.week is None):
        parser.error("--season and --week must be given together")
    log = weekly.run_weekly(season=args.season, week=args.week, dry_run=args.dry_run)
    print(json.dumps({k: v for k, v in log.items() if k != "steps"}, indent=2))
    for s in log["steps"]:
        print(
            f"- {s['step']}: "
            + ", ".join(f"{k}={v}" for k, v in s.items() if k != "step" and k != "positions")
        )
    return 2 if log["action"] == weekly.ACTION_HOLD else 0


if __name__ == "__main__":
    sys.exit(main())
