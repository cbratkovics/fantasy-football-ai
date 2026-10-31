"""Bounded explicit source probe. Exit 0: required sources observed; 2: required source failure.

A zero exit does not establish complete coverage or current recommendation actionability.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from ffai.context.probe import live_sources, probe
from ffai.context.snapshot import iso, utc
from ffai.decision_lab.canonical import content_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int)
    parser.add_argument(
        "--as-of", required=True, help="Timezone-aware decision time or now; normalized to UTC"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--fixtures", type=Path, help="JSON synthetic source envelopes; never live verified"
    )
    mode.add_argument("--live", action="store_true", help="Explicit bounded provider calls")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/context"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--forecast", type=Path, help="Decision Lab inputs snapshot JSON; exact canonical link"
    )
    args = parser.parse_args()
    if args.as_of != "now":
        utc(args.as_of)
    now = iso(dt.datetime.now(dt.UTC))
    if args.fixtures:
        sources = json.loads(args.fixtures.read_text(encoding="utf-8"))["sources"]
        run_mode = "fixture"
    else:
        sources = live_sources(
            args.season,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            now=now,
            clock=lambda: iso(dt.datetime.now(dt.UTC)),
        )
        run_mode = "live"
    forecast = None
    if args.forecast:
        inputs = json.loads(args.forecast.read_text(encoding="utf-8"))
        forecast = {
            "snapshot_id": inputs["snapshot_id"],
            "sha256": content_id(inputs),
            "season": inputs["season"],
            "week": inputs["week"],
            "rows": inputs["rows"],
        }
    now = iso(dt.datetime.now(dt.UTC))
    decision_time = now if args.as_of == "now" else args.as_of
    report, snapshot, code = probe(
        sources,
        season=args.season,
        week=args.week,
        as_of=decision_time,
        observed_at=now,
        mode=run_mode,
        forecast=forecast,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    # Always overwrite current attempt outputs, including failures. No last-success fallback.
    (args.out / "capabilities.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": run_mode,
                "required_sources_usable": report["required_sources_usable"],
                "sources": {k: v["state"] for k, v in report["sources"].items()},
                "exit_code": code,
            }
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
