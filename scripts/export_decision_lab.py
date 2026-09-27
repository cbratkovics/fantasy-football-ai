#!/usr/bin/env python
"""Export the Decision Lab evidence bundle from committed artifacts (ADR-0032).

    python scripts/export_decision_lab.py                                   # artifacts/decision_lab
    python scripts/export_decision_lab.py --public-copy frontend-next/public/decision-lab
    python scripts/export_decision_lab.py --check  [--public-copy DIR]      # rebuild + compare, no writes
    python scripts/export_decision_lab.py --verify [--public-copy DIR]      # load + verify only

Offline: reads artifacts/marts, artifacts/predictions, artifacts/eval, artifacts/models/*.csv and
metadata, artifacts/runs and artifacts/manifest.json. Never writes outside --out (and the public
copy). ``--check`` builds into a temporary sibling directory and compares every file except
``_build.json`` / ``_last_attempt.json`` (``manifest.json`` is compared with
``code_revision.produced_at`` neutralised, because the export records the git HEAD that produced
it and the commit that contains it necessarily differs); exit 1 on any difference.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

import argparse

from ffai.config import REPO_ROOT
from ffai.decision_lab import bundle, exporter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "decision_lab")
    parser.add_argument("--public-copy", type=Path, default=None)
    parser.add_argument(
        "--check", action="store_true", help="rebuild and compare; never writes --out"
    )
    parser.add_argument("--verify", action="store_true", help="load and verify --out only")
    args = parser.parse_args(argv)

    if args.verify:
        try:
            report = exporter.verify_bundle(args.out, public_copy=args.public_copy)
        except (bundle.BundleError, exporter.ExportError) as exc:
            print(f"VERIFY FAILED: {exc}")
            return 1
        print(
            f"verified {args.out}: {len(report['snapshots'])} snapshots, "
            f"{sum(s['inputs_rows'] for s in report['snapshots'])} input rows, {report['cases']} cases"
            + (f"; public copy {args.public_copy} identical" if args.public_copy else "")
        )
        return 0

    if args.check:
        try:
            diffs = exporter.check_bundle(args.artifacts, args.out, public_copy=args.public_copy)
        except exporter.ExportError as exc:
            print(f"CHECK FAILED: {exc}")
            return 1
        if diffs:
            print("CHECK FAILED: the committed bundle differs from a fresh build:")
            for d in diffs:
                print(f"  {d}")
            return 1
        print(
            f"check ok: {args.out} matches a fresh build"
            + (" and its public copy" if args.public_copy else "")
        )
        return 0

    try:
        manifest = exporter.export_bundle(args.artifacts, args.out, public_copy=args.public_copy)
    except exporter.ExportError as exc:
        print(
            f"EXPORT FAILED (previous bundle untouched; see {args.out / bundle.LAST_ATTEMPT_NAME}): {exc}"
        )
        return 1
    print(exporter.summary(manifest))
    print(
        f"wrote {args.out}" + (f" and mirrored to {args.public_copy}" if args.public_copy else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
