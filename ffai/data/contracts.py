"""Data contracts for the weekly stats frame — a thin wrapper over the dbt silver layer.

The checks themselves are dbt tests in ``dbt/models/silver/_silver.yml`` and
``dbt/tests/silver/`` (ADR-0015): grain uniqueness, required columns, value ranges,
non-negative counting stats, accepted positions, not-null, freshness through the expected
week, newest-week row count versus the prior week, row-count monotonicity, and the scoring-rules
reconciliation. This module only runs ``dbt build --select +tag:silver`` with the run's
parameters as dbt vars and maps ``target/run_results.json`` into the report shape the weekly
policy consumes (``{"ok": bool, "checks": [...], "summary": {...}}``), so a failing silver test
still produces a HOLD.

``summarise_run_results`` is pure (dict in, report out) and unit-tested on a fixture;
``run_silver_contracts`` is the side-effecting entry point the weekly job calls. Drift is not a
contract and stays in ``ffai.eval.drift``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ffai.config import REPO_ROOT

DBT_PROJECT_DIR = REPO_ROOT / "dbt"
DBT_SELECT = "+tag:silver"
DEFAULT_TARGET = "dev"
FAILING_STATUSES = frozenset({"fail", "error"})
# dbt test names -> the check names the weekly log and ADR-0015 mapping table use.
CHECK_NAME_HINTS: tuple[tuple[str, str], ...] = (
    ("assert_stats_fresh_through_expected_week", "freshness"),
    ("row_count_within_pct_of_prior_period", "newest_week_row_count"),
    ("assert_stats_row_count_not_below_prior_run", "row_count_monotonic"),
    ("assert_scoring_rules_reconcile_to_nflverse", "scoring_reconciliation"),
    ("unique_combination_of_columns_slv_player_stats", "grain_unique_player_season_week"),
    ("expect_table_columns_to_contain_set", "required_columns"),
    ("accepted_values_slv_player_stats_position", "positions_in_scope"),
)


def dbt_target() -> str:
    """dbt target for the contracts run: ``FFAI_DBT_TARGET`` (the weekly workflow sets prod)."""
    return os.environ.get("FFAI_DBT_TARGET", DEFAULT_TARGET)


def dbt_full_refresh() -> bool:
    """``FFAI_DBT_FULL_REFRESH=1`` rebuilds the incremental silver model from scratch (ADR-0023)."""
    return os.environ.get("FFAI_DBT_FULL_REFRESH", "").strip().lower() in {"1", "true", "yes"}


def _short_name(unique_id: str) -> str:
    # test.ffai_dbt.not_null_slv_player_stats_player_id.4f2a1b -> not_null_slv_player_stats_player_id
    parts = unique_id.split(".")
    if parts[0] == "test" and len(parts) >= 4:
        return parts[2]
    return parts[-1]


def check_name(unique_id: str) -> str:
    """Map a dbt node id to a stable contract-check name (the dbt test name by default)."""
    short = _short_name(unique_id)
    for needle, name in CHECK_NAME_HINTS:
        if needle in short:
            return name
    return short


def summarise_run_results(run_results: dict[str, Any]) -> dict[str, Any]:
    """Map a dbt ``run_results.json`` dict to the contract report shape.

    A check is ``ok`` unless its status is ``fail`` or ``error``. Skipped tests (their model
    failed to build) are recorded as skipped and do not count on their own; the model error
    already fails the report. Models that errored appear as ``model:<name>`` checks.
    """
    checks: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for r in run_results.get("results", []):
        uid = r.get("unique_id", "")
        status = str(r.get("status", "")).lower()
        kind = uid.split(".")[0]
        counts[status] = counts.get(status, 0) + 1
        if kind == "test" or kind == "unit_test":
            checks.append(
                {
                    "name": check_name(uid),
                    "ok": status not in FAILING_STATUSES,
                    "detail": {
                        "status": status,
                        "failures": r.get("failures"),
                        "message": r.get("message"),
                        "node": uid,
                    },
                }
            )
        elif kind == "model" and status in FAILING_STATUSES:
            checks.append(
                {
                    "name": f"model:{uid.split('.')[-1]}",
                    "ok": False,
                    "detail": {"status": status, "message": r.get("message"), "node": uid},
                }
            )
    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "summary": {
            "dbt": {
                "elapsed_s": run_results.get("elapsed_time"),
                "invocation_id": (run_results.get("metadata") or {}).get("invocation_id"),
                "status_counts": counts,
                "n_checks": len(checks),
            }
        },
    }


def dbt_vars(
    *,
    stats_path: str | Path | None = None,
    target_season: int | None = None,
    target_week: int | None = None,
    expected_through: tuple[int, int] | None = None,
    prior_row_count: int | None = None,
) -> dict[str, Any]:
    """The dbt vars a contracts run needs (paths relative to the repo root)."""
    v: dict[str, Any] = {}
    if stats_path is not None:
        p = Path(stats_path)
        v["stats_path"] = p.relative_to(REPO_ROOT).as_posix() if p.is_absolute() else p.as_posix()
    if target_season is not None and target_week is not None:
        v["target_season"], v["target_week"] = int(target_season), int(target_week)
    if expected_through is not None:
        v["expected_season"], v["expected_week"] = int(expected_through[0]), int(
            expected_through[1]
        )
    if prior_row_count is not None:
        v["prior_row_count"] = int(prior_row_count)
    return v


def dbt_command(
    vars_: dict[str, Any],
    *,
    target: str,
    project_dir: Path = DBT_PROJECT_DIR,
    full_refresh: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "dbt.cli.main",
        "build",
        "--select",
        DBT_SELECT,
        "--indirect-selection",
        "cautious",
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(project_dir),
        "--target",
        target,
        "--vars",
        json.dumps(vars_),
    ]
    if full_refresh:
        cmd.append("--full-refresh")
    return cmd


def run_silver_contracts(
    *,
    stats_path: str | Path | None = None,
    target_season: int | None = None,
    target_week: int | None = None,
    expected_through: tuple[int, int] | None = None,
    prior_row_count: int | None = None,
    target: str | None = None,
    project_dir: Path = DBT_PROJECT_DIR,
) -> dict[str, Any]:
    """Build bronze + silver with their tests and return the contract report.

    Runs dbt as a subprocess from the repository root (file sources are relative to it). A dbt
    exit status of 1 (test failures) or 2 (unhandled error) is mapped through the run results;
    if no run results were written at all the report carries a single failing ``dbt`` check.
    """
    target = target or dbt_target()
    vars_ = dbt_vars(
        stats_path=stats_path,
        target_season=target_season,
        target_week=target_week,
        expected_through=expected_through,
        prior_row_count=prior_row_count,
    )
    cmd = dbt_command(
        vars_, target=target, project_dir=project_dir, full_refresh=dbt_full_refresh()
    )
    results_path = project_dir / "target" / "run_results.json"
    results_path.unlink(missing_ok=True)  # never read a previous invocation's results
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if not results_path.exists():
        return {
            "ok": False,
            "checks": [
                {
                    "name": "dbt",
                    "ok": False,
                    "detail": {"returncode": proc.returncode, "stderr": proc.stderr[-2000:]},
                }
            ],
            "summary": {"dbt": {"target": target, "vars": vars_}},
        }
    report = summarise_run_results(json.loads(results_path.read_text(encoding="utf-8")))
    report["summary"]["dbt"].update(
        {"target": target, "vars": vars_, "returncode": proc.returncode}
    )
    for check in report["checks"]:
        if check["name"] == "freshness" and expected_through is not None:
            check["detail"]["expected_through"] = [
                int(expected_through[0]),
                int(expected_through[1]),
            ]
    if proc.returncode not in (0, 1) and report["ok"]:
        report["ok"] = False
        report["checks"].append(
            {
                "name": "dbt",
                "ok": False,
                "detail": {"returncode": proc.returncode, "stderr": proc.stderr[-2000:]},
            }
        )
    return report
