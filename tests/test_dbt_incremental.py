"""Incremental equivalence for slv_player_stats (ADR-0023).

Runs dbt against a scratch DuckDB file (``FFAI_DUCKDB_PATH``) over the committed 40-player
fixture and proves three things about the incremental materialisation:

1. **Append equivalence.** Build from a fixture truncated to all but the last two periods, then
   run incrementally over the full fixture: row count and a content checksum equal a full refresh
   over the full fixture.
2. **Restated week inside the lookback.** Change a stat of an already-loaded row in one of the
   newest ``stats_lookback_periods`` periods and run incrementally: the table equals a full refresh
   over the restated fixture (the correction was picked up, and the derived points moved with it).
3. **Restated week outside the lookback.** A correction older than the lookback is *not* picked up
   by an incremental run (the documented limit of the lookback; a full refresh is the remedy) and
   is picked up by ``--full-refresh``.

Each dbt invocation takes a few seconds; the module runs five. Skipped when dbt is not importable.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from ffai.config import REPO_ROOT

pytest.importorskip("dbt.cli.main")

FIXTURE = Path(__file__).parent / "fixtures" / "player_stats_sample.csv"
MODEL = "slv_player_stats"
KEYS = ["player_id", "season", "week"]
LOOKBACK = 2  # periods; smaller than the project default so case 3 fits in the fixture
DBT_DIR = REPO_ROOT / "dbt"
PACKAGES_DIR = DBT_DIR / "dbt_packages"


def _dbt_bin() -> list[str]:
    """The dbt console script next to the current interpreter, else ``python -m dbt``."""
    script = Path(sys.executable).parent / "dbt"
    if script.exists():
        return [str(script)]
    return [sys.executable, "-m", "dbt"]


def _required_packages() -> list[str]:
    """Package directory names dbt expects: every entry of package-lock.yml (transitive
    dependencies included), else packages.yml."""
    import yaml

    for name in ("package-lock.yml", "packages.yml"):
        path = DBT_DIR / name
        if path.exists():
            entries = yaml.safe_load(path.read_text(encoding="utf-8")).get("packages", [])
            names = [e["package"].split("/")[-1] for e in entries if "package" in e]
            if names:
                return names
    return []


def _packages_installed() -> bool:
    required = _required_packages()
    return bool(required) and all(
        (PACKAGES_DIR / name).is_dir() and any((PACKAGES_DIR / name).iterdir()) for name in required
    )


@pytest.fixture(scope="session", autouse=True)
def dbt_packages() -> None:
    """Install the dbt packages once per session, exactly like ``make dbt-deps``.

    A test that shells out to dbt owns its dependency install (ADR-0030): CI job boundaries are
    not a test's dependency manager. Never skipped: a missing package directory must fail loudly.
    """
    if _packages_installed():
        return
    cmd = [*_dbt_bin(), "deps", "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    # macOS occasionally leaves empty "<pkg> 2" copies next to the installed packages, which makes
    # dbt refuse to run ("expects 3 package(s) ... found 6"); empty directories are safe to drop.
    for d in PACKAGES_DIR.iterdir() if PACKAGES_DIR.exists() else []:
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    if proc.returncode != 0 or not _packages_installed():
        pytest.fail(
            "dbt deps failed or left packages missing\n"
            f"required: {_required_packages()}\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}",
            pytrace=False,
        )


def _dbt_run(db_path: Path, stats_path: Path, *, full_refresh: bool) -> None:
    cmd = [
        *_dbt_bin(),
        "run",
        "--select",
        f"+{MODEL}",
        "--project-dir",
        "dbt",
        "--profiles-dir",
        "dbt",
        "--target",
        "dev",
        "--target-path",
        str(db_path.parent / "target"),
        "--log-path",
        str(db_path.parent / "logs"),
        "--vars",
        json.dumps({"stats_path": str(stats_path), "stats_lookback_periods": LOOKBACK}),
    ]
    if full_refresh:
        cmd.append("--full-refresh")
    env = {**os.environ, "FFAI_DUCKDB_PATH": str(db_path)}
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


def _table(db_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(f"select * from silver.{MODEL}").df()
    finally:
        con.close()
    return df.sort_values(KEYS).reset_index(drop=True)


def _checksum(df: pd.DataFrame) -> str:
    ordered = df[sorted(df.columns)].sort_values(KEYS).reset_index(drop=True)
    return hashlib.sha256(
        pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()
    ).hexdigest()


@pytest.fixture(scope="module")
def fixture_frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


@pytest.fixture(scope="module")
def periods(fixture_frame: pd.DataFrame) -> list[int]:
    keys = sorted(
        {
            int(s) * 100 + int(w)
            for s, w in zip(fixture_frame["season"], fixture_frame["week"], strict=True)
        }
    )
    return keys


def _write(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def _restate(frame: pd.DataFrame, period_key: int) -> tuple[pd.DataFrame, pd.Series]:
    """Add 50 rushing yards (and 1 rushing TD) to the first regular-season fantasy row of a period."""
    out = frame.copy()
    mask = (out["season"] * 100 + out["week"] == period_key) & (out["season_type"] == "REG")
    idx = out[mask].index[0]
    out.loc[idx, "rushing_yards"] = float(out.loc[idx, "rushing_yards"] or 0) + 50.0
    out.loc[idx, "rushing_tds"] = float(out.loc[idx, "rushing_tds"] or 0) + 1.0
    return out, out.loc[idx, KEYS]


def test_incremental_run_equals_full_refresh_after_append(
    tmp_path: Path, fixture_frame: pd.DataFrame, periods: list[int]
) -> None:
    full = _write(fixture_frame, tmp_path / "full.csv")
    cutoff = periods[-2]
    truncated = fixture_frame[fixture_frame["season"] * 100 + fixture_frame["week"] < cutoff]
    trunc = _write(truncated, tmp_path / "truncated.csv")

    ref_db = tmp_path / "ref" / "w.duckdb"
    ref_db.parent.mkdir()
    _dbt_run(ref_db, full, full_refresh=True)
    reference = _table(ref_db)

    inc_db = tmp_path / "inc" / "w.duckdb"
    inc_db.parent.mkdir()
    _dbt_run(inc_db, trunc, full_refresh=True)
    before = _table(inc_db)
    assert len(before) < len(reference)
    _dbt_run(inc_db, full, full_refresh=False)
    after = _table(inc_db)

    assert len(after) == len(reference)
    assert _checksum(after) == _checksum(reference)
    pd.testing.assert_frame_equal(after, reference, check_like=True)


def test_restated_week_inside_lookback_is_picked_up(
    tmp_path: Path, fixture_frame: pd.DataFrame, periods: list[int]
) -> None:
    full = _write(fixture_frame, tmp_path / "full.csv")
    restated_frame, key = _restate(
        fixture_frame, periods[-LOOKBACK]
    )  # newest-but-one period: inside the lookback
    restated = _write(restated_frame, tmp_path / "restated.csv")

    db = tmp_path / "w.duckdb"
    _dbt_run(db, full, full_refresh=True)
    original = _table(db)
    _dbt_run(db, restated, full_refresh=False)
    incremental = _table(db)

    ref_db = tmp_path / "ref.duckdb"
    _dbt_run(ref_db, restated, full_refresh=True)
    reference = _table(ref_db)

    row = incremental.merge(key.to_frame().T.astype(incremental[KEYS].dtypes.to_dict()), on=KEYS)
    orig_row = original.merge(key.to_frame().T.astype(original[KEYS].dtypes.to_dict()), on=KEYS)
    assert len(row) == 1 and len(orig_row) == 1
    assert row["rushing_yards"].iloc[0] == orig_row["rushing_yards"].iloc[0] + 50.0
    assert row["points_ppr"].iloc[0] == pytest.approx(orig_row["points_ppr"].iloc[0] + 5.0 + 6.0)
    assert _checksum(incremental) == _checksum(reference)


def test_restated_week_outside_lookback_needs_full_refresh(
    tmp_path: Path, fixture_frame: pd.DataFrame, periods: list[int]
) -> None:
    full = _write(fixture_frame, tmp_path / "full.csv")
    restated_frame, key = _restate(
        fixture_frame, periods[-(LOOKBACK + 3)]
    )  # older than the lookback
    restated = _write(restated_frame, tmp_path / "restated.csv")

    db = tmp_path / "w.duckdb"
    _dbt_run(db, full, full_refresh=True)
    original = _table(db)
    _dbt_run(db, restated, full_refresh=False)
    incremental = _table(db)
    assert _checksum(incremental) == _checksum(
        original
    ), "a correction older than the lookback must not change the table"

    _dbt_run(db, restated, full_refresh=True)
    refreshed = _table(db)
    row = refreshed.merge(key.to_frame().T.astype(refreshed[KEYS].dtypes.to_dict()), on=KEYS)
    orig_row = original.merge(key.to_frame().T.astype(original[KEYS].dtypes.to_dict()), on=KEYS)
    assert row["rushing_yards"].iloc[0] == orig_row["rushing_yards"].iloc[0] + 50.0
