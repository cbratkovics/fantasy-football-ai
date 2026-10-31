"""Explicit bounded ingestion, isolated from application startup and requests.

Only sanitized normalized context and capability metadata are exported. Raw payloads remain in
an ignored private cache. Each source call runs in a fresh worker with the client's cache OFF.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ffai.context.snapshot import MAX_FETCH_AGE_SECONDS, SOURCES, build_snapshot, iso, utc
from ffai.decision_lab.canonical import content_id

WORKER = """
import json, sys
import nflreadpy as nfl
from nflreadpy.config import update_config
update_config(cache_mode='off', timeout=15, verbose=False)
name, season = sys.argv[1], int(sys.argv[2])
f = getattr(nfl, 'load_' + name)([season])
print(f.write_json())
"""


def fetch_source(name: str, season: int) -> list[dict]:
    """One official loader call; 15s socket timeout, 30s process limit, no retries."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", WORKER, name, str(season)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            env={**os.environ, "NFLREADPY_CACHE": "off", "NFLREADPY_TIMEOUT": "15"},
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("source_deadline_exceeded") from exc
    except subprocess.CalledProcessError as exc:
        evidence = exc.stderr or ""
        if "404" in evidence:
            raise FileNotFoundError("provider_release_unavailable") from exc
        if "403" in evidence:
            raise PermissionError("source_access_denied") from exc
        raise ConnectionError("provider_request_failed") from exc
    if len(proc.stdout) > 32 * 1024 * 1024:
        raise ValueError("source exceeds 32 MiB context limit")
    rows = json.loads(proc.stdout)
    if not isinstance(rows, list):
        raise ValueError("source is not a row array")
    return rows


def live_sources(
    season: int,
    *,
    cache_dir: Path,
    refresh: bool,
    now: str,
    fetch=fetch_source,
    clock=None,
) -> dict[str, dict]:
    """Same-day cache keeps first observation and fetch time; refresh bypasses both caches.

    Content-identical refresh keeps the original first-observed time. A failed refresh produces
    an explicit current failure, never falls back to a prior successful payload.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in SOURCES:
        path = cache_dir / f"{name}_{season}.json"
        try:
            previous = json.loads(path.read_text()) if path.exists() else None
            if previous is not None:
                utc(previous["fetched_at_utc"])
                content_id(previous["rows"])
        except (OSError, ValueError, KeyError, TypeError):
            previous = None
        if (
            previous
            and not refresh
            and utc(previous["fetched_at_utc"]).date() == utc(now).date()
            and 0
            <= (utc(now) - utc(previous["fetched_at_utc"])).total_seconds()
            <= MAX_FETCH_AGE_SECONDS[name]
        ):
            result[name] = previous
            continue
        try:
            rows = fetch(name, season)
            fetched_at = clock() if clock else now
            digest = content_id(rows)
            first = (
                previous["first_observed_at_utc"]
                if previous and content_id(previous["rows"]) == digest
                else fetched_at
            )
            result[name] = {
                "source_id": f"nflverse:{name}:{season}",
                "state": "partial" if rows else "unavailable",
                "reasons": ["coverage_not_independently_verified", "provider_update_time_unknown"],
                "rows": rows,
                "first_observed_at_utc": first,
                "fetched_at_utc": fetched_at,
                "provider_updated_at_utc": None,
                "coverage_complete": False,
            }
            path.write_text(json.dumps(result[name], allow_nan=False), encoding="utf-8")
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            # Exception class is safe/public; HTTP URLs, private network details and tokens are not.
            result[name] = {
                "source_id": f"nflverse:{name}:{season}",
                "state": (
                    "unavailable" if isinstance(exc, FileNotFoundError) else "network-unverified"
                ),
                "reasons": ["source_fetch_failed", type(exc).__name__],
                "rows": [],
                "first_observed_at_utc": None,
                "fetched_at_utc": now,
                "coverage_complete": False,
            }
    return result


def probe(
    sources: dict[str, dict],
    *,
    season: int,
    week: int | None,
    as_of: str,
    observed_at: str,
    mode: str,
    forecast: dict | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    snapshot = build_snapshot(
        sources,
        season=season,
        week=week,
        as_of=as_of,
        observed_at=observed_at,
        mode=mode,
        forecast=forecast,
    )
    status = {k: snapshot["sources"][k]["state"] for k in SOURCES}
    # Presence is demonstrable, independent completeness is a separate claim. Injuries optional.
    required_ok = all(snapshot["sources"][k]["season_rows"] > 0 for k in ("schedules", "rosters"))
    code = 0 if required_ok else 2
    report = {
        "schema_version": "context_capabilities/1.0",
        "mode": mode,
        "observed_at_utc": iso(utc(observed_at)),
        "requested_as_of_utc": iso(utc(as_of)),
        "target": snapshot["target"],
        "context_snapshot_id": snapshot["context_snapshot_id"],
        "sources": snapshot["sources"],
        "required_sources_usable": required_ok,
        "process_exit_code": code,
        "optional_injuries_state": status["injuries"],
        "coverage": snapshot["coverage"],
        "live_verified": mode == "live" and required_ok,
        "interpretation": "Observed source coverage only; not verified game-day eligibility or historical publication availability",
    }
    return report, snapshot, code
