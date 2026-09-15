"""Population Stability Index (PSI) per feature against the training reference.

PSI on 10 bins, ``psi = sum_i (cur_i - ref_i) * ln(cur_i / ref_i)`` with ``ref_i = 0.1`` by
construction and a small epsilon to guard empty bins. ``drift_status`` turns per-feature values
into ok / warn / hold (ADR-0010: the median of the monitored features, not any single feature).

Reference (ADR-0031, hybrid): the window is the last ``WINDOW_WEEKS`` played weeks. While the
window sits inside one season, the reference is the week-of-season bucket recorded at training
time (``metadata.json -> positions[pos].drift_reference.buckets``). When the window crosses a
season boundary (target weeks 2-4: last season's weeks 16-18 plus this season's week 1), the
reference is rebuilt from the training seasons' rows at the same week-of-season positions the
window contains, so late-season rows are compared with late-season training rows and week-1
rows with week-1 rows. :func:`reference_for_window` decides and records which mode was used;
every run writes ``artifacts/drift/<run_id>.json`` (``artifacts/schemas/drift_report.schema.json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WARN_PSI = 0.10  # any monitored feature above this, or median above it, is a warning
HOLD_PSI = 0.25  # median of the monitored features above this is a HOLD (broad shift)
SEVERE_PSI = 0.50  # two or more monitored features above this is a HOLD (severe shift)
MIN_SEVERE_FEATURES = 2
MIN_MONITORED_FOR_MEDIAN = 3  # below this the median is not a robust statistic
BUCKET_WEEKS = 4
WINDOW_WEEKS = 4
N_DECILES = 10
# A matched reference needs at least this many training rows, else the bucket reference is used.
MIN_REFERENCE_ROWS = 100
REPORT_VERSION = "1.1"
_EPS = 1e-6


def week_bucket(week: int, size: int = BUCKET_WEEKS) -> int:
    """Week-of-season bucket (weeks 1-4 → 0, 5-8 → 1, 9-12 → 2, 13-16 → 3, 17+ → 4).

    Feature distributions depend on the phase of the season (rolling windows fill up, rosters
    settle), so PSI compares a window of recent weeks against training rows from the same
    bucket. Buckets are computed at training time and stored in the model metadata.
    """
    return min((int(week) - 1) // size, 4)


def drift_status(psi: dict[str, float], monitored: list[str]) -> tuple[str, list[str], list[str]]:
    """Robust status over the monitored features.

    Single-feature PSI on a few hundred rows is noisy (the audit's scan over 60 normal windows
    found a single feature above 0.25 in 78% of them), so a HOLD requires a *broad* shift —
    the median monitored PSI above ``HOLD_PSI`` — or a *severe* one — at least
    ``MIN_SEVERE_FEATURES`` features above ``SEVERE_PSI``. Anything above ``HOLD_PSI`` on a single
    feature, or a median above ``WARN_PSI``, is a warning.
    """
    vals = {
        f: psi[f] for f in monitored if f in psi and psi[f] is not None and not np.isnan(psi[f])
    }
    if not vals:
        return "ok", [], []
    med = float(np.median(list(vals.values())))
    flagged = sorted(f for f, v in vals.items() if v > HOLD_PSI)
    severe = sorted(f for f, v in vals.items() if v > SEVERE_PSI)
    broad = med > HOLD_PSI and len(vals) >= MIN_MONITORED_FOR_MEDIAN
    if broad or len(severe) >= MIN_SEVERE_FEATURES:
        return "hold", flagged, severe
    if flagged or med > WARN_PSI:
        return "warn", flagged, severe
    return "ok", flagged, severe


def psi_from_deciles(values: np.ndarray, decile_edges: list[float]) -> float:
    """PSI of ``values`` against a reference whose decile edges are ``decile_edges`` (11 edges)."""
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    edges = np.asarray(decile_edges, dtype="float64")
    # Collapse duplicate edges (e.g. many zeros) so bins stay well defined.
    uniq = np.unique(edges)
    if uniq.size < 2:
        return 0.0
    ref_counts = np.histogram(edges[:-1], bins=uniq)[0]  # how many original deciles per bin
    ref = ref_counts / ref_counts.sum()
    inner = uniq.copy()
    inner[0], inner[-1] = -np.inf, np.inf
    cur = np.histogram(values, bins=inner)[0] / values.size
    ref = np.clip(ref, _EPS, None)
    cur = np.clip(cur, _EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def deciles(frame: pd.DataFrame, features: list[str]) -> dict[str, list[float]]:
    """Decile edges (11 points) per feature; the same construction as the training reference."""
    qs = np.linspace(0, 1, N_DECILES + 1)
    return {f: [float(v) for v in np.quantile(frame[f].to_numpy(), qs)] for f in features}


def window_weeks(window: pd.DataFrame) -> list[int]:
    """The window's weeks in time order (season, week), e.g. ``[16, 17, 18, 1]``."""
    periods = window[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    return [int(w) for w in periods["week"]]


def reference_for_window(
    window: pd.DataFrame,
    bucket_reference: dict[str, Any],
    training_rows: pd.DataFrame,
    monitored: list[str],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """The hybrid reference (ADR-0031) and a description of how it was chosen.

    ``bucket_reference`` is ``metadata.json -> drift_reference`` for the cohort (``buckets`` and
    ``global``); ``training_rows`` are the cohort's feature rows of the training seasons, with
    ``season`` and ``week`` columns. When the window spans more than one season the reference is
    the training rows at the window's week-of-season positions (mode ``matched``); otherwise, or
    when fewer than ``MIN_REFERENCE_ROWS`` matched rows exist, the training-time bucket of the
    newest week (mode ``bucket``).
    """
    weeks = window_weeks(window)
    seasons = sorted(int(s) for s in window["season"].unique())
    crosses = len(seasons) > 1
    training_seasons = sorted(int(s) for s in training_rows["season"].unique())
    if crosses:
        rows = training_rows[training_rows["week"].isin(weeks)]
        if len(rows) >= MIN_REFERENCE_ROWS:
            return deciles(rows, monitored), {
                "mode": "matched",
                "window_seasons": seasons,
                "window_weeks": weeks,
                "reference_weeks": sorted(set(weeks)),
                "training_seasons": training_seasons,
                "n_reference_rows": int(len(rows)),
            }
    bucket = week_bucket(weeks[-1]) if weeks else 0
    ref = bucket_reference["buckets"].get(str(bucket)) or bucket_reference["global"]
    return ref, {
        "mode": "bucket",
        "window_seasons": seasons,
        "window_weeks": weeks,
        "reference_weeks": list(
            range(bucket * BUCKET_WEEKS + 1, bucket * BUCKET_WEEKS + BUCKET_WEEKS + 1)
        ),
        "bucket": int(bucket),
        "training_seasons": training_seasons,
        "n_reference_rows": None,
        "fallback": "fewer than MIN_REFERENCE_ROWS matched rows" if crosses else None,
    }


def drift_report(
    current: pd.DataFrame,
    reference_deciles: dict[str, list[float]],
    monitored: list[str],
    *,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """PSI for every feature in ``reference_deciles``; status from :func:`drift_status`.

    Returns ``{"n", "psi": {feature: value}, "monitored", "median_monitored", "flagged" (> HOLD_PSI),
    "severe" (> SEVERE_PSI), "thresholds", "status": "ok" | "warn" | "hold", "reference"}`` where
    ``reference`` is the description returned by :func:`reference_for_window` (or None).
    """
    psi: dict[str, float] = {}
    for feature, edges in reference_deciles.items():
        if feature in current.columns:
            psi[feature] = psi_from_deciles(current[feature].to_numpy(), edges)
    status, flagged, severe = drift_status(psi, monitored)
    vals = [psi[f] for f in monitored if f in psi and not np.isnan(psi[f])]
    return {
        "n": int(len(current)),
        "psi": {k: (None if np.isnan(v) else round(v, 4)) for k, v in psi.items()},
        "monitored": list(monitored),
        "median_monitored": round(float(np.median(vals)), 4) if vals else None,
        "flagged": flagged,
        "severe": severe,
        "thresholds": {
            "warn": WARN_PSI,
            "hold_median": HOLD_PSI,
            "severe": SEVERE_PSI,
            "min_severe_features": MIN_SEVERE_FEATURES,
        },
        "status": status,
        "reference": reference,
    }


def run_report(
    *,
    run_id: str,
    at_utc: str,
    season: int,
    week: int,
    model_version: str,
    status: str,
    positions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """The per-run monitoring artifact written to ``artifacts/drift/<run_id>.json``."""
    return {
        "drift_report_version": REPORT_VERSION,
        "run_id": run_id,
        "at_utc": at_utc,
        "season": int(season),
        "week": int(week),
        "model_version": model_version,
        "window_weeks_max": WINDOW_WEEKS,
        "status": status,
        "positions": positions,
    }


def write_run_report(report: dict[str, Any], artifacts: Path) -> Path:
    path = artifacts / "drift" / f"{report['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path
