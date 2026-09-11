"""Population Stability Index (PSI) per feature against the training reference.

The reference is the decile edges recorded at training time
(``metadata.json -> positions[pos].drift_reference_deciles``). PSI on 10 bins:

    psi = sum_i (cur_i - ref_i) * ln(cur_i / ref_i)

with ``ref_i = 0.1`` by construction and a small epsilon to guard empty bins. See
``drift_status`` for how per-feature values are turned into ok / warn / hold, and ADR-0010 for why
the rule is based on the median of the monitored features rather than any single feature.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

WARN_PSI = 0.10  # any monitored feature above this, or median above it, is a warning
HOLD_PSI = 0.25  # median of the monitored features above this is a HOLD (broad shift)
SEVERE_PSI = 0.50  # two or more monitored features above this is a HOLD (severe shift)
MIN_SEVERE_FEATURES = 2
MIN_MONITORED_FOR_MEDIAN = 3  # below this the median is not a robust statistic
BUCKET_WEEKS = 4
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


def drift_report(
    current: pd.DataFrame,
    reference_deciles: dict[str, list[float]],
    monitored: list[str],
) -> dict[str, Any]:
    """PSI for every feature in ``reference_deciles``; status from :func:`drift_status`.

    Returns ``{"n", "psi": {feature: value}, "monitored", "median_monitored", "flagged" (> HOLD_PSI),
    "severe" (> SEVERE_PSI), "thresholds", "status": "ok" | "warn" | "hold"}``.
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
    }
