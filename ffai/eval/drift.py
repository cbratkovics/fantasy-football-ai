"""Population Stability Index (PSI) per feature against the training reference.

The reference is the decile edges recorded at training time
(``metadata.json -> positions[pos].drift_reference_deciles``). PSI on 10 bins:

    psi = sum_i (cur_i - ref_i) * ln(cur_i / ref_i)

with ``ref_i = 0.1`` by construction and a small epsilon to guard empty bins. Conventional
thresholds: < 0.10 no change, 0.10-0.25 moderate (warn), > 0.25 large (HOLD in the weekly job
when it hits a top-10-importance feature).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

WARN_PSI = 0.10
HOLD_PSI = 0.25
_EPS = 1e-6


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
    *,
    warn: float = WARN_PSI,
    hold: float = HOLD_PSI,
) -> dict[str, Any]:
    """PSI for every feature in ``reference_deciles``; flags for the ``monitored`` subset.

    Returns ``{"psi": {feature: value}, "warn": [...], "hold": [...], "n": int, "status": ...}``
    where ``status`` is ``"ok" | "warn" | "hold"`` based only on monitored features.
    """
    psi: dict[str, float] = {}
    for feature, edges in reference_deciles.items():
        if feature in current.columns:
            psi[feature] = psi_from_deciles(current[feature].to_numpy(), edges)
    warned = sorted(f for f in monitored if f in psi and warn <= psi[f] <= hold)
    held = sorted(f for f in monitored if f in psi and psi[f] > hold)
    status = "hold" if held else ("warn" if warned else "ok")
    return {
        "n": int(len(current)),
        "psi": {k: (None if np.isnan(v) else round(v, 4)) for k, v in psi.items()},
        "monitored": list(monitored),
        "warn": warned,
        "hold": held,
        "thresholds": {"warn": warn, "hold": hold},
        "status": status,
    }
