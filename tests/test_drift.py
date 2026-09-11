"""PSI drift checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffai.eval import drift


def _deciles(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.quantile(values, np.linspace(0, 1, 11))]


def test_same_distribution_has_near_zero_psi() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20000)
    cur = rng.normal(0, 1, 5000)
    assert drift.psi_from_deciles(cur, _deciles(ref)) < 0.02


def test_shifted_distribution_has_large_psi() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 20000)
    cur = rng.normal(2.0, 1, 5000)
    assert drift.psi_from_deciles(cur, _deciles(ref)) > 0.25


def test_constant_feature_is_stable() -> None:
    ref = np.zeros(1000)
    assert drift.psi_from_deciles(np.zeros(100), _deciles(ref)) == 0.0


def test_report_flags_only_monitored_features() -> None:
    rng = np.random.default_rng(2)
    ref = {"a": _deciles(rng.normal(0, 1, 5000)), "b": _deciles(rng.normal(0, 1, 5000))}
    cur = pd.DataFrame({"a": rng.normal(3, 1, 1000), "b": rng.normal(0, 1, 1000)})
    r = drift.drift_report(cur, ref, monitored=["b"])
    assert r["status"] == "ok" and r["hold"] == [] and r["psi"]["a"] > 0.25
    r = drift.drift_report(cur, ref, monitored=["a", "b"])
    assert r["status"] == "hold" and r["hold"] == ["a"]
