"""PSI drift checks and the robust ok / warn / hold rule."""

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


def test_week_bucket() -> None:
    assert [drift.week_bucket(w) for w in (1, 4, 5, 8, 9, 12, 13, 16, 17, 18)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
    ]


def test_status_rule_single_spike_is_a_warning_not_a_hold() -> None:
    psi = {"a": 0.9, "b": 0.05, "c": 0.04, "d": 0.03}
    status, flagged, severe = drift.drift_status(psi, ["a", "b", "c", "d"])
    assert status == "warn" and flagged == ["a"] and severe == ["a"]


def test_status_rule_broad_shift_holds() -> None:
    psi = {"a": 0.3, "b": 0.28, "c": 0.26, "d": 0.05}
    assert drift.drift_status(psi, ["a", "b", "c", "d"])[0] == "hold"  # median 0.27


def test_status_rule_two_severe_features_hold() -> None:
    psi = {"a": 0.6, "b": 0.55, "c": 0.02, "d": 0.01, "e": 0.01}
    status, _, severe = drift.drift_status(psi, list(psi))
    assert status == "hold" and severe == ["a", "b"]


def test_status_rule_ok_and_median_warning() -> None:
    assert drift.drift_status({"a": 0.05, "b": 0.02}, ["a", "b"])[0] == "ok"
    assert drift.drift_status({"a": 0.15, "b": 0.12}, ["a", "b"])[0] == "warn"
    assert drift.drift_status({"a": 0.9}, ["zzz"])[0] == "ok"  # nothing monitored present


def test_report_flags_only_monitored_features() -> None:
    rng = np.random.default_rng(2)
    ref = {k: _deciles(rng.normal(0, 1, 5000)) for k in ("a", "b", "c")}
    cur = pd.DataFrame(
        {"a": rng.normal(3, 1, 1000), "b": rng.normal(0, 1, 1000), "c": rng.normal(0, 1, 1000)}
    )
    r = drift.drift_report(cur, ref, monitored=["b", "c"])
    assert r["status"] == "ok" and r["flagged"] == [] and r["psi"]["a"] > 0.25
    r = drift.drift_report(cur, ref, monitored=["a", "b", "c"])
    assert r["status"] == "warn" and r["flagged"] == ["a"] and r["median_monitored"] is not None
    # With only two monitored features the median is not used; one spike stays a warning.
    r = drift.drift_report(cur, ref, monitored=["a", "b"])
    assert r["status"] == "warn"
