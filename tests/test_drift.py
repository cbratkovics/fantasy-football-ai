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


def _seasonal_world(
    rng: np.random.Generator, seasons: list[int], n_players: int = 40
) -> pd.DataFrame:
    """A cohort whose feature level rises through the season: week-1 rows sit near 10, week-18 rows
    near 10 + 17 * 0.6. Late-season rows compared with an early-season reference look shifted;
    compared with late-season rows they do not."""
    rows = []
    for season in seasons:
        for week in range(1, 19):
            level = 10.0 + 0.6 * (week - 1)
            for _p in range(n_players):
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "position": "QB",
                        "f_a": rng.normal(level, 2.0),
                        "f_b": rng.normal(level * 2, 5.0),
                        "f_c": rng.normal(3.0, 1.0),
                    }
                )
    return pd.DataFrame(rows)


def _bucket_reference(train: pd.DataFrame, feats: list[str]) -> dict:
    buckets = {}
    b = train["week"].map(drift.week_bucket)
    for k in sorted(b.unique()):
        buckets[str(int(k))] = drift.deciles(train[b == k], feats)
    return {
        "bucket_weeks": drift.BUCKET_WEEKS,
        "global": drift.deciles(train, feats),
        "buckets": buckets,
    }


def test_early_season_window_holds_under_bucket_reference_and_clears_under_hybrid() -> None:
    rng = np.random.default_rng(3)
    feats = ["f_a", "f_b", "f_c"]
    train = _seasonal_world(rng, [2019, 2020, 2021, 2022])
    bucket_ref = _bucket_reference(train, feats)
    current = _seasonal_world(rng, [2025, 2026])
    # target 2026 week 2: last four played weeks are 2025 wk16-18 and 2026 wk1
    window = current[
        ((current["season"] == 2025) & (current["week"] >= 16))
        | ((current["season"] == 2026) & (current["week"] == 1))
    ]
    pooled = drift.drift_report(window, bucket_ref["buckets"]["0"], feats)
    assert pooled["status"] == "hold", pooled["median_monitored"]
    ref, meta = drift.reference_for_window(window, bucket_ref, train, feats)
    hybrid = drift.drift_report(window, ref, feats, reference=meta)
    assert meta["mode"] == "matched"
    assert meta["window_weeks"] == [16, 17, 18, 1] and meta["reference_weeks"] == [1, 16, 17, 18]
    assert hybrid["status"] == "ok", hybrid["median_monitored"]
    assert hybrid["reference"]["training_seasons"] == [2019, 2020, 2021, 2022]


def test_mid_season_window_is_identical_under_both_references() -> None:
    rng = np.random.default_rng(4)
    feats = ["f_a", "f_b", "f_c"]
    train = _seasonal_world(rng, [2019, 2020, 2021, 2022])
    bucket_ref = _bucket_reference(train, feats)
    current = _seasonal_world(rng, [2026])
    window = current[current["week"].between(5, 8)]  # target week 9, window inside the season
    bucket_only = drift.drift_report(window, bucket_ref["buckets"]["1"], feats)
    ref, meta = drift.reference_for_window(window, bucket_ref, train, feats)
    hybrid = drift.drift_report(window, ref, feats, reference=meta)
    assert (
        meta["mode"] == "bucket" and meta["bucket"] == 1 and meta["reference_weeks"] == [5, 6, 7, 8]
    )
    assert hybrid["psi"] == bucket_only["psi"] and hybrid["status"] == bucket_only["status"]


def test_matched_reference_falls_back_to_bucket_when_thin() -> None:
    rng = np.random.default_rng(5)
    feats = ["f_a", "f_b", "f_c"]
    train = _seasonal_world(rng, [2019], n_players=3)  # 3 rows per week: far below the minimum
    bucket_ref = _bucket_reference(train, feats)
    current = _seasonal_world(rng, [2025, 2026], n_players=10)
    window = current[
        ((current["season"] == 2025) & (current["week"] >= 16))
        | ((current["season"] == 2026) & (current["week"] == 1))
    ]
    _, meta = drift.reference_for_window(window, bucket_ref, train, feats)
    assert meta["mode"] == "bucket" and meta["fallback"]
