"""Actual scoring uses observed evidence, never the forecast's reception estimate."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ffai.scoring import REQUIRED_COLUMNS, RULES_ID
from ffai.serve import app as api
from ffai.serve.actuals import resolve_actual


def http_actual_client(monkeypatch, rows, *, marts=None):
    """Exercise HTTP serialization with local synthetic evidence, without model loading."""
    payload = {
        "predictions": rows,
        "model_version": "synthetic-acceptance-model",
        "feature_version": "asof_v1",
        "generated_at_utc": "2026-10-31T00:00:00Z",
        "data_through": {"season": 2026, "week": 7},
        "scoring": {"derivation": "estimated receptions", "interval": "residual offsets"},
    }
    state = SimpleNamespace(
        predictions={(2026, 8): payload},
        marts=marts,
        scored_seasons=[],
        model_version=payload["model_version"],
        metadata={"feature_version": "asof_v1"},
        loaded_at_utc="2026-10-31T13:00:00Z",
    )
    monkeypatch.setattr(api, "_state", lambda: state)
    return TestClient(api.app)


def receiver_http_row(**overrides):
    return {
        "player_id": "synthetic-acceptance-receiver",
        "name": "Synthetic acceptance receiver",
        "position": "WR",
        "prediction": 20,
        "floor": 10,
        "ceiling": 30,
        "receptions_estimate": 2,
        "candidate": "rf",
        "actual": 12,
        "actual_receptions": 6,
        "actual_rules_id": RULES_ID,
        "actual_source_revision": "same-outcome",
        "actual_receptions_revision": "same-outcome",
        **overrides,
    }


def test_http_receiver_actuals_are_6_9_12_with_estimated_receptions_two(monkeypatch):
    client = http_actual_client(monkeypatch, [receiver_http_row()])
    for fmt, expected in (("standard", 6), ("half", 9), ("ppr", 12)):
        response = client.get(f"/predictions/2026/8?scoring={fmt}")
        assert response.status_code == 200
        record = response.json()["predictions"][0]
        assert record["actual"] == expected
        assert record["actual_scoring_format"] == fmt
        assert record["receptions_estimate"] == 2
        assert record["actual_null_reason"] is None


def test_http_player_history_does_not_reintroduce_conflicting_weekly_actual(monkeypatch):
    client = http_actual_client(
        monkeypatch, [receiver_http_row(actual_receptions_revision="different-outcome")]
    )
    prediction = client.get("/predictions/2026/8?scoring=ppr").json()["predictions"][0]
    history = client.get("/players/synthetic-acceptance-receiver").json()["history"][0]
    assert prediction["actual"] is None
    assert history["actual"] is None
    assert history["actual_null_reason"] == "outcome_revision_mismatch"
    assert history["actual_scoring_format"] == "ppr"


@pytest.mark.parametrize(
    "fmt,actual,mart_rows,expected,reason",
    [
        ("ppr", 0, None, 0, None),
        ("ppr", -2, None, -2, None),
        ("ppr", 12, None, 12, None),
        ("standard", 12, None, None, "conversion_evidence_missing"),
        ("half", None, None, None, "outcome_unavailable"),
        (
            "standard",
            12,
            [
                {"actual": 12, "actual_standard": 6, "candidate": "rf"},
                {"actual": 12, "actual_standard": 6, "candidate": "xgb"},
            ],
            6,
            None,
        ),
        (
            "standard",
            12,
            [{"actual": 12, "actual_standard": 6}, {"actual": 12, "actual_standard": 7}],
            None,
            "conflicting_mart_outcomes",
        ),
    ],
)
def test_http_legacy_absent_marts_and_duplicate_outcome_boundaries(
    monkeypatch, fmt, actual, mart_rows, expected, reason
):
    row = receiver_http_row(actual=actual)
    for key in list(row):
        if key.startswith("actual_"):
            del row[key]
    mart = (
        None
        if mart_rows is None
        else SimpleNamespace(actual_outcomes=lambda season, week: {row["player_id"]: mart_rows})
    )
    client = http_actual_client(monkeypatch, [row], marts=mart)
    response = client.get(f"/predictions/2026/8?scoring={fmt}")
    assert response.status_code == 200
    observation = response.json()["predictions"][0]
    assert observation["actual"] == expected
    assert observation["actual_null_reason"] == reason
    assert observation["actual_scoring_format"] == fmt


def test_receiver_actual_uses_six_real_receptions_not_two_estimated(monkeypatch):
    row = {
        "player_id": "receiver",
        "position": "WR",
        "prediction": 20,
        "floor": 10,
        "ceiling": 30,
        "receptions_estimate": 2,
        "candidate": "rf",
        "actual": 12,
        "actual_receptions": 6,
        "actual_rules_id": "ffai-nflverse-v1",
        "actual_source_revision": "same-outcome",
        "actual_receptions_revision": "same-outcome",
    }
    payload = {
        "predictions": [row],
        "model_version": "m",
        "feature_version": "f",
        "generated_at_utc": "2026-10-31T00:00:00Z",
        "data_through": {"season": 2026, "week": 7},
        "scoring": {"derivation": "estimated receptions", "interval": "residual offsets"},
    }
    monkeypatch.setattr(
        api, "_state", lambda: SimpleNamespace(predictions={(2026, 8): payload}, marts=None)
    )
    for fmt, expected in (("standard", 6), ("half", 9), ("ppr", 12)):
        result = api.predictions(2026, 8, fmt, None)
        assert result.predictions[0].actual == expected


def stat_record(**values):
    stats = dict.fromkeys(REQUIRED_COLUMNS, None)
    stats.update(values)
    return {
        "actual_stats": stats,
        "actual_rules_id": RULES_ID,
        "actual_source_revision": "r1",
        "actual_stats_revision": "r1",
    }


@pytest.mark.parametrize("fmt,expected", [("standard", 6), ("half", 9), ("ppr", 12)])
def test_hand_calculated_complete_receiver_stat_row(fmt, expected):
    record = stat_record(receiving_yards=60, receptions=6)
    record["receptions_estimate"] = 2
    assert resolve_actual(record, fmt).actual == expected


@pytest.mark.parametrize("fmt,expected", [("standard", 21), ("half", 21), ("ppr", 21)])
def test_qb_rushing_turnover_two_point_and_special_teams(fmt, expected):
    # 250/25 + 2*4 - 2 + 30/10 + 6 - 2*2 = 21; +2 conversion -2 sack = 21.
    r = stat_record(
        passing_yards=250,
        passing_tds=2,
        passing_interceptions=1,
        rushing_yards=30,
        rushing_tds=1,
        rushing_fumbles_lost=2,
        passing_2pt_conversions=1,
        sack_fumbles_lost=1,
    )
    assert resolve_actual(r, fmt).actual == expected


@pytest.mark.parametrize("value", [0, -2])
def test_legacy_ppr_preserves_observed_zero_and_negative_without_marts(value):
    result = resolve_actual({"actual": value}, "ppr")
    assert result.actual == value
    assert result.actual_null_reason is None
    assert result.actual_scoring_format == "ppr"


def test_missing_conversion_never_uses_estimated_receptions():
    assert resolve_actual({"actual": 12, "receptions_estimate": 2}, "standard").actual is None
    assert (
        resolve_actual({"actual": 12}, "half").actual_null_reason == "conversion_evidence_missing"
    )
    assert resolve_actual({}, "standard").actual_null_reason == "outcome_unavailable"


def test_missing_stat_column_is_different_from_provider_null_no_opportunity():
    r = stat_record()
    assert resolve_actual(r, "standard").actual == 0
    del r["actual_stats"]["receptions"]
    assert resolve_actual(r, "standard").actual_null_reason == "invalid_stat_components"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_nonfinite_outcomes_are_rejected(value):
    assert resolve_actual({"actual": value}, "ppr").actual is None
    assert resolve_actual({"actual": value}, "standard").actual_null_reason == "nonfinite_actual"


def test_nonfinite_components_are_not_provider_nulls():
    assert resolve_actual(stat_record(receiving_yards=float("inf")), "standard").actual is None


def test_actual_receptions_require_same_revision():
    r = {
        "actual": 12,
        "actual_receptions": 6,
        "actual_source_revision": "old",
        "actual_receptions_revision": "corrected",
    }
    assert resolve_actual(r, "standard").actual_null_reason == "outcome_revision_mismatch"


def test_component_disagreement_with_stored_ppr_is_exposed():
    r = stat_record(receiving_yards=60, receptions=6)
    r["actual"] = 13
    assert resolve_actual(r, "standard").actual_null_reason == "conflicting_outcome_sources"


def test_validated_format_artifact_and_rules():
    r = {
        "actual": 12,
        "actual_rules_id": RULES_ID,
        "actuals_by_format": {"standard": 6, "half": 9, "ppr": 12},
    }
    assert resolve_actual(r, "half").actual == 9
    r["actual_rules_id"] = "custom-six-point-pass-td"
    assert resolve_actual(r, "standard").actual_null_reason == "scoring_rules_mismatch"


def test_duplicate_mart_outcomes_reduce_only_after_agreement():
    rows = [
        {"actual": 12, "actual_standard": 6, "actual_half": 9, "candidate": c}
        for c in ("rf", "xgb")
    ]
    r = resolve_actual({"actual": 12}, "standard", outcomes=rows)
    assert r.actual == 6 and r.actual_source == "exported_actuals_fact"
    rows[1]["actual_standard"] = 7
    assert (
        resolve_actual({"actual": 12}, "standard", outcomes=rows).actual_null_reason
        == "conflicting_mart_outcomes"
    )


def test_corrected_mart_does_not_mix_with_old_artifact():
    assert (
        resolve_actual(
            {"actual": 12}, "half", outcomes=[{"actual": 13, "actual_half": 10}]
        ).actual_null_reason
        == "conflicting_outcome_sources"
    )


def test_different_conversion_sources_must_agree():
    r = {"actual": 12, "actual_rules_id": RULES_ID, "actuals_by_format": {"standard": 6, "ppr": 12}}
    result = resolve_actual(r, "standard", outcomes=[{"actual": 12, "actual_standard": 7}])
    assert result.actual is None and result.actual_null_reason == "conflicting_outcome_sources"


def test_unsupported_actual_format_rejected():
    with pytest.raises(ValueError):
        resolve_actual({"actual": 12}, "custom")


def test_future_attachment_adds_format_values_without_changing_legacy_ppr():
    import pandas as pd

    from ffai.pipeline.score import attach_actuals

    row = dict.fromkeys(REQUIRED_COLUMNS, None)
    row.update(
        player_id="receiver",
        season=2026,
        week=8,
        fantasy_points_ppr=12,
        receiving_yards=60,
        receptions=6,
    )
    payload = {
        "season": 2026,
        "week": 8,
        "predictions": [
            {"player_id": "receiver", "receptions_estimate": 2},
            {"player_id": "missing"},
        ],
    }
    result, count = attach_actuals(payload, pd.DataFrame([row]))
    observed, absent = result["predictions"]
    assert count == 1
    assert observed["actual"] == 12
    assert observed["actuals_by_format"] == {"ppr": 12, "half": 9, "standard": 6}
    assert observed["actual_receptions"] == 6
    assert observed["actual_source_revision"] == observed["actual_receptions_revision"]
    assert absent["actual"] is None


def test_future_attachment_rejects_conflicting_grain():
    import pandas as pd

    from ffai.pipeline.score import attach_actuals

    payload = {"season": 2026, "week": 8, "predictions": [{"player_id": "a"}]}
    rows = [
        {"player_id": "a", "season": 2026, "week": 8, "fantasy_points_ppr": n} for n in (12, 13)
    ]
    with pytest.raises(ValueError, match="conflicting actual"):
        attach_actuals(payload, pd.DataFrame(rows))


def test_optional_mart_with_missing_format_columns_is_usable_for_ppr_only():
    assert resolve_actual({}, "ppr", outcomes=[{"actual": 0}]).actual == 0
    r = resolve_actual({"actual": 12}, "standard", outcomes=[{"actual": 12}])
    assert r.actual is None and r.actual_null_reason == "conversion_evidence_missing"


def test_response_declares_requested_actual_and_rules(monkeypatch):
    row = {
        "player_id": "a",
        "position": "WR",
        "prediction": 12,
        "floor": 0,
        "ceiling": 20,
        "receptions_estimate": 2,
        "candidate": "rf",
        "actual": 0,
    }
    payload = {
        "predictions": [row],
        "model_version": "m",
        "feature_version": "f",
        "generated_at_utc": "2026-10-31T00:00:00Z",
        "data_through": {"season": 2026, "week": 7},
        "scoring": {"derivation": "estimated receptions", "interval": "residual offsets"},
    }
    monkeypatch.setattr(
        api, "_state", lambda: SimpleNamespace(predictions={(2026, 8): payload}, marts=None)
    )
    body = api.predictions(2026, 8, "standard", None).model_dump()
    assert body["scoring_rules_id"] == RULES_ID
    assert "4/TD" in body["scoring_rules_description"]
    actual = body["predictions"][0]
    assert actual["actual"] is None
    assert actual["actual_scoring_format"] == "standard"
    assert actual["actual_null_reason"] == "conversion_evidence_missing"
    assert api.predictions(2026, 8, "ppr", None).predictions[0].actual == 0


@pytest.mark.parametrize("fmt", ["ppr", "half", "standard"])
def test_ppr_also_validates_explicit_conflicts_and_duplicate_outcomes(fmt):
    record = {
        "actual": 12,
        "actual_rules_id": RULES_ID,
        "actuals_by_format": {"ppr": 13, "half": 10, "standard": 7},
    }
    assert resolve_actual(record, fmt).actual_null_reason == "conflicting_outcome_sources"
    record = {"actual": 12}
    rows = [
        {"actual": 12, "actual_half": 9, "actual_standard": 6},
        {"actual": 13, "actual_half": 10, "actual_standard": 7},
    ]
    assert (
        resolve_actual(record, fmt, outcomes=rows).actual_null_reason == "conflicting_mart_outcomes"
    )


def test_ppr_validates_conflicting_realized_components():
    r = stat_record(receiving_yards=60, receptions=6)
    r["actual"] = 13
    assert resolve_actual(r, "ppr").actual_null_reason == "conflicting_outcome_sources"


def test_repeat_attachment_drops_old_revision_when_corrected_row_incomplete_or_missing():
    import pandas as pd

    from ffai.pipeline.score import attach_actuals

    row = dict.fromkeys(REQUIRED_COLUMNS, None)
    row.update(
        player_id="a", season=2026, week=8, fantasy_points_ppr=12, receiving_yards=60, receptions=6
    )
    payload = {"season": 2026, "week": 8, "predictions": [{"player_id": "a"}]}
    payload, _ = attach_actuals(payload, pd.DataFrame([row]))
    old_revision = payload["predictions"][0]["actual_source_revision"]
    row.update(fantasy_points_ppr=13, receiving_yards=70)
    payload, _ = attach_actuals(payload, pd.DataFrame([row]))
    record = payload["predictions"][0]
    assert record["actual_source_revision"] != old_revision
    assert record["actual"] == 13
    assert record["actuals_by_format"]["standard"] == 7
    del row["receptions"]
    payload, _ = attach_actuals(payload, pd.DataFrame([row]))
    record = payload["predictions"][0]
    assert record["actual"] == 13
    assert "actuals_by_format" not in record and "actual_source_revision" not in record
    payload, count = attach_actuals(payload, pd.DataFrame([row]).iloc[:0])
    assert count == 0 and payload["predictions"][0]["actual"] is None
