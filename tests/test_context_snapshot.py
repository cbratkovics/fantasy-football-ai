"""Independent synthetic expectations; no provider access or historical publication claims."""

import copy
import json
from pathlib import Path

import pytest

from ffai.context.probe import live_sources, probe
from ffai.context.snapshot import build_snapshot, kickoff, load_snapshot, validate_snapshot
from ffai.decision_lab.canonical import content_id

FIXTURE = Path(__file__).parent / "fixtures/context/sources.json"
AS_OF = "2026-10-31T13:00:00Z"


def sources():
    return json.loads(FIXTURE.read_text())["sources"]


def build(src=None, **kwargs):
    return build_snapshot(
        src or sources(), season=2026, week=9, as_of=AS_OF, observed_at=AS_OF, **kwargs
    )


def player(s, pid):
    return next(p for p in s["players"] if p["player_id"] == pid)


def forecast():
    return {
        "snapshot_id": "weekly-9",
        "sha256": "a" * 64,
        "season": 2026,
        "week": 9,
        "rows": [{"player_id": "traded", "team": "OLD"}, {"player_id": "ghost", "team": "A"}],
    }


def test_roster_universe_retains_rookie_and_trade_independent_of_forecast():
    s = build(forecast=forecast())
    assert player(s, "rookie")["forecast"]["state"] == "missing_history"
    traded = player(s, "traded")
    assert traded["current_team"] == "B"
    assert traded["forecast"]["forecast_team"] == "OLD"
    assert traded["game"]["opponent"] == "A"
    assert player(s, "ghost")["identity_state"] == "unresolved"
    assert s["coverage"]["roster_identities"] == 4


def test_duplicate_names_never_join_and_availability_distinct():
    s = build()
    assert player(s, "rookie")["name"] == player(s, "traded")["name"]
    assert player(s, "rookie")["availability"]["state"] == "unknown"
    assert player(s, "traded")["availability"] == {
        "state": "questionable",
        "designation": "Questionable",
        "basis": "provider_report",
    }
    assert player(s, "out")["availability"]["state"] == "unavailable"


def test_complete_schedule_bye_differs_from_incomplete_schedule():
    assert player(build(), "bye")["game"]["state"] == "bye"
    src = sources()
    src["schedules"]["coverage_complete"] = False
    assert player(build(src), "bye")["game"]["state"] == "unknown"


def test_conflicting_roster_identity_is_unresolved_regardless_of_order():
    src = sources()
    src["rosters"]["rows"].append(
        {"season": 2026, "gsis_id": "traded", "team": "A", "position": "RB"}
    )
    one = player(build(src), "traded")
    src["rosters"]["rows"].reverse()
    two = player(build(src), "traded")
    assert one["identity_state"] == "unresolved"
    assert one["current_team"] is None
    assert one["game"] == two["game"]


def test_duplicate_equivalent_roster_and_game_rows_do_not_multiply():
    src = sources()
    src["rosters"]["rows"].append(copy.deepcopy(src["rosters"]["rows"][0]))
    src["schedules"]["rows"].append(copy.deepcopy(src["schedules"]["rows"][0]))
    assert len(build(src)["players"]) == 4
    assert player(build(src), "rookie")["game"]["game_id"] == "2026_09_A_B"


def test_schedule_revision_conflict_not_arbitrary_selection():
    src = sources()
    variant = copy.deepcopy(src["schedules"]["rows"][0])
    variant["gametime"] = "16:25"
    src["schedules"]["rows"].append(variant)
    assert "schedule_conflict" in player(build(src), "rookie")["reasons"]
    assert player(build(src), "rookie")["game"]["kickoff_utc"] is None


@pytest.mark.parametrize(
    ("day", "time", "expected"),
    [
        ("2026-11-01", "09:30", "2026-11-01T14:30:00+00:00"),
        ("2026-10-25", "09:30", "2026-10-25T13:30:00+00:00"),
        ("2026-11-01", "01:30", None),
        ("2026-03-08", "02:30", None),
        ("2026-11-01", None, None),
    ],
)
def test_timezone_and_early_kickoff(day, time, expected):
    actual, reason = kickoff({"gameday": day, "gametime": time})
    assert actual == expected
    assert (reason is None) == (expected is not None)


def test_postponement_preserved_and_outcomes_not_projected():
    src = sources()
    src["schedules"]["rows"][0]["schedule_state"] = "postponed"
    s = build(src)
    assert player(s, "rookie")["game"]["state"] == "postponed"
    assert "home_score" not in json.dumps(s)
    assert "away_score" not in json.dumps(s)


def test_future_row_does_not_change_earlier_evidence_or_identity():
    src = sources()
    old = build(src)
    src["injuries"]["rows"].append(
        {
            "season": 2026,
            "week": 9,
            "gsis_id": "rookie",
            "report_status": "Out",
            "first_observed_at_utc": "2026-11-01T12:00:00Z",
        }
    )
    new = build(src)
    assert new == old


def test_fresh_download_cannot_be_backdated_to_pre_game_availability():
    src = sources()
    src["rosters"]["first_observed_at_utc"] = "2026-11-01T12:00:00Z"
    s = build(src)
    assert s["sources"]["rosters"]["as_of_rows"] == 0
    assert s["players"] == []


def test_old_content_fresh_fetch_keeps_content_observation_age_visible():
    src = sources()
    src["rosters"]["first_observed_at_utc"] = "2026-01-01T00:00:00Z"
    s = build(src)
    assert s["sources"]["rosters"]["first_observed_at_utc"] == "2026-01-01T00:00:00Z"
    assert s["sources"]["rosters"]["provider_updated_at_utc"] is None
    assert "Provider publication time unknown" in s["sources"]["rosters"]["provider_time_note"]


def test_stale_fetch_is_not_a_refresh_guarantee():
    src = sources()
    src["rosters"]["fetched_at_utc"] = "2026-10-29T12:00:00Z"
    assert "fetch_stale" in build(src)["sources"]["rosters"]["reasons"]


def test_no_next_regular_period_and_unknown_schedule_are_distinct():
    src = sources()
    opts = dict(
        season=2026, week=None, as_of="2027-02-01T12:00:00Z", observed_at="2027-02-01T12:00:00Z"
    )
    assert build_snapshot(src, **opts)["target"]["state"] == "no_target"
    src["schedules"]["coverage_complete"] = False
    assert build_snapshot(src, **opts)["target"]["state"] == "target_unknown"
    with pytest.raises(ValueError, match="regular-season"):
        build_snapshot(src, **{**opts, "week": 19})


def test_forecast_link_and_duplicate_rows_fail_closed():
    f = forecast()
    f["week"] = 8
    assert player(build(forecast=f), "traded")["forecast"]["state"] == "unresolved_reference"
    f = forecast()
    f["rows"].append(f["rows"][0])
    assert player(build(forecast=f), "traded")["forecast"]["state"] == "unresolved_reference"


def test_digest_contract_load_and_tamper(tmp_path):
    s = build()
    p = tmp_path / "snapshot.json"
    p.write_text(json.dumps(s))
    assert load_snapshot(p) == s
    s["players"][0]["current_team"] = "FAKE"
    with pytest.raises(ValueError, match="digest"):
        validate_snapshot(s)


def test_probe_fixture_never_live_verified_and_optional_failure_preserves_required():
    src = sources()
    src["injuries"] = {
        "state": "network-unverified",
        "rows": [],
        "reasons": ["provider_unavailable"],
    }
    report, snapshot, code = probe(
        src, season=2026, week=9, as_of=AS_OF, observed_at=AS_OF, mode="fixture"
    )
    assert code == 0 and report["required_sources_usable"] is True
    assert report["live_verified"] is False
    assert len(snapshot["players"]) == 4
    src["rosters"] = {}
    assert probe(src, season=2026, week=9, as_of=AS_OF, observed_at=AS_OF, mode="fixture")[2] == 2


def test_same_day_cache_refresh_and_failed_attempt_do_not_relabel_old_success(tmp_path):
    calls = []

    def fetch(name, season):
        calls.append(name)
        return [{"season": season, "id": name}]

    old = live_sources(
        2026, cache_dir=tmp_path, refresh=False, now="2026-10-31T12:00:00Z", fetch=fetch
    )
    cached = live_sources(
        2026, cache_dir=tmp_path, refresh=False, now="2026-10-31T13:00:00Z", fetch=fetch
    )
    assert len(calls) == 3 and cached == old
    refreshed = live_sources(
        2026, cache_dir=tmp_path, refresh=True, now="2026-10-31T13:00:00Z", fetch=fetch
    )
    assert len(calls) == 6
    assert refreshed["rosters"]["first_observed_at_utc"] == old["rosters"]["first_observed_at_utc"]
    assert refreshed["rosters"]["fetched_at_utc"] == "2026-10-31T13:00:00Z"

    def failed(*args):
        raise OSError("blocked")

    failed_result = live_sources(
        2026, cache_dir=tmp_path, refresh=True, now="2026-10-31T14:00:00Z", fetch=failed
    )
    assert failed_result["rosters"]["state"] == "network-unverified"
    assert failed_result["rosters"]["rows"] == []


def test_latest_known_roster_week_preserves_trade_and_same_week_conflict():
    src = sources()
    src["rosters"]["rows"].append(
        {"season": 2026, "week": 7, "gsis_id": "traded", "team": "OLD", "position": "RB"}
    )
    assert player(build(src), "traded")["current_team"] == "B"
    src["rosters"]["rows"].append(
        {"season": 2026, "week": 8, "gsis_id": "traded", "team": "OTHER", "position": "RB"}
    )
    assert player(build(src), "traded")["identity_state"] == "unresolved"


def test_unknown_kickoff_prevents_offseason_claim():
    src = sources()
    src["schedules"]["rows"][0]["gametime"] = None
    s = build_snapshot(src, season=2026, week=None, as_of=AS_OF, observed_at=AS_OF)
    assert s["target"]["state"] == "target_unknown"


def test_same_day_source_expiry_and_broken_cache_are_refetched(tmp_path):
    calls = []

    def fetch(name, season):
        calls.append(name)
        return [{"season": season}]

    live_sources(2026, cache_dir=tmp_path, refresh=False, now="2026-10-31T01:00:00Z", fetch=fetch)
    calls.clear()
    live_sources(2026, cache_dir=tmp_path, refresh=False, now="2026-10-31T08:00:00Z", fetch=fetch)
    assert calls == ["rosters", "injuries"]
    (tmp_path / "schedules_2026.json").write_text("broken json")
    live_sources(2026, cache_dir=tmp_path, refresh=False, now="2026-10-31T08:30:00Z", fetch=fetch)
    assert calls[-1] == "schedules"


def test_structurally_invalid_signed_snapshot_rejected():
    s = build()
    del s["sources"]["schedules"]["fetched_at_utc"]
    s["context_snapshot_id"] = content_id(
        {k: v for k, v in s.items() if k != "context_snapshot_id"}
    )
    with pytest.raises(ValueError, match="malformed"):
        validate_snapshot(s)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("players", 0, "name"), {"unexpected": "object"}),
        (("players", 0, "provider_id"), "different-canonical-id"),
        (("players", 0, "game", "state"), "unchecked-provider-state"),
        (("sources", "extra"), {"unexpected": "source"}),
        (("sources", "schedules", "sha256"), ["a"] * 64),
    ],
)
def test_resealed_malformed_companion_rejected_before_api_publication(path, replacement):
    value = build()
    current = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement
    value["context_snapshot_id"] = content_id(
        {k: v for k, v in value.items() if k != "context_snapshot_id"}
    )
    with pytest.raises(ValueError):
        validate_snapshot(value)


def test_stats_cache_contract_remains_same_day_and_refresh(tmp_path, monkeypatch):
    import pandas as pd

    from ffai.data import nflverse

    monkeypatch.setattr(nflverse, "CACHE_DIR", tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return pd.DataFrame({"x": [1]})

    nflverse._cached("test_stats", [2026], fetch, False)
    nflverse._cached("test_stats", [2026], fetch, False)
    assert len(calls) == 1
    nflverse._cached("test_stats", [2026], fetch, True)
    assert len(calls) == 2


def test_missing_columns_and_missing_team_fail_conservatively():
    src = sources()
    del src["schedules"]["rows"][0]["home_team"]
    s = build(src)
    assert s["sources"]["schedules"]["state"] == "unsupported"
    assert player(s, "bye")["game"]["state"] == "unknown"
    src = sources()
    src["rosters"]["rows"][0]["team"] = None
    assert player(build(src), "rookie")["identity_state"] == "unresolved"


def test_injury_from_wrong_team_or_postseason_not_applied():
    src = sources()
    src["injuries"]["rows"].append(
        {"season": 2026, "week": 9, "gsis_id": "rookie", "team": "OLD", "report_status": "Out"}
    )
    src["injuries"]["rows"].append(
        {
            "season": 2026,
            "week": 9,
            "season_type": "POST",
            "gsis_id": "rookie",
            "team": "A",
            "report_status": "Out",
        }
    )
    assert player(build(src), "rookie")["availability"]["state"] == "unknown"


def test_capabilities_expose_missing_target_injury_coverage():
    src = sources()
    for row in src["injuries"]["rows"]:
        row["week"] = 8
    s = build(src)
    assert s["sources"]["injuries"]["target_rows"] == 0
    assert s["sources"]["injuries"]["observed_weeks"] == [8]
    assert "target_period_coverage_unverified" in s["sources"]["injuries"]["reasons"]
    assert player(s, "out")["availability"]["state"] == "unknown"


def test_cli_overwrites_failed_current_attempt_without_stale_success(tmp_path):
    import subprocess
    import sys

    command = [
        sys.executable,
        "scripts/probe_context.py",
        "--season",
        "2026",
        "--week",
        "9",
        "--fixtures",
        str(FIXTURE),
        "--out",
        str(tmp_path),
    ]
    success = subprocess.run(command + ["--as-of", AS_OF], capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    assert (
        json.loads((tmp_path / "capabilities.json").read_text())["required_sources_usable"] is True
    )
    earlier = subprocess.run(
        command + ["--as-of", "2026-10-30T12:00:00Z"], capture_output=True, text=True
    )
    assert earlier.returncode == 2, earlier.stderr
    report = json.loads((tmp_path / "capabilities.json").read_text())
    assert report["required_sources_usable"] is False
    assert report["live_verified"] is False
    assert json.loads((tmp_path / "snapshot.json").read_text())["players"] == []


@pytest.mark.parametrize("invalid_week", [0, 19, 20, -1, True])
def test_derived_target_never_selects_invalid_regular_week(invalid_week):
    src = sources()
    src["schedules"]["rows"][0]["week"] = invalid_week
    s = build_snapshot(src, season=2026, week=None, as_of=AS_OF, observed_at=AS_OF)
    assert s["target"]["week"] is None
    assert s["target"]["state"] == "target_unknown"
    assert "unsupported_regular_period" in s["sources"]["schedules"]["reasons"]


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("provider_updated_at_utc", "provider_updated_after_as_of"),
        ("effective_at_utc", "source_effective_after_as_of"),
    ],
)
def test_future_source_revision_cannot_enter_earlier_context(field, reason):
    src = sources()
    src["injuries"][field] = "2026-11-01T12:00:00Z"
    # Even a claimed earlier first observation cannot make the later source revision usable.
    s = build(src)
    assert s["sources"]["injuries"]["as_of_rows"] == 0
    assert reason in s["sources"]["injuries"]["reasons"]
    assert player(s, "out")["availability"]["state"] == "unknown"


def test_future_updated_injury_row_is_excluded_without_changing_earlier_identity():
    src = sources()
    before = build(src)
    src["injuries"]["rows"].append(
        {
            "season": 2026,
            "week": 9,
            "gsis_id": "rookie",
            "report_status": "Out",
            "provider_updated_at_utc": "2026-11-01T12:00:00Z",
        }
    )
    assert build(src) == before
