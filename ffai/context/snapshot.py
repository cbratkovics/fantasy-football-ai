"""Pure context normalization. All time, coverage assertions and forecast links are explicit.

Source envelopes contain rows plus observation metadata. Unknown publication times stay unknown.
A source observed after the requested decision time cannot supply earlier decision evidence.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ffai.config import POSITIONS, regular_season_weeks
from ffai.decision_lab.canonical import content_id

VERSION = "context_snapshot/1.0"
SOURCES = ("schedules", "rosters", "injuries")
# Operational assumptions, not measured provider refresh guarantees.
MAX_FETCH_AGE_SECONDS = {"schedules": 86400, "rosters": 21600, "injuries": 21600}


def utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timezone offset required")
    return parsed.astimezone(dt.UTC)


def iso(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).isoformat(timespec="seconds")


def kickoff(row: dict) -> tuple[str | None, str | None]:
    """nflverse gametime is Eastern; reject DST ambiguity/gaps instead of guessing a fold."""
    try:
        if row.get("kickoff_utc"):
            return iso(utc(row["kickoff_utc"])), None
        if not row.get("gameday") or not row.get("gametime"):
            return None, "kickoff_unknown"
        naive = dt.datetime.fromisoformat(f"{row['gameday']}T{row['gametime']}")
        if naive.tzinfo is not None:
            return None, "kickoff_timezone_invalid"
        tz = ZoneInfo("America/New_York")
        a, b = naive.replace(tzinfo=tz, fold=0), naive.replace(tzinfo=tz, fold=1)
        if a.utcoffset() != b.utcoffset():
            return None, "kickoff_ambiguous_or_nonexistent"
        if a.astimezone(dt.UTC).astimezone(tz).replace(tzinfo=None) != naive:
            return None, "kickoff_ambiguous_or_nonexistent"
        return iso(a), None
    except (ValueError, TypeError):
        return None, "kickoff_invalid"


def _source(name: str, envelope: dict, as_of: dt.datetime, mode: str) -> tuple[dict, list]:
    rows, reasons = [], list(envelope.get("reasons", []))
    state = envelope.get("state", "unavailable")
    first = envelope.get("first_observed_at_utc")
    fetched = envelope.get("fetched_at_utc")
    if not first or not fetched:
        reasons.append("observation_time_unknown")
    else:
        try:
            if utc(first) > as_of:
                reasons.append("source_observed_after_as_of")
            if utc(fetched) > as_of:
                reasons.append("fetch_after_as_of")
            if (as_of - utc(fetched)).total_seconds() > MAX_FETCH_AGE_SECONDS[name]:
                reasons.append("fetch_stale")
        except (ValueError, TypeError):
            reasons.append("observation_time_invalid")
    for field, reason in (
        ("provider_updated_at_utc", "provider_updated_after_as_of"),
        ("effective_at_utc", "source_effective_after_as_of"),
    ):
        if envelope.get(field):
            try:
                if utc(envelope[field]) > as_of:
                    reasons.append(reason)
            except (ValueError, TypeError):
                reasons.append("source_revision_time_invalid")
    usable = state in {"verified", "partial", "fixture"} and not any(
        r in reasons
        for r in (
            "source_observed_after_as_of",
            "fetch_after_as_of",
            "observation_time_unknown",
            "observation_time_invalid",
            "provider_updated_after_as_of",
            "source_effective_after_as_of",
            "source_revision_time_invalid",
        )
    )
    if usable:
        for row in envelope.get("rows", []):
            observed = row.get("first_observed_at_utc", first)
            effective = row.get("effective_at_utc")
            updated = row.get("provider_updated_at_utc")
            try:
                if (
                    utc(observed) <= as_of
                    and (not effective or utc(effective) <= as_of)
                    and (not updated or utc(updated) <= as_of)
                ):
                    rows.append(row)
            except (ValueError, TypeError):
                reasons.append("row_time_invalid")
    required = {
        "schedules": {"season", "week", "game_type", "game_id", "home_team", "away_team"},
        "rosters": {"season", "team", "position"},
        "injuries": {"season", "week", "report_status"},
    }[name]
    if rows and any(
        not required.issubset(r)
        or (name != "schedules" and "gsis_id" not in r and "player_id" not in r)
        for r in rows
    ):
        rows = []
        state, usable = "unsupported", False
        reasons.append("required_fields_missing")
    if mode == "fixture":
        state = (
            "fixture" if usable else ("unsupported" if state == "unsupported" else "unavailable")
        )
        reasons.append("fixture_not_live_verified")
    elif not usable:
        state = "unavailable" if state not in {"network-unverified", "unsupported"} else state
    return {
        "source_id": envelope.get("source_id", f"nflverse:{name}"),
        "sha256": content_id(rows),
        "state": state,
        "reasons": sorted(set(reasons)),
        "provider_updated_at_utc": envelope.get("provider_updated_at_utc"),
        "provider_time_note": envelope.get(
            "provider_time_note", "Provider publication time unknown"
        ),
        "first_observed_at_utc": first,
        "fetched_at_utc": fetched,
        "effective_at_utc": envelope.get("effective_at_utc"),
        "coverage_complete": bool(envelope.get("coverage_complete")) and usable,
        "input_rows": len(rows),
        "as_of_rows": len(rows),
        "freshness_max_age_seconds": MAX_FETCH_AGE_SECONDS[name],
    }, rows


def build_snapshot(
    sources: dict[str, dict],
    *,
    season: int,
    week: int | None,
    as_of: str,
    observed_at: str,
    mode: str = "fixture",
    forecast: dict | None = None,
) -> dict[str, Any]:
    """Build a roster-first universe; never accept statistical candidates as current rosters."""
    if mode not in {"fixture", "live"}:
        raise ValueError("mode must be fixture or live")
    at, observed = utc(as_of), utc(observed_at)
    if week is not None and not 1 <= week <= regular_season_weeks(season):
        raise ValueError("only supported regular-season weeks may be requested")
    metadata, data = {}, {}
    for name in SOURCES:
        metadata[name], data[name] = _source(name, sources.get(name, {}), at, mode)
    regular_games = [
        r for r in data["schedules"] if r.get("season") == season and r.get("game_type") == "REG"
    ]
    games = [
        r
        for r in regular_games
        if type(r.get("week")) is int and 1 <= r["week"] <= regular_season_weeks(season)
    ]
    if len(games) != len(regular_games):
        metadata["schedules"]["coverage_complete"] = False
        metadata["schedules"]["reasons"] = sorted(
            set(metadata["schedules"]["reasons"] + ["unsupported_regular_period"])
        )
    if week is None:
        pending = [r for r in games if kickoff(r)[0] and utc(kickoff(r)[0]) > at]
        week = min((int(r["week"]) for r in pending), default=None)
    schedule_identities = defaultdict(set)
    for game in games:
        schedule_identities[game.get("game_id")].add(
            content_id(
                {
                    k: game.get(k)
                    for k in (
                        "season",
                        "week",
                        "home_team",
                        "away_team",
                        "gameday",
                        "gametime",
                        "kickoff_utc",
                        "schedule_state",
                    )
                }
            )
        )
    if None in schedule_identities or any(
        len(variants) > 1 for variants in schedule_identities.values()
    ):
        metadata["schedules"]["coverage_complete"] = False
        metadata["schedules"]["reasons"] = sorted(
            set(metadata["schedules"]["reasons"] + ["schedule_conflict"])
        )
    target_games = [r for r in games if r.get("week") == week] if week else []
    for name in SOURCES:
        periods = sorted(
            {
                int(r["week"])
                for r in data[name]
                if r.get("season") == season and type(r.get("week")) is int
            }
        )
        target_count = (
            sum(r.get("season") == season and r.get("week") == week for r in data[name])
            if week is not None
            else 0
        )
        metadata[name]["season_rows"] = sum(r.get("season") == season for r in data[name])
        metadata[name]["observed_weeks"] = periods
        metadata[name]["target_rows"] = target_count
        if week is not None and not target_count:
            metadata[name]["reasons"] = sorted(
                set(metadata[name]["reasons"] + ["target_period_coverage_unverified"])
            )
    target_state = (
        "target"
        if week
        else (
            "no_target"
            if metadata["schedules"]["coverage_complete"]
            and games
            and all(kickoff(r)[0] for r in games)
            else "target_unknown"
        )
    )
    # Never resolve incompatible schedule records by row order, including conflicting revisions.
    by_team = defaultdict(list)
    by_game = defaultdict(list)
    for game in target_games:
        by_game[game.get("game_id")].append(game)
    for variants in by_game.values():
        unique = {
            content_id(
                {
                    k: r.get(k)
                    for k in (
                        "game_id",
                        "home_team",
                        "away_team",
                        "gameday",
                        "gametime",
                        "kickoff_utc",
                        "schedule_state",
                    )
                }
            ): r
            for r in variants
        }
        for row in unique.values():
            for team in (row.get("home_team"), row.get("away_team")):
                by_team[team].append(row)
    roster = defaultdict(list)
    unresolved = []
    for row in data["rosters"]:
        if row.get("season") != season or row.get("position") not in POSITIONS:
            continue
        pid = row.get("gsis_id", row.get("player_id"))
        if not isinstance(pid, str) or not pid:
            unresolved.append({"source": "rosters", "reason": "canonical_id_missing"})
            continue
        if row.get("week") is not None and week is not None and row["week"] > week:
            continue
        roster[pid].append(row)
    forecasts = defaultdict(list)
    forecast = forecast or {}
    link_ok = (
        forecast.get("season") == season
        and forecast.get("week") == week
        and bool(forecast.get("snapshot_id"))
        and bool(forecast.get("sha256"))
    )
    for row in forecast.get("rows", []):
        forecasts[row.get("player_id")].append(row)
    injuries = defaultdict(list)
    for row in data["injuries"]:
        if (
            row.get("season") == season
            and row.get("week") == week
            and row.get("season_type", "REG") == "REG"
        ):
            pid = row.get("gsis_id", row.get("player_id"))
            if pid:
                injuries[pid].append(row)
    players = []
    for pid in sorted(set(roster) | set(forecasts)):
        rows = roster.get(pid, [])
        known_weeks = [r["week"] for r in rows if r.get("week") is not None]
        if known_weeks:
            latest_week = max(known_weeks)
            rows = [r for r in rows if r.get("week") in (None, latest_week)]
        identities = {(r.get("team"), r.get("position")) for r in rows}
        team, position = next(iter(identities)) if len(identities) == 1 else (None, None)
        identity = "resolved" if team and position in POSITIONS else "unresolved"
        reasons = []
        if not rows:
            reasons.append("projection_without_roster_support")
        if identity != "resolved":
            reasons.append("identity_unresolved")
        matches = by_team.get(team, []) if team and week else []
        game = {
            "game_id": None,
            "opponent": None,
            "kickoff_utc": None,
            "state": "unknown",
            "schedule_revision": None,
        }
        if len(matches) == 1 and identity == "resolved":
            r = matches[0]
            instant, reason = kickoff(r)
            game = {
                "game_id": r.get("game_id"),
                "opponent": (
                    r.get("away_team") if r.get("home_team") == team else r.get("home_team")
                ),
                "kickoff_utc": instant,
                "state": r.get("schedule_state", "scheduled"),
                "schedule_revision": r.get("revision"),
            }
            if reason:
                reasons.append(reason)
        elif len(matches) > 1:
            reasons.append("schedule_conflict")
        elif week and metadata["schedules"]["coverage_complete"] and identity == "resolved":
            game["state"] = "bye"
            reasons.append("known_bye")
        else:
            reasons.append("game_coverage_unknown")
        statuses = {
            str(r.get("report_status"))
            for r in injuries.get(pid, [])
            if r.get("report_status") and r.get("team") in (None, team)
        }
        designation = next(iter(statuses)) if len(statuses) == 1 else None
        state = "unknown"
        if len(statuses) > 1:
            reasons.append("availability_conflict")
        elif designation:
            if designation.lower() in {"out", "inactive", "injured reserve", "ir", "suspended"}:
                state = "unavailable"
            elif designation.lower() in {"questionable", "doubtful"}:
                state = "questionable"
        if state == "unknown":
            reasons.append("availability_unverified")
        fr = forecasts.get(pid, [])
        fstate = (
            "present"
            if link_ok and len(fr) == 1
            else ("missing_history" if not fr else "unresolved_reference")
        )
        if fstate != "present":
            reasons.append("forecast_" + fstate)
        players.append(
            {
                "player_id": pid,
                "provider_id": pid,
                "name": (
                    sorted({r.get("full_name") for r in rows if r.get("full_name")})[0]
                    if any(r.get("full_name") for r in rows)
                    else None
                ),
                "position": position,
                "current_team": team,
                "identity_state": identity,
                "roster_week": rows[0].get("week") if len(rows) == 1 else None,
                "game": game,
                "forecast": {
                    "state": fstate,
                    "forecast_team": fr[0].get("team") if len(fr) == 1 else None,
                },
                "availability": {
                    "state": state,
                    "designation": designation,
                    "basis": "provider_report" if designation else "no_provider_evidence",
                },
                "reasons": sorted(set(reasons)),
            }
        )
    result = {
        "schema_version": VERSION,
        "mode": mode,
        "target": {"season": season, "season_type": "REG", "week": week, "state": target_state},
        "requested_as_of_utc": iso(at),
        "observed_at_utc": iso(observed),
        "forecast": {k: forecast.get(k) for k in ("snapshot_id", "sha256", "season", "week")},
        "sources": metadata,
        "players": players,
        "unresolved_mappings": unresolved,
        "coverage": {
            "roster_identities": len(roster),
            "players": len(players),
            "unscored": sum(p["forecast"]["state"] != "present" for p in players),
            "unresolved_identities": sum(p["identity_state"] != "resolved" for p in players),
        },
        "limitations": [
            "Roster membership does not establish availability or fantasy ownership",
            "Kickoff is a conservative deadline, not verified platform lock rules",
            "Fetch age thresholds are assumptions; content update time may be unknown",
            "Context is not persisted in v1 receipts",
        ],
    }
    result["context_snapshot_id"] = content_id(result)
    validate_snapshot(result)
    return result


def _validate(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != VERSION:
        raise ValueError("unsupported context schema")
    if content_id({k: v for k, v in value.items() if k != "context_snapshot_id"}) != value.get(
        "context_snapshot_id"
    ):
        raise ValueError("context digest mismatch")
    if value["mode"] not in {"fixture", "live"}:
        raise ValueError("unsupported context mode")
    utc(value["requested_as_of_utc"])
    utc(value["observed_at_utc"])
    target = value["target"]
    if (
        type(target["season"]) is not int
        or target["season_type"] != "REG"
        or target["state"] not in {"target", "no_target", "target_unknown"}
    ):
        raise ValueError("unsupported target")
    if target["week"] is not None and (
        type(target["week"]) is not int
        or not 1 <= target["week"] <= regular_season_weeks(target["season"])
    ):
        raise ValueError("unsupported target week")
    if (target["state"] == "target") != (target["week"] is not None):
        raise ValueError("target state disagrees with period")
    if set(value["sources"]) != set(SOURCES):
        raise ValueError("unsupported context sources")
    for name in SOURCES:
        source = value["sources"][name]
        if source["state"] not in {
            "fixture",
            "verified",
            "partial",
            "unavailable",
            "unsupported",
            "network-unverified",
        }:
            raise ValueError("unsupported source state")
        if (
            not isinstance(source["source_id"], str)
            or not isinstance(source["sha256"], str)
            or len(source["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in source["sha256"])
            or not isinstance(source["coverage_complete"], bool)
        ):
            raise ValueError("invalid source identity or coverage")
        if not isinstance(source["reasons"], list) or any(
            not isinstance(r, str) for r in source["reasons"]
        ):
            raise ValueError("invalid source reasons")
        for field in (
            "provider_updated_at_utc",
            "first_observed_at_utc",
            "fetched_at_utc",
            "effective_at_utc",
        ):
            if source[field] is not None:
                utc(source[field])
    forecast = value["forecast"]
    if forecast["sha256"] is not None and (
        not isinstance(forecast["sha256"], str) or len(forecast["sha256"]) != 64
    ):
        raise ValueError("invalid forecast digest")
    if forecast["snapshot_id"] is not None and not isinstance(forecast["snapshot_id"], str):
        raise ValueError("invalid forecast identity")
    if any(forecast[k] is not None and type(forecast[k]) is not int for k in ("season", "week")):
        raise ValueError("invalid forecast period")
    ids = [p["player_id"] for p in value["players"]]
    if len(ids) != len(set(ids)) or any(not isinstance(p, str) or not p for p in ids):
        raise ValueError("context player IDs must be unique nonempty canonical IDs")
    for player in value["players"]:
        if any(
            player[k] is not None and not isinstance(player[k], str)
            for k in ("name", "provider_id", "position", "current_team")
        ):
            raise ValueError("invalid player identity fields")
        if player["identity_state"] not in {"resolved", "unresolved"} or player["availability"][
            "state"
        ] not in {"unknown", "questionable", "unavailable"}:
            raise ValueError("unsupported player state")
        if player["identity_state"] == "resolved" and (
            not player["current_team"]
            or player["position"] not in POSITIONS
            or player["provider_id"] != player["player_id"]
        ):
            raise ValueError("resolved identity requires team and supported position")
        if player["forecast"]["state"] not in {
            "present",
            "missing_history",
            "unresolved_reference",
        }:
            raise ValueError("unsupported forecast state")
        if (
            player["game"]["state"]
            not in {"scheduled", "bye", "unknown", "postponed", "cancelled", "ambiguous"}
            or not isinstance(player["reasons"], list)
            or any(not isinstance(r, str) for r in player["reasons"])
        ):
            raise ValueError("invalid player context")
        if any(
            player["game"][k] is not None and not isinstance(player["game"][k], str)
            for k in ("game_id", "opponent", "kickoff_utc")
        ):
            raise ValueError("invalid game identity fields")
        if (
            player["availability"]["designation"] is not None
            and not isinstance(player["availability"]["designation"], str)
        ) or not isinstance(player["availability"]["basis"], str):
            raise ValueError("invalid availability fields")
        if player["game"]["kickoff_utc"]:
            utc(player["game"]["kickoff_utc"])
    return value


def validate_snapshot(value: dict) -> dict:
    """Reject malformed/digest-mismatched companions; never repairs or rehashes evidence."""
    try:
        return _validate(value)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("malformed context snapshot") from exc


def load_snapshot(path: Path) -> dict:
    return validate_snapshot(json.loads(path.read_text(encoding="utf-8")))
