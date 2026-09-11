"""The autonomous weekly job.

Steps (each appends to the run log):

1. Determine the current season / next week from nflverse schedules (or the CLI overrides).
2. Load stats through the last completed week; run the data contracts; failure → HOLD.
3. Drift: PSI of the last four completed weeks of played rows against the week-of-season
   matched training deciles; median monitored PSI > 0.25 (or ≥ 2 features > 0.5) → HOLD,
   any feature > 0.25 → warn (``ffai.eval.drift.drift_status``, ADR-0010).
4. Build as-of features for the upcoming week and score the champion →
   ``artifacts/predictions/<season>/week_<ww>.json``.
5. Attach last week's actuals to last week's predictions file and append champion + challenger
   MAE / baseline MAE / within-±3 to ``artifacts/eval/rolling_<season>.json``.
6. PROMOTE rule (``registry.should_promote``): challenger beats champion on rolling MAE in each
   of the last four scored weeks and on the frozen test set → swap manifest slots and rescore.
7. Update ``manifest.json`` and regenerate ``docs/MODEL_CARD.md``.
8. The workflow (not this module) commits the artifacts and opens an issue on HOLD, using the run
   log written to ``artifacts/runs/<run_id>.json``.

``decide`` is a pure function over the step results so the policy is unit-testable without data.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ffai.config import ARTIFACTS_DIR, MIN_SEASON, POSITIONS, REPO_ROOT, regular_season_weeks
from ffai.data import contracts, nflverse
from ffai.eval import drift, evaluator, model_card
from ffai.features import asof
from ffai.models import registry
from ffai.pipeline import score

ACTION_PUBLISH, ACTION_HOLD, ACTION_PROMOTE = "PUBLISH", "HOLD", "PROMOTE"
# Drift is measured on the last N completed weeks of played rows, against the training deciles
# of the same week-of-season bucket; structural features (week_of_season, games_played_prior) and
# season-to-date averages (which reset every season) are excluded. See ADR-0010.
DRIFT_WINDOW_WEEKS = 4
DRIFT_EXCLUDED_FEATURES = frozenset(asof.TEMPORAL_FEATURES)


def decide(
    *,
    contract_ok: bool,
    drift_status: str,
    promote_ok: bool,
    promote_reason: str = "",
    contract_failures: list[str] | None = None,
    drift_features: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Pure policy: map step outcomes to (action, reasons)."""
    if not contract_ok:
        return ACTION_HOLD, [f"data contract failed: {', '.join(contract_failures or ['unknown'])}"]
    if drift_status == "hold":
        return ACTION_HOLD, [f"drift PSI > hold threshold on {', '.join(drift_features or [])}"]
    reasons = []
    if drift_status == "warn":
        reasons.append(f"drift warning on {', '.join(drift_features or [])}")
    if promote_ok:
        return ACTION_PROMOTE, reasons + [f"promotion rule satisfied: {promote_reason}"]
    return ACTION_PUBLISH, reasons + [
        "contracts ok, drift within limits",
        f"no promotion: {promote_reason}".rstrip(": "),
    ]


def rolling_path(season: int, artifacts: Path) -> Path:
    return artifacts / "eval" / f"rolling_{season}.json"


def read_rolling(season: int, artifacts: Path) -> dict[str, Any]:
    p = rolling_path(season, artifacts)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"season": season, "weeks": []}


def _weighted_test_mae(meta: dict[str, Any], candidate: str | dict[str, str]) -> float:
    num = den = 0.0
    for pos in POSITIONS:
        cand = registry.candidate_for(candidate, pos)
        p = meta["positions"][pos]
        num += p["candidates"][cand]["test_mae"] * p["n_test"]
        den += p["n_test"]
    return num / den


def _week_metrics(payload: dict[str, Any], history: pd.DataFrame) -> dict[str, Any] | None:
    rows = [
        {
            "player_id": r["player_id"],
            "season": payload["season"],
            "week": payload["week"],
            "position": r["position"],
            "prediction": r["prediction"],
            "actual": r["actual"],
        }
        for r in payload["predictions"]
        if r.get("actual") is not None
    ]
    if not rows:
        return None
    enriched = evaluator.add_causal_baseline(rows, history.to_dict("records"))
    m = evaluator._metrics(enriched, "prediction")
    b = evaluator._metrics(enriched, "trailing_mean_baseline")
    return {
        "n": m["n"],
        "mae": m["mae"],
        "within_3_rate": m["within_3_rate"],
        "baseline_mae": b["mae"],
    }


def run_weekly(
    *,
    artifacts: Path = ARTIFACTS_DIR,
    season: int | None = None,
    week: int | None = None,
    today: dt.date | None = None,
    dry_run: bool = False,
    write_model_card: bool = True,
) -> dict[str, Any]:
    """Execute the weekly job; return the run log (also written to artifacts/runs/ unless dry)."""
    now = dt.datetime.now(dt.UTC)
    run_id = f"run-{now:%Y%m%dT%H%M%SZ}"
    log: dict[str, Any] = {
        "run_id": run_id,
        "at_utc": now.isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "steps": [],
    }

    def step(name: str, **info: Any) -> None:
        log["steps"].append({"step": name, **info})

    manifest = registry.read_manifest(artifacts / "manifest.json")
    champ_version, champ_cand = registry.slot(manifest, "champion")
    chall_version, chall_cand = registry.slot(manifest, "challenger")
    meta = registry.read_model_metadata(champ_version, artifacts)

    # 1. season / week
    if season is None or week is None:
        year = (today or dt.date.today()).year
        schedules = nflverse.load_schedules([year - 1, year])
        season, week = nflverse.current_season_week(schedules, today)
    log["season"], log["week"] = int(season), int(week)
    step("season_week", season=season, week=week)

    # 2. stats + contracts
    stats = nflverse.load_weekly_stats(range(MIN_SEASON, season + 1))
    if week > 1:
        expected_through = (season, week - 1)
    else:
        expected_through = (season - 1, regular_season_weeks(season - 1))
    prior_rows = (manifest.get("last_run") or {}).get("stats_rows")
    report = contracts.check_stats_contract(
        stats, expected_through=expected_through, prior_row_count=prior_rows
    )
    failures = [c["name"] for c in report["checks"] if not c["ok"]]
    step("contracts", ok=report["ok"], failures=failures, summary=report["summary"])
    contract_ok = report["ok"]

    drift_status, drift_feats, promote_ok, promote_reason = "skipped", [], False, "not evaluated"
    predictions_payload = None
    rolling = read_rolling(season, artifacts)
    if contract_ok:
        # 3. drift: last DRIFT_WINDOW_WEEKS completed weeks of *played* rows (the same population
        # the training deciles describe) against the week-of-season-matched training reference.
        prior = stats[
            (stats["season"] < season) | ((stats["season"] == season) & (stats["week"] < week))
        ]
        played = asof.training_frame(asof.build_features(prior))
        periods = played[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
        recent = periods.tail(DRIFT_WINDOW_WEEKS)
        window = played.merge(recent, on=["season", "week"])
        bucket = str(drift.week_bucket(int(recent["week"].iloc[-1])))
        drift_by_pos = {}
        worst = "ok"
        for pos in POSITIONS:
            cand = registry.candidate_for(champ_cand, pos)
            pm = meta["positions"][pos]
            monitored = [
                f
                for f in pm["candidates"][cand]["top_feature_importance"]
                if f not in DRIFT_EXCLUDED_FEATURES and not f.endswith("_season_avg")
            ]
            ref = pm["drift_reference"]["buckets"].get(bucket) or pm["drift_reference"]["global"]
            rep = drift.drift_report(window[window["position"] == pos], ref, monitored)
            drift_by_pos[pos] = {
                "status": rep["status"],
                "median_monitored": rep["median_monitored"],
                "flagged": rep["flagged"],
                "severe": rep["severe"],
                "n": rep["n"],
            }
            drift_feats += [f"{pos}:{f}" for f in rep["flagged"]]
            if rep["status"] == "hold" or (rep["status"] == "warn" and worst == "ok"):
                worst = rep["status"]
        drift_status = worst
        step("drift", status=drift_status, positions=drift_by_pos)

        if drift_status != "hold":
            # 5. last week's actuals + rolling evaluation (champion and challenger shadow)
            if week > 1:
                last_path = score.predictions_path(season, week - 1, artifacts)
                history = stats[
                    (stats["season"] < season)
                    | ((stats["season"] == season) & (stats["week"] < week - 1))
                ][["player_id", "season", "week", "position", "fantasy_points_ppr"]].rename(
                    columns={"fantasy_points_ppr": "actual"}
                )
                if last_path.exists():
                    last_payload = json.loads(last_path.read_text(encoding="utf-8"))
                    last_payload, n_attached = score.attach_actuals(last_payload, stats)
                    if not dry_run:
                        score.write_predictions(last_payload, artifacts)
                    champ_m = _week_metrics(last_payload, history)
                else:
                    last_payload, n_attached, champ_m = None, 0, None
                shadow = score.score_week(
                    stats,
                    season,
                    week - 1,
                    model_version=chall_version,
                    candidate=chall_cand,
                    artifacts=artifacts,
                )
                shadow, _ = score.attach_actuals(shadow, stats)
                chall_m = _week_metrics(shadow, history)
                if champ_m and chall_m:
                    rolling["weeks"] = [w for w in rolling["weeks"] if w["week"] != week - 1]
                    rolling["weeks"].append(
                        {
                            "week": week - 1,
                            "n": champ_m["n"],
                            "champion": champ_m,
                            "challenger": chall_m,
                            "scored_at_utc": now.isoformat(timespec="seconds"),
                        }
                    )
                    rolling["weeks"].sort(key=lambda w: w["week"])
                    if not dry_run:
                        rolling_path(season, artifacts).write_text(
                            json.dumps(rolling, indent=2) + "\n", encoding="utf-8"
                        )
                step(
                    "rolling_eval",
                    week=week - 1,
                    actuals_attached=n_attached,
                    champion=champ_m,
                    challenger=chall_m,
                )
            else:
                step("rolling_eval", week=None, note="week 1: no prior week to score")

            # 6. promotion rule
            recent_c = [w["champion"]["mae"] for w in rolling["weeks"]]
            recent_x = [w["challenger"]["mae"] for w in rolling["weeks"]]
            promote_ok, promote_reason = registry.should_promote(
                recent_c,
                recent_x,
                _weighted_test_mae(meta, champ_cand),
                _weighted_test_mae(meta, chall_cand),
            )
            step(
                "promotion_rule",
                promote=promote_ok,
                reason=promote_reason,
                weeks_scored=len(rolling["weeks"]),
            )

    action, reasons = decide(
        contract_ok=contract_ok,
        drift_status=drift_status,
        promote_ok=promote_ok,
        promote_reason=promote_reason,
        contract_failures=failures,
        drift_features=drift_feats,
    )
    if action == ACTION_PROMOTE:
        manifest["champion"], manifest["challenger"] = (
            {"model_version": chall_version, "candidate": chall_cand},
            {"model_version": champ_version, "candidate": champ_cand},
        )
        champ_version, champ_cand = chall_version, chall_cand

    if action in (ACTION_PUBLISH, ACTION_PROMOTE):
        # 4. score the upcoming week with the (possibly new) champion
        predictions_payload = score.score_week(
            stats,
            season,
            week,
            model_version=champ_version,
            candidate=champ_cand,
            artifacts=artifacts,
        )
        if not dry_run:
            path = score.write_predictions(predictions_payload, artifacts)
            manifest["predictions"] = {"latest": str(path.relative_to(artifacts))}
        step("score", week=week, n=predictions_payload["n"], model_version=champ_version)

    log["action"], log["reasons"] = action, reasons
    manifest["last_run"] = {
        "run_id": run_id,
        "at_utc": log["at_utc"],
        "season": season,
        "week": week,
        "action": action,
        "reasons": reasons,
        "stats_rows": int(len(stats)),
    }
    manifest["data_through"] = {
        "season": int(stats["season"].max()),
        "week": int(stats[stats["season"] == stats["season"].max()]["week"].max()),
    }
    if not dry_run:
        registry.write_manifest(manifest, artifacts / "manifest.json")
        if write_model_card and manifest.get("eval_id"):
            model_card.write_model_card(
                champ_version,
                manifest["eval_id"],
                tiers_version=manifest.get("tiers_version"),
                artifacts=artifacts,
                out_path=REPO_ROOT / "docs" / "MODEL_CARD.md",
            )
        runs = artifacts / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"{run_id}.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return log
