"""Offline evidence adapter: builds the Decision Lab bundle from committed artifacts only.

Sources (all under ``artifacts/``; nothing else is opened — no network, no MotherDuck, no model
loading, no stats cache):

* ``marts/fct_player_week.parquet`` (+ ``marts/_export_manifest.json``) — projections, intervals,
  the causal baseline and, separately, the observed outcomes;
* ``predictions/<season>/week_*.json`` — the published weekly files (candidate per position,
  data cutoff, generation time, display names);
* ``eval/*.json`` and ``eval/rolling_<season>.json`` — the reference aggregates the baseline
  provenance is reconciled against;
* ``manifest.json``, ``models/<version>/metadata.json`` (feature version),
  ``models/<version>/*.csv`` (period-specific display names), ``runs/*.json`` (the run that
  published a weekly file), and ``marts/dim_player.parquet`` only as a display-name fallback.

The build is atomic: files go to a sibling temporary directory, the result is loaded with the
fail-closed :mod:`ffai.decision_lab.bundle` loader, and only then is the directory swapped into
place. A failure leaves the previous bundle untouched and records ``_last_attempt.json``.
Everything except ``_build.json`` is deterministic for a given set of inputs.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import duckdb

from ffai.config import REPO_ROOT, regular_season_weeks
from ffai.decision_lab import POLICY_VERSION, SCHEMA_VERSIONS, SPEC_PATH, bundle, cases, contracts
from ffai.decision_lab.canonical import content_id, file_sha256, q

EXPORTER_VERSION = "1.0.0"
POSITIONS = ("QB", "RB", "WR", "TE")
BASELINE_NAME = "causal_trailing_mean"
RECONCILE_TOLERANCE = 1e-4
FAMILY_PREFIX = {"frozen_test": "frozen", "out_of_sample_season": "oos", "weekly": "weekly"}
MART_BASELINE_SOURCE = {
    "player_history": "player_history",
    "position_history": "position_history",
    "prediction": "model_prediction_fallback",
}
VOLATILE = {bundle.BUILD_NAME, bundle.LAST_ATTEMPT_NAME}

DIGEST_COVERAGE = {
    "file_sha256": "SHA-256 of the file's bytes exactly as written to the bundle.",
    "content_sha256": (
        "SHA-256 of canonical_json(parsed document) (ffai/decision_lab/canonical.py: sorted keys, "
        "no whitespace, numbers rounded half-up at six decimals); independent of formatting."
    ),
    "decision_id": (
        "content_sha256 of the normalized decision_inputs document: schema and policy versions, "
        "the snapshot reference including the inputs snapshot's content digest, slot, every "
        "alternative with its availability fields, and the parameters; never outcomes, events, "
        "timestamps or the computed result (ffai/decision_lab/receipts.py)."
    ),
    "outcome_id": "content_sha256 of the outcome snapshot document.",
    "inputs_content_sha256": (
        "Recorded inside each outcome snapshot: the content_sha256 of the inputs snapshot it pairs with."
    ),
    "manifest": (
        "The manifest's own content digest is recorded in _build.json, never inside the manifest "
        "(no self-reference)."
    ),
    "volatile": "_build.json and _last_attempt.json are outside every digest.",
}


class ExportError(RuntimeError):
    """The export could not be completed.

    ``restored`` says what the published destinations hold afterwards: ``True`` — the previous
    canonical bundle and public copy (or their absence, on a first build) were re-established and
    verified; ``False`` — a restoration step or its verification failed, so the destinations must
    be treated as unknown and must not be published; ``None`` — the new bundle is in place and
    verified but a post-success cleanup failed. ``stage`` names where the failure happened
    (``staging``, ``replacement``, ``verification``, ``cleanup``).
    """

    def __init__(self, message: str, *, restored: bool | None = True, stage: str = "staging"):
        super().__init__(message)
        self.restored = restored
        self.stage = stage


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def git_head(repo: Path = REPO_ROOT) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExportError(f"missing source file {path}") from exc
    except json.JSONDecodeError as exc:
        raise ExportError(f"{path} is not valid JSON: {exc}") from exc


def _candidate_map(cand: str | dict[str, str]) -> dict[str, str]:
    if isinstance(cand, str):
        return {p: cand for p in POSITIONS}
    return {p: str(cand[p]) for p in sorted(cand)}


def _candidate_tag(cand_by_pos: dict[str, str]) -> str:
    values = set(cand_by_pos.values())
    return next(iter(values)) if len(values) == 1 else "mixed"


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


# --- evidence --------------------------------------------------------------------------------


class Evidence:
    """Every committed source the exporter reads, with digests recorded as they are consumed."""

    def __init__(self, artifacts: Path):
        self.artifacts = Path(artifacts)
        if not self.artifacts.is_dir():
            raise ExportError(f"artifacts directory not found: {self.artifacts}")
        self.sources: dict[str, dict[str, Any]] = {}
        self.manifest = self._json("manifest.json", role="artifact_manifest")
        mart_dir = self.artifacts / "marts"
        self.mart_path = mart_dir / "fct_player_week.parquet"
        if not self.mart_path.exists():
            raise ExportError(f"missing mart {self.mart_path}")
        export_entries = self._json("marts/_export_manifest.json", role="mart_export_manifest")
        by_model = {e["model"]: e for e in export_entries}
        first = export_entries[0] if export_entries else {}
        self.mart_export = {
            "exported_at_utc": first.get("exported_at_utc"),
            "target": first.get("target"),
            "code_commit": first.get("code_commit") or None,
            "invocation_id": first.get("invocation_id"),
        }
        self.mart_row_counts = {m: int(e["row_count"]) for m, e in by_model.items()}
        self.con = duckdb.connect(database=":memory:")
        quoted = str(self.mart_path).replace("'", "''")
        self.con.execute(f"create view fct_player_week as select * from read_parquet('{quoted}')")
        self.mart_columns = [
            r[0] for r in self.con.execute("describe select * from fct_player_week").fetchall()
        ]
        n_mart = self.con.execute("select count(*) from fct_player_week").fetchone()[0]
        self._record("marts/fct_player_week.parquet", role="mart", rows=int(n_mart))
        formats = {
            r[0]
            for r in self.con.execute(
                "select distinct target_scoring_format from fct_player_week"
            ).fetchall()
        }
        if formats - {"ppr"}:
            raise ExportError(f"mart carries non-PPR rows: {sorted(formats)}")
        self._dim_player: dict[str, tuple[str | None, str | None]] | None = None
        self._csv_display: dict[str, dict[tuple[str, int, int, str], tuple[Any, Any]]] = {}

    # -- bookkeeping -------------------------------------------------------------------------
    def _record(self, rel: str, *, role: str, rows: int | None = None) -> dict[str, Any]:
        path = self.artifacts / rel
        key = "artifacts/" + rel
        if key not in self.sources:
            self.sources[key] = {
                "role": role,
                "path": key,
                "sha256": file_sha256(path),
                "rows": rows,
            }
        return self.sources[key]

    def _json(self, rel: str, *, role: str, rows: int | None = None) -> Any:
        doc = _read_json(self.artifacts / rel)
        self._record(rel, role=role, rows=rows)
        return doc

    def source_ref(self, rel: str) -> dict[str, Any]:
        return dict(self.sources[rel])

    # -- evaluation artifacts ----------------------------------------------------------------
    def eval_artifacts(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted((self.artifacts / "eval").glob("*.json")):
            if p.name.startswith("rolling_"):
                continue
            doc = _read_json(p)
            if doc.get("kind") not in FAMILY_PREFIX or doc.get("kind") == "weekly":
                continue
            self._record(f"eval/{p.name}", role="eval_artifact")
            doc["_path"] = f"artifacts/eval/{p.name}"
            out.append(doc)
        return out

    def rolling(self, season: int) -> dict[str, Any] | None:
        p = self.artifacts / "eval" / f"rolling_{season}.json"
        if not p.exists():
            return None
        doc = self._json(f"eval/rolling_{season}.json", role="rolling_eval")
        doc["_path"] = f"artifacts/eval/rolling_{season}.json"
        return doc

    def predictions_files(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted((self.artifacts / "predictions").glob("*/week_*.json")):
            doc = _read_json(p)
            rel = f"predictions/{p.parent.name}/{p.name}"
            self._record(rel, role="predictions", rows=len(doc.get("predictions", [])))
            doc["_path"] = "artifacts/" + rel
            out.append(doc)
        return out

    def run_logs(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted((self.artifacts / "runs").glob("*.json")):
            doc = _read_json(p)
            doc["_path"] = f"artifacts/runs/{p.name}"
            doc["_rel"] = f"runs/{p.name}"
            out.append(doc)
        return out

    def feature_version(self, model_version: str) -> str | None:
        rel = f"models/{model_version}/metadata.json"
        if not (self.artifacts / rel).exists():
            return None
        return self._json(rel, role="model_metadata").get("feature_version")

    # -- display names -----------------------------------------------------------------------
    def csv_display(self, rel: str) -> dict[tuple[str, int, int, str], tuple[Any, Any]]:
        if rel not in self._csv_display:
            path = self.artifacts / rel
            table: dict[tuple[str, int, int, str], tuple[Any, Any]] = {}
            n = 0
            with path.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    n += 1
                    key = (
                        str(row["player_id"]),
                        int(row["season"]),
                        int(row["week"]),
                        str(row["candidate"]),
                    )
                    table[key] = (row.get("player_display_name") or None, row.get("team") or None)
            self._record(rel, role="prediction_csv", rows=n)
            self._csv_display[rel] = table
        return self._csv_display[rel]

    def dim_player(self) -> dict[str, tuple[str | None, str | None]]:
        if self._dim_player is None:
            path = self.artifacts / "marts" / "dim_player.parquet"
            if not path.exists():
                self._dim_player = {}
            else:
                quoted = str(path).replace("'", "''")
                rows = self.con.execute(
                    f"select player_id, player_display_name, team from read_parquet('{quoted}')"
                ).fetchall()
                self._record("marts/dim_player.parquet", role="dim_player", rows=len(rows))
                self._dim_player = {str(r[0]): (r[1], r[2]) for r in rows}
        return self._dim_player

    # -- mart queries ------------------------------------------------------------------------
    def mart_rows(self, source: str, season: int, model_version: str) -> list[dict[str, Any]]:
        cur = self.con.execute(
            "select * from fct_player_week where source = ? and season = ? and model_version = ? "
            "order by week, position, player_id, candidate",
            [source, season, model_version],
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


# --- reconciliation ----------------------------------------------------------------------------


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """n, mae, baseline_mae for ALL and per position over the observed champion rows."""
    out: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, Any]]] = {"ALL": []}
    for r in rows:
        if r.get("actual") is None:
            continue
        groups["ALL"].append(r)
        groups.setdefault(r["position"], []).append(r)
    for key, grp in groups.items():
        out[key] = {
            "n": len(grp),
            "mae": _mean([abs(float(r["actual"]) - float(r["prediction"])) for r in grp]),
            "baseline_mae": _mean([abs(float(r["actual"]) - float(r["baseline"])) for r in grp]),
        }
    return out


def _compare(expected: dict[str, dict[str, Any]], observed: dict[str, dict[str, Any]]) -> bool:
    for key, exp in expected.items():
        obs = observed.get(key)
        if obs is None:
            return False
        if int(exp["n"]) != int(obs["n"]):
            return False
        for metric in ("mae", "baseline_mae"):
            e, o = exp.get(metric), obs.get(metric)
            if e is None or o is None or abs(float(e) - float(o)) > RECONCILE_TOLERANCE:
                return False
    return True


def reconcile_eval(rows: list[dict[str, Any]], artifact: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "ALL": {
            "n": artifact["metrics"]["n"],
            "mae": artifact["metrics"]["mae"],
            "baseline_mae": artifact["baseline"]["mae"],
        }
    }
    for pos, coh in (artifact.get("cohorts") or {}).items():
        expected[pos] = {"n": coh["n"], "mae": coh["mae"], "baseline_mae": coh["baseline"]["mae"]}
    observed = _aggregate(rows)
    ok = _compare(expected, observed)
    return {
        "status": "matched" if ok else "mismatch",
        "reference": artifact["eval_id"],
        "expected": expected,
        "observed": {k: observed[k] for k in expected if k in observed},
        "tolerance": RECONCILE_TOLERANCE,
    }


def reconcile_rolling(
    rows: list[dict[str, Any]], rolling: dict[str, Any] | None, season: int, week: int
) -> dict[str, Any]:
    entry = None
    for w in (rolling or {}).get("weeks", []):
        if int(w.get("week")) == week:
            entry = w
    reference = f"rolling_{season}.json week {week}"
    if entry is None or not entry.get("champion"):
        return {
            "status": "unavailable",
            "reference": reference,
            "expected": None,
            "observed": _aggregate(rows).get("ALL"),
            "tolerance": RECONCILE_TOLERANCE,
        }
    champ = entry["champion"]
    expected = {
        "ALL": {"n": champ["n"], "mae": champ["mae"], "baseline_mae": champ["baseline_mae"]}
    }
    observed = _aggregate(rows)
    ok = _compare(expected, observed)
    return {
        "status": "matched" if ok else "mismatch",
        "reference": reference,
        "expected": expected,
        "observed": {"ALL": observed.get("ALL")},
        "tolerance": RECONCILE_TOLERANCE,
    }


POPULATION_RULE = (
    "every mart row belongs to a player with at least one prior game (as-of HISTORY_FLAG), so the "
    "causal trailing mean has player history whenever the warehouse holds the full stats history"
)


def baseline_meta(reconciliation: dict[str, Any], *, mart_column: bool) -> dict[str, Any]:
    status = reconciliation["status"]
    reconciled = {"matched": True, "mismatch": False, "unavailable": None}[status]
    reconciled_to = reconciliation["reference"] if status == "matched" else None
    if mart_column:
        return {
            "name": BASELINE_NAME,
            "provenance_basis": "mart_column",
            "reconciled": reconciled,
            "reconciled_to": reconciled_to,
            "note": (
                "per-row provenance read from fct_player_week.baseline_source; aggregate "
                f"reconciliation against {reconciliation['reference']}: {status}"
            ),
        }
    if status == "matched":
        return {
            "name": BASELINE_NAME,
            "provenance_basis": "population_rule_and_reconciliation",
            "reconciled": True,
            "reconciled_to": reconciled_to,
            "note": (
                f"{POPULATION_RULE}; the champion-candidate aggregates of the whole evaluation window "
                f"(n exact, MAE and baseline MAE within {RECONCILE_TOLERANCE}) match "
                f"{reconciliation['reference']}, so every row's baseline is player history"
            ),
        }
    if status == "unavailable":
        return {
            "name": BASELINE_NAME,
            "provenance_basis": "population_rule_unreconciled",
            "reconciled": None,
            "reconciled_to": None,
            "note": (
                f"{POPULATION_RULE}; no reference aggregate exists yet for this week "
                f"({reconciliation['reference']}), so the baseline values for this snapshot have not "
                "been reconciled to an evaluation artifact"
            ),
        }
    return {
        "name": BASELINE_NAME,
        "provenance_basis": "unknown",
        "reconciled": False,
        "reconciled_to": None,
        "note": (
            f"the champion-candidate aggregates do not match {reconciliation['reference']}; the "
            "baseline branch cannot be established, so every row is labelled unknown"
        ),
    }


def _row_provenance(row: dict[str, Any], meta: dict[str, Any]) -> str:
    if meta["provenance_basis"] == "mart_column":
        src = row.get("baseline_source")
        if src not in MART_BASELINE_SOURCE:
            raise ExportError(f"unexpected baseline_source {src!r} in the mart")
        return MART_BASELINE_SOURCE[src]
    if meta["provenance_basis"] == "unknown":
        return "unknown"
    return "player_history"


# --- snapshot construction -------------------------------------------------------------------


def _snapshot_id(
    prefix: str, season: int, week: int, model_version: str, cand: dict[str, str]
) -> str:
    return f"{prefix}-{season}-w{week:02d}-{model_version}-{_candidate_tag(cand)}"


def _q(x: Any) -> float | None:
    if x is None:
        return None
    return q(float(x))


def _display(
    ev: Evidence,
    pid: str,
    lookup: tuple[Any, Any] | None,
    used_dim: list[bool],
) -> tuple[str | None, str | None, str | None]:
    if lookup is not None and (lookup[0] or lookup[1]):
        return lookup[0], lookup[1], "prediction_source"
    fallback = ev.dim_player().get(pid)
    if fallback is not None:
        used_dim[0] = True
        return fallback[0], fallback[1], "dim_player_current"
    return None, None, None


def _rows_to_snapshot_rows(
    ev: Evidence,
    rows: list[dict[str, Any]],
    *,
    meta: dict[str, Any],
    availability_default: str,
    display_lookup: dict[str, tuple[Any, Any]],
    used_dim: list[bool],
) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        pid = str(r["player_id"])
        name, team, src = _display(ev, pid, display_lookup.get(pid), used_dim)
        out.append(
            {
                "player_id": pid,
                "display_name": name,
                "team": team,
                "position": r["position"],
                "model_version": r["model_version"],
                "candidate": r["candidate"],
                "projection": _q(r["prediction"]),
                "floor": _q(r.get("prediction_floor")),
                "ceiling": _q(r.get("prediction_ceiling")),
                "baseline": _q(r.get("baseline")),
                "baseline_provenance": _row_provenance(r, meta),
                "availability_default": availability_default,
                "display_source": src,
                "source_row_ref": (
                    f"fct_player_week:{pid}:{r['season']}:{r['week']}:{r['model_version']}:{r['candidate']}"
                ),
            }
        )
    out.sort(key=lambda x: x["player_id"])
    ids = [x["player_id"] for x in out]
    if len(set(ids)) != len(ids):
        raise ExportError("duplicate player_id within one snapshot after candidate filtering")
    return out


def _outcome_doc(
    inputs: dict[str, Any],
    inputs_digest: str,
    rows: list[dict[str, Any]],
    files: list[dict[str, Any]],
    observed_at: str | None,
) -> dict[str, Any] | None:
    by_pid = {str(r["player_id"]): r for r in rows}
    out_rows = []
    observed = 0
    for r in inputs["rows"]:
        m = by_pid[r["player_id"]]
        actual = m.get("actual")
        actual = None if actual is None else q(float(actual))
        src = m.get("actual_source") if actual is not None else None
        if actual is not None:
            observed += 1
        out_rows.append({"player_id": r["player_id"], "actual": actual, "actual_source": src})
    if observed == 0:
        return None
    return {
        "schema_version": SCHEMA_VERSIONS["outcome_snapshot"],
        "snapshot_id": inputs["snapshot_id"],
        "inputs_content_sha256": inputs_digest,
        "season": inputs["season"],
        "week": inputs["week"],
        "model_version": inputs["model_version"],
        "scoring_format": inputs["scoring_format"],
        "observed_at_utc": observed_at,
        "source": {"mart": "fct_player_week", "files": files},
        "coverage": {"scored": len(out_rows), "observed": observed},
        "rows": out_rows,
    }


class Builder:
    """Assembles every snapshot, the cases file and the manifest in memory."""

    def __init__(self, ev: Evidence):
        self.ev = ev
        self.inputs: dict[str, dict[str, Any]] = {}
        self.digests: dict[str, str] = {}
        self.outcomes: dict[str, dict[str, Any] | None] = {}
        self.entries: dict[str, dict[str, Any]] = {}
        self.mart_column = "baseline_source" in ev.mart_columns
        self.latest_weekly: str | None = None

    # -- historical windows (frozen test, out-of-sample season) ------------------------------
    def add_eval_window(self, artifact: dict[str, Any]) -> None:
        ev = self.ev
        kind, season = artifact["kind"], int(artifact["season"])
        model_version = artifact["model"]["version"]
        cand = _candidate_map(artifact["model"]["candidate"])
        rows = [
            r
            for r in ev.mart_rows(kind, season, model_version)
            if r["candidate"] == cand.get(r["position"])
        ]
        if not rows:
            raise ExportError(f"no mart rows for {kind} {season} {model_version}")
        reconciliation = reconcile_eval(rows, artifact)
        meta = baseline_meta(reconciliation, mart_column=self.mart_column)
        csv_rel = None
        input_path = (artifact.get("input") or {}).get("path")
        if input_path:
            csv_rel = input_path.split("artifacts/", 1)[-1]
            if not (ev.artifacts / csv_rel).exists():
                csv_rel = None
        table = ev.csv_display(csv_rel) if csv_rel else {}
        feature_version = ev.feature_version(model_version)
        weeks = sorted({int(r["week"]) for r in rows})
        for week in weeks:
            wk_rows = [r for r in rows if int(r["week"]) == week]
            lookup = {
                str(r["player_id"]): table.get(
                    (str(r["player_id"]), season, week, r["candidate"]), (None, None)
                )
                for r in wk_rows
            }
            used_dim = [False]
            snap_rows = _rows_to_snapshot_rows(
                ev,
                wk_rows,
                meta=meta,
                availability_default="realized_stats_row",
                display_lookup=lookup,
                used_dim=used_dim,
            )
            files = [
                ev.source_ref("artifacts/marts/fct_player_week.parquet"),
                ev.source_ref("artifacts/marts/_export_manifest.json"),
                ev.source_ref(artifact["_path"]),
            ]
            if csv_rel:
                files.append(ev.source_ref("artifacts/" + csv_rel))
            if used_dim[0]:
                files.append(ev.source_ref("artifacts/marts/dim_player.parquet"))
            cutoff = (
                {"season": season, "week": week - 1}
                if week > 1
                else {"season": season - 1, "week": regular_season_weeks(season - 1)}
            )
            sid = _snapshot_id(FAMILY_PREFIX[kind], season, week, model_version, cand)
            inputs = {
                "schema_version": SCHEMA_VERSIONS["inputs_snapshot"],
                "snapshot_id": sid,
                "mode": "historical_replay",
                "source_family": kind,
                "season": season,
                "week": week,
                "model_version": model_version,
                "feature_version": feature_version,
                "candidate_by_position": cand,
                "scoring_format": "ppr",
                "data_cutoff": cutoff,
                "generated_at_utc": None,
                "publication": {
                    "status": "recorded",
                    "run_id": None,
                    "action": None,
                    "note": f"retrospective evaluation artifact {artifact['eval_id']}",
                },
                "population": {
                    "conditioning": "realized_stats_rows",
                    "description": (
                        "players with a realized nflverse stats row in this week and at least one "
                        "prior game, scored by the frozen model; not a reconstructed pregame roster "
                        "universe. data_cutoff is the last completed week before this one: each row's "
                        "features use only strictly earlier weeks (as-of)."
                    ),
                    "n_rows": len(snap_rows),
                    "exclusions": [
                        "players without a stats row this week (inactive, bye, injured, not rostered)",
                        "players without at least one prior game (no as-of history)",
                    ],
                },
                "baseline": meta,
                "champion_at_source": {
                    "model_version": model_version,
                    "candidate_by_position": cand,
                    "basis": "eval_artifact",
                },
                "source": {
                    "mart": "fct_player_week",
                    "files": files,
                    "mart_export": dict(ev.mart_export),
                },
                "rows": snap_rows,
            }
            outcome_files = [
                ev.source_ref("artifacts/marts/fct_player_week.parquet"),
                ev.source_ref("artifacts/marts/_export_manifest.json"),
            ]
            digest = self._check_inputs(inputs)
            outcomes = _outcome_doc(
                inputs, digest, wk_rows, outcome_files, ev.mart_export["exported_at_utc"]
            )
            self._add(inputs, outcomes, reconciliation, digest=digest)

    # -- weekly files ------------------------------------------------------------------------
    def add_weekly(
        self, pred: dict[str, Any], runs: list[dict[str, Any]], manifest: dict[str, Any]
    ) -> None:
        ev = self.ev
        season, week = int(pred["season"]), int(pred["week"])
        model_version = pred["model_version"]
        cand = _candidate_map(pred["candidate"])
        rows = [
            r
            for r in ev.mart_rows("weekly", season, model_version)
            if int(r["week"]) == week and r["candidate"] == cand.get(r["position"])
        ]
        if not rows:
            raise ExportError(
                f"no mart rows for weekly {season} week {week} {model_version}: the mart export "
                "predates this predictions file"
            )
        observed = sum(1 for r in rows if r.get("actual") is not None)
        rolling = ev.rolling(season) if observed else None
        reconciliation = reconcile_rolling(rows, rolling, season, week)
        meta = baseline_meta(reconciliation, mart_column=self.mart_column)
        run = _matching_run(runs, season, week, pred.get("generated_at_utc"))
        lookup = {str(p["player_id"]): (p.get("name"), p.get("team")) for p in pred["predictions"]}
        used_dim = [False]
        snap_rows = _rows_to_snapshot_rows(
            ev,
            rows,
            meta=meta,
            availability_default="unknown",
            display_lookup=lookup,
            used_dim=used_dim,
        )
        files = [
            ev.source_ref("artifacts/marts/fct_player_week.parquet"),
            ev.source_ref("artifacts/marts/_export_manifest.json"),
            ev.source_ref(pred["_path"]),
        ]
        if run is not None:
            ev._record(run["_rel"], role="run_log")
            files.append(ev.source_ref(run["_path"]))
        if rolling is not None:
            files.append(ev.source_ref(rolling["_path"]))
        if used_dim[0]:
            files.append(ev.source_ref("artifacts/marts/dim_player.parquet"))
        last = manifest.get("last_run") or {}
        last_note = (
            f"manifest last_run at export: {last.get('run_id')} {last.get('action')} "
            f"season {last.get('season')} week {last.get('week')} at {last.get('at_utc')}"
        )
        pred_rel = pred["_path"].split("artifacts/", 1)[-1]
        if run is None:
            pub_note = f"{pred_rel}: no PUBLISH/PROMOTE run log matches this week; {last_note}"
        else:
            pub_note = f"{pred_rel} produced by {run['run_id']} ({run['action']}); {last_note}"
        sid = _snapshot_id("weekly", season, week, model_version, cand)
        inputs = {
            "schema_version": SCHEMA_VERSIONS["inputs_snapshot"],
            "snapshot_id": sid,
            "mode": "historical_replay" if observed else "published_weekly",
            "source_family": "weekly",
            "season": season,
            "week": week,
            "model_version": model_version,
            "feature_version": pred.get("feature_version") or ev.feature_version(model_version),
            "candidate_by_position": cand,
            "scoring_format": "ppr",
            "data_cutoff": (
                {
                    "season": int(pred["data_through"]["season"]),
                    "week": int(pred["data_through"]["week"]),
                }
                if pred.get("data_through")
                else None
            ),
            "generated_at_utc": pred.get("generated_at_utc"),
            "publication": {
                "status": "published",
                "run_id": run["run_id"] if run else None,
                "action": run["action"] if run else None,
                "note": pub_note,
            },
            "population": {
                "conditioning": "scored_eligible_players",
                "description": (
                    "every player with a stat row in this or the previous season and at least one "
                    "prior game, scored regardless of roster status, injury, bye or availability"
                ),
                "n_rows": len(snap_rows),
                "exclusions": [
                    "players without a stats row in this or the previous season",
                    "players without at least one prior game (no as-of history)",
                ],
            },
            "baseline": meta,
            "champion_at_source": {
                "model_version": model_version,
                "candidate_by_position": cand,
                "basis": "predictions_file",
            },
            "source": {
                "mart": "fct_player_week",
                "files": files,
                "mart_export": dict(ev.mart_export),
            },
            "rows": snap_rows,
        }
        outcome_files = [
            ev.source_ref("artifacts/marts/fct_player_week.parquet"),
            ev.source_ref("artifacts/marts/_export_manifest.json"),
        ]
        digest = self._check_inputs(inputs)
        outcomes = _outcome_doc(
            inputs, digest, rows, outcome_files, ev.mart_export["exported_at_utc"]
        )
        self._add(inputs, outcomes, reconciliation, digest=digest)
        latest = (manifest.get("predictions") or {}).get("latest")
        if latest and latest.replace("\\", "/").endswith(pred_rel):
            self.latest_weekly = sid

    # -- synthetic ---------------------------------------------------------------------------
    def add_synthetic(self) -> None:
        for inputs, outcomes in cases.synthetic_snapshots():
            self._add(inputs, outcomes, None)

    def _check_inputs(self, inputs: dict[str, Any]) -> str:
        """Contract-check an inputs snapshot before anything derives from it; returns its digest."""
        sid = inputs["snapshot_id"]
        if sid in self.inputs:
            raise ExportError(f"duplicate snapshot id {sid}")
        try:
            contracts.check_inputs_snapshot(inputs)
        except contracts.ContractError as exc:
            raise ExportError(f"snapshot {sid} violates its contract: {exc}") from exc
        return content_id(inputs)

    def _add(
        self,
        inputs: dict[str, Any],
        outcomes: dict[str, Any] | None,
        reconciliation: dict[str, Any] | None,
        *,
        digest: str | None = None,
    ) -> None:
        sid = inputs["snapshot_id"]
        if digest is None:
            digest = self._check_inputs(inputs)
        if outcomes is not None:
            try:
                contracts.check_outcome_snapshot(outcomes)
            except contracts.ContractError as exc:
                raise ExportError(f"snapshot {sid} violates its contract: {exc}") from exc
            if outcomes["inputs_content_sha256"] != digest:
                raise ExportError(f"snapshot {sid}: outcome digest does not pair with its inputs")
        self.inputs[sid] = inputs
        self.digests[sid] = digest
        self.outcomes[sid] = outcomes
        self.entries[sid] = {
            "snapshot_id": sid,
            "mode": inputs["mode"],
            "source_family": inputs["source_family"],
            "season": inputs["season"],
            "week": inputs["week"],
            "model_version": inputs["model_version"],
            "candidate_by_position": dict(inputs["candidate_by_position"]),
            "population": dict(inputs["population"]),
            "baseline": dict(inputs["baseline"]),
            "reconciliation": reconciliation,
        }

    # -- files ---------------------------------------------------------------------------------
    def write(self, out: Path, *, code_revision: str | None) -> dict[str, Any]:
        ev = self.ev
        (out / "inputs").mkdir(parents=True, exist_ok=True)
        (out / "outcomes").mkdir(parents=True, exist_ok=True)
        snapshots = []
        order = sorted(
            self.entries.values(),
            key=lambda e: (e["mode"], e["source_family"], e["season"], e["week"], e["snapshot_id"]),
        )
        for entry in order:
            sid = entry["snapshot_id"]
            inputs, outcomes = self.inputs[sid], self.outcomes[sid]
            rel = f"inputs/{sid}.json"
            (out / rel).write_text(dumps(inputs), encoding="utf-8")
            e = dict(entry)
            e["inputs"] = {
                "path": rel,
                "file_sha256": file_sha256(out / rel),
                "content_sha256": self.digests[sid],
                "n_rows": len(inputs["rows"]),
            }
            if outcomes is None:
                e["outcomes"] = None
            else:
                orel = f"outcomes/{sid}.json"
                (out / orel).write_text(dumps(outcomes), encoding="utf-8")
                e["outcomes"] = {
                    "path": orel,
                    "file_sha256": file_sha256(out / orel),
                    "content_sha256": content_id(outcomes),
                    "n_rows": len(outcomes["rows"]),
                    "n_observed": outcomes["coverage"]["observed"],
                }
            snapshots.append(e)
        cases_doc = cases.build_cases(
            self.inputs, self.outcomes, published_snapshot_id=self.latest_weekly
        )
        try:
            contracts.check_cases(cases_doc)
        except contracts.ContractError as exc:
            raise ExportError(f"cases.json violates its contract: {exc}") from exc
        (out / "cases.json").write_text(dumps(cases_doc), encoding="utf-8")
        manifest = ev.manifest
        champion = manifest.get("champion") or {}
        challenger = manifest.get("challenger") or {}
        last = manifest.get("last_run") or {}
        lineage = {
            "model_version": champion.get("model_version"),
            "feature_version": manifest.get("feature_version"),
            "champion_candidate_by_position": (
                _candidate_map(champion["candidate"]) if champion else None
            ),
            "challenger_model_version": challenger.get("model_version"),
            "challenger_candidate_by_position": (
                _candidate_map(challenger["candidate"]) if challenger else None
            ),
            "manifest_last_run": {
                "run_id": last.get("run_id"),
                "action": last.get("action"),
                "season": last.get("season"),
                "week": last.get("week"),
                "at_utc": last.get("at_utc"),
            },
            "data_through": manifest.get("data_through"),
            "predictions_latest": (manifest.get("predictions") or {}).get("latest"),
            "mart_export": {
                **ev.mart_export,
                "row_counts": dict(sorted(ev.mart_row_counts.items())),
            },
            "baseline_provenance_method": (
                "mart_column" if self.mart_column else "population_rule_and_reconciliation"
            ),
        }
        doc = {
            "schema_version": SCHEMA_VERSIONS["manifest"],
            "exporter_version": EXPORTER_VERSION,
            "policy_version": POLICY_VERSION,
            "schema_versions": dict(SCHEMA_VERSIONS),
            "code_revision": {
                "produced_at": code_revision,
                "note": (
                    "revision that produced this export; the commit that later contains it may differ "
                    "(compare with artifacts/marts/_export_manifest.json code_commit, which also "
                    "predates its containing commit)"
                ),
            },
            "lineage": lineage,
            "sources": [ev.sources[k] for k in sorted(ev.sources)],
            "snapshots": snapshots,
            "cases": {
                "path": "cases.json",
                "file_sha256": file_sha256(out / "cases.json"),
                "content_sha256": content_id(cases_doc),
                "n_rows": len(cases_doc["cases"]),
            },
            "latest_weekly_snapshot_id": self.latest_weekly,
            "policy_spec": {
                "path": "ffai/decision_lab/policy_spec.json",
                "sha256": file_sha256(SPEC_PATH),
            },
            "digest_coverage": dict(DIGEST_COVERAGE),
        }
        try:
            contracts.check_manifest(doc)
        except contracts.ContractError as exc:
            raise ExportError(f"manifest violates its contract: {exc}") from exc
        (out / bundle.MANIFEST_NAME).write_text(dumps(doc), encoding="utf-8")
        build = {
            "built_at_utc": utc_now(),
            "exporter_version": EXPORTER_VERSION,
            "python_version": platform.python_version(),
            "duckdb_version": duckdb.__version__,
            "manifest_content_sha256": content_id(doc),
            "manifest_file_sha256": file_sha256(out / bundle.MANIFEST_NAME),
            "code_revision": code_revision,
        }
        (out / bundle.BUILD_NAME).write_text(dumps(build), encoding="utf-8")
        return doc


def _matching_run(
    runs: list[dict[str, Any]], season: int, week: int, generated_at: str | None
) -> dict[str, Any] | None:
    """The PUBLISH/PROMOTE run for (season, week): the latest one started at or before the
    predictions file's generation time, else the latest matching run."""
    matching = [
        r
        for r in runs
        if r.get("action") in ("PUBLISH", "PROMOTE")
        and int(r.get("season", -1)) == season
        and int(r.get("week", -1)) == week
        and r.get("run_id")
    ]
    if not matching:
        return None
    matching.sort(key=lambda r: str(r.get("at_utc") or ""))
    if generated_at:
        before = [r for r in matching if str(r.get("at_utc") or "") <= generated_at]
        if before:
            return before[-1]
    return matching[-1]


# --- building, checking, mirroring -----------------------------------------------------------


def build_into(artifacts: Path, out: Path, *, code_revision: str | None) -> dict[str, Any]:
    """Build a complete bundle into ``out`` (must not exist) and verify it with the loader."""
    out.mkdir(parents=True, exist_ok=False)
    ev = Evidence(artifacts)
    builder = Builder(ev)
    for artifact in ev.eval_artifacts():
        builder.add_eval_window(artifact)
    runs = ev.run_logs()
    for pred in ev.predictions_files():
        builder.add_weekly(pred, runs, ev.manifest)
    builder.add_synthetic()
    manifest = builder.write(out, code_revision=code_revision)
    try:
        report = bundle.load_bundle(out).verify_all()
    except bundle.BundleError as exc:
        raise ExportError(f"freshly built bundle failed verification: {exc}") from exc
    manifest["_verify_report"] = report
    return manifest


def _write_last_attempt(
    out: Path,
    error: str,
    code_revision: str | None,
    *,
    stage: str,
    restoration: dict[str, Any],
    published_state: str,
) -> None:
    """Failure diagnostics next to (never inside) the evidence: ``_last_attempt.json`` is outside
    every digest (see ``DIGEST_COVERAGE``) and is deleted by the next successful export."""
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / bundle.LAST_ATTEMPT_NAME).write_text(
            dumps(
                {
                    "status": "failed",
                    "attempted_at_utc": utc_now(),
                    "error": error,
                    "stage": stage,
                    "restoration": restoration,
                    "published_state": published_state,
                    "code_revision": code_revision,
                    "exporter_version": EXPORTER_VERSION,
                    "note": (
                        "diagnostics only; outside every digest; published_state says whether the "
                        "previously published bundle was re-established (previous_bundle_preserved), "
                        "was never there (no_previous_bundle), or could not be verified (unknown)"
                    ),
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        # Diagnostics must never mask the original failure (the destination may be unusable).
        pass


# Filesystem seams: tests inject failures here (a rename that fails after the first swap, a copy
# that fails, a verification that fails) without touching the real filesystem semantics.
def _move(src: Path, dst: Path) -> None:
    os.rename(src, dst)


def _copytree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst)


def _verify_published(out: Path, public_copy: Path | None) -> dict[str, Any]:
    return verify_bundle(out, public_copy=public_copy)


def _rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _restore(
    *,
    out: Path,
    public: Path | None,
    tmp_out: Path,
    tmp_pub: Path | None,
    prev_out: Path,
    prev_pub: Path | None,
    placed: dict[str, bool],
    had_out: bool,
    had_pub: bool,
) -> dict[str, Any]:
    """Undo whatever ``export_bundle`` had swapped, in reverse order, then *verify* the result.

    Returns ``{"status": "verified" | "failed", "steps": [...], "errors": [...]}``. ``verified``
    means: every destination that existed before holds a bundle that loads fail-closed (canonical)
    or equals the canonical (public copy); every destination that did not exist before is absent
    again; no ``.previous-*`` sibling remains. Anything else is ``failed``.
    """
    steps: list[str] = []
    errors: list[str] = []

    def attempt(label: str, fn) -> None:  # noqa: ANN001
        try:
            fn()
            steps.append(label)
        except Exception as exc:  # noqa: BLE001 - collect everything, verify afterwards
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    if public is not None and tmp_pub is not None and prev_pub is not None:
        if placed["pub_placed"]:
            attempt("move new public copy aside", lambda: _move(public, tmp_pub))
        if placed["pub_moved"]:
            attempt("restore previous public copy", lambda: _move(prev_pub, public))
    if placed["out_placed"]:
        attempt("move new canonical bundle aside", lambda: _move(out, tmp_out))
    if placed["out_moved"]:
        attempt("restore previous canonical bundle", lambda: _move(prev_out, out))

    # Verification of the restored state (never trust the renames alone).
    # The state to re-establish is the one recorded *before* any rename (had_out / had_pub), not
    # whatever the interrupted sequence happened to reach.
    try:
        if had_out:
            if not out.is_dir():
                raise ExportError("previous canonical bundle is missing after restoration")
            bundle.load_bundle(out).verify_all()
        elif out.exists():
            raise ExportError("a partial canonical bundle remains where none existed before")
        if public is not None:
            if had_pub:
                if not public.is_dir():
                    raise ExportError("previous public copy is missing after restoration")
                if had_out:
                    diffs = compare_dirs(out, public, ignore={bundle.LAST_ATTEMPT_NAME})
                    if diffs:
                        raise ExportError(
                            f"restored public copy differs from the bundle: {diffs[:5]}"
                        )
            elif public.exists():
                raise ExportError("a partial public copy remains where none existed before")
        for leftover in (prev_out, prev_pub):
            if leftover is not None and leftover.exists():
                raise ExportError(f"previous version still parked at {leftover.name}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"verification: {type(exc).__name__}: {exc}")
    return {"status": "failed" if errors else "verified", "steps": steps, "errors": errors}


def export_bundle(
    artifacts: Path, out: Path, *, public_copy: Path | None = None, repo: Path = REPO_ROOT
) -> dict[str, Any]:
    """Build, stage, verify, then replace both published destinations; restore both on failure.

    Sequence:

    1. **Staging.** Build into ``<out>.build-<pid>`` and verify it with the fail-closed loader;
       copy it to ``<public>.build-<pid>`` and verify the copy is byte-identical. Nothing published
       has been touched yet; a failure here just removes the staging directories.
    2. **Replacement.** Park the previous canonical bundle as ``<out>.previous-<pid>``, move the
       staged bundle in; park the previous public copy, move the staged copy in. Then run the
       final verification on the *published* paths (loader over ``out``; byte comparison with the
       public copy).
    3. **Cleanup.** Only after that verification succeeds are the parked previous versions deleted.

    What this does and does not guarantee: each directory rename is atomic on its own, but the
    two-to-four renames are **not one transaction** — a reader can observe a momentarily missing
    or mismatched destination between them, and a process killed mid-swap leaves ``.previous-*``
    / ``.build-*`` siblings behind (the weekly job excludes such siblings from its commit and
    restores both paths from git). Every *caught* failure during replacement or verification is
    rolled back in reverse order and the rollback is verified (:func:`_restore`); the exception
    then reports ``restored=True``. If the rollback itself cannot be verified the exception
    reports ``restored=False`` and the caller must not publish either path. Diagnostics go to
    ``_last_attempt.json`` (outside every digest); the immutable evidence files are never edited.
    """
    artifacts, out = Path(artifacts), Path(out)
    public = Path(public_copy) if public_copy is not None else None
    code_revision = git_head(repo)
    tmp_out = out.parent / f"{out.name}.build-{os.getpid()}"
    tmp_pub = public.parent / f"{public.name}.build-{os.getpid()}" if public else None
    prev_out = out.parent / f"{out.name}.previous-{os.getpid()}"
    prev_pub = public.parent / f"{public.name}.previous-{os.getpid()}" if public else None
    had_out, had_pub = out.exists(), bool(public and public.exists())
    published_state = "previous_bundle_preserved" if had_out else "no_previous_bundle"
    for stale in (tmp_out, tmp_pub, prev_out, prev_pub):
        if stale is not None and stale.exists():
            raise ExportError(
                f"leftover directory {stale} from an interrupted export; inspect and remove it first",
                restored=None,
                stage="staging",
            )

    # 1. staging -----------------------------------------------------------------------------
    try:
        manifest = build_into(artifacts, tmp_out, code_revision=code_revision)
        if public is not None and tmp_pub is not None:
            _copytree(tmp_out, tmp_pub)
            diffs = compare_dirs(tmp_out, tmp_pub)
            if diffs:
                raise ExportError(f"staged public copy differs from the staged bundle: {diffs[:5]}")
    except Exception as exc:  # noqa: BLE001 - any failure must leave the published paths untouched
        for tmp in (tmp_out, tmp_pub):
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
        _write_last_attempt(
            out,
            f"{type(exc).__name__}: {exc}",
            code_revision,
            stage="staging",
            restoration={"status": "not_needed", "steps": [], "errors": []},
            published_state=published_state,
        )
        raise ExportError(str(exc), restored=True, stage="staging") from exc

    # 2. replacement + final verification ------------------------------------------------------
    placed = {"out_moved": False, "out_placed": False, "pub_moved": False, "pub_placed": False}
    stage = "replacement"
    try:
        if had_out:
            _move(out, prev_out)
            placed["out_moved"] = True
        _move(tmp_out, out)
        placed["out_placed"] = True
        if public is not None and tmp_pub is not None and prev_pub is not None:
            if had_pub:
                _move(public, prev_pub)
                placed["pub_moved"] = True
            _move(tmp_pub, public)
            placed["pub_placed"] = True
        stage = "verification"
        report = _verify_published(out, public)
    except Exception as exc:  # noqa: BLE001
        restoration = _restore(
            out=out,
            public=public,
            tmp_out=tmp_out,
            tmp_pub=tmp_pub,
            prev_out=prev_out,
            prev_pub=prev_pub,
            placed=placed,
            had_out=had_out,
            had_pub=had_pub,
        )
        for tmp in (tmp_out, tmp_pub):
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
        restored = restoration["status"] == "verified"
        state = published_state if restored else "unknown"
        message = f"{type(exc).__name__}: {exc}"
        if not restored:
            message += (
                "; restoration could not be established (" + "; ".join(restoration["errors"]) + ")"
            )
        _write_last_attempt(
            out,
            message,
            code_revision,
            stage=stage,
            restoration=restoration,
            published_state=state,
        )
        raise ExportError(message, restored=restored, stage=stage) from exc

    # 3. cleanup of the parked previous versions ------------------------------------------------
    try:
        _rmtree(prev_out)
        if prev_pub is not None:
            _rmtree(prev_pub)
    except OSError as exc:
        raise ExportError(
            f"new bundle is in place and verified, but a parked previous version could not be "
            f"removed: {exc}; remove it before publishing",
            restored=None,
            stage="cleanup",
        ) from exc
    manifest["_verify_report"] = report
    return manifest


def mirror(src: Path, dest: Path) -> None:
    """Replace ``dest`` with a byte-identical copy of ``src``: stage the copy beside ``dest``,
    verify it, park the previous ``dest``, swap, and restore the previous ``dest`` if the swap or
    the verification fails. Used for ad-hoc copies; ``export_bundle`` stages the public copy
    itself so both destinations are replaced under one rollback."""
    src, dest = Path(src), Path(dest)
    tmp = dest.parent / f"{dest.name}.build-{os.getpid()}"
    prev = dest.parent / f"{dest.name}.previous-{os.getpid()}"
    for stale in (tmp, prev):
        if stale.exists():
            raise ExportError(f"leftover directory {stale} from an interrupted copy", restored=None)
    try:
        _copytree(src, tmp)
        diffs = compare_dirs(src, tmp)
        if diffs:
            raise ExportError(f"staged copy differs from the bundle: {diffs[:5]}")
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(tmp, ignore_errors=True)
        raise ExportError(str(exc), restored=True) from exc
    had = dest.exists()
    moved = placed_new = False
    try:
        if had:
            _move(dest, prev)
            moved = True
        _move(tmp, dest)
        placed_new = True
        diffs = compare_dirs(src, dest)
        if diffs:
            raise ExportError(f"public copy differs from the bundle: {diffs[:5]}")
    except Exception as exc:  # noqa: BLE001
        errors: list[str] = []
        try:
            if placed_new:
                _move(dest, tmp)
            if moved:
                _move(prev, dest)
            if had and not dest.is_dir():
                errors.append("previous copy missing after restoration")
            if not had and dest.exists():
                errors.append("partial copy remains where none existed")
        except Exception as inner:  # noqa: BLE001
            errors.append(f"{type(inner).__name__}: {inner}")
        shutil.rmtree(tmp, ignore_errors=True)
        if errors:
            raise ExportError(
                f"{exc}; restoration could not be established ({'; '.join(errors)})",
                restored=False,
                stage="replacement",
            ) from exc
        raise ExportError(str(exc), restored=True, stage="replacement") from exc
    _rmtree(prev)


def _files(root: Path) -> dict[str, Path]:
    return {
        p.relative_to(root).as_posix(): p
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.startswith(".")
    }


def _manifest_comparable(text: bytes) -> bytes:
    """manifest.json with ``code_revision.produced_at`` neutralised: the git HEAD recorded at
    export time necessarily differs from the HEAD of the commit that contains the export."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(doc, dict) and isinstance(doc.get("code_revision"), dict):
        doc["code_revision"] = {**doc["code_revision"], "produced_at": None}
    return dumps(doc).encode("utf-8")


def compare_dirs(a: Path, b: Path, *, ignore: set[str] = frozenset()) -> list[str]:
    """Relative paths that differ (missing on one side or different bytes)."""
    fa, fb = _files(Path(a)), _files(Path(b))
    diffs = []
    for rel in sorted(set(fa) | set(fb)):
        if rel in ignore:
            continue
        if rel not in fa or rel not in fb:
            diffs.append(rel)
            continue
        ba, bb = fa[rel].read_bytes(), fb[rel].read_bytes()
        if rel == bundle.MANIFEST_NAME and ba != bb:
            ba, bb = _manifest_comparable(ba), _manifest_comparable(bb)
        if ba != bb:
            diffs.append(rel)
    return diffs


def check_bundle(artifacts: Path, out: Path, *, public_copy: Path | None = None) -> list[str]:
    """Rebuild into a temporary sibling and compare with ``out`` (and ``out`` with the public
    copy). Never writes to ``out``. Returns the differing paths (empty when current)."""
    artifacts, out = Path(artifacts), Path(out)
    tmp = out.parent / f"{out.name}.check-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        build_into(artifacts, tmp, code_revision=git_head())
        diffs = [f"{out.name}/{d}" for d in compare_dirs(tmp, out, ignore=VOLATILE)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if public_copy is not None:
        diffs += [
            f"{Path(public_copy).name}/{d}"
            for d in compare_dirs(out, Path(public_copy), ignore={bundle.LAST_ATTEMPT_NAME})
        ]
    return diffs


def verify_bundle(out: Path, *, public_copy: Path | None = None) -> dict[str, Any]:
    """Load and verify ``out`` with the fail-closed loader; check the public copy is identical."""
    report = bundle.load_bundle(Path(out)).verify_all()
    if public_copy is not None:
        diffs = compare_dirs(Path(out), Path(public_copy), ignore={bundle.LAST_ATTEMPT_NAME})
        if diffs:
            raise ExportError(f"public copy differs from the bundle: {diffs}")
    return report


def summary(manifest: dict[str, Any]) -> str:
    """Short human summary of a built manifest."""
    lines = []
    snaps = manifest["snapshots"]
    n_rows = sum(s["inputs"]["n_rows"] for s in snaps)
    lines.append(
        f"snapshots: {len(snaps)} ({sum(1 for s in snaps if s['mode'] != 'synthetic')} real, "
        f"{sum(1 for s in snaps if s['mode'] == 'synthetic')} synthetic); input rows: {n_rows}; "
        f"cases: {manifest['cases']['n_rows']}; latest weekly: {manifest['latest_weekly_snapshot_id']}"
    )
    for s in snaps:
        if s["mode"] == "synthetic":
            continue
        rec = s.get("reconciliation") or {}
        out = s.get("outcomes")
        lines.append(
            f"  {s['snapshot_id']}: {s['mode']} rows={s['inputs']['n_rows']} "
            f"observed={out['n_observed'] if out else 0} reconciliation={rec.get('status')} "
            f"baseline={s['baseline']['provenance_basis']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "decision_lab")
    parser.add_argument("--public-copy", type=Path, default=None)
    args = parser.parse_args(argv)
    manifest = export_bundle(args.artifacts, args.out, public_copy=args.public_copy)
    print(summary(manifest))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
