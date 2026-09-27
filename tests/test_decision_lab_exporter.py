"""Exporter tests over the committed artifacts (offline; the bundle is built once per module)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import pytest

from ffai.config import REPO_ROOT
from ffai.decision_lab import bundle, cases, exporter, policy
from ffai.decision_lab.canonical import content_id, file_sha256, q

ARTIFACTS = REPO_ROOT / "artifacts"
COMMITTED = ARTIFACTS / "decision_lab"
PUBLIC = REPO_ROOT / "frontend-next" / "public" / "decision-lab"
MART = ARTIFACTS / "marts" / "fct_player_week.parquet"
MODEL_VERSION = "20260911-asof_v1-d333de20"

pytestmark = pytest.mark.skipif(not MART.exists(), reason="committed marts are required")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("lab") / "bundle"
    manifest = exporter.export_bundle(ARTIFACTS, out)
    return out, manifest


def _snapshots(manifest: dict) -> list[dict]:
    return [s for s in manifest["snapshots"] if s["mode"] != "synthetic"]


# --- real-data reconciliation ------------------------------------------------------------------


def test_reconciliation_statuses_on_committed_evidence(built):
    _, manifest = built
    real = _snapshots(manifest)
    assert len(real) == 39
    for s in real:
        rec = s["reconciliation"]
        if s["source_family"] in ("frozen_test", "out_of_sample_season"):
            assert rec["status"] == "matched", s["snapshot_id"]
            assert s["baseline"]["provenance_basis"] == "population_rule_and_reconciliation"
            assert s["baseline"]["reconciled"] is True
            assert rec["reference"].startswith("eval-")
            assert set(rec["expected"]) == {"ALL", "QB", "RB", "WR", "TE"}
        elif s["week"] in (1, 2):
            assert rec["status"] == "matched", s["snapshot_id"]
            assert rec["reference"] == f"rolling_2026.json week {s['week']}"
            assert s["mode"] == "historical_replay"
        else:
            assert rec["status"] == "unavailable", s["snapshot_id"]
            assert s["baseline"]["provenance_basis"] == "population_rule_unreconciled"
            assert s["baseline"]["reconciled"] is None
            assert s["mode"] == "published_weekly"
            assert s["outcomes"] is None
    assert manifest["latest_weekly_snapshot_id"] == f"weekly-2026-w03-{MODEL_VERSION}-rf"
    assert manifest["lineage"]["model_version"] == MODEL_VERSION
    assert manifest["lineage"]["feature_version"] == "asof_v1"


def test_champion_at_source_and_row_provenance(built):
    out, manifest = built
    b = bundle.load_bundle(out)
    for s in _snapshots(manifest):
        doc = b.inputs_snapshot(s["snapshot_id"])
        basis = "predictions_file" if s["source_family"] == "weekly" else "eval_artifact"
        assert doc["champion_at_source"]["basis"] == basis
        assert doc["champion_at_source"]["candidate_by_position"] == {
            p: "rf" for p in ("QB", "RB", "WR", "TE")
        }
        assert {r["candidate"] for r in doc["rows"]} == {"rf"}
        assert {r["baseline_provenance"] for r in doc["rows"]} == {"player_history"}
        default = "unknown" if s["source_family"] == "weekly" else "realized_stats_row"
        assert {r["availability_default"] for r in doc["rows"]} == {default}
        assert {r["display_source"] for r in doc["rows"]} == {"prediction_source"}
        assert doc["publication"]["status"] == ("published" if default == "unknown" else "recorded")
        if s["source_family"] == "weekly":
            assert doc["publication"]["run_id"] is not None
            assert doc["publication"]["action"] == "PUBLISH"
            assert doc["generated_at_utc"] is not None
        else:
            assert doc["generated_at_utc"] is None
            assert doc["data_cutoff"]["week"] >= 1


def test_grain_uniqueness_and_no_outcome_fields(built):
    out, manifest = built
    forbidden = {
        "actual",
        "abs_error",
        "regret",
        "hit",
        "downside",
        "prediction_rank",
        "actual_source",
    }
    for s in manifest["snapshots"]:
        doc = json.loads((out / s["inputs"]["path"]).read_text(encoding="utf-8"))
        ids = [r["player_id"] for r in doc["rows"]]
        assert len(ids) == len(set(ids)), s["snapshot_id"]
        assert ids == sorted(ids)
        for r in doc["rows"]:
            assert not (forbidden & set(r)), r
            assert r["model_version"] == doc["model_version"]
        assert doc["population"]["n_rows"] == len(ids)


def test_manifest_digests_equal_recomputation(built):
    out, manifest = built
    for s in manifest["snapshots"]:
        for key in ("inputs", "outcomes"):
            ref = s[key]
            if ref is None:
                continue
            path = out / ref["path"]
            assert file_sha256(path) == ref["file_sha256"]
            doc = json.loads(path.read_text(encoding="utf-8"))
            assert content_id(doc) == ref["content_sha256"]
            assert len(doc["rows"]) == ref["n_rows"]
            if key == "outcomes":
                assert doc["coverage"]["observed"] == ref["n_observed"]
                inputs = json.loads((out / s["inputs"]["path"]).read_text(encoding="utf-8"))
                assert doc["inputs_content_sha256"] == content_id(inputs)
    cases_doc = json.loads((out / "cases.json").read_text(encoding="utf-8"))
    assert content_id(cases_doc) == manifest["cases"]["content_sha256"]
    assert file_sha256(out / "cases.json") == manifest["cases"]["file_sha256"]
    build = json.loads((out / "_build.json").read_text(encoding="utf-8"))
    on_disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert build["manifest_content_sha256"] == content_id(on_disk)
    assert "manifest_content_sha256" not in json.dumps(on_disk)


def test_build_is_deterministic(built, tmp_path):
    out, _ = built
    again = tmp_path / "again"
    exporter.export_bundle(ARTIFACTS, again)
    assert exporter.compare_dirs(out, again, ignore=exporter.VOLATILE) == []
    assert (out / "manifest.json").read_bytes() == (again / "manifest.json").read_bytes()
    build_a = json.loads((out / "_build.json").read_text(encoding="utf-8"))
    build_b = json.loads((again / "_build.json").read_text(encoding="utf-8"))
    assert build_a["manifest_content_sha256"] == build_b["manifest_content_sha256"]


def test_tampered_inputs_file_fails_closed(built, tmp_path):
    out, manifest = built
    copy = tmp_path / "tampered"
    shutil.copytree(out, copy)
    sid = manifest["snapshots"][0]["snapshot_id"]
    path = copy / manifest["snapshots"][0]["inputs"]["path"]
    data = bytearray(path.read_bytes())
    idx = data.index(b'"projection": ') + len(b'"projection": ')
    data[idx] = ord("9") if data[idx] != ord("9") else ord("1")
    path.write_bytes(bytes(data))
    with pytest.raises(bundle.BundleError, match="digest mismatch"):
        bundle.load_bundle(copy).inputs_snapshot(sid)
    with pytest.raises(bundle.BundleError):
        bundle.load_bundle(copy).verify_all()


def test_failed_build_leaves_existing_bundle_untouched(built, tmp_path):
    out, _ = built
    existing = tmp_path / "existing"
    shutil.copytree(out, existing)
    before = {p.relative_to(existing): p.read_bytes() for p in existing.rglob("*") if p.is_file()}
    empty = tmp_path / "empty_artifacts"
    empty.mkdir()
    with pytest.raises(exporter.ExportError):
        exporter.export_bundle(empty, existing)
    after = {p.relative_to(existing): p.read_bytes() for p in existing.rglob("*") if p.is_file()}
    attempt = existing / bundle.LAST_ATTEMPT_NAME
    assert attempt.exists()
    note = json.loads(attempt.read_text(encoding="utf-8"))
    assert note["status"] == "failed" and "missing" in note["error"]
    after.pop(Path(bundle.LAST_ATTEMPT_NAME))
    assert after == before
    assert not list(tmp_path.glob("existing.build-*"))
    bundle.load_bundle(existing).cases()


def _small_artifacts(tmp_path: Path, mart_sql: str) -> Path:
    """A copy of the artifacts tree with a duckdb-built mart in place of the committed one."""
    root = tmp_path / "artifacts"
    (root / "marts").mkdir(parents=True)
    (root / "eval").mkdir()
    shutil.copy(ARTIFACTS / "manifest.json", root / "manifest.json")
    shutil.copy(ARTIFACTS / "marts" / "_export_manifest.json", root / "marts")
    for p in (ARTIFACTS / "eval").glob("*.json"):
        if "oos2025" in p.name or p.name.startswith("rolling_"):
            shutil.copy(p, root / "eval" / p.name)
    shutil.copytree(ARTIFACTS / "predictions", root / "predictions")
    shutil.copytree(ARTIFACTS / "runs", root / "runs")
    model = root / "models" / MODEL_VERSION
    model.mkdir(parents=True)
    for name in ("metadata.json", "oos_predictions_2025.csv"):
        shutil.copy(ARTIFACTS / "models" / MODEL_VERSION / name, model / name)
    con = duckdb.connect()
    con.execute(
        f"copy ({mart_sql.format(src=str(MART))}) to '{root / 'marts' / 'fct_player_week.parquet'}' (format parquet)"
    )
    return root


def test_mismatched_mart_yields_unknown_baseline(tmp_path):
    # the oos window is shortened to four weeks (n differs) and its baseline is shifted by one
    # point (baseline MAE differs): both must be reported, never explained away
    sql = (
        "select * replace (case when source = 'out_of_sample_season' then baseline + 1.0 "
        "else baseline end as baseline) from read_parquet('{src}') "
        "where source = 'weekly' or (source = 'out_of_sample_season' and week <= 4)"
    )
    root = _small_artifacts(tmp_path, sql)
    out = tmp_path / "bundle"
    out.mkdir()
    marker = out / bundle.LAST_ATTEMPT_NAME
    marker.write_text("{}", encoding="utf-8")
    manifest = exporter.export_bundle(root, out)
    assert not marker.exists(), "a successful build removes the failure marker"
    oos = [s for s in manifest["snapshots"] if s["source_family"] == "out_of_sample_season"]
    weekly = [s for s in manifest["snapshots"] if s["source_family"] == "weekly"]
    assert len(oos) == 4 and len(weekly) == 3
    for s in oos:
        assert s["reconciliation"]["status"] == "mismatch"
        assert s["reconciliation"]["expected"]["ALL"]["n"] == 5914
        assert s["reconciliation"]["observed"]["ALL"]["n"] < 5914
        assert s["baseline"]["provenance_basis"] == "unknown"
        assert s["baseline"]["reconciled"] is False
    assert {s["reconciliation"]["status"] for s in weekly} == {"matched", "unavailable"}
    b = bundle.load_bundle(out)
    snap = b.inputs_snapshot(oos[0]["snapshot_id"])
    assert {r["baseline_provenance"] for r in snap["rows"]} == {"unknown"}
    # fail-closed baseline claims: the policy refuses to present a baseline comparison
    top = [r["player_id"] for r in snap["rows"] if r["position"] == "RB"][:2]
    case = {
        "slot": "RB",
        "alternatives": top,
        "overrides": {},
        "parameters": policy.default_parameters(),
    }
    result = policy.evaluate(cases.case_inputs(case, snap))
    assert result["status"] == "review"
    assert "baseline_unavailable" in {r["code"] for r in result["reasons"]}
    assert "baseline_unreconciled" in result["limitations"]
    assert result["baseline"]["status"] == "unavailable"
    # check / verify / mirror helpers on this small bundle
    assert exporter.check_bundle(root, out) == []
    mirror = tmp_path / "mirror"
    exporter.mirror(out, mirror)
    assert exporter.verify_bundle(out, public_copy=mirror)["cases"] >= 10
    (mirror / "cases.json").write_text("{}", encoding="utf-8")
    with pytest.raises(exporter.ExportError, match="public copy differs"):
        exporter.verify_bundle(out, public_copy=mirror)
    assert exporter.check_bundle(root, out, public_copy=mirror) == ["mirror/cases.json"]
    assert not list(out.parent.glob(f"{out.name}.check-*"))


# --- committed bundle ----------------------------------------------------------------------------


def test_committed_bundle_equals_fresh_build_and_public_copy(built):
    out, _ = built
    assert COMMITTED.exists(), "run scripts/export_decision_lab.py"
    assert exporter.compare_dirs(COMMITTED, out, ignore=exporter.VOLATILE) == []
    assert (
        PUBLIC.exists()
    ), "run scripts/export_decision_lab.py --public-copy frontend-next/public/decision-lab"
    assert exporter.compare_dirs(COMMITTED, PUBLIC, ignore={bundle.LAST_ATTEMPT_NAME}) == []
    assert not (COMMITTED / bundle.LAST_ATTEMPT_NAME).exists()


def test_committed_oos_inputs_equal_mart_rows():
    b = bundle.load_bundle(COMMITTED)
    con = duckdb.connect()
    con.execute(f"create view m as select * from read_parquet('{MART}')")
    counts = dict(
        con.execute(
            "select week, count(*) from m where source = 'out_of_sample_season' and candidate = 'rf' group by 1"
        ).fetchall()
    )
    assert len(counts) == 18
    csv_names: dict[tuple[str, int], str] = {}
    for pid, week, name in con.execute(
        "select player_id, week, player_display_name from read_csv_auto(?) where candidate = 'rf'",
        [str(ARTIFACTS / "models" / MODEL_VERSION / "oos_predictions_2025.csv")],
    ).fetchall():
        csv_names[(pid, int(week))] = name
    for week, n in sorted(counts.items()):
        sid = f"oos-2025-w{week:02d}-{MODEL_VERSION}-rf"
        snap = b.inputs_snapshot(sid)
        assert len(snap["rows"]) == n
        sample = con.execute(
            "select player_id, prediction, prediction_floor, prediction_ceiling, baseline, position from m "
            "where source = 'out_of_sample_season' and candidate = 'rf' and week = ? order by player_id limit 7",
            [week],
        ).fetchall()
        rows = {r["player_id"]: r for r in snap["rows"]}
        for pid, pred, floor, ceil, base, pos in sample:
            r = rows[pid]
            assert r["projection"] == q(pred)
            assert r["floor"] == q(floor)
            assert r["ceiling"] == q(ceil)
            assert r["baseline"] == q(base)
            assert r["position"] == pos
            assert r["display_name"] == csv_names[(pid, week)]
            assert r["source_row_ref"] == f"fct_player_week:{pid}:2025:{week}:{MODEL_VERSION}:rf"
        outcome = b.outcome_snapshot(sid)
        assert outcome is not None and outcome["coverage"]["observed"] == n
