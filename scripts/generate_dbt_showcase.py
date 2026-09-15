#!/usr/bin/env python3
"""Build the safe, deterministic dbt metadata used by the static case study.

This deliberately reads repository definitions, not run_results/catalog data.  It publishes an
allow-list of structural fields and short raw-SQL excerpts; credentials, compiled SQL, absolute
paths, invocation ids, timestamps and warehouse row counts can therefore never enter the site.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend-next/src/data/dbt-showcase.json"
PROJECT = "ffai_dbt"
GRAINS = {
    "slv_player_stats": "player_id × season × week",
    "slv_actuals": "player_id × season × week × scoring_format",
    "slv_predictions": "player_id × season × week × model_version × candidate",
    "fct_player_week": "player_id × season × week × model_version × candidate",
    "fct_player_decisions": "player_id × season × week × model_version × candidate",
    "fct_weekly_eval": "eval_window × season × week × model_version × candidate × cohort",
    "fct_decision_policy": "season × week × position × model_version × candidate × min_floor",
    "dim_player": "player_id",
    "dim_player_current": "player_id",
    "dim_player_asof": "player_id × period_key",
    "dim_model_version": "model_version",
    "fct_tier_outcomes": "tiers_version × position × player_id",
    "snp_player": "player_id × dbt_valid_from (captured attribute version)",
}
CONSUMERS = {
    "fct_weekly_eval": "/marts/weekly_eval and the Evaluation page",
    "fct_player_week": "/marts/player_week/{player_id} and Player history",
    "fct_decision_policy": "/marts/decisions (explicitly pinned to v1)",
    "dim_player": "player mart endpoints; latest-seen identity",
    "dim_player_current": "Warehouse view; not exported/currently consumed by this page",
    "dim_player_asof": "Warehouse view; not exported/currently consumed by this page",
}


def load_specs() -> dict[str, dict]:
    specs: dict[str, dict] = {}
    for path in sorted((ROOT / "dbt/models").glob("**/*.yml")):
        doc = yaml.safe_load(path.read_text()) or {}
        for model in doc.get("models", []):
            specs[model["name"]] = {**model, "yaml_path": path.relative_to(ROOT).as_posix()}
    path = ROOT / "dbt/snapshots/_snapshots.yml"
    for snapshot in (yaml.safe_load(path.read_text()) or {}).get("snapshots", []):
        specs[snapshot["name"]] = {**snapshot, "yaml_path": path.relative_to(ROOT).as_posix()}
    return specs


def generate() -> dict:
    specs = load_specs()
    nodes = []
    inputs: list[Path] = []
    sql_files = sorted((ROOT / "dbt/models").glob("**/*.sql")) + sorted(
        (ROOT / "dbt/snapshots").glob("*.sql")
    )
    for path in sql_files:
        name = (
            "fct_decision_policy"
            if path.stem in {"fct_decision_policy_v1", "fct_decision_policy_v2"}
            else path.stem
        )
        spec = specs[name]
        text = path.read_text()
        inputs.extend([path, ROOT / spec["yaml_path"]])
        layer = "snapshot" if "snapshots" in path.parts else path.parent.name
        versions = spec.get("versions") or [None]
        if spec.get("versions"):
            file_version = int(path.stem.rsplit("_v", 1)[1])
            versions = [v for v in versions if v["v"] == file_version]
        for ver in versions:
            version = ver.get("v") if ver else None
            unique_id = f"{'snapshot' if layer == 'snapshot' else 'model'}.{PROJECT}.{name}" + (
                f".v{version}" if version else ""
            )
            alias = (ver or {}).get("config", {}).get("alias") or (
                path.stem if version is None or version == 1 else f"{name}_v{version}"
            )
            configured = spec.get("config", {}).get("materialized")
            materialization = configured or (
                "snapshot"
                if layer == "snapshot"
                else (
                    "view"
                    if name in {"dim_player_current", "dim_player_asof"}
                    else "incremental" if name == "slv_player_stats" else "table"
                )
            )
            export = layer == "gold" and spec.get("config", {}).get("meta", {}).get("export", True)
            columns = [
                {k: c.get(k) for k in ("name", "data_type", "description") if c.get(k) is not None}
                for c in spec.get("columns", [])
            ]
            refs = sorted(set(__import__("re").findall(r"ref\(['\"]([^'\"]+)", text)))
            nodes.append(
                {
                    "uniqueId": unique_id,
                    "name": name,
                    "version": version,
                    "alias": alias,
                    "resourceType": "snapshot" if layer == "snapshot" else "model",
                    "layer": layer,
                    "materialization": materialization,
                    "description": " ".join(str(spec.get("description", "")).split()),
                    "grain": GRAINS.get(name, "See model SQL and contract"),
                    "columns": columns,
                    "upstream": refs,
                    "constraints": spec.get("constraints", []),
                    "tests": spec.get("data_tests", []),
                    "sqlPath": path.relative_to(ROOT).as_posix(),
                    "yamlPath": spec["yaml_path"],
                    "sqlExcerpt": "\n".join(text.splitlines()[:24]),
                    "exported": bool(export),
                    "consumer": CONSUMERS.get(name, "No direct application consumer documented"),
                }
            )
    nodes.sort(key=lambda n: n["uniqueId"])
    unique_inputs = sorted(set(inputs))
    digest = hashlib.sha256()
    for path in unique_inputs:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    layers = {
        layer: sum(n["layer"] == layer for n in nodes) for layer in ("bronze", "silver", "gold")
    }
    return {
        "schemaVersion": 1,
        "sourceFingerprint": digest.hexdigest(),
        "inventory": {
            "executableModelNodes": sum(n["resourceType"] == "model" for n in nodes),
            "logicalModelNames": len({n["name"] for n in nodes if n["resourceType"] == "model"}),
            "snapshotNodes": sum(n["resourceType"] == "snapshot" for n in nodes),
            "layers": layers,
        },
        "nodes": nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(generate(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != rendered:
            print(f"stale showcase metadata: run {Path(__file__).relative_to(ROOT)}")
            return 1
        print(f"showcase metadata is current: {OUT.relative_to(ROOT)}")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(rendered)
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
