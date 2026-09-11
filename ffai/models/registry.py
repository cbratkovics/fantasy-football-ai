"""Artifact manifest: what the API serves and what the weekly job promotes.

``artifacts/manifest.json`` is the single source of truth for the served configuration:

    {
      "manifest_version": "1.0",
      "updated_at_utc": "...",
      "feature_version": "asof_v1",
      "champion":   {"model_version": "...", "candidate": "rf"},
      "challenger": {"model_version": "...", "candidate": "xgb"},
      "tiers_version": "...",
      "eval_id": "...",
      "data_through": {"season": 2024, "week": 18},
      "predictions": {"latest": "predictions/2025/week_01.json"},
      "last_run": {"run_id": "...", "at_utc": "...", "action": "PUBLISH|HOLD|PROMOTE",
                   "reasons": [...], "season": ..., "week": ...}
    }

Model artifacts live in ``artifacts/models/<model_version>/`` with ``<position>_<candidate>.pkl``
pipelines and one ``metadata.json``. Paths inside the manifest are relative to ``artifacts/``.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import joblib

from ffai.config import ARTIFACTS_DIR, MANIFEST_PATH, POSITIONS

MANIFEST_VERSION = "1.0"
ACTIONS = ("PUBLISH", "HOLD", "PROMOTE")


def utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def read_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"manifest not found at {path}; train models and run scripts/evaluate.py first"
        )
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_manifest(manifest: dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    manifest = dict(manifest)
    manifest["manifest_version"] = MANIFEST_VERSION
    manifest["updated_at_utc"] = utc_now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def model_dir(model_version: str, artifacts: Path = ARTIFACTS_DIR) -> Path:
    return artifacts / "models" / model_version


def read_model_metadata(model_version: str, artifacts: Path = ARTIFACTS_DIR) -> dict[str, Any]:
    with (model_dir(model_version, artifacts) / "metadata.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def candidate_for(candidate: str | dict[str, str], position: str) -> str:
    """Resolve a slot's candidate spec (a name, or a per-position mapping) for one position."""
    if isinstance(candidate, str):
        return candidate
    return candidate[position]


def load_pipelines(
    model_version: str, candidate: str | dict[str, str], artifacts: Path = ARTIFACTS_DIR
) -> dict[str, Any]:
    """Load ``{position: sklearn Pipeline}`` for a candidate spec of one model version."""
    d = model_dir(model_version, artifacts)
    out = {}
    for pos in POSITIONS:
        p = d / f"{pos}_{candidate_for(candidate, pos)}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"missing model artifact {p}")
        out[pos] = joblib.load(p)
    return out


def slot(manifest: dict[str, Any], name: str) -> tuple[str, str | dict[str, str]]:
    """(model_version, candidate spec) for ``champion`` or ``challenger``."""
    s = manifest.get(name)
    if not s:
        raise KeyError(f"manifest has no {name!r} slot")
    return s["model_version"], s["candidate"]


def should_promote(
    champion_recent_mae: list[float],
    challenger_recent_mae: list[float],
    champion_frozen_test_mae: float,
    challenger_frozen_test_mae: float,
    *,
    min_weeks: int = 4,
) -> tuple[bool, str]:
    """Deterministic promotion rule.

    Promote when the challenger beats the champion on rolling MAE in each of the last
    ``min_weeks`` scored weeks **and** on the frozen test set. Returns (decision, reason).
    """
    if len(champion_recent_mae) < min_weeks or len(challenger_recent_mae) < min_weeks:
        return False, f"fewer than {min_weeks} scored weeks available"
    recent_c = champion_recent_mae[-min_weeks:]
    recent_x = challenger_recent_mae[-min_weeks:]
    wins = sum(x < c for c, x in zip(recent_c, recent_x, strict=True))
    if wins < min_weeks:
        return (
            False,
            f"challenger won {wins}/{min_weeks} recent weeks (needs {min_weeks}/{min_weeks})",
        )
    if not challenger_frozen_test_mae < champion_frozen_test_mae:
        return False, (
            f"challenger frozen-test MAE {challenger_frozen_test_mae:.3f} not below champion "
            f"{champion_frozen_test_mae:.3f}"
        )
    return True, (
        f"challenger won all {min_weeks} recent weeks and frozen test "
        f"({challenger_frozen_test_mae:.3f} < {champion_frozen_test_mae:.3f})"
    )
