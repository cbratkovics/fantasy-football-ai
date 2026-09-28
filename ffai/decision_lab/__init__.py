"""Decision Lab: one-slot start/sit comparisons over committed evidence (ADR-0032, ADR-0033).

The product question: *given these players as my actual alternatives for one starting slot,
what does the available evidence recommend, what assumptions could change that recommendation,
what did I choose, and what can we honestly conclude after outcomes become available?*

Layout (every module is pure Python over dicts; nothing here trains a model or opens a socket):

* :mod:`ffai.decision_lab.canonical` — canonical JSON, deterministic number formatting, and the
  SHA-256 content identities every other module hashes with. Mirrored in TypeScript.
* :mod:`ffai.decision_lab.policy` — the reference decision policy (``policy_spec.json`` is the
  versioned specification; the TypeScript implementation must agree on the golden fixtures).
* :mod:`ffai.decision_lab.metrics` — outcome metrics for a saved choice set and an explicit action.
* :mod:`ffai.decision_lab.receipts` — decision receipts: immutable inputs and result plus
  appended action / outcome events; validation, replay, and idempotent merge.
* :mod:`ffai.decision_lab.contracts` — JSON Schemas and semantic validation for the bundle files.
* :mod:`ffai.decision_lab.bundle` — fail-closed loader for an exported bundle directory.
* :mod:`ffai.decision_lab.exporter` — builds the bundle from the committed gold marts, the
  predictions files, and the evaluation artifacts (``scripts/export_decision_lab.py``).
* :mod:`ffai.decision_lab.cases` — the curated case library and its deterministic discovery.
* :mod:`ffai.decision_lab.replay` — the CLI that re-validates and recomputes a receipt.

Inputs never contain target-week actuals, error metrics, hindsight ranks, or regret; outcomes are
a separate snapshot with its own identity, so attaching them later never changes a decision id.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SPEC_PATH = Path(__file__).with_name("policy_spec.json")


@lru_cache(maxsize=1)
def spec() -> dict[str, Any]:
    """The versioned policy specification (``policy_spec.json``), loaded once."""
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


POLICY_VERSION: str = spec()["policy_version"]
SCHEMA_VERSIONS: dict[str, str] = dict(spec()["schema_versions"])
