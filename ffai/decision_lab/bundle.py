"""Fail-closed loader for an exported Decision Lab bundle directory.

A bundle is ``manifest.json`` plus the files it lists (``inputs/*.json``, ``outcomes/*.json``,
``cases.json``). Every file is verified on load: byte digest, content digest (canonical JSON), row
count, schema, semantic checks, and cross-file compatibility (an outcome snapshot must name the
content digest of its inputs snapshot). Any mismatch raises :class:`BundleError`; nothing is
substituted, defaulted, or partially loaded. ``_build.json`` (volatile build metadata) and
``_last_attempt.json`` (a failed export attempt) are informational and never affect identities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ffai.decision_lab import SPEC_PATH, contracts
from ffai.decision_lab.canonical import content_id, file_sha256

MANIFEST_NAME = "manifest.json"
BUILD_NAME = "_build.json"
LAST_ATTEMPT_NAME = "_last_attempt.json"


class BundleError(RuntimeError):
    """The bundle is missing, corrupt, or inconsistent with this code."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BundleError(f"missing file {path}") from exc
    except json.JSONDecodeError as exc:
        raise BundleError(f"{path} is not valid JSON: {exc}") from exc


class Bundle:
    def __init__(self, root: Path, *, verify_spec: bool = True):
        self.root = Path(root)
        self.manifest: dict[str, Any] = _read_json(self.root / MANIFEST_NAME)
        try:
            contracts.check_manifest(self.manifest)
        except contracts.ContractError as exc:
            raise BundleError(f"manifest: {exc}") from exc
        if verify_spec:
            expected = file_sha256(SPEC_PATH)
            if self.manifest["policy_spec"]["sha256"] != expected:
                raise BundleError(
                    "bundle was exported with a different policy_spec.json; rebuild it with this code"
                )
        self._by_id = {s["snapshot_id"]: s for s in self.manifest["snapshots"]}
        self._inputs: dict[str, dict[str, Any]] = {}
        self._outcomes: dict[str, dict[str, Any] | None] = {}
        self._cases: dict[str, Any] | None = None

    # -- helpers -------------------------------------------------------------------------
    def _verified(self, ref: dict[str, Any], *, rows_key: str = "n_rows") -> dict[str, Any]:
        path = self.root / ref["path"]
        if not path.exists():
            raise BundleError(f"listed file is missing: {ref['path']}")
        actual_file = file_sha256(path)
        if actual_file != ref["file_sha256"]:
            raise BundleError(f"{ref['path']}: byte digest mismatch (corrupt or edited)")
        doc = _read_json(path)
        actual_content = content_id(doc)
        if actual_content != ref["content_sha256"]:
            raise BundleError(f"{ref['path']}: content digest mismatch")
        rows = doc.get("rows", doc.get("cases"))
        if not isinstance(rows, list) or len(rows) != ref[rows_key]:
            raise BundleError(
                f"{ref['path']}: row count {len(rows) if isinstance(rows, list) else '?'} != manifest {ref[rows_key]}"
            )
        return doc

    # -- accessors -----------------------------------------------------------------------
    @property
    def snapshot_ids(self) -> list[str]:
        return list(self._by_id)

    def snapshot_entry(self, snapshot_id: str) -> dict[str, Any]:
        try:
            return self._by_id[snapshot_id]
        except KeyError as exc:
            raise BundleError(f"unknown snapshot {snapshot_id}") from exc

    def inputs_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        if snapshot_id not in self._inputs:
            entry = self.snapshot_entry(snapshot_id)
            doc = self._verified(entry["inputs"])
            try:
                contracts.check_inputs_snapshot(doc)
            except contracts.ContractError as exc:
                raise BundleError(str(exc)) from exc
            if doc["snapshot_id"] != snapshot_id or doc["mode"] != entry["mode"]:
                raise BundleError(
                    f"{entry['inputs']['path']}: snapshot identity differs from the manifest"
                )
            self._inputs[snapshot_id] = doc
        return self._inputs[snapshot_id]

    def outcome_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        if snapshot_id not in self._outcomes:
            entry = self.snapshot_entry(snapshot_id)
            ref = entry.get("outcomes")
            if ref is None:
                self._outcomes[snapshot_id] = None
            else:
                doc = self._verified(ref)
                if doc["coverage"]["observed"] != ref["n_observed"]:
                    raise BundleError(f"{ref['path']}: observed count differs from the manifest")
                inputs = self.inputs_snapshot(snapshot_id)
                try:
                    contracts.check_outcome_snapshot(doc, inputs)
                except contracts.ContractError as exc:
                    raise BundleError(str(exc)) from exc
                self._outcomes[snapshot_id] = doc
        return self._outcomes[snapshot_id]

    def outcome_id(self, snapshot_id: str) -> str | None:
        doc = self.outcome_snapshot(snapshot_id)
        return content_id(doc) if doc is not None else None

    def cases(self) -> dict[str, Any]:
        if self._cases is None:
            doc = self._verified(self.manifest["cases"])
            try:
                contracts.check_cases(doc)
            except contracts.ContractError as exc:
                raise BundleError(str(exc)) from exc
            for c in doc["cases"]:
                if c["snapshot_id"] not in self._by_id:
                    raise BundleError(
                        f"case {c['case_id']} references unknown snapshot {c['snapshot_id']}"
                    )
            self._cases = doc
        return self._cases

    def build_info(self) -> dict[str, Any] | None:
        p = self.root / BUILD_NAME
        return _read_json(p) if p.exists() else None

    def last_attempt(self) -> dict[str, Any] | None:
        p = self.root / LAST_ATTEMPT_NAME
        return _read_json(p) if p.exists() else None

    def verify_all(self) -> dict[str, Any]:
        """Load and verify every listed file; returns a report (raises on the first failure)."""
        report = {"snapshots": [], "cases": None}
        for sid in self.snapshot_ids:
            inp = self.inputs_snapshot(sid)
            out = self.outcome_snapshot(sid)
            report["snapshots"].append(
                {
                    "snapshot_id": sid,
                    "inputs_rows": len(inp["rows"]),
                    "outcome_rows": len(out["rows"]) if out else None,
                    "observed": out["coverage"]["observed"] if out else None,
                }
            )
        report["cases"] = len(self.cases()["cases"])
        return report


def load_bundle(root: Path, *, verify_spec: bool = True) -> Bundle:
    root = Path(root)
    if not (root / MANIFEST_NAME).exists():
        raise BundleError(f"no bundle at {root} (missing {MANIFEST_NAME})")
    return Bundle(root, verify_spec=verify_spec)
