"""Frozen v1 receipts replay against their original evidence, without today's context."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ffai.decision_lab import receipts, replay

FIXTURES = Path(__file__).parent / "fixtures" / "start_sit_compat"
INDEX = json.loads((FIXTURES / "index.json").read_text())


def read(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.parametrize("example", INDEX["examples"], ids=lambda x: x["label"])
def test_original_v1_receipts_keep_identity_events_and_evidence(example):
    inputs = read(example["inputs"]["path"])
    outcomes = read(example["outcomes"]["path"])
    for kind in ("inputs", "outcomes"):
        data = (FIXTURES / example[kind]["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == example[kind]["sha256"]
    for entry in example["receipts"]:
        receipt = read(entry["path"])
        checks = receipts.validate_receipt(
            receipt,
            inputs_lookup=lambda sid: inputs if sid == example["snapshot_id"] else None,
            outcome_lookup=lambda sid: outcomes if sid == example["snapshot_id"] else None,
        )
        assert all(check["ok"] for check in checks)
        assert "snapshot_digest" in {check["check"] for check in checks}
        assert replay.replayed_result(receipt) == receipt["result"]
        assert receipt["decision_id"] == entry["decision_id"]
        assert receipt["result_sha256"] == entry["result_sha256"]
        assert [event["event_id"] for event in receipt["events"]] == entry["event_ids"]
        if receipt["outcome"]["state"] == "attached":
            assert "metrics_replay" in {check["check"] for check in checks}
        else:
            assert receipt["action"]["state"] == "not_recorded"
            assert receipt["action"]["chosen_player_id"] is None


def test_frozen_receipt_refuses_replacement_evidence():
    example = INDEX["examples"][0]
    receipt = read(example["receipts"][0]["path"])
    changed = copy.deepcopy(read(example["inputs"]["path"]))
    changed["rows"][0]["projection"] += 1
    with pytest.raises(receipts.ReceiptError, match="snapshot_digest"):
        receipts.validate_receipt(receipt, inputs_lookup=lambda _: changed)


def test_recorded_choice_remains_an_explicit_event_not_the_recommendation():
    initial = read("historical-receipt-0.json")
    recorded = read("historical-receipt-1.json")
    assert initial["action"]["state"] == "not_recorded"
    assert recorded["action"]["state"] == "recorded"
    assert recorded["action"]["kind"] == "hypothetical_replay"
    assert [event["event_type"] for event in recorded["events"]] == [
        "action_recorded",
        "outcome_attached",
    ]
    assert recorded["inputs"] == initial["inputs"]
    assert recorded["result"] == initial["result"]
