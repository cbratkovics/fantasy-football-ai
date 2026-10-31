"""Explicit local browser-review server with synthetic outcomes, never a production entry point.

FFAI_ACCEPTANCE_FIXTURE=synthetic .venv/bin/python -m uvicorn \
    --app-dir tests acceptance_api:create_app --factory --port 7861
The normal application still reads its committed artifacts. Only this isolated test process
replaces its prediction state; no publication files are written and no provider calls occur.
"""

import os
from contextlib import asynccontextmanager
from copy import deepcopy

from ffai.serve import app as api


def create_app():
    if os.environ.get("FFAI_ACCEPTANCE_FIXTURE") != "synthetic":
        raise RuntimeError("Explicit FFAI_ACCEPTANCE_FIXTURE=synthetic is required")

    @asynccontextmanager
    async def fixture_lifespan(app):
        state = api.load_state()
        payload = deepcopy(state.predictions[(2026, 8)])
        base = {
            "position": "WR",
            "team": "LOCAL",
            "prediction": 20,
            "floor": 10,
            "ceiling": 30,
            "receptions_estimate": 2,
            "candidate": "rf",
        }
        receiver = {
            **base,
            "player_id": "LOCAL-RECEIVER",
            "name": "LOCAL FIXTURE receiver",
            "actual": 12,
            "actual_receptions": 6,
            "actual_rules_id": "ffai-nflverse-v1",
            "actual_source_revision": "fixture-r1",
            "actual_receptions_revision": "fixture-r1",
        }
        payload["predictions"] = [
            receiver,
            {
                **base,
                "player_id": "LOCAL-MISSING",
                "name": "LOCAL FIXTURE missing evidence",
                "actual": 12,
            },
            {
                **base,
                "player_id": "LOCAL-ZERO",
                "name": "LOCAL FIXTURE zero",
                "actual": 0,
                "actual_receptions": 0,
                "actual_source_revision": "fixture-r1",
                "actual_receptions_revision": "fixture-r1",
            },
            {
                **base,
                "player_id": "LOCAL-NEGATIVE",
                "name": "LOCAL FIXTURE negative",
                "actual": -2,
                "actual_receptions": 0,
                "actual_source_revision": "fixture-r1",
                "actual_receptions_revision": "fixture-r1",
            },
        ]
        payload["scoring"]["derivation"] = "LOCAL SYNTHETIC FIXTURE: not published model output"
        state.predictions = {(2026, 8): payload}
        state.marts = None
        app.state.s = state
        yield

    api.app.router.lifespan_context = fixture_lifespan
    return api.app
