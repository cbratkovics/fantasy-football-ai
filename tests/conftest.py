"""Shared fixtures.

``stats`` is the weekly stats frame. By default it is the small committed fixture
(``tests/fixtures/player_stats_sample.csv``: 40 players, 2023-2024, real nflverse rows) so the
suite runs offline in CI. Set ``FFAI_TEST_DATA=full`` to run the same tests on the full
2019-2024 pull through ``ffai.data.nflverse`` (uses the local parquet cache, network on a miss).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "player_stats_sample.csv"


@pytest.fixture(scope="session")
def stats() -> pd.DataFrame:
    if os.environ.get("FFAI_TEST_DATA") == "full":
        from ffai.data import nflverse

        return nflverse.load_weekly_stats(range(2019, 2025))
    df = pd.read_csv(FIXTURE)
    df["season"] = df["season"].astype("int64")
    df["week"] = df["week"].astype("int64")
    return df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def using_full_data() -> bool:
    return os.environ.get("FFAI_TEST_DATA") == "full"
