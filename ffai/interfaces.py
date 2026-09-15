"""The seams a different domain plugs into (ADR-0028; docs/TEMPLATE_BOUNDARY.md §2).

Three contracts, each implemented once in this repository:

* :class:`SourceLoader` — ``ffai.data.nflverse.LOADER``. Yields one row per
  ``(entity_key, *period_columns)`` with the raw columns every feature and target derive from,
  cached as dated parquet, and knows the current period from the source's own schedule.
* :class:`TargetSpec` — ``ffai.scoring.TARGET_SPEC``. Names the target column and its units,
  derives it from raw columns by explicit rules, and reconciles those rules against the value
  the source publishes (row for row; a non-empty disagreement frame is a stop, never a fudge).
* :class:`FeatureModule` — ``ffai.features.asof``. The one feature builder training, evaluation,
  and serving all call. Every feature of a row uses only rows strictly earlier within the entity.

The artifact shapes that flow between the layers are JSON Schemas under ``artifacts/schemas/``
(evaluation artifact, manifest, model metadata, predictions file, drift report); the drift-hold
rule's inputs and thresholds are documented there and in ``ffai.eval.drift``.

``tests/test_interfaces.py`` asserts the three implementations satisfy these protocols and that
every committed artifact validates against its schema.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class SourceLoader(Protocol):
    """Reads the source into the period-row frame every other layer consumes."""

    LIBRARY: str
    """``<client>==<version>`` recorded in training metadata for provenance."""

    ID_COLUMNS: tuple[str, ...]
    """Identifier / context columns: entity key, display name, cohort, period columns, ..."""

    STAT_COLUMNS: tuple[str, ...]
    """Raw numeric columns. Every feature and the target derive from these."""

    def load_period_rows(
        self, seasons: int | Iterable[int], *, refresh: bool = False
    ) -> pd.DataFrame:
        """One row per (entity_key, *period_columns), ``ID_COLUMNS + STAT_COLUMNS``, sorted by
        grain, raw values as published (numeric columns as float64). Raises ``KeyError`` when
        the source lacks an expected column."""

    def cache_path_for(self, name: str, seasons: int | Iterable[int]) -> Path:
        """The dated cache file a same-day load reads or writes (dbt's stats source)."""

    def current_period(self, today: Any = None) -> tuple[int, int]:
        """``(season, next_period_to_play)`` from the source's own schedule, not the clock."""

    def periods_in_season(self, season: int) -> int:
        """Number of regular periods in ``season`` (the freshness contract's ceiling)."""


@dataclass(frozen=True)
class TargetSpec:
    """What the model predicts and how the number is derived from raw columns."""

    column: str
    units: str
    format: str
    formats: tuple[str, ...]
    required_columns: tuple[str, ...]
    derive: Callable[[pd.DataFrame, str], pd.Series]
    """``derive(df, fmt) -> Series`` of the target under a scoring format, by explicit rules."""
    reconcile: Callable[[pd.DataFrame], pd.DataFrame]
    """Rows where the rules disagree with the source's own published value; empty = ok."""


@runtime_checkable
class FeatureModule(Protocol):
    """The one feature builder. Version it; never write a second one."""

    FEATURE_VERSION: str
    KEY_COLUMNS: tuple[str, ...]
    CONTEXT_COLUMNS: tuple[str, ...]
    HISTORY_FLAG: str
    TARGET_FLAG: str

    def all_feature_names(self) -> list[str]: ...

    def features_for_position(self, position: str) -> list[str]: ...

    def build_features(
        self, stats: pd.DataFrame, targets: pd.DataFrame | None = None
    ) -> pd.DataFrame: ...

    def training_frame(self, features: pd.DataFrame) -> pd.DataFrame: ...
