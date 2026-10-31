"""Resolve observed points without borrowing forecast estimates or fetching source data.

Legacy ``actual`` is PPR. Format-specific artifact evidence has precedence, then a complete
same-revision stat row, then same-revision realized receptions, then matching exported facts.
Every available conversion source must agree; duplicate mart rows are validated before reducing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from ffai.scoring import FORMATS, REQUIRED_COLUMNS, RULES_ID, ScoringFormat, score


@dataclass(frozen=True)
class ActualResolution:
    actual: float | None
    actual_scoring_format: ScoringFormat
    actual_source: str | None
    actual_null_reason: str | None = None

    def fields(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value) if math.isfinite(value) else None


def resolve_actual(
    record: Mapping[str, Any],
    fmt: ScoringFormat,
    *,
    outcomes: Sequence[Mapping[str, Any]] = (),
) -> ActualResolution:
    """Resolve one identified player-period; callers must select matching outcome provenance.

    ``outcomes`` must be facts for the same player, season, week and source family. The serving
    adapter enforces this join. Exported historical facts use this repository's fixed scoring
    rules. Unknown custom rules never qualify. Nullable stat components mean no opportunity;
    a missing row or column does not. Numeric zero and negative scores remain observations.
    """
    if fmt not in FORMATS:
        raise ValueError(f"unsupported actual scoring format {fmt!r}")

    def missing(reason: str) -> ActualResolution:
        return ActualResolution(None, fmt, None, reason)

    legacy = record.get("actual")
    ppr = _number(legacy)
    if legacy is not None and ppr is None:
        return missing("nonfinite_actual")
    rules = record.get("actual_rules_id", RULES_ID)
    if rules != RULES_ID:
        return missing("scoring_rules_mismatch")
    # Legacy PPR needs no optional exports, but supplied conflicting evidence fails closed.
    candidates: list[tuple[float, str]] = []
    if fmt == "ppr" and ppr is not None:
        candidates.append((ppr, "prediction_artifact_ppr"))
    explicit = record.get("actuals_by_format")
    if explicit is not None:
        if not isinstance(explicit, Mapping) or record.get("actual_rules_id") != RULES_ID:
            return missing("scoring_rules_mismatch")
        if explicit.get("ppr") is not None:
            same_ppr = _number(explicit["ppr"])
            if same_ppr is None:
                return missing("nonfinite_actual")
            if ppr is not None and abs(same_ppr - ppr) > 0.011:
                return missing("conflicting_outcome_sources")
            ppr = same_ppr
        if explicit.get(fmt) is not None:
            value = _number(explicit[fmt])
            if value is None:
                return missing("nonfinite_actual")
            candidates.append((value, "prediction_artifact_formats"))

    stats = record.get("actual_stats")
    if stats is not None:
        if not isinstance(stats, Mapping) or any(c not in stats for c in REQUIRED_COLUMNS):
            return missing("invalid_stat_components")
        if record.get("actual_rules_id") != RULES_ID:
            return missing("scoring_rules_mismatch")
        # A component snapshot must explicitly accompany the same outcome revision.
        if not record.get("actual_source_revision") or record.get(
            "actual_stats_revision"
        ) != record.get("actual_source_revision"):
            return missing("outcome_revision_mismatch")
        if any(v is not None and _number(v) is None for v in stats.values()):
            return missing("invalid_stat_components")
        frame = pd.DataFrame([{c: stats[c] for c in REQUIRED_COLUMNS}])
        calculated_ppr = float(score(frame, "ppr").iloc[0])
        if ppr is not None and abs(calculated_ppr - ppr) > 0.011:
            return missing("conflicting_outcome_sources")
        ppr = calculated_ppr
        candidates.append((float(score(frame, fmt).iloc[0]), "realized_stat_components"))

    if record.get("actual_receptions") is not None and ppr is not None:
        if not record.get("actual_source_revision") or record.get(
            "actual_receptions_revision"
        ) != record.get("actual_source_revision"):
            return missing("outcome_revision_mismatch")
        rec = _number(record["actual_receptions"])
        if rec is None or rec < 0 or rec != int(rec):
            return missing("invalid_stat_components")
        candidates.append(
            (
                ppr - {"ppr": 0, "half": 0.5, "standard": 1}[fmt] * rec,
                "realized_receptions_same_revision",
            )
        )

    # A model/candidate dimension may duplicate one outcome. Validate all rows, never take first.
    mart_values: list[float] = []
    mart_ppr: list[float] = []
    for row in outcomes:
        if row.get("actual_rules_id", RULES_ID) != RULES_ID:
            return missing("scoring_rules_mismatch")
        value = row.get("actual" if fmt == "ppr" else f"actual_{fmt}")
        observed = row.get("actual")
        if observed is not None:
            parsed = _number(observed)
            if parsed is None:
                return missing("nonfinite_actual")
            mart_ppr.append(parsed)
        if value is None:
            continue
        parsed = _number(value)
        if parsed is None:
            return missing("nonfinite_actual")
        if observed is None:
            return missing("conversion_evidence_missing")
        mart_values.append(parsed)
    if mart_ppr and max(mart_ppr) - min(mart_ppr) > 0.011:
        return missing("conflicting_mart_outcomes")
    if mart_values and max(mart_values) - min(mart_values) > 0.011:
        return missing("conflicting_mart_outcomes")
    if ppr is not None and mart_ppr and abs(ppr - mart_ppr[0]) > 0.011:
        return missing("conflicting_outcome_sources")
    if mart_values:
        candidates.append((mart_values[0], "exported_actuals_fact"))
    if not candidates:
        return missing("outcome_unavailable" if ppr is None else "conversion_evidence_missing")
    if max(v for v, _ in candidates) - min(v for v, _ in candidates) > 0.011:
        return missing("conflicting_outcome_sources")
    value, source = candidates[0]
    return ActualResolution(round(value, 2), fmt, source)
