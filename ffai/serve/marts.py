"""Gold marts for serving: parquet files exported by the weekly dbt build, queried in-process.

``dbt run-operation export_gold`` writes ``artifacts/marts/<model>.parquet`` and
``_export_manifest.json`` (row counts, export time, target, dbt invocation id, commit). The API
opens them with an in-memory DuckDB connection at startup — no MotherDuck token, no network
(ADR-0018). If the directory is absent the ``/marts/*`` routes answer 404 and everything else
works as before.

Query helpers return plain dicts so the route layer only shapes responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

MART_TABLES: tuple[str, ...] = (
    "dim_player",
    "dim_model_version",
    "fct_player_week",
    "fct_weekly_eval",
    "fct_player_decisions",
    "fct_decision_policy",
    "fct_tier_outcomes",
)
EXPORT_MANIFEST = "_export_manifest.json"


class MartStore:
    """In-process DuckDB views over the exported gold parquet files."""

    def __init__(self, marts_dir: Path):
        self.dir = Path(marts_dir)
        manifest_path = self.dir / EXPORT_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"no exported marts at {self.dir} (missing {EXPORT_MANIFEST})")
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.export: dict[str, Any] = {
            "exported_at_utc": entries[0]["exported_at_utc"] if entries else None,
            "target": entries[0]["target"] if entries else None,
            "invocation_id": entries[0]["invocation_id"] if entries else None,
            "code_commit": (entries[0].get("code_commit") or None) if entries else None,
            "row_counts": {e["model"]: int(e["row_count"]) for e in entries},
        }
        self.con = duckdb.connect(database=":memory:")
        self.tables: list[str] = []
        for name in MART_TABLES:
            path = self.dir / f"{name}.parquet"
            if path.exists():
                # DDL cannot take bound parameters; the path is ours, quoted for SQL.
                quoted = str(path).replace("'", "''")
                self.con.execute(f"create view {name} as select * from read_parquet('{quoted}')")
                self.tables.append(name)

    # -- helpers -------------------------------------------------------------------------
    def _rows(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        cur = self.con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def min_floors(self) -> list[float]:
        return [
            float(r["min_floor"])
            for r in self._rows("select distinct min_floor from fct_decision_policy order by 1")
        ]

    # -- queries -------------------------------------------------------------------------
    def weekly_eval(
        self,
        *,
        cohort: str = "ALL",
        season: int | None = None,
        candidate: str | None = None,
        model_version: str | None = None,
    ) -> list[dict[str, Any]]:
        where, params = ["cohort = ?"], [cohort]
        if season is not None:
            where.append("season = ?")
            params.append(season)
        if candidate is not None:
            where.append("candidate = ?")
            params.append(candidate)
        if model_version is not None:
            where.append("model_version = ?")
            params.append(model_version)
        return self._rows(
            "select * from fct_weekly_eval where "
            + " and ".join(where)
            + " order by period_key, model_version, candidate",
            params,
        )

    def player_week(
        self, player_id: str, *, candidate: str | dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        rows = self._rows(
            "select * from fct_player_week where player_id = ? order by period_key, model_version, candidate",
            [player_id],
        )
        if candidate is None:
            return rows
        if isinstance(candidate, str):
            return [r for r in rows if r["candidate"] == candidate]
        return [r for r in rows if r["candidate"] == candidate.get(r["position"])]

    def decisions(
        self,
        *,
        min_floor: float,
        season: int | None = None,
        position: str | None = None,
        candidate: str | None = None,
    ) -> dict[str, Any]:
        where, params = ["min_floor = ?"], [min_floor]
        if season is not None:
            where.append("season = ?")
            params.append(season)
        if position is not None:
            where.append("position = ?")
            params.append(position)
        if candidate is not None:
            where.append("candidate = ?")
            params.append(candidate)
        clause = " and ".join(where)
        rows = self._rows(
            f"select * from fct_decision_policy where {clause} order by period_key, position, candidate",
            params,
        )
        # Count-weighted roll-ups of the same rows, in SQL, so no number is invented client-side.
        summary_sql = f"""
            select
                {{group}} as cohort,
                sum(eligible_decisions) as eligible_decisions,
                sum(recommendations) as recommendations,
                sum(recommendations) * 1.0 / nullif(sum(eligible_decisions), 0) as recommendation_rate,
                sum(recommendation_mae * recommendations_with_outcome)
                    / nullif(sum(recommendations_with_outcome), 0) as recommendation_mae,
                sum(mean_regret * recommendations_with_outcome)
                    / nullif(sum(recommendations_with_outcome), 0) as mean_regret,
                sum(hit_rate * recommendations_with_outcome)
                    / nullif(sum(recommendations_with_outcome), 0) as hit_rate,
                sum(downside_rate * recommendations_with_outcome)
                    / nullif(sum(recommendations_with_outcome), 0) as downside_rate,
                sum(recommendations_with_outcome) as recommendations_with_outcome
            from fct_decision_policy where {clause}
            group by 1 order by 1
        """
        overall = self._rows(summary_sql.format(group="'ALL'"), params)
        by_position = self._rows(summary_sql.format(group="position"), params)
        return {"rows": rows, "summary": overall + by_position}


def load_marts(marts_dir: Path) -> MartStore | None:
    """Open the marts if they were exported; ``None`` (not an error) when they were not."""
    try:
        return MartStore(marts_dir)
    except FileNotFoundError:
        return None
