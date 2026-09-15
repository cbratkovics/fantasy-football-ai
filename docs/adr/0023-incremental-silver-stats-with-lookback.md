# ADR-0023 — `slv_player_stats` is incremental with a restatement lookback (2026-09-15)

**Context.** The warehouse rebuilt every table on every run. `slv_player_stats` is the largest
table (40,330 rows in prod, growing ~600 rows a week) and the only one whose transformation is
local to its own grain: typing, the `(player_id, season, week)` dedup window, and the scoring
macro never look at another partition. `fct_player_week` cannot be incremental: its causal
baseline is a running window over an entire evaluation window. Bronze is a one-to-one copy of
the newest snapshot file, whose name changes with every pull, so "new rows" is undefined there.
nflverse also restates earlier weeks (stat corrections), so a pure `period > max(period)` filter
would silently miss corrections.

**Decision.** `slv_player_stats` is `materialized: incremental`, `incremental_strategy:
delete+insert`, `unique_key: [player_id, season, week]`, `on_schema_change: fail`. An
incremental run reprocesses every period whose `period_key` is at or above the smallest of the
newest `stats_lookback_periods` distinct periods already in the table (var, default 4, chosen
conservatively and not calibrated: the model header records what was measured and how to
calibrate it), plus
anything newer; delete+insert replaces those rows. Full-refresh policy: `--full-refresh` at the
start of a season, after any change to the model's SQL or columns, or after a restatement older
than the lookback. The weekly workflow exposes `full_refresh` as a dispatch input
(`FFAI_DBT_FULL_REFRESH=1` for the contracts wrapper, `--full-refresh` for the prod build). CI
always builds a fresh DuckDB file, so every CI build is a full refresh.

Proof: `tests/test_dbt_incremental.py` builds a scratch warehouse (`FFAI_DUCKDB_PATH`) from the
committed fixture and asserts (1) truncate-then-incremental equals a full refresh in row count,
frame equality, and a content checksum; (2) a restated stat inside the lookback is picked up and
the derived points move with it, equal to a full refresh over the restated file; (3) a
restatement older than the lookback is *not* picked up until `--full-refresh`. There is
deliberately no dbt unit test of the filter with `is_incremental` overridden to true: a unit test
whose `given` includes `this` needs the model to already exist in the target (dbt reads its
columns from the relation), so it errors on every fresh warehouse (CI's fallback build, a fresh
clone, the template smoke test). Verified on a scratch MotherDuck database on 2026-09-15; the
pytest module covers the same behaviour on real rows.

**Consequences.** Weekly prod runs rewrite ~4 weeks of stats instead of six seasons. The
`dbt build --select +tag:silver` contracts run and the full build both hit the incremental path
and are idempotent. The first prod run after this change finds the existing table and appends;
no manual migration. Corrections older than four weeks need the documented full refresh.
