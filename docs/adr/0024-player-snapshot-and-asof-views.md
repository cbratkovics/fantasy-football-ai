# ADR-0024 — `dim_player` history as an SCD2 snapshot with current and as-of views (2026-09-15)

**Context.** `dim_player` holds latest-seen attributes with no history. Team changes on a
trade, position on a role change, and display names on a rename; a mart that joins a 2025
prediction to today's team is quietly wrong. dbt snapshots are the standard answer, but the
snapshot only knows history from its first run onward, and the as-of question for a
*scored period* is asked in period terms (season, week), not wall-clock terms.

**Decision.** `dbt/snapshots/snp_player.sql`: `strategy: check` on `[player_display_name,
position, team]`, `unique_key: player_id`, `hard_deletes: ignore`, schema `snapshots`. Each row
also carries `as_of_period_key`, the warehouse's data-through period (`max(period_key)` of
`slv_player_stats`) when that version was first captured; it is not a check column, so it
records *when* each version appeared. Two gold views (contracted, `meta: {export: false}` so
`artifacts/marts` is unchanged): `dim_player_current` (the open version per player; tests:
unique/not-null `player_id`, `equal_rowcount` with `dim_player`, relationship to it) and
`dim_player_asof` (one row per `(player_id, period_key)` for every scored period). A version
captured with data through `K` is the state used to score periods `K+1 .. K'` where `K'` is
the next version's capture period. `is_exact_asof` is true when such a version exists and false
when the view fell back to the current record (periods before the first capture, or a player
captured after the period). The singular test `assert_asof_exact_after_first_capture` fails if
any period after the snapshot's first capture, for a player in the snapshot, is not exact.

**Consequences.** History accrues from the first prod build after this commit; every 2024 and
2025 row resolves to the current record with `is_exact_asof = false`, and every period after
the first capture resolves exactly (in the dev build: 6,524 exact, 5,747 fallback rows). The
API does not read the views yet; they exist so downstream marts can choose "now" or "then"
explicitly. The Space mirror and `_export_manifest.json` are unaffected.
