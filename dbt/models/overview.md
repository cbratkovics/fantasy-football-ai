{% docs __overview__ %}

# ffai_dbt — analytics warehouse for the fantasy-football projection project

This is the analytics layer of
[fantasy-football-ai](https://github.com/cbratkovics/fantasy-football-ai), a weekly
fantasy-points projection model with a committed evaluation trail. The warehouse is a
bronze / silver / gold medallion built with dbt Core and dbt-duckdb: DuckDB locally and in CI,
MotherDuck in production. GitHub Actions builds it once a week after the scoring job runs.
Every source is a file the repository already owns: the committed evaluation, prediction, tier,
manifest, and model-metadata artifacts under `artifacts/`, plus the nflverse weekly player-stats
cache the weekly job refreshes.

## Layers

**Bronze** (`brz_*`, 9 models) — typed one-to-one copies of the source files. Every row carries a
`source_file` column with the repo-relative path it was loaded from, so any number can be traced
back to a committed file.

**Silver** (`slv_*`, 7 models) — conformed, deduplicated, and grain-enforced. Each grain is a
`unique_combination_of_columns` test. `slv_player_stats` is the one incremental model: delete+insert
on its grain key with a lookback of `stats_lookback_periods` (default 4) so nflverse stat
corrections to recent weeks are reprocessed; `tests/test_dbt_incremental.py` proves an incremental
run and a full refresh produce identical rows, including a restated week (ADR-0023):

- `slv_player_stats` — (player_id, season, week)
- `slv_actuals` — (player_id, season, week, scoring_format)
- `slv_predictions` — (player_id, season, week, model_version, candidate)
- `slv_eval_metrics` — (eval_id, cohort)
- `slv_eval_folds` — (eval_id, season, week)
- `slv_rolling_weeks` — (season, week)
- `slv_tiers` — (tiers_version, position, player_id)

**Snapshot** (`snp_player`) — SCD Type 2 history of `dim_player`'s slowly changing attributes
(team, position, display name), captured on every build; each version records the data-through
period it was first seen at (ADR-0024).

**Gold** (9 models, contracts enforced) — what the API and the site read, plus the two
snapshot-backed views:

- `dim_player` — one row per player with the latest-seen name, team, and position.
- `dim_model_version` — one row per trained model version: training provenance plus the
  manifest's champion and challenger slots.
- `fct_player_week` — one row per (player, season, week, model_version, candidate): prediction,
  interval, realised points, error, and the causal trailing-mean baseline.
- `fct_weekly_eval` — MAE, within-3, and within-5 per evaluation window, week, model version,
  candidate, and cohort, computed in SQL from `fct_player_week`.
- `fct_player_decisions` — the floor policy applied to every scored player-week (recommend when
  the prediction floor clears the threshold, else review) with regret, hit, and downside outcomes.
- `fct_decision_policy` — the same policy swept over a grid of floor thresholds: recommendation
  rates, recommendation MAE, mean regret, hit rate, and downside rate per week and position.
  Versioned (ADR-0025): v1 is the served shape and keeps the plain relation name; v2 (latest) adds
  `recommendation_within_3_rate` and `recommendation_interval_coverage`. The API pins v1.
- `fct_tier_outcomes` — each preseason draft tier and rank next to the realised rank of the
  season that followed.
- `dim_player_current` / `dim_player_asof` (views, not exported) — the current snapshot version per
  player, and the version in force when each scored period was scored, with `is_exact_asof` saying
  whether history could answer or the current record was used as a fallback.

## How trust is established

Counts from `dbt ls` and the `dbt build` output of this project:

| Resource | Count |
| --- | --- |
| Models | 26 (9 bronze, 7 silver, 9 gold) |
| Snapshots | 1 |
| Source tables | 9 (one source, `repo_files`) |
| Data tests | 96 (89 generic, 7 singular) |
| Unit tests | 3 |
| Exposures | 2 |

The 89 generic tests: 28 `expect_column_values_to_be_between`, 27 `not_null`, 14
`accepted_values`, 8 `unique_combination_of_columns`, 4 `expression_is_true`, 3 `unique`,
2 `relationships`, 1 `equal_rowcount`, 1 `expect_table_columns_to_contain_set`, and 1 custom
`row_count_within_pct_of_prior_period`. The 7 singular tests reconcile the warehouse to things
outside it: the scoring rules against nflverse's own points, recorded actuals against the stats,
stats freshness and row count against the previous run, the causal baseline against the
evaluation artifacts, the marts against the evaluation artifacts, and the as-of dimension against
the snapshot's first capture.

- **Contracts.** Every gold mart has an enforced dbt contract (column names, types, and
  primary-key / not-null constraints), so a schema change fails the build instead of the API.
- **Source freshness.** The player-stats source declares a freshness rule derived from the
  newest (season, week) it holds; `dbt source freshness` warns when the pull is more than ten
  days behind.
- **Artifact reconciliation.** The singular test `assert_marts_reconcile_to_eval_artifacts`
  recomputes n, MAE, within-3, and within-5 from `fct_weekly_eval` for every committed
  evaluation artifact and fails the build if any of them differs from the published JSON by more
  than 1e-4. The warehouse cannot publish a number the evaluation artifacts do not already carry.
- **Unit tests** cover the scoring-rules macro under all three formats, the prediction
  deduplication rule, and the weekly MAE / within-k aggregation on fixed rows.
- **Versions.** The public decisions mart is versioned; a version is served for at least one
  season after its successor ships, and the API names the version it reads.
- **Slim CI.** Pull requests build only `state:modified+` against the last `main` dev build
  (manifest and DuckDB file from the Actions cache), deferring everything else to it; `main`
  itself always does a full dev build. Production builds happen once a week on MotherDuck.

## From model to mart to API

A prediction is written by the weekly job to `artifacts/predictions/<season>/week_<ww>.json` →
`brz_predictions_weekly` → `slv_predictions` (deduplicated across the frozen-test, out-of-sample,
and weekly families) → `fct_player_week` (joined to the realised points from `slv_actuals`, with
the causal baseline) → `fct_weekly_eval` and `fct_player_decisions` / `fct_decision_policy` →
`export_gold` writes each gold table to `artifacts/marts/<alias>.parquet` → the FastAPI service
opens those files in-process and serves `/marts/weekly_eval`, `/marts/player_week/{id}`, and
`/marts/decisions`. The evaluation numbers on the site come from `artifacts/eval/*.json` through
`/performance`; the reconciliation test is what lets the two paths coexist.

## How to navigate

Start at `fct_weekly_eval` (the evaluation numbers) or `fct_player_week` (one row per
prediction) and expand the lineage graph upstream to see which artifact files feed them. The
two exposures, `api` and `decision_lab_site`, show which marts the FastAPI service and the
Next.js site read.

## Links

- Repository: <https://github.com/cbratkovics/fantasy-football-ai>
- Live site: <https://fantasy-football-ai.vercel.app>
- API docs: <https://cbratkovics-fantasy-football-ai.hf.space/docs>
- Model card: <https://github.com/cbratkovics/fantasy-football-ai/blob/main/docs/MODEL_CARD.md>
- This docs site: <https://cbratkovics.github.io/fantasy-football-ai/>

{% enddocs %}
