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
`unique_combination_of_columns` test:

- `slv_player_stats` — (player_id, season, week)
- `slv_actuals` — (player_id, season, week, scoring_format)
- `slv_predictions` — (player_id, season, week, model_version, candidate)
- `slv_eval_metrics` — (eval_id, cohort)
- `slv_eval_folds` — (eval_id, season, week)
- `slv_rolling_weeks` — (season, week)
- `slv_tiers` — (tiers_version, position, player_id)

**Gold** (7 marts, contracts enforced) — what the API and the site read:

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
- `fct_tier_outcomes` — each preseason draft tier and rank next to the realised rank of the
  season that followed.

## How trust is established

Counts from `dbt ls` and the `dbt build` output of this project:

| Resource | Count |
| --- | --- |
| Models | 23 (9 bronze, 7 silver, 7 gold) |
| Source tables | 9 (one source, `repo_files`) |
| Data tests | 82 (76 generic, 6 singular) |
| Unit tests | 3 |
| Exposures | 2 |

The 76 generic tests: 24 `expect_column_values_to_be_between`, 23 `not_null`, 13
`accepted_values`, 7 `unique_combination_of_columns`, 4 `expression_is_true`, 2 `unique`,
1 `relationships`, 1 `expect_table_columns_to_contain_set`, and 1 custom
`row_count_within_pct_of_prior_period`. The 6 singular tests reconcile the warehouse to things
outside it: the scoring rules against nflverse's own points, recorded actuals against the stats,
stats freshness and row count against the previous run, the causal baseline against the
evaluation artifacts, and the marts against the evaluation artifacts.

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
