# Template boundary audit

Read-only inventory of this repository, classified for extraction into the reusable
`ds-dbt-stack-template` (copier). Written 2026-09-14 against commit `9bd196a` before any
Phase B/C change. Nothing in this document changes behaviour; it decides what the template
parameterises, what it keeps as-is, and what stays fantasy-football-specific behind an interface.

Classification:

- **Generic** — reusable as-is (no domain word inside, or only in comments).
- **Parameterizable** — reusable once one or more template variables are substituted.
- **Domain** — fantasy-football specific; the template exposes an interface and a stub.

Baseline counts (from `dbt ls` and `pytest --collect-only` at this commit): 23 models
(9 bronze, 7 silver, 7 gold), 82 data tests (76 generic, 6 singular), 3 unit tests, 0 snapshots,
2 exposures, 9 source tables; 68 pytest tests. Tool versions: dbt-core 1.12.4, dbt-duckdb 1.11.0,
DuckDB via dbt-duckdb, Python 3.11.13, Node 23.9 locally (CI pins 20), copier 9.18.2 (via `uvx`).

## 1. Proposed template variables

| Variable | Meaning | Value in this repo | Used by |
|---|---|---|---|
| `project_slug` | repo / URL slug | `fantasy-football-ai` | README, badges, workflows, exposures, Pages URL |
| `project_name` | display name | `Win My League · Decision Lab` (site) / `ffai` (API title) | frontend `SITE.name`, API title, docs |
| `package_name` | Python package | `ffai` | package dir, imports, `pyproject`, Dockerfile, env-var prefix `FFAI_` |
| `dbt_project_name` | derived: `{{ package_name }}_dbt` | `ffai_dbt` | `dbt_project.yml`, `profiles.yml`, node ids in `contracts.py` |
| `domain_summary` | one sentence | "weekly PPR fantasy-points projections for QB/RB/WR/TE" | README, API description, layout metadata |
| `entity_name` / `entity_key` | the thing predicted | `player` / `player_id` | grain keys, dim name, API routes `/players/{id}`, `/marts/player_week`, frontend labels |
| `entity_display_column` | | `player_display_name` | dim, prediction records, tiers |
| `period_name` / `period_key_columns` | the time grain | `week` / `(season, week)` | `period_key = season*100+week`, file names `week_%02d.json`, rolling file per season, freshness vars |
| `season_name` | outer period | `season` | split config, eval `kind` suffix `oos<season>` |
| `cohort_name` / `cohorts` | the per-model grouping | `position` / `QB, RB, WR, TE` | one model per cohort, `accepted_values` tests, `cohort` column in eval marts, API query patterns, drift by cohort |
| `target_column` / `target_units` | | `fantasy_points_ppr` / `points` | asof `TARGET`, model metadata `target`, KPI labels |
| `within_k_thresholds` | tolerance-band metrics | `(3, 5)` | `within_3_rate`, `within_5_rate` in evaluator, marts, UI |
| `prediction_range` | sanity bounds | `(-10, 80)` | `slv_predictions` test |
| `source_name` / `source_library` | | `nflverse` / `nflreadpy` | loader, `data_library` in metadata, DATA_SOURCES.md |
| `min_season`, `train_seasons`, `val_season`, `test_season` | split design | 2019, 2019-2022, 2023, 2024 | config, dbt test `season >= 2019`, README |
| `motherduck_database` | | `ffai` | `profiles.yml` prod path `md:ffai` |
| `hf_space_name` | | `cbratkovics/fantasy-football-ai` | `weekly.yml`, `ci.yml` deploy-space, exposure URL `cbratkovics-fantasy-football-ai.hf.space` |
| `vercel_project_name` / `site_url` | | `fantasy-football-ai` / `https://www.winmyleague.ai` | CORS origins, exposure URL, overview links |
| `github_owner` | | `cbratkovics` | badges, Pages URL, exposures owner, repo URL |
| `owner_name` / `owner_email` | | Christopher Bratkovics / cbratkovics@gmail.com | dbt source `meta.owner`, exposures `owner` |
| `schedule_cron` / `schedule_note` | scheduled job | `0 10 * 9-12,1 2` / "Tuesdays in season" | `weekly.yml` |
| `python_version` | | 3.11 | CI, Dockerfile, `pyproject` |
| `node_version` | | 20 | CI |
| `include_frontend` | | true | frontend dir, CI job, exposure |
| `include_tiers` | GMM tier module | true here, template default false | `ffai/models/tiers.py`, tiers source/bronze/silver/gold, `/tiers` route, tiers UI |
| `candidates` | model candidates | `rf, xgb` | `accepted_values` tests, API query regex `^(rf|xgb)$`, artifact file names `{cohort}_{candidate}.pkl` |
| `drift_thresholds` | PSI warn/hold/severe | 0.10 / 0.25 / 0.50 (calibrated on this data, ADR-0010) | `drift.py`; the template must leave these as a calibration TODO, not carry them |
| `decision_policy` | floor threshold grid, replacement ranks | `min_floor 6`, grid `0..14`, ranks `QB 12 / RB 24 / WR 24 / TE 12` | dbt vars, API default, DecisionsPanel default |

## 2. Proposed interfaces (Domain items plug in here)

| Interface | Contract | Implemented today by |
|---|---|---|
| `SourceLoader` | `load_period_rows(seasons, *, refresh=False) -> DataFrame` with one row per `(entity_key, *period_key_columns)`, columns = `ID_COLUMNS + STAT_COLUMNS`, sorted by grain, parquet-cached under `CACHE_DIR/<name>_<first>-<last>_<date>.parquet`; `cache_path_for(name, seasons)`; `current_period(schedule_frame, today) -> (season, next_period)`; `LIBRARY` string for provenance. Must raise on missing columns. | `ffai/data/nflverse.py` |
| `TargetSpec` | `target_column`, `target_units`, `derive(df) -> Series` (rules-based target from raw columns), `reconcile(df) -> DataFrame` of disagreements with the source's own published target (empty = ok), optional `formats` (alternative targets derived by the *same* rules) | `ffai/scoring.py` (`score`, `reconcile`, `FORMATS`) |
| Features contract | `FEATURE_VERSION`; `KEY_COLUMNS`, `CONTEXT_COLUMNS`, `HISTORY_FLAG`, `TARGET_FLAG`; `all_feature_names()`; `features_for_cohort(cohort) -> list[str]`; `build_features(stats, targets=None) -> DataFrame` (keys + context + `target` + flags + features; every feature of row *t* from rows strictly before *t* within the entity); `training_frame(features)`. Leakage test recomputes N random rows from earlier periods only. | `ffai/features/asof.py` |
| Eval artifact schema | JSON Schema for `artifact_version 2.1`: `eval_id, kind, season, generated_at_utc, code_commit, input{path (repo-relative), sha256, n_rows}, model{version, feature_version, candidate}, split, metric_definitions, metrics, baseline, cohorts{cohort: metrics+baseline}, rolling_origin{folds…}, policy_sweep` | `ffai/eval/evaluator.py::build_artifact` (no schema file committed yet) |
| Manifest schema | JSON Schema for `manifest_version 1.0`: `champion/challenger{model_version, candidate}`, `feature_version`, `tiers_version?`, `eval_id`, `evaluations[]`, `data_through{season, period}`, `predictions{latest}`, `last_run{run_id, at_utc, action, reasons, season, week, stats_rows}`, `updated_at_utc` | `ffai/models/registry.py` (docstring only) |
| Model metadata schema | `model_version, feature_version, *_version, data_library, trained_at_utc, data_through, seasons, target, input_sha256, input_rows, code_commit, interval_method, candidates, positions{cohort: n_*, features, champion, challenger, candidates{…}, drift_reference}` | `ffai/models/train.py` |
| Drift-hold rule inputs | `drift_report(current_frame, reference_deciles, monitored) -> {status, psi, flagged, severe, median_monitored}`; thresholds as module constants with a documented calibration procedure (backtest every N-period window in the training seasons; a rule that holds normal windows is wrong) | `ffai/eval/drift.py`, ADR-0010 |
| Weekly policy | `decide(contract_ok, drift_status, promote_ok, …) -> (action, reasons)` pure; `should_promote(...)` pure | `ffai/pipeline/weekly.py`, `registry.should_promote` |
| Contracts wrapper | `run_silver_contracts(**dbt_vars) -> {"ok", "checks", "summary"}` over `dbt build --select +tag:silver`; `CHECK_NAME_HINTS` maps dbt test names to stable check names | `ffai/data/contracts.py` |
| Mart store | `MartStore(marts_dir)` opening `MART_TABLES` parquet as DuckDB views; query helpers return dicts | `ffai/serve/marts.py` |

## 3. Python package `ffai/`

| Module | Class | Notes / variables / interface |
|---|---|---|
| `ffai/__init__.py` | Parameterizable | version string; docstring names the sport |
| `ffai/config.py` | Parameterizable | `POSITIONS` → `cohorts`; `TARGET`; seasons split; `SEASON_TYPE="REG"`, `REGULAR_SEASON_WEEKS` (Domain: NFL calendar → `SourceLoader.periods_in_season(season)`); env prefix `FFAI_` |
| `ffai/scoring.py` | Domain | `TargetSpec` implementation (rules reproduce nflverse; formats standard/half/ppr) |
| `ffai/data/nflverse.py` | Domain | `SourceLoader` implementation; `ID_COLUMNS`, `STAT_COLUMNS`, `current_season_week` via schedules |
| `ffai/data/contracts.py` | Generic | dbt wrapper; `CHECK_NAME_HINTS` contains model name `slv_player_stats` and `position` → parameterize |
| `ffai/features/asof.py` | Domain mechanism is Generic | shift-inside-groupby / rolling / expanding pattern is generic; `BASE_STATS`, ratio features, per-position exclusion lists, `"fantasy_points_ppr_L1"` history flag are Domain → features contract |
| `ffai/models/train.py` | Parameterizable | one model per cohort; RF/XGB params; season split from config; artifact naming `{cohort}_{candidate}.pkl`; `"target": "fantasy_points_ppr (nflverse, regular season)"` literal |
| `ffai/models/tiers.py` | Domain (optional module) | GMM tiers; `_OPPORTUNITY` per position; behind `include_tiers` |
| `ffai/models/registry.py` | Generic | uses `POSITIONS` only in `load_pipelines` → cohorts |
| `ffai/eval/evaluator.py` | Generic | column names `player_id/season/week/position` → grain/entity/cohort variables; `within_3/5` thresholds; artifact schema |
| `ffai/eval/drift.py` | Generic | thresholds are calibrated constants (do not copy values); `week_bucket` size 4 |
| `ffai/eval/model_card.py` | Parameterizable | template structure generic; prose ("PPR", "QB/RB/WR/TE", "Not for betting", nflverse) Domain → moves to a content file |
| `ffai/eval/oos.py` | Generic | `POSITIONS` → cohorts |
| `ffai/pipeline/score.py` | Parameterizable | eligible-entity rule (rows in season or season-1); Domain: `receptions_estimate`, `scoring.derivation` text, `_reception_weight`; `PREDICTIONS_VERSION` |
| `ffai/pipeline/weekly.py` | Generic shape | Domain bits: `nflverse.load_schedules`, `regular_season_weeks`, `fantasy_points_ppr` rename, `DRIFT_EXCLUDED_FEATURES` from asof |
| `ffai/serve/app.py` | Parameterizable | `DEFAULT_ORIGINS` (winmyleague.ai, vercel app) → site vars; title/description; `Query(pattern="^(ALL\|QB\|RB\|WR\|TE)$")`, `^(rf\|xgb)$`, `/tiers/{position}` (optional), `/predictions` scoring param (Domain: formats); `POLICY_TEXT`/`REPLACEMENT_TEXT` literals |
| `ffai/serve/marts.py` | Generic | `MART_TABLES` list is a variable |
| `ffai/serve/schemas.py` | Parameterizable | `ScoringFormat` Literal (Domain), `receptions_estimate` (Domain), rest generic |
| `ffai/_legacy/` | Domain | excluded from the template |
| `scripts/*.py` | Parameterizable | `sys.path` bootstrap pattern generic; seasons/positions defaults from config |
| `scripts/check_dbt_descriptions.py` | Generic | |

## 4. dbt project `dbt/`

| Object | Class | Notes |
|---|---|---|
| `dbt_project.yml` | Parameterizable | name, vars (`stats_path` glob, `min_floor*`, `replacement_rank` per cohort, `full_stats`) |
| `profiles.yml` | Parameterizable | `md:{{ motherduck_database }}`; dev path |
| `packages.yml` | Generic | dbt_utils, dbt_expectations |
| `macros/export_gold.sql` | Generic | writes `<name>.parquet`; must switch to `alias` once a versioned model exists (Phase B) |
| `macros/generate_schema_name.sql` | Generic | |
| `macros/fantasy_points.sql` | Domain | SQL twin of `TargetSpec.derive` → template ships `target_rules.sql` stub |
| `macros/stat_columns.sql` | Domain | `stat_columns()`, `non_negative_stat_columns()` → generated from the loader contract |
| sources `player_stats` | Domain | column list, `season_type`, freshness `loaded_at_field` expression |
| sources `predictions_weekly` | Parameterizable | struct spec includes `receptions_estimate` (Domain column), `name/team/position` |
| sources `predictions_test`, `predictions_oos`, `eval_artifacts`, `eval_rolling`, `manifest`, `model_metadata` | Generic | file globs follow the artifact contract |
| sources `tiers` | Domain (optional) | |
| `brz_player_stats` | Domain | typed copy; newest snapshot by filename |
| `brz_predictions_*` | Parameterizable | grain columns, `position` |
| `brz_eval_artifacts`, `brz_eval_rolling`, `brz_manifest` | Generic | |
| `brz_model_metadata` | Parameterizable | `for pos in ['QB','RB','WR','TE']` loop and `champion_qb…` columns → cohorts |
| `brz_tiers` | Domain (optional) | |
| `slv_player_stats` | Domain | `season_type = 'REG'`, position filter, dedup by `fantasy_points_ppr`, scoring macro; `period_key = season*100+week` (Parameterizable) |
| `slv_actuals` | Domain | long by scoring format |
| `slv_predictions` | Parameterizable | family union + dedup rule (Generic); `receptions_estimate` |
| `slv_eval_metrics` | Parameterizable | cohort loop over positions |
| `slv_eval_folds`, `slv_rolling_weeks` | Generic | |
| `slv_tiers` | Domain (optional) | |
| `dim_player` | Parameterizable | entity dimension; sources (stats > predictions > tiers) |
| `dim_model_version` | Parameterizable | `champion_qb…n_features_te` columns per cohort |
| `fct_player_week` | Parameterizable | causal baseline in SQL (Generic, fsum); `prediction_half/standard`, `actual_half/standard`, `receptions_estimate`, `within_3/5` thresholds (Domain/variables) |
| `fct_weekly_eval` | Generic | `cohort` column |
| `fct_player_decisions`, `fct_decision_policy` | Parameterizable | floor policy is generic; `replacement_rank` per cohort; `min_floor` in target units |
| `fct_tier_outcomes` | Domain (optional) | |
| `tests/generic/row_count_within_pct_of_prior_period.sql` | Generic | |
| `tests/gold/assert_marts_reconcile_to_eval_artifacts.sql` | Generic | the artifact-reconciliation test |
| `tests/gold/assert_baseline_reconciles_to_eval_artifacts.sql` | Generic | `full_stats` gate; tolerance note is DuckDB/MotherDuck-specific, keep |
| `tests/gold/assert_recorded_actuals_match_stats.sql` | Generic | `scoring_format = 'ppr'` literal → target format var |
| `tests/silver/assert_scoring_rules_reconcile_to_nflverse.sql` | Domain | `TargetSpec.reconcile` twin |
| `tests/silver/assert_stats_fresh_through_expected_week.sql`, `assert_stats_row_count_not_below_prior_run.sql` | Generic | period naming |
| unit tests: `scoring_rules_standard_half_ppr` (Domain), `predictions_dedup_prefers_weekly_then_latest` (Generic), `weekly_eval_mae_and_within_k` (Generic) | | |
| `_exposures.yml` | Parameterizable | URLs, owner |
| `models/overview.md` | Parameterizable | counts and links are literals (must be regenerated per project) |
| `.sqlfluff`, `.sqlfluffignore` | Generic | |

## 5. Workflows

| File | Class | Variables |
|---|---|---|
| `.github/workflows/ci.yml` | Parameterizable | fixture path, `full_stats` var, Space name, node version, frontend dir, `NEXT_PUBLIC_API_URL` |
| `.github/workflows/weekly.yml` | Parameterizable | cron, `FFAI_DBT_TARGET`, Space name, `hf upload` excludes, issue labels, `docs/MODEL_CARD.md` path; the outcome-reading step parses `artifacts/runs/*.json` (Generic) |

## 6. Frontend `frontend-next/`

| Route / component | Class | Notes |
|---|---|---|
| `src/lib/api/client.ts`, `format.ts` | Generic | |
| `src/lib/constants.ts` | Parameterizable | `SITE` (name, support email, repo, author links), `ROUTES` (Domain routes tiers/draft/start-sit) |
| `src/lib/api/types.ts` | Parameterizable | `Position` union, `SCORING_FORMATS`, `receptions_estimate`; mart types Generic |
| `src/lib/api/players.ts`, `marts.ts`, `tiers.ts` | Parameterizable / Domain (tiers) | route paths use `players`, `player_week` |
| `/performance` + `PerformanceDashboard.tsx` | Parameterizable | literals: `COHORTS`, `METRIC_LABEL` (`Within ±3`), "fantasy points", "player-game rows", eyebrow copy |
| `DecisionsPanel.tsx` | Parameterizable | `DEFAULT_MIN_FLOOR = 6`, `SUMMARY_ORDER` positions, "player-weeks", "pts" |
| `RollingOriginChart.tsx`, `PlayerHistoryChart.tsx` | Generic | "week" axis label |
| `/player/[id]`, `PlayerProfile.tsx` | Parameterizable | entity naming |
| `/predictions`, `PredictionsBoard.tsx` | Domain | scoring formats, positions |
| `/tiers`, `/draft`, `/start-sit` + `TierBoard`, `TierChart`, `DraftSimulator`, `StartSitEngine` | Domain | not in the template (tiers behind `include_tiers` only for the API/dbt side) |
| `/`, `PortfolioHome.tsx` | Domain copy | template ships a generic landing page |
| `/about`, `/help`, `/learn`, `/how-it-works`, `/privacy`, `/terms` | Domain copy | one `content/` file in the template |
| `layout.tsx`, `Navigation.tsx`, `Footer.tsx` | Parameterizable | title, keywords, nav items |
| `ui/*` (Controls, PageShell, Prose, Provenance, States) | Generic | |
| `next.config.js`, `tailwind.config.ts`, `.eslintrc.json`, `Dockerfile` | Generic | |

## 7. Docs

| Doc | Class |
|---|---|
| `README.md` | Domain copy over a Parameterizable structure (hero, badges, results table with eval ids, architecture Mermaid, warehouse section, API table, run locally, limitations, provenance) |
| `docs/ARCHITECTURE.md` | Parameterizable (module table, artifact contract table) |
| `docs/DATA_SOURCES.md` | Domain |
| `docs/MODEL_CARD.md` | Generated (Generic generator, Domain prose) |
| `docs/DECISIONS.md` (ADR-0001…0021) | Domain history; the template ships `docs/adr/0000-template.md` |
| `docs/PORTFOLIO_CASE_STUDY.md`, `docs/REBUILD_REPORT.md`, `docs/COMMERCIAL_USE_COMPLIANCE.md`, `AUDIT.md` | Domain, not templated |
| `frontend-next/VERCEL_ENVIRONMENT_SETUP.md` | Parameterizable |

## 8. Hard-coded strings, paths, and assumptions that break on a second domain

Grouped by the variable that would replace them.

- **Cohorts `QB / RB / WR / TE`**: `ffai/config.py::POSITIONS`; `asof.features_for_position` exclusion lists; `train.py` loop and `positions` metadata key; `registry.load_pipelines`; `app.py` Query regexes (`^(ALL|QB|RB|WR|TE)$`, `^(QB|RB|WR|TE)$`); dbt `accepted_values` on 8 columns; `brz_model_metadata`, `brz_tiers`, `slv_eval_metrics` Jinja loops; `dim_model_version` `champion_*`/`n_features_*` columns; `dbt_project.yml` `replacement_rank`; frontend `POSITIONS`, `COHORTS`, `SUMMARY_ORDER`, `Position` type; model-card template `positions.QB…`.
- **Entity `player` / `player_id` / `player_display_name` / `team`**: every dbt model, `KEY_COLUMNS`, `CONTEXT_COLUMNS`, API routes `/players/{player_id}` and `/marts/player_week/{player_id}`, `dim_player`, frontend route `/player/[id]`, GSIS-id descriptions.
- **Period `season` / `week`**: `period_key = season * 100 + week` (silver + gold), `week_%02d.json`, `rolling_<season>.json`, `regular_season_weeks`, `SEASON_TYPE = "REG"`, dbt test `week between 1 and 22`, `season >= 2019`, `week_bucket` size 4, `DRIFT_WINDOW_WEEKS = 4`, freshness expression `make_timestamp(season, 9, 1…)`, cron months `9-12,1`.
- **Target `fantasy_points_ppr` and scoring formats**: `TARGET`, `HISTORY_FLAG` keyed on `fantasy_points_ppr_L1`, `scoring.py`, `fantasy_points` macro, `slv_actuals`, `fct_player_week.prediction_half/standard`, `score.py` reception derivation, `app.py::_reception_weight`, `ScoringFormat` in schemas/types, `receptions_estimate` in the predictions source struct.
- **Metric thresholds in target units**: `within_3_rate`, `within_5_rate` (evaluator, marts, YAML, UI labels "Within ±3"), `prediction between -10 and 80`, `min_floor` grid `0..14`, `min_floor = 6`, `DEFAULT_MIN_FLOOR = 6`.
- **Drift thresholds**: `WARN_PSI 0.10`, `HOLD_PSI 0.25`, `SEVERE_PSI 0.50`, `MIN_SEVERE_FEATURES 2`, `MIN_MONITORED_FOR_MEDIAN 3` (calibrated on nflverse backtests, ADR-0010; must not be copied as defaults).
- **Source**: `nflreadpy` / `nfl_data_py` fallback, `STAT_COLUMNS` (23), `load_schedules` for the current period, `data/cache/player_stats_*.parquet`, fixture `tests/fixtures/player_stats_sample.csv` (40 players), `data_library` string.
- **Candidates `rf` / `xgb`**: file names `{pos}_{cand}.pkl`, `accepted_values`, API regex `^(rf|xgb|all)$`, model-card table columns "RF val MAE / XGB val MAE".
- **Deployment names**: Space `cbratkovics/fantasy-football-ai`, `https://cbratkovics-fantasy-football-ai.hf.space`, `https://www.winmyleague.ai`, `https://fantasy-football-ai.vercel.app`, `https://cbratkovics.github.io/fantasy-football-ai/`, `support@winmyleague.ai`, GitHub owner `cbratkovics`, CORS `DEFAULT_ORIGINS`, exposures `owner`, `meta.owner`.
- **Names**: `ffai` (package, MotherDuck db, Docker image `ffai-api`, env prefix `FFAI_ARTIFACTS_DIR / FFAI_CACHE_DIR / FFAI_CORS_ORIGINS / FFAI_DBT_TARGET`, dbt project `ffai_dbt`, contracts node ids `test.ffai_dbt.…`), `frontend-next` directory, "Win My League", "Decision Lab".
- **Version literals**: `artifact_version "2.1"`, `manifest_version "1.0"`, `PREDICTIONS_VERSION "1.0"`, `FEATURE_VERSION "asof_v1"`, `TIER_FEATURE_VERSION`.
- **Counts in prose**: `overview.md` and README state "23 models · 82 data tests · 3 unit tests · 2 exposures", "68 tests", "9 runtime packages"; these must be regenerated, never copied.
- **Calendar**: NFL 17/18-week regular season, "Tuesdays 10:00 UTC", off-season Feb–Aug note, "GitHub disables schedules after 60 days".

## 9. Surprises and design gaps found during the audit

1. **ADRs live in `docs/DECISIONS.md`, not `docs/adr/`.** The brief asks for `docs/adr/NNNN-*.md`. Proposal: new ADRs (0022 onwards) are written as individual files under `docs/adr/`, and `docs/DECISIONS.md` gains an index line per new ADR so the existing links keep working. Recorded as the first new ADR.
2. **`--defer` cannot resolve prod relations from the dev target without a MotherDuck token.** The dev target is a local DuckDB file (catalog `ffai_dev`); prod relations render as `"ffai"."<schema>"."<model>"`. A deferred `ref()` to an unmodified upstream model would point at a catalog that does not exist in CI. dbt-duckdb supports `attach:` in the profile (`ATTACH 'md:ffai' (READ_ONLY)`), but that needs `MOTHERDUCK_TOKEN` in CI, consumes MotherDuck compute on every push, and is unavailable to fork PRs. Options for Checkpoint A (see the checkpoint report): defer to the last `main` **dev** build instead (manifest + `.duckdb` file restored from the CI cache, no secret, no MotherDuck usage), or attach MotherDuck read-only in CI with a read-scoped token. The prod manifest can still be committed for `state:modified+` *selection*; only the deferral target changes.
3. **Prod manifest size**: `dbt/target/manifest.json` is 1.72 MB (169 KB gzipped) with 943 macros (dbt_utils, dbt_expectations, dbt_date). Its `metadata` block (`invocation_id`, `generated_at`, `env`, …) changes on every run, so a raw copy would change every week even when the project did not. Proposal: commit it under `artifacts/dbt/prod_manifest/manifest.json` with the volatile `metadata` fields nulled by a small script, so the file only changes when models, macros, tests, or vars change. The Space mirror must exclude it.
4. **Versioned models and `export_gold`.** The macro writes `<node.name>.parquet`; both versions of a versioned model share `name`, so the macro must write `<node.alias>.parquet`. Keeping `alias: fct_decision_policy` on v1 keeps the served parquet file name, `_export_manifest.json` row, and API responses byte-for-byte identical; v2 exports as `fct_decision_policy_v2.parquet`.
5. **Snapshots and history.** `dim_player` is "latest-seen" attributes with no history; the columns that actually change week to week are `team`, `position` (rare), and `player_display_name` (rare renames). A `check`-strategy snapshot over those three columns captures history only from its first prod run onward; an "as of period" view can therefore be exact only for periods scored after the snapshot started, and must fall back to the current record before that (documented in the view description).
6. **Incremental candidate.** `slv_player_stats` (40,330 rows, the largest table) is the right incremental candidate: its transformation is partition-local to `(player_id, season, week)` so appending by `period_key` is safe. `fct_player_week` (23,932 rows) is *not* safe to make incremental: its causal baseline is a running window over the whole evaluation window. Bronze is a one-to-one copy of the newest snapshot file, whose name changes daily.
7. **Frontend labels**: `/performance` already reads metric *definitions* from the artifact, but the words "fantasy points", "player-game rows", "player-weeks", `Within ±3`, and the cohort list are literals in `PerformanceDashboard.tsx` and `DecisionsPanel.tsx` (Phase C moves them to a config module).
8. **Memory note verified false today**: the `dbt_packages/<pkg> 2` duplicate-directory quirk did not reproduce; `dbt ls` reports the expected 23/82/3/2 counts.
9. `copier` is not installed in the repo venv; `uvx copier` (9.18.2) works and is the zero-footprint way to run it.
