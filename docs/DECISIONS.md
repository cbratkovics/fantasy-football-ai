# Architecture decision records

Short, dated records of decisions that shape this repository. Newest last. ADR-0001 … ADR-0021
are sections of this file; from ADR-0022 on each record is a file under `docs/adr/` and is
indexed at the end of this file (ADR-0022).

## ADR-0001 — `main` is fast-forwarded to the production branch (2026-09-10)

**Context.** The live site (www.winmyleague.ai) was built from
`origin/codex/polish-ui-components-including-header-and-footer-mlj0ni`, nine commits ahead of
`main`, containing `backend/evaluation/decision_evaluator.py`, `tests/test_decision_evaluator.py`,
`analytics/sql/risk_strategy.sql`, `docs/PORTFOLIO_CASE_STUDY.md`, and the "Decision Lab"
frontend (see `AUDIT.md` §6). `main` and production had diverged.

**Decision.** `main` was fast-forwarded (`git merge --ff-only`) to that branch head (`403d7ca`).
No conflicts existed because the branch already contained `main`. From this commit on, `main`
matches what production was built from, and the rebuild proceeds from that state.

**Consequences.** The branch's README, dashboard, and evaluator are the starting point for the
slim rebuild. The other four `codex/*` remote branches are ancestors of this one and are left
untouched.

## ADR-0002 — Rescue the git-ignored production trainer (2026-09-10)

**Context.** The only real-data models were trained by
`backend/scripts/training/train_production_ml_models.py`, which `backend/.gitignore` excluded via
a blanket `scripts/` rule. The artifacts were committed; their producer was not.

**Decision.** The trainer is copied verbatim to `ffai/_legacy/train_production_ml_models.py` and
committed as the reference for the as-of feature scheme. The `scripts/` and `archive/` ignore
lines are removed from `backend/.gitignore`. The remaining untracked `backend/scripts/**` files
are **not** added: several embed a database credential and all of them are superseded by the
rebuild; they are deleted with the rest of `backend/` in Phase 4.

## ADR-0003 — Remove committed credentials, leave rotation to the owner (2026-09-10)

**Context.** `scripts/run/start_backend.sh` and `scripts/run_with_env.sh` contained a Supabase
Postgres connection string with a password (committed 2025-07-31). Live-looking OpenAI and
Anthropic keys exist in the untracked `backend/.env.local`.

**Decision.** Both scripts are deleted from the working tree. History is **not** rewritten and no
credential is rotated by the rebuild; both actions require the owner's accounts and are listed in
`docs/REBUILD_REPORT.md`. `.env*` files stay ignored; `*.env.example` files with placeholder values
remain tracked.

## ADR-0004 — Slim rebuild instead of a refactor (2026-09-10)

**Context.** `AUDIT.md` found a backend that served constants and hash-seeded random numbers,
four incompatible feature vocabularies, leakage in every tracked training path, 52 legacy
launchers, no tests, no CI, and published metrics with no computational source.

**Decision.** Rebuild around two seeds — the git-ignored production trainer's lagged feature
scheme and the branch's decision evaluator — as a single package `ffai/` with committed,
versioned artifacts. Everything else was deleted rather than repaired.

**Consequences.** ~180 files removed; the repository now has one feature module, one trainer,
one evaluator, one server, and one weekly job. Old branches remain on the remote for reference.

## ADR-0005 — RandomForest champion, XGBoost challenger (2026-09-10)

**Context.** The legacy production models were RandomForests selected over XGBoost by
validation MAE. Re-running both candidates on the rebuilt nflverse data with the same
hyper-parameters gave the same outcome at every position (`docs/MODEL_CARD.md`).

**Decision.** Champion = lower validation-season MAE per position; the other candidate is
persisted as the challenger and shadow-scored weekly. Promotion is a deterministic rule
(`ffai/models/registry.py: should_promote`): the challenger must win each of the last four
scored weeks *and* the frozen test set.

**Consequences.** Both candidates ship in every model version (≈5 MB). No hyper-parameter
search was performed; that is future work and must be evaluated with the same artifact.

## ADR-0006 — No database, no cache, no queue (2026-09-10)

**Context.** Runtime budget is $0/month and the audit showed Postgres/Redis/Celery added failure
modes without adding served value.

**Decision.** The server reads committed JSON/pickle artifacts at startup. State changes happen
only through the weekly job committing new artifacts. History per player is served from the
predictions files (with actuals attached after each week) and the frozen test predictions.

**Consequences.** Every deploy is a git commit; the API is stateless and horizontally trivial;
per-request latency is a dictionary lookup. There is no user state and no write path.

## ADR-0007 — nflverse is the only data source (2026-09-10)

**Context.** The old code touched Sleeper, ESPN (authenticated), sportsdata.io, OpenWeather,
Open-Meteo, collegefootballdata, and scraped Pro-Football-Reference, mostly unreachably and
with unresolved licence questions.

**Decision.** Use nflverse releases through `nflreadpy` only (`docs/DATA_SOURCES.md`). Scoring
rules are reconciled against nflverse's own points on every row.

**Consequences.** No injury, weather, opponent, or market features in `asof_v1`; the model card
lists them as absent. Adding a source requires a licence note in `DATA_SOURCES.md` and a
contract check.

## ADR-0008 — Draft-tier inputs are prior-season aggregates only (2026-09-10)

**Context.** Preseason tiers are consumed before any current-season game is played; using
in-season data would be leakage by construction.

**Decision.** Tier inputs for season `S` are computed from season `S-1` rows only (PPR/game,
its dispersion, games played, opportunity share, age when rosters provide it). Component count
is chosen by BIC; tiers are evaluated honestly against realised season-`S` PPR/game and the
numbers are recorded in the tiers metadata whatever they turn out to be.

**Consequences.** Rookies and players without four prior games get no tier. The 2024 evaluation
shows moderate rank correlation and low within-band rates for some positions; that is reported,
not hidden.

## ADR-0009 — Owner's local env files are preserved, not deleted (2026-09-10)

**Context.** `backend/.env` and `backend/.env.local` (untracked) contained live-looking API keys.
Phase 4 deletes `backend/`.

**Decision.** The two files were moved to the repository root as `.env.backend` and
`.env.backend.local`, both matched by the existing `.env.*` ignore rule, so the owner does not
lose the only copy. The owner should rotate the keys and delete both files
(`docs/REBUILD_REPORT.md`).

## ADR-0010 — Drift HOLD is based on the median monitored PSI, not any single feature (2026-09-10)

**Context.** The rebuild brief specified "PSI > 0.25 on any top-10-importance feature → HOLD".
Measured on this data the rule is not usable: over 60 rolling four-week windows in 2021–2024
(15 end-weeks × 4 seasons, four positions each, week-of-season-matched training reference,
season-to-date and structural features excluded), **47 of 60 windows** had at least one monitored
feature above 0.25 — including windows drawn from the training seasons themselves. Per-position
samples are a few hundred rows (QB ≈ 145), ten-bin PSI is noisy at that size, and several
features carry point masses at zero. A single-week comparison is worse still (a single position
week is ≈ 40–150 rows).

**Decision.** The weekly job measures PSI on the last four completed weeks of *played* rows (the
population the training deciles describe) against the training deciles of the same
week-of-season bucket (stored per bucket in `metadata.json` → `drift_reference.buckets`).
`week_of_season`, `games_played_prior`, and every `*_season_avg` feature (which resets each
season) are excluded from monitoring. Status per position:

* **HOLD** — median PSI of the monitored features > 0.25 (broad shift), or ≥ 2 monitored features
  > 0.50 (severe shift such as a schema or units change);
* **WARN** — any monitored feature > 0.25, or median > 0.10;
* **OK** otherwise.

On the same 60-window scan this holds 2 windows (median rule) plus a small number of severe
cases, all in the QB cohort, and warns on most weeks — which is the honest description of a
noisy input stream.

**Consequences.** The job publishes on ordinary weeks and still stops on a genuine population
change. Every run log records the per-position median, the flagged features, and the severe
features so the threshold can be revisited with evidence. This departs from the brief's literal
rule and is flagged in `docs/REBUILD_REPORT.md`.

## ADR-0011 — Remove the 2025 week-1 predictions file (2026-09-11)

**Context.** `artifacts/predictions/2025/week_01.json` was produced during the rebuild by scoring
2025 week 1 from 2024 history as a serving demonstration. The 2025 season is now complete and the
frozen model has been evaluated on all of it out of sample
(`artifacts/eval/…-oos2025.json`, `models/<version>/oos_predictions_2025.csv`), which
supersedes a single pre-season week and carries actuals.

**Decision.** The file is deleted; the manifest's `predictions.latest` points at the first 2026
weekly artifact. Player history for 2025 is served from the out-of-sample predictions CSV.

**Consequences.** No prediction file exists for a season without a matching evaluation; the
`predictions/` tree holds only live-season weekly outputs of the job.

## ADR-0012 — Minimum players per GMM component (2026-09-11)

**Context.** BIC alone chose 10 components for 60 quarterbacks (2024 preseason tiers), giving
tiers of two or three players and a within-band rate of 29.6 %.

**Decision.** The BIC search is capped at `n_components <= n_players // 8` (constant
`MIN_PLAYERS_PER_COMPONENT`). Tiers are retrained alone (models untouched) into a new
`tiers_version`; each position is compared against the previous artifact on Spearman and
within-band, and a position that gets worse on both keeps its previous tiers (recorded as
`kept: "previous"` in `metadata.json -> comparison_to_previous`). The old artifact directory
stays committed.

**Consequences.** The before/after table is in the model card. The cap is a structural
constraint, not a tuned hyper-parameter; changing it requires the same comparison.

## ADR-0013 — dbt Core + dbt-duckdb warehouse, developed on a DuckDB file, deployed to MotherDuck (2026-09-14)

**Context.** The evaluation and decision analytics lived in Python (`ffai/eval`) and a SQL
sketch (`analytics/sql/risk_strategy.sql`). The portfolio targets analytics-engineering roles,
where dbt sources, tests, contracts, unit tests, exposures, and generated docs are the expected
vocabulary. Runtime must stay $0/month and serving must never need a secret.

**Decision.** A dbt project `dbt/` (`ffai_dbt`, dbt Core ≥ 1.9, adapter `dbt-duckdb`) with two
targets in a committed `dbt/profiles.yml`: `dev` → `.duckdb/ffai_dev.duckdb` (git-ignored);
`prod` → `md:ffai` on MotherDuck, token read from `MOTHERDUCK_TOKEN` via `env_var`. Four
threads. The warehouse of record is MotherDuck (free tier: 10 GB, 10 compute-hours/month); the
weekly job builds it once a week and CI builds only `dev`. Every dbt command runs from the
repository root so the file-based sources resolve by relative path (Makefile `dbt-*` targets,
CI, and the weekly wrapper all do this).

**Consequences.** Local development needs no account; the prod target needs one environment
variable. The API does not use dbt or MotherDuck at all (ADR-0018). Compute usage is one build
per week, well inside the free tier.

## ADR-0014 — Medallion layers as schemas; sources are the repository's own files, copied into bronze tables (2026-09-14)

**Context.** dbt-duckdb can query files in place through `external_location`. If silver and gold
read files directly, MotherDuck would hold no copy of the inputs and lineage would stop at a
glob.

**Decision.** Schemas are exactly `bronze`, `silver`, `gold` (custom `generate_schema_name`,
no `main_` prefix), matching the model folders and tags. Sources are declared once in
`models/bronze/_sources.yml` over the nflverse parquet cache (`data/cache/`), the prediction
files, the frozen-test and out-of-sample CSVs, the evaluation JSONs, the rolling files, the
tiers JSONs, the manifest, and the model metadata. Bronze models (`brz_*`) materialise each
source as a **table**, typed, one-to-one with the file family; downstream layers never touch
files. `brz_player_stats` keeps only the newest cache snapshot (max filename = widest season
range at the latest load date). The JSON sources with optional keys (`actual` in weekly
prediction records, an empty `weeks` list in a new rolling file) are read with explicit column
types so a missing key is a NULL rather than a schema change. Because dbt-duckdb formats
`external_location` with Python string formatting, the sources use `formatter: template` so
DuckDB's `columns = {...}` specs survive; the stats path is a dbt var (`stats_path`) the
weekly wrapper and CI override.

**Consequences.** MotherDuck holds a copy of every input the marts depend on; `dbt docs`
lineage starts at a named source with a description and freshness. `dbt source freshness` on
the stats maps the newest `(season, week)` to an approximate game date (warn after 10 days);
the HOLD-grade freshness contract is a var-driven test (ADR-0015).

## ADR-0015 — Data contracts are dbt tests on the silver layer; Python keeps a thin wrapper (2026-09-14)

**Context.** `ffai/data/contracts.py` implemented nine checks in pandas. dbt has first-class
equivalents (generic tests, packages, singular tests, unit tests) and running them in the
warehouse means the same tests guard MotherDuck and the local build. The weekly policy must
still HOLD on a failure.

**Decision.** Each Python check has one dbt test; the Python module now only runs
`dbt build --select +tag:silver --indirect-selection cautious` with the run's parameters as
dbt vars (`stats_path`, `target_season`/`target_week`, `expected_season`/`expected_week`,
`prior_row_count`) and maps `target/run_results.json` into the same report shape
(`{"ok", "checks", "summary"}`) the policy consumed before. Nothing was lost; two checks got
stricter because the data allows it:

| Python check (deleted) | dbt test (now) |
|---|---|
| `required_columns` | `dbt_expectations.expect_table_columns_to_contain_set` on `slv_player_stats` (a missing column also fails the bronze cast at build time, reported as `model:brz_player_stats`) |
| `grain_unique_player_season_week` | `dbt_utils.unique_combination_of_columns(player_id, season, week)` |
| `value_ranges` (season ≥ 2019, 1 ≤ week ≤ 22) | `dbt_expectations.expect_column_values_to_be_between` on `season` and `week` |
| `non_negative_counting_stats` (16 columns) | `expect_column_values_to_be_between(min_value: 0)` per column |
| `positions_in_scope` | `accepted_values` on `position` |
| `null_rates` (≤ 2 % on `fantasy_points_ppr`, `position`, `team`) | `not_null` on the same three columns — the 2019–2025 pull has zero nulls, so the tolerance became exact |
| `freshness` (latest ≥ expected (season, week)) | singular `assert_stats_fresh_through_expected_week` driven by vars; `dbt source freshness` (date-mapped, warn-only) is declared separately |
| `newest_week_row_count` (absolute 200–600 band) | custom generic `row_count_within_pct_of_prior_period(period_columns=[season, week], tolerance_pct=0.35)` — relative, so it also holds on the CI fixture; the largest real consecutive-week swing is 23 % |
| `row_count_monotonic` (rows never shrink vs the prior run) | singular `assert_stats_row_count_not_below_prior_run` driven by `prior_row_count` |
| — (was `tests/test_scoring_reconciliation.py` only) | singular `assert_scoring_rules_reconcile_to_nflverse` (max abs diff ≤ 0.01 on every row) + unit test `scoring_rules_standard_half_ppr` |

The weekly job drops the target week's partial rows through the `target_season`/`target_week`
vars (same rule as before). Drift stays in Python (`ffai/eval/drift.py`): it is a statistical
monitor, not a contract. `tests/test_contracts.py` tests the mapping on a fixture
`run_results.json` (fail/error → HOLD reasons; warn/skipped → not failures).

**Consequences.** The contracts run needs dbt in the weekly environment (added to
`requirements-train.txt`; the API image is unchanged). A contract failure appears in the run log
as the dbt test name plus its failing-row count; the freshness reason still names the expected
season and week. The scoring rules now exist twice by design — `ffai/scoring.py` for the
trainer and `macros/fantasy_points.sql` for the warehouse — and both are reconciled row-for-row
against nflverse, which is the guard against drift between them.

## ADR-0016 — Gold marts with enforced contracts, and a reconciliation test against the evaluation artifacts (2026-09-14)

**Context.** The published numbers live in `artifacts/eval/*.json` and must stay the source of
truth (claim discipline). A warehouse that recomputes them is only trustworthy if it is forced to
agree with them.

**Decision.** Gold has seven contracted models (`contract: enforced`, DuckDB-enforced
`not_null` / composite `primary_key` / `check` constraints, every column typed and described):
`dim_player`, `dim_model_version`, `fct_player_week`, `fct_weekly_eval`,
`fct_player_decisions`, `fct_decision_policy` (ADR-0017), `fct_tier_outcomes`. Grain
interpretations that differ from the brief's shorthand:

* `fct_player_week` is one row per (player, season, week, model_version, **candidate**): a model
  version ships both candidates and the artifacts record both. "Per scoring format" is met by
  realised points under all three formats (`actual`, `actual_half`, `actual_standard`, rules
  macro) and by derived `prediction_half` / `prediction_standard` where the weekly file recorded
  a reception estimate; the frozen-test and out-of-sample CSVs carry no estimate, so the
  prediction row stays at the model target (`target_scoring_format = 'ppr'`) rather than
  fabricating other formats.
* `actual` comes from the warehouse stats when present, else from the value the artifact
  recorded (`actual_source`); a singular test asserts the two agree wherever both exist. This is
  what lets CI build gold from the 40-player fixture and still reconcile.
* `baseline` reproduces the evaluator's causal trailing mean in SQL (seed = stats strictly before
  the window; running = evaluated rows of the same window and candidate from earlier periods;
  position fallback; prediction fallback), with `fsum` so exact-rational means and DuckDB
  doubles agree.
* `fct_weekly_eval` adds `eval_window` and a `cohort` column ('ALL' plus positions) to the
  per-season-week-model grain so cohorts are pre-aggregated and the reconciliation can join.

The reconciliation is a singular test, `assert_marts_reconcile_to_eval_artifacts`: for every
committed artifact and cohort, the n-weighted aggregate of `fct_weekly_eval` over the window
must match the published n exactly and MAE / within-3 / within-5 to 1e-4 (the artifacts are
rounded to 4 dp). It passes with max |Δ| = 3.1e-5 on MAE and ≤ 4.3e-5 on the rates. A second
test reconciles the baseline MAE to 1e-4 and the baseline within-3 rate to within three rows,
because an error of exactly 3.0 is decided by floating-point rounding (Python's exact-fraction
mean vs `fsum(x)/n`): one WR row of 2,391 differs on local DuckDB and two on MotherDuck, whose
summation order differs. That test needs the full stats history and is
disabled under `full_stats: false` (CI).

**Consequences.** A gold build that disagrees with a published artifact fails. The marts never
replace the artifacts: `/performance` still serves the JSON verbatim, and the model card is
unchanged. Unit tests (`weekly_eval_mae_and_within_k`) pin the metric definitions on fixed rows.

## ADR-0017 — Decision marts replace the risk-strategy SQL sketch; replacement level is a rank (2026-09-14)

**Context.** `analytics/sql/risk_strategy.sql` sketched a policy-metrics mart over decision and
outcome facts that never existed (`fct_player_decisions`, `fct_player_outcomes`, league and
lineup-slot columns). It was kept as a design note.

**Decision.** The sketch is deleted and ported to two gold models. `fct_player_decisions` applies
the floor policy to every scored player-week: `recommend` when the prediction floor clears the
`min_floor` var (default 6.0), else `review`; `replacement_level_points` is the prediction of the
rank-k player at the position that week (`replacement_rank` var: QB 12, RB 24, WR 24, TE 12 —
twelve-team starters, no flex), `best_eligible_points` is the best realised score at the
position that week (the sketch's per-slot maximum), and `regret`, `hit`, `downside` follow the
sketch's definitions. `fct_decision_policy` is the sketch's aggregate (counts, rates,
recommendation MAE, mean regret, hit and downside rates) swept over `min_floor_grid`
(0–14 by 2) so the API can answer `?min_floor=` without a rebuild. Weeks without actuals keep
null outcomes.

**Consequences.** The Decisions panel on the site and `/marts/decisions` read
`fct_decision_policy`; the per-player rows stay in the warehouse and the parquet export. The
league / lineup-slot dimensions of the sketch are not modelled (no league data exists).
Replacement level is a modelling choice recorded here, not a published metric.

## ADR-0018 — Serving reads exported gold parquet in-process; no MotherDuck token on the Space (2026-09-14)

**Context.** The warehouse of record is MotherDuck, but the API runs on a public Hugging Face
Space with no secrets, must cost $0, and must keep serving if MotherDuck is unreachable. The
published numbers must stay the evaluation artifacts.

**Decision.** After `dbt build --target prod`, the weekly job runs
`dbt run-operation export_gold`, which `COPY`s every gold table to `artifacts/marts/<model>.parquet`
(zstd) plus `_export_manifest.json` (row counts, export time, target, dbt invocation id, git
commit). Those files are committed with the other artifacts and mirrored to the Space like
everything else. The API opens them at startup with an in-memory DuckDB connection
(`ffai/serve/marts.py`; `duckdb` added to `requirements-api.txt`) and serves three routes:
`/marts/weekly_eval` (`fct_weekly_eval`), `/marts/player_week/{player_id}`
(`fct_player_week`, champion candidate per position by default), and
`/marts/decisions?min_floor=` (`fct_decision_policy`, one value of the exported grid, with
count-weighted roll-ups computed in SQL). Every mart response carries
`source: "gold marts exported by the weekly build"` and the export provenance. `/performance`
and `/performance/{eval_id}` are unchanged and still return the artifacts verbatim; the site
labels the marts panel accordingly. If `artifacts/marts/` is absent the mart routes answer 404
and nothing else changes. A `TestClient` test restates the reconciliation at the API level (the
n-weighted MAE / within-3 over a window's mart rows equals the artifact to 1e-4).

**Consequences.** No MotherDuck token exists on the Space or in the image. The marts are
~1.3 MB per week of git history. The frontend's player history chart and the new Decisions
panel read the marts; the evaluation dashboard still reads the artifacts.

## ADR-0019 — dbt docs on GitHub Pages from a CI artifact, not a branch or a committed folder (2026-09-14)

**Context.** The brief offered two publishing routes: a `gh-pages` branch or a committed
`docs/dbt/` folder. Both put ~3.6 MB of generated HTML into git on every change.

**Decision.** `ci.yml` runs `dbt docs generate --static` on the `dev` build and publishes the
single-file site with `actions/upload-pages-artifact` + `actions/deploy-pages` (Pages source
"GitHub Actions") on every push to `main`. No generated HTML is committed and no extra branch
exists. The catalog statistics on the site describe the CI build (fixture stats, committed
artifacts); model, column, test, and exposure documentation is identical to prod.

**Consequences.** The owner must set Settings → Pages → Source to "GitHub Actions" once. The
site URL is `https://cbratkovics.github.io/fantasy-football-ai/`. This departs from the two
listed options for the reasons above.

## ADR-0020 — CI builds the dev target from the committed fixture; prod builds happen only in the weekly job (2026-09-14)

**Context.** MotherDuck's free tier allows 10 compute-hours a month; the nflverse cache is
git-ignored; CI must stay offline and secret-free.

**Decision.** CI runs `dbt build --target dev` with `stats_path` pointing at
`tests/fixtures/player_stats_sample.csv` (the stats source reads CSV or parquet by extension)
and `full_stats: false`, which disables only the baseline reconciliation (it needs every
player's full history). Everything else — contracts, generic and singular tests, unit tests,
the artifact reconciliation — runs on the fixture plus the committed artifacts because
`fct_player_week.actual` falls back to the artifact-recorded value (ADR-0016). Prod builds run
only in `weekly.yml` (one per week). `sqlfluff` with the dbt templater lints the project in CI;
`RF01` and `AL09` are excluded because they misread DuckDB struct access (`a.metrics.n`) as
table references and self-aliases.

**Consequences.** CI cannot catch a stats-only regression that the fixture does not contain;
the weekly job can, and HOLDs. Compute on MotherDuck stays at minutes per month.

## ADR-0021 — Description coverage is enforced by a manifest/catalog script in CI, not a pre-commit hook (2026-09-14)

**Context.** Every bronze model shipped with undocumented columns and most silver columns had no
description; the docs site's Columns tab was empty for them. The brief offered `dbt-checkpoint`
hooks (`check-model-columns-have-desc`, `check-model-has-all-columns`) in a pre-commit config, or
a script over `target/manifest.json` run in CI.

**Decision.** `scripts/check_dbt_descriptions.py` reads `dbt/target/manifest.json` and
`dbt/target/catalog.json` after `dbt docs generate` and exits 1 if any model, source table,
exposure, or model column lacks a description, if a built column has no YAML entry, or if YAML
documents a column the model no longer has. The CI `dbt` job runs it right after docs generate.
No pre-commit framework is introduced: the repo has none today, `dbt-checkpoint` needs its own
hook environment and a parsed manifest at commit time, and the catalog comparison (columns that
exist versus columns that are documented) needs a built warehouse, which CI already has.

**Consequences.** A new column without a description fails CI, not the commit. The check
covers models, sources, and exposures; it does not require descriptions on tests or macros.

## Index of records under `docs/adr/`

- [ADR-0022](adr/0022-adr-files-under-docs-adr.md) — New decision records are files under `docs/adr/`, indexed here (2026-09-15)
- [ADR-0023](adr/0023-incremental-silver-stats-with-lookback.md) — `slv_player_stats` is incremental with a restatement lookback (2026-09-15)
- [ADR-0024](adr/0024-player-snapshot-and-asof-views.md) — `dim_player` history as an SCD2 snapshot with current and as-of views (2026-09-15)
- [ADR-0025](adr/0025-versioned-decisions-mart.md) — `fct_decision_policy` is versioned; v1 keeps its relation name, the API pins a version (2026-09-15)
- [ADR-0026](adr/0026-slim-ci-deferred-to-cached-main-dev-build.md) — Slim CI compares and defers to the cached `main` dev build, not a prod manifest (2026-09-15)
- [ADR-0027](adr/0027-single-constraints-file-for-the-dbt-toolchain.md) — One `constraints.txt` pins dbt-core, dbt-duckdb, and DuckDB for CI and the weekly job (2026-09-15)
- [ADR-0028](adr/0028-single-project-config-and-interfaces.md) — One project config, three named seams, and JSON Schemas for the artifacts (2026-09-15)
- [ADR-0029](adr/0029-reusable-stack-template-extracted.md) — The stack is extracted into a private copier template; this repository keeps its names (2026-09-15)
- [ADR-0030](adr/0030-tests-that-shell-out-to-dbt-own-their-dependencies.md) — A test that shells out to dbt installs its own packages; CI job boundaries are not a dependency manager (2026-09-15)
