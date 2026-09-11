# Architecture decision records

Short, dated records of decisions that shape this repository. Newest last.

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
