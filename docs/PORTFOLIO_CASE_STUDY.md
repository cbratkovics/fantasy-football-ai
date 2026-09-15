# Win My League: Forecasting and Analytics Architecture

## Purpose and implemented scope

Win My League produces weekly fantasy-football point forecasts and exposes the evidence needed
to interpret them. The repository contains two related systems with a deliberate boundary:

1. Python ingests nflverse data, builds time-aware features, trains and scores models, evaluates
   forecasts, and writes versioned artifacts.
2. dbt transforms repository-owned source files and artifacts into tested bronze, silver, and
   gold relations for analysis. It does not train models or compute features used for inference.

The product is an experimental, read-only analytics application. It is not a guarantee of player
performance, a transaction system, or evidence of operation at commercial scale.

## Architecture and data flow

```text
nflverse weekly player statistics
  -> Python contracts and strictly-as-of features
  -> per-position RandomForest and XGBoost candidates
  -> prediction, model-metadata, and evaluation artifacts
  -> FastAPI /predictions and /performance

repository data and artifacts
  -> dbt bronze copies
  -> dbt silver conformance and grain tests
  -> dbt gold evaluation and decision marts
  -> exported Parquet relations
  -> FastAPI /marts/*
```

`ffai/data/nflverse.py` is the source adapter. The feature builder in
`ffai/features/asof.py` uses only observations earlier than the row being predicted; training,
evaluation, and weekly scoring share that implementation. `ffai/models/train.py` fits a
RandomForest and XGBoost candidate for each supported position using temporal season splits and
records feature version, input hash, code commit, and data-through period in model metadata.
Residual quantiles from validation data define forecast intervals.

The weekly pipeline validates data, checks feature drift, scores the registered champion and
challenger, attaches newly available outcomes, and applies a deterministic publish/hold/promote
policy. A successful publication updates committed artifacts before dbt builds and exports its
analytical relations. See `docs/ARCHITECTURE.md` for module and artifact contracts.

## Transformation layers and grains

The dbt project reads nflverse cache files plus committed prediction, evaluation, tier, manifest,
and model-metadata artifacts. Its layers have separate responsibilities:

- **Bronze** models are typed copies of source file families and retain `source_file` provenance.
- **Silver** models reconcile schemas, deterministic source precedence, and explicit grains.
  Important grains include player × season × week for statistics; player × season × week ×
  model version × candidate for predictions; eval id × cohort for evaluation metrics; and tier
  version × position × player for tiers.
- **Gold** models provide contracted dimensions and facts including `fct_player_week`,
  `fct_weekly_eval`, `fct_player_decisions`, the versioned `fct_decision_policy`, and
  `fct_tier_outcomes`.

Gold contracts fix column names and types. Generic and singular tests enforce uniqueness,
accepted values, relationships, scoring reconciliation, freshness, and agreement between marts
and evaluation artifacts. Python model training remains upstream of these transformations.

### Corrections and captured history

`slv_player_stats` uses incremental delete-and-insert processing at its player-week grain. Each
run reprocesses the newest four distinct loaded periods plus new periods, allowing recent
nflverse corrections to replace existing rows. The four-period window is a conservative policy,
not a claim that older corrections cannot occur; older restatements require `--full-refresh`.
Downstream facts that depend on earlier observations remain full-table transformations.

The `snp_player` SCD2 snapshot records changes to player name, position, and team from the time
snapshotting begins. `dim_player_asof.is_exact_asof` distinguishes periods supported by an actual
capture from older periods that must fall back to the current record. The repository does not
claim historical dimension state before its first capture, and snapshot-backed views are not
currently exported to the serving layer.

## Evaluation and provenance

Evaluation artifacts under `artifacts/eval/` are the computational source for forecast metrics.
They contain metric definitions, cohort results, a causal trailing-mean baseline, rolling-origin
folds, input hashes, model and feature versions, and a code commit. The generated model card and
`/performance` read those artifacts rather than duplicating figures in prose or UI constants.

The frozen-test evaluation reflects the declared historical training, validation, and test
seasons. A separately typed out-of-sample-season artifact evaluates the same frozen model on a
later completed season. These are historical evaluations; they are not prospective weekly
performance. Rolling weekly artifacts accumulate only after outcomes become available and must
be interpreted using their recorded evaluation window and sample size.

dbt independently recomputes analytical metrics from prediction and outcome rows. Singular tests
reconcile aligned artifact and mart scopes within the declared tolerance. This provides two
inspectable computation paths without treating either path as evidence outside its recorded
population.

## Decision and serving contracts

FastAPI is stateless with respect to the analytics warehouse: it loads committed files at
startup and performs no MotherDuck query on a request.

- `/performance` serves full evaluation JSON artifacts registered by `artifacts/manifest.json`.
- `/marts/weekly_eval`, `/marts/player_week/{player_id}`, and `/marts/decisions` query Parquet
  relations exported by `dbt run-operation export_gold` and opened in an in-memory DuckDB
  connection. These routes return 404 when no export is present.
- `/predictions/*` and `/players/*` use committed prediction and model artifacts rather than dbt
  relations.

An exported relation is not automatically an API contract. The latest dbt decision-policy model
is v2, while `/marts/decisions` deliberately pins the backwards-compatible v1 relation. The API
reports that served mart version, and both relation aliases may coexist in an export. The floor
policy applies an implemented threshold to forecast intervals; it should not be confused with
the evaluator's optional `decision_score` sweep, for which current artifacts contain no scorer.

## Validation and reproducible evidence

Important decisions have executable checks:

- leakage tests perturb future rows and assert that earlier as-of features do not change;
- artifact schema, grain, scoring, evaluator, registry, drift, and weekly-policy tests run in the
  Python suite;
- an incremental dbt regression compares correction handling with a full refresh, including the
  documented older-correction boundary;
- dbt contracts, unit tests, and reconciliation tests run before documentation or production
  exports are published;
- API tests cover both artifact-backed endpoints and exported-mart behavior; and
- the frontend reads versions and metrics from API responses and is type-checked during build.

Reproduction commands and environment assumptions live in `README.md`. Counts and metric values
should be taken from the current test output and artifacts, not copied from this brief.

## Operational boundaries

- nflverse is the only statistics source. Licensing and attribution are documented in
  `docs/DATA_SOURCES.md` and `docs/COMMERCIAL_USE_COMPLIANCE.md`.
- The development warehouse is DuckDB; the scheduled workflow uses MotherDuck for transformation
  and then exports files. No database or credential is required on the API serving path.
- The model has no injury, weather, opponent, betting, roster, authentication, payment, or
  automated lineup-submission integration.
- Drift thresholds and the recent-correction lookback are explicit operating policies, not
  universal performance guarantees.
- A branch build, cached page, and deployed production alias can refer to different revisions.
  The artifact commit and export manifest are the authoritative revision metadata when present.

## Scoped technical next steps

1. Accumulate prospective weekly evaluation windows and report interval calibration only after
   enough outcome-bearing rows exist.
2. Calibrate the incremental restatement window from observed correction timing while preserving
   the documented full-refresh escape hatch.
3. Migrate the decisions endpoint to mart v2 only after its compatibility and reconciliation
   requirements are satisfied for a full deprecation window.
4. Use snapshot-backed player attributes in a downstream export only when consumers need them,
   retaining `is_exact_asof` so pre-capture fallback remains visible.
5. Add contextual features only with source licensing, as-of availability, contracts, and the
   same leakage checks as the current feature set.
