# ADR-0028 — One project config, three named seams, and JSON Schemas for the artifacts (2026-09-15)

**Context.** The boundary audit (`docs/TEMPLATE_BOUNDARY.md`) found the project vocabulary
(entity, period, cohorts, target, tolerance bands, candidates) and the deployment names (Space,
site, Pages, MotherDuck database, schedule) repeated as literals across the package, the dbt
project, both workflows, and the frontend. The reusable template needs one place to substitute
them, and this repository must behave identically afterwards.

**Decision.** Three things, all behaviour-preserving:

1. `ffai/config.py` holds `PROJECT: ProjectConfig`, a frozen dataclass with every project-wide
   constant; the legacy names (`POSITIONS`, `TARGET`, `TRAIN_SEASONS`, …) are aliases of its
   fields so no caller changed. dbt and the site cannot import Python, so they carry *mirrors*
   and `tests/test_project_config.py` fails when a mirror drifts: `dbt_project.yml` `vars`
   (`entity_key`, `cohorts`, `candidates`, `target_scoring_format`, `within_k`) must equal
   `ffai.config.dbt_vars()`; `frontend-next/src/lib/project.config.json` must equal
   `ffai.config.frontend_config()` (regenerate with `python -m ffai.config --frontend`); the
   weekly cron must equal `PROJECT.schedule_cron`; the workflows read the Space name with
   `python -m ffai.config hf_space`; the API's CORS default comes from `cors_origins()`. The
   dbt cohort loops and the target scoring format read the vars (compiled SQL is unchanged).
2. `ffai/interfaces.py` names the seams: `SourceLoader` (implemented by
   `ffai.data.nflverse.LOADER`, delegating to the unchanged module functions), `TargetSpec`
   (`ffai.scoring.TARGET_SPEC`: column, units, formats, `derive`, `reconcile`), and
   `FeatureModule` (satisfied by `ffai.features.asof`). `tests/test_interfaces.py` asserts the
   implementations satisfy the protocols on the fixture.
3. `artifacts/schemas/*.schema.json` (JSON Schema 2020-12): the evaluation artifact, the
   manifest, model metadata, the scored-period file, and the drift report (with the hold rule's
   inputs and thresholds documented as calibration output, not defaults). Every committed
   artifact validates against its schema in the test suite.

On the site, `/performance`, the decisions panel, and the player history read entity, period,
cohort, metric, and unit labels from `lib/project.ts`; the remaining sport-specific sentences of
those pages live in one file, `src/content/performance.ts`. The other pages (landing, tiers,
draft, start/sit, help, learn, about) keep their domain copy: they are Domain in the boundary
audit and are not part of the template.

**Consequences.** Behaviour is proven unchanged four ways: the API contract tests pass
unchanged, `artifacts/` shows no diff apart from the new `schemas/` directory, the compiled
dbt SQL is identical before and after, and the gold marts exported from the same dev warehouse
before and after have identical content checksums. What could not be made generic without a
behaviour change stays Domain and is listed in the Checkpoint C report.
