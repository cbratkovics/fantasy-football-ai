# Decision Lab — implementation report (2026-09-27)

Technical report for the change set on branch `lab/decision-lab`. It distinguishes what is
implemented, what was verified by running it, and what remains unverified. It contains no
personal notes and no local paths; the private review helper (`make lab-notes`) writes those
outside the repository.

## Repository state and discrepancies

* The external review referenced `94bb33b` ("weekly: 2026 wk03 PUBLISH"). The local `main`
  checkout was eight commits behind it (at `56ed5cb`, a strict ancestor; no divergence). The lab
  branch `lab/decision-lab` was created from `origin/main` = `94bb33b`; local `main` was not
  moved. The working tree was clean before the change.
* Every review observation checked against the checkout held: the Decisions panel selects a
  precomputed `min_floor` and season over `fct_decision_policy` v1; `fct_player_decisions.regret`
  uses the best scorer of the whole position pool and a rank-k projection replacement; weekly
  targets are players with recent stat rows; historical prediction sources have null reception
  estimates and the predictions endpoint passes the stored actual through under non-PPR formats;
  floors are position/candidate residual quantiles clipped at zero; `rolling_2026.json` shows
  baseline MAE 5.0275 vs champion 5.1046 (week 1) and 4.7397 vs 4.8282 (week 2); CI's fixture
  build sets `full_stats=false`; the API pins mart v1; the frontend had lint/build only.
* One useful example of "export revision ≠ containing commit" already existed:
  `artifacts/marts/_export_manifest.json` records `code_commit ca8b345`, while the commit that
  contains it is `94bb33b`. The lab manifest records `code_revision.produced_at` the same way and
  the bundle check neutralises that field when comparing.
* The committed `fct_player_week.parquet` has no baseline provenance column. The mart's
  champion-candidate aggregates reconcile exactly (n, MAE, baseline MAE, overall and per
  position) to both evaluation artifacts and to both rolling weeks, so provenance is derived by
  population rule plus reconciliation; a `baseline_source` column was added to the dbt model for
  future exports.

## What was built (paths)

Python (`ffai/decision_lab/`): `policy_spec.json` (versioned specification), `canonical.py`,
`policy.py` (reference), `metrics.py`, `receipts.py`, `contracts.py` (JSON Schemas + semantic
checks), `bundle.py` (fail-closed loader), `exporter.py`, `cases.py`, `replay.py` (CLI),
`golden.py` (shared fixtures). Scripts: `scripts/export_decision_lab.py`,
`scripts/lab_review_notes.py`. Bundle: `artifacts/decision_lab/` (manifest, 49 inputs snapshots,
40 outcome snapshots, `cases.json`, `_build.json`) mirrored to `frontend-next/public/decision-lab/`.

Frontend (`frontend-next/`): `src/lib/decision-lab/` (types, policy_spec mirror, sha256,
canonical, policy, metrics, validate, receipts, cases, bundle, storage, explain, spec_digest),
`src/app/decision-lab/page.tsx`, `src/components/decision-lab/*`, `src/content/decision-lab.ts`,
navigation and footer entries, `ROUTES.decisionLab`, vitest and Playwright configuration,
`package.json` scripts `test`, `test:watch`, `typecheck`, `test:e2e`.

dbt: `fct_player_week.baseline_source` (contracted, described, accepted-values and consistency
tests), exposure `decision_lab`, regenerated `frontend-next/src/data/dbt-showcase.json`.

Build/CI: Makefile targets `lab-build`, `lab-check`, `lab-verify`, `lab-test`, `lab-replay`,
`lab-golden`, `lab-e2e`, `lab-notes`; `ci.yml` (golden + bundle check in the python job;
typecheck, vitest, Playwright and the extended publication scan in the frontend job);
`weekly.yml` (lab export after the gold export with `continue-on-error`, public copy staged,
issue on failure).

Docs: `docs/DECISION_LAB.md`, ADR-0032/0033/0034 (+ index), README section and run block.

Tests: `tests/test_decision_lab_{policy,canonical,metrics,receipts,contracts,exporter,cases,bundle,replay}.py`,
`tests/test_lab_review_notes.py`, `tests/fixtures/decision_lab/golden.json`; frontend
`src/lib/decision-lab/__tests__/*.test.ts`, `src/components/decision-lab/__tests__/*.test.tsx`,
`e2e/decision-lab.spec.ts`.

## Architectural decisions

* Inputs and outcomes are separate documents with separate content identities; the inputs
  contract rejects any outcome-derived field, so the policy cannot see actuals.
* Decision identity = digest of the canonical inputs; events (action, decline, outcome) are
  appended with their own identities; a changed assumption is a child decision.
* Canonical JSON with one number rule in both languages (half-up at six decimals) so digests
  agree across Python and TypeScript; verified by the golden fixtures.
* The Python policy is the reference; the TypeScript mirror must reproduce every golden case
  (status, recommended id, ordered reasons with values, gates, baseline block, limitations,
  explanation text) and the receipt lifecycle byte-for-byte.
* Baseline provenance is derived, never fabricated: reconciled windows → `player_history`;
  unreconciled → flagged limitation; mismatch → `unknown` (REVIEW unless model-only exploration).
* The bundle lives twice (canonical + public copy) because the deployment root is `frontend-next`;
  equality is checked in CI and the weekly job stages the copy.
* Curated real cases are discovered deterministically (top-four projections per position and
  week, first occurrence of each pattern); sealed patterns are shown only after reveal and the
  documentation states that the static files are inspectable.

## Commands run and outcomes (local, this session)

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest tests -q` | all passed (461 collected; 360 lab-specific), exit 0 |
| `.venv/bin/python -m ruff check ffai tests scripts` · `black --check ffai tests scripts` | pass |
| `.venv/bin/python -m ffai.decision_lab.golden --check` | current |
| `scripts/export_decision_lab.py --out artifacts/decision_lab --public-copy frontend-next/public/decision-lab` | 49 snapshots (39 real, 10 synthetic), 13,626 input rows, 18 cases; every historical snapshot `matched`; weekly 2026 w01/w02 `matched` to `rolling_2026.json`; w03 `unavailable` (published, no outcomes) |
| `scripts/export_decision_lab.py --check …` · `--verify …` | check ok; verified, public copy identical |
| Replay round trip on `syn-ambiguity-floor-relaxation` (new → record SYN-A → attach outcomes → validate) | decision_id and result digest unchanged; regret 8.0; points vs baseline choice −8.0; tampering `result.status` → exit 1 |
| Fixture dbt build (isolated DuckDB, `full_stats=false`, `--full-refresh`) | PASS=128 WARN=0 ERROR=0 (131 nodes) with the new column and tests |
| `dbt docs generate --static` + `scripts/check_dbt_descriptions.py` | ok: every model, column (466), source, exposure described |
| `sqlfluff lint dbt/models dbt/tests dbt/macros` | pass |
| `scripts/generate_dbt_showcase.py` + `pytest tests/test_dbt_showcase.py tests/test_project_config.py` | regenerated; 8 passed |
| `scripts/check_publication.py` (tracked + new deployable paths incl. `frontend-next/public/decision-lab`) | passed |
| `npm test` (vitest) | 9 files, 150 tests passed (137 library/parity + 13 component), exit 0 |
| `npm run typecheck` · `npm run lint` · `npm run build` | no ESLint warnings; tsc clean; `/decision-lab` prerendered (37.6 kB route, 166 kB first load), exit 0 |
| `npm run test:e2e` (Playwright, chromium) | 5 passed (34.6 s): full flow incl. export → clear → import → reload, stale state, published-weekly availability, keyboard, 390×844 mobile; no non-local requests |

## Stable case ids

1. `syn-ambiguity-floor-relaxation` — REVIEW at gap 0.5 for floor 8 and floor 5; RECOMMEND SYN-A
   at gap 0.3 / floor 5 with the baseline preferring SYN-B; outcomes A=8, B=16.
2. `real-oos-2025-w01-rb-bca017` — 2025 week 1 RB, four highest projections (sealed pattern
   revealed after an action).
3. `real-oos-2025-w02-wr-efa4e0` — 2025 week 2 WR, four highest projections.

Others: `syn-exact-tie`, `syn-threshold-equality`, `syn-missing-and-negative-outcomes`,
`syn-slot-and-exclusion`, `syn-unknown-availability`, `syn-baseline-fallback`,
`syn-blocked-evidence`, `syn-one-option`, `syn-floor-missing`, `real-frozen-2024-w01-wr-043738`,
`real-frozen-2024-w01-qb-2b2f06`, `real-weekly-2026-w02-qb-6710a9`, `real-oos-2025-w02-qb-20d8b2`
(agreement), `real-oos-2025-w01-qb-95cef7` (ambiguity), `real-weekly-2026-w03-rb-1ed3fb`
(published snapshot, availability unresolved until assumed).

## Known blockers and unverified items

* GitHub Actions runs of `ci.yml` and `weekly.yml` were not executed (nothing was pushed).
* No production dbt build ran; `baseline_source` exists in the model and the fixture build, not
  yet in the committed parquet. The exporter's `mart_column` path is implemented but only the
  derivation path is exercised by real data.
* Full-history baseline reconciliation inside dbt (`full_stats=true`) was not run locally; the
  lab's Python reconciliation of the committed marts to the evaluation artifacts is the check
  that ran.
* Vercel deployment of the new route and the Space mirror were not exercised.
* The reveal sequence is educational; the outcome files are public.
* Curated real cases illustrate patterns; they are not a performance sample.

## Working tree

Modified: `.github/workflows/ci.yml`, `.github/workflows/weekly.yml`, `Makefile`, `README.md`,
`dbt/models/gold/_exposures.yml`, `dbt/models/gold/_gold.yml`, `dbt/models/gold/fct_player_week.sql`,
`docs/DECISIONS.md`, `frontend-next/package.json`, `frontend-next/package-lock.json`,
`frontend-next/tsconfig.json`, `frontend-next/src/lib/constants.ts`,
`frontend-next/src/components/layout/{Navigation,Footer}.tsx`, `frontend-next/src/data/dbt-showcase.json`.
Added: everything listed under "What was built". Nothing committed.
