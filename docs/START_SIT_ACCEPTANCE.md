# Start/Sit acceptance review

Verdict: **READY_FOR_REVIEW_WITH_EXTERNAL_CHECKS**. This permits human review, not merge
approval. The supported release is a conditional weekly comparator using the implemented PPR
rules. It is not a verified game-day service. The three incremental-dbt test bodies remain
unexecuted because dependency installation is denied by the current environment's network policy.

## Preserved work and comparison method

At review start the branch was `work`, full HEAD
`3ddbb36dbf67f33c769c45645957aa92b690b457`. There were 15 modified tracked paths, no staged
changes, and 36 untracked files from the implementation. The complete initial inventory and
tracked/staged binary diffs were recorded under `/tmp/ffai-acceptance/initial-*`. Existing work
was preserved. No reset, clean, stash, staging, commit, push, merge, deployment or production
workflow was performed.

As explicitly requested, a detached baseline worktree was created at
`/workspace/ffai-acceptance-baseline`, at that exact verified commit. It remains clean. Both
checkouts used `/workspace/fantasy-football-ai/.venv/bin/python` with checkout-specific cwd and
`PYTHONPATH`. Before tests, `ffai.__file__`, `ffai.config.__file__`, HEAD, relevant variable
presence and installed versions were recorded; baseline imports resolved into the baseline,
not the editable modified checkout. Python was 3.11.17; pytest 9.2.0; pandas 3.1.0; numpy 2.4.6;
DuckDB 1.5.5; dbt-core 1.12.4; dbt-duckdb 1.11.0. Fixture CSV, mart and constraints bytes matched.
No FFAI artifact/test-data/cache/warehouse overrides were present in the comparison commands.

Browser comparison used Node 24.19.0, Next 14.2.25, Playwright 1.63.0 and system Chromium 154.
Baseline `npm ci` used the same unchanged lockfile. The original spec, cases, manifest and served
case bytes matched. Both servers were run sequentially at port 3100. The failed Lab flow uses
same-origin artifacts, not the deployed API. Both runs used the same host clock; source periods
and policy computation are artifact-bound, not implicit current-time selections. The explicit
new integration suite fixes browser time to its clearly synthetic 2099 test period.

Raw logs, traces, screenshots, environment manifests and live payloads remain outside tracked
publication paths. No new live ingestion was needed for this review. Detailed comparison evidence
is `/tmp/ffai-acceptance/python-comparison.md` and `browser-comparison.md`.

## Findings, ordered by severity

All application findings below were fixed and have regressions that failed before the fixes.
No unresolved application merge blocker was identified within the conditional-comparator scope.

| Severity | Finding and release relevance | Minimal correction and reference |
|---|---|---|
| High | Malformed context became “missing,” allowing manual assumptions to remove a corruption gate. Introduced by the optional companion implementation. | Preserve incompatible state until a valid replacement succeeds; `frontend-next/src/components/start-sit/ContextEvidence.tsx:31`. |
| High | A failed refresh discarded a previously known bye/out/passed kickoff, reopening the dated real-action surface. Introduced by companion refresh handling. | Retain last valid facts and original times; failures never refresh verification; `ContextEvidence.tsx:25`. Requests are sequenced so old responses cannot win. |
| High | Weekly snapshot rows could silently encode assumed/hindsight availability even though the quick UI displayed unknown. The broad v1 snapshot contract permits those values, but this new weekly boundary must not. | Reject non-unknown defaults in the quick loader and direct adapter; only explicit user overrides establish assumptions; `frontend-next/src/lib/start-sit/adapter.ts:16`, `:44`, `:68`. Lab/policy contracts are unchanged. |
| Medium | A new snapshot retained the previous week's user assumptions. Result invalidation alone did not make the assumptions period-specific. | Clear overrides/historical-mode intent on changed content digest, retain nominations, preserve same-evidence assumptions; `frontend-next/src/components/start-sit/StartSitEngine.tsx:54`. |
| Medium | `/players` returned a weekly numeric actual that `/predictions` withheld for a conflicting outcome revision. This was an inconsistency with the newly introduced resolver. | Reuse the PPR resolver and matching outcomes for history, add format/source/null-reason metadata; `ffai/serve/app.py:268`, `:295`, `ffai/serve/schemas.py:80`. Frozen values remain unchanged. |
| Medium | JS date normalization accepted impossible calendar dates; future observation/fetch metadata could appear usable. Malformed source extensions/identity primitives could reach rendering or API serving. | Strict aware calendar validation, explicit as-of ordering, closed source membership and identity validation; `frontend-next/src/lib/start-sit/readiness.ts:50`, `:111`, `:149`, `:159`; `ffai/context/snapshot.py:438`. |
| Medium | Inherited exporter and weekly-browser tests no longer exercised the published evidence correctly. Inheritance did not make these acceptance failures harmless. | Repair expected source-period/provenance assertions and manifest-linked case selection, preserving mismatch refusal and behavioral assertions; `tests/test_decision_lab_exporter.py:39`, `:93`, `:256`; `frontend-next/e2e/decision-lab.spec.ts:155`. |

## Baseline versus modified failures

The three exporter and missing-case failures are called inherited only because they were actually
reproduced on the clean baseline. The dbt entries are setup errors, not failed assertions or passes.

| Exact node | Clean baseline | Modified before review fixes | Final modified | Signature, origin and recommended action |
|---|---|---|---|---|
| `tests/test_decision_lab_exporter.py::test_reconciliation_statuses_on_committed_evidence` | FAIL | Same FAIL | PASS | Expected39, actual44; later assertions also assumed week3-era provenance/scored weeks. Reproduced on baseline. Fixed exact period set/latest publication checks from raw source facts and source manifest, without rewriting evidence. |
| `tests/test_decision_lab_exporter.py::test_champion_at_source_and_row_provenance` | FAIL | Same FAIL | PASS | Expected only player_history, but source facts include position_history (408 rows). Reproduced on baseline. Now compare every exported branch to the raw mart, retaining independent-baseline checks. |
| `tests/test_decision_lab_exporter.py::test_mismatched_mart_yields_unknown_baseline` | FAIL | Same FAIL | PASS | Expected3 weekly snapshots, actual8; intended legacy fixture accidentally inherited modern explicit baseline_source. Reproduced on baseline. Exact weekly set checked; the synthetic legacy fixture explicitly omits the new column so original mismatch/null/refusal assertions still run. |
| `e2e/decision-lab.spec.ts:133` (original), now `:155`, “published weekly case: review on availability until every alternative is assumed available” | FAIL | Same FAIL | PASS | 15s timeout locating absent `case-real-weekly-2026-w03-rb-1ed3fb`; unchanged bundle links week8. Reproduced on baseline. Repaired selection still requires the same scenario: exact nominees, unknown→REVIEW, intentional assumptions→stored expected starter, no outcome/action fabrication. |
| `tests/test_dbt_incremental.py::test_incremental_run_equals_full_refresh_after_append` | SETUP ERROR | Same ERROR | BLOCKED | Locked dbt packages unavailable: proxy CONNECT403 for `hub.getdbt.com/api/v1/index.json`. Environment-blocked, reproduced on baseline. Run in an environment permitting the normal registry route; body has not run. |
| `tests/test_dbt_incremental.py::test_restated_week_inside_lookback_is_picked_up` | SETUP ERROR | Same ERROR | BLOCKED | Same registry denial. Correction-inside-lookback verification remains a merge gate. No pins/checks changed. |
| `tests/test_dbt_incremental.py::test_restated_week_outside_lookback_needs_full_refresh` | SETUP ERROR | Same ERROR | BLOCKED | Same registry denial. Correction-outside-lookback/full-refresh remedy remains a merge gate. No checks bypassed. |

Exact Python comparison commands, executed from **each** checkout with its own `PYTHONPATH`:

```bash
# Set this to /workspace/ffai-acceptance-baseline, then /workspace/fantasy-football-ai.
CHECKOUT=/workspace/ffai-acceptance-baseline
cd "$CHECKOUT"
PYTHONPATH="$CHECKOUT" /workspace/fantasy-football-ai/.venv/bin/python -m pytest -vv \
  tests/test_decision_lab_exporter.py::test_reconciliation_statuses_on_committed_evidence \
  tests/test_decision_lab_exporter.py::test_champion_at_source_and_row_provenance \
  tests/test_decision_lab_exporter.py::test_mismatched_mart_yields_unknown_baseline
PYTHONPATH="$CHECKOUT" /workspace/fantasy-football-ai/.venv/bin/python -m pytest -vv \
  tests/test_dbt_incremental.py::test_incremental_run_equals_full_refresh_after_append \
  tests/test_dbt_incremental.py::test_restated_week_inside_lookback_is_picked_up \
  tests/test_dbt_incremental.py::test_restated_week_outside_lookback_needs_full_refresh
```

Exact original browser comparison commands (cwd each checkout's `frontend-next`):

```bash
# Baseline was built with the unchanged lockfile and served on port3100 first.
NEXT_TELEMETRY_DISABLED=1 NEXT_PUBLIC_API_URL=http://localhost:7861 npm run build
NEXT_TELEMETRY_DISABLED=1 npm run start -- --hostname 127.0.0.1 --port 3100
# Another terminal; baseline config's testDir points into baseline checkout:
npm exec playwright test -- --config=/tmp/ffai-acceptance/baseline-browser.config.cjs \
  e2e/decision-lab.spec.ts --grep 'published weekly case'
# After stopping only that task-started server and serving modified3100:
npm exec playwright test -- --config=/tmp/ffai-acceptance/modified-browser.config.cjs \
  e2e/decision-lab.spec.ts --grep 'published weekly case'
```

The temporary configs use `defineConfig`, that checkout's installed Playwright, `testDir` pointing
to its e2e directory, baseURL `http://localhost:3100`, one worker, zero retries, expect15s, test90s,
and `launchOptions.executablePath='/usr/bin/chromium'`. They and result logs remain under `/tmp`.

## Product boundaries and browser evidence

Actuals support standard, half-PPR and PPR; recommendations support only implemented PPR rules.
The hand receiver example was exercised through HTTP and the running browser/API with actual
receptions6 versus estimate2: standard6, half9, PPR12. Zero/negative observations remained numeric;
missing conversion displayed its reason. HTTP tests additionally cover legacy/no marts,
conflicting revisions, matching/conflicting duplicate outcomes and schema metadata.

The verified quick adapter still calls the canonical Lab builder and policy. Tests exercise a
clear preference, ties in either nomination order, exact/below gap boundary, FLEX exclusions,
duplicate/oversized/one-option sets, missing selected rows and unresolved availability. A nomination
is user-nominated ownership only. No winner is selected by display order and no action is recorded
by computing. v1 receipt import/replay against paired original evidence passes in both languages.

Five explicit integration browser flows ran against frontend3300 and changed API7861:

1. Predictions requested the changed local API (asserted origin on every predictions HTTP request),
   returned the new rules metadata, displayed uniquely labeled LOCAL FIXTURE players and format
   values6/9/12, zero/negative and missing evidence. This was not the deployed older service.
2. No companion404: unknown availability→intentional assumptions→conditional dated suggestion,
   keyboard Compute, mobile390px without page overflow, no persisted action.
3. Valid partial companion: sources remain unverified; no current recommendation; intentional
   historical hypothetical exploration reveals only hypothetical recording controls.
4. Known bye and a later passed kickoff: assumptions cannot override either; clock advancement
   and focus503 retain the disqualifying facts with original timestamps.
5. Malformed digest and incompatible forecast reference: incompatible rather than missing context,
   no ordinary action panel; failed refresh cannot cleanse corrupt evidence.

All seven existing Lab/StartSit browser flows also passed: receipt record/reveal/export/import,
stale controls, published unknown-availability comparison, keyboard, mobile scrolling, API-outage
same-origin comparison and corrupt bundle failure. The browser used system Chromium154; the
standard CI managed-browser download/install remains a normal CI environment operation.

## Precisely how optional context reaches the browser

The component fetches **same-origin `/context/snapshot.json`**, no-store. This comes from an explicit,
reviewed export to `frontend-next/public/context/snapshot.json`. API `GET /context` independently
reads `artifacts/context/snapshot.json`; the frontend does **not** proxy or automatically fetch it.
Exporting only to API artifacts therefore does not enable frontend context.

No file is published by this review. The browser integration uses Playwright-local clearly
synthetic response fixtures, not provider payloads or edited real projections. The local API
fixture factory replaces only in-memory prediction state in its isolated process; normal API
startup and repository artifacts are unchanged.

Absent companion stays unverified and permits an intentional conditional dated comparison.
Partial companions never turn row counts into availability proof. Malformed/incompatible context
stays blocked until a valid compatible replacement. Failed refresh retains previous known facts
without resetting first-observed/fetched times or claiming fresh verification. Game deadlines
reassess locally every second and on focus/visibility; snapshots/assumptions are not silently
rewritten. Identity/cardinality, UTC/DST, target-week coverage, source/content age and no-target
cases have offline tests. A user assertion cannot change source verification or league lock rules.

## Exact local review configuration and commands

Commands run from `/workspace/fantasy-football-ai`, with the existing prepared dependencies.
The synthetic API is opt-in and its helper is outside the default application entry point:

```bash
FFAI_ACCEPTANCE_FIXTURE=synthetic \
FFAI_CORS_ORIGINS=http://localhost:3300,http://127.0.0.1:3300,http://localhost:3100 \
  .venv/bin/python -m uvicorn --app-dir tests acceptance_api:create_app \
  --factory --host 127.0.0.1 --port 7861
```

Another terminal:

```bash
cd /workspace/fantasy-football-ai/frontend-next
NEXT_PUBLIC_API_URL=http://127.0.0.1:7861 npm run build
npm run start -- --hostname 127.0.0.1 --port 3300
```

Another terminal, same frontend directory:

```bash
FFAI_CHROMIUM_EXECUTABLE=/usr/bin/chromium \
  npm exec playwright test -- --config=playwright.acceptance.config.ts
```

This complete opt-in suite is separate from the ordinary e2e directory because it requires the
explicit local fixture API. It contains no skipped tests. Results default to `/tmp/ffai-acceptance`.
With a normally installed Playwright browser, omit `FFAI_CHROMIUM_EXECUTABLE`. Ordinary development
continues to use `make api` and `npm run dev`; rebuild without the acceptance API URL before any
normal publication. No build or server started here is a deployment.

To disable optional context, stop exporting/serving its companion. Do not remove correct scoring
or restore the independent highest-projection rule. Do not publish live normalized data before
establishing rights and appropriate source coverage. This review adds no automated live schedule,
receipt-v2 persistence, custom-league support or verified game-day mode.

## Commands and final outcomes

Passing and blocked outcomes are recorded separately below. A full-suite nonzero status caused
by setup errors is not a passing full suite.

```bash
# cwd repository root, executed after review fixes:
PYTHONPATH=/workspace/fantasy-football-ai .venv/bin/python -m pytest tests
make lint publication-check lab-check lab-verify
# cwd frontend-next:
npm run typecheck
npm run lint
npm test
NEXT_PUBLIC_API_URL=http://127.0.0.1:7861 npm run build
FFAI_CHROMIUM_EXECUTABLE=/usr/bin/chromium npm exec playwright test -- --config=playwright.acceptance.config.ts
npm exec playwright test -- --config=/tmp/ffai-acceptance/modified-browser.config.cjs --output=/tmp/ffai-acceptance/e2e-final-results
```

Final counts, protected-file checks and final Git inventory follow in the verification record.

## External merge gates and PR preparation

The remaining merge gate is normal-network execution of the **three exact dbt nodes above** after
`make dbt-deps` succeeds using existing `constraints.txt` and `dbt/package-lock.yml`. Registry access
to `hub.getdbt.com` is currently denied with proxy403. Preserve TLS/package integrity verification;
no alternate unchecked package or skipped assertion is an acceptable substitute. Run the full CI
workflow in its normal supported environment before merge, including its managed Chromium setup.
The scoped browser checks already passed with system Chromium; no standard CI success is claimed.

Suggested commit boundaries:

1. Format-correct actual resolver, attachment, API/history consistency, displays and HTTP tests.
2. Verified quick comparator, explicit assumptions/actions, period reset, baseline-proven test repairs
   and v1 compatibility fixtures.
3. Context probe/contract/readiness, validation/refresh hardening and optional artifact API.
4. Acceptance integration harness and review documentation (may be folded into the relevant slices).

PR title: **Make weekly Start/Sit comparisons use verified PPR evidence and format-correct actuals**

Suggested PR body:

> Prediction browsing could return a stored PPR actual under half/standard, while Start/Sit chose
> the highest forecast independently of the Lab policy. Resolve actuals from realized, compatible
> evidence and route the quick one-slot PPR comparison through the verified weekly bundle and
> canonical policy. Keep availability assumptions intentional and period-specific; retain separate
> optional context readiness so malformed data, known impossibility and passed deadlines cannot
> become actionable advice. Add explicit fixture/live capability probing and offline/browser
> coverage while preserving historical evidence and v1 receipt semantics.
>
> Validation: see this acceptance report for passing tests and baseline comparison. Three dbt
> integration checks remain blocked by package-registry policy and must pass in normal CI before
> merge. The release is a conditional weekly comparator; no verified game-day service or accuracy
> improvement is claimed.

## Final verification record

| Outcome | Command/check | Result |
|---|---|---|
| PASS portions; overall BLOCKED/nonzero | `PYTHONPATH=/workspace/fantasy-football-ai .venv/bin/python -m pytest tests` | **585 passed, 3 setup errors, no assertion failures**, 159.26s. All three setup errors are the exact dbt nodes listed above. Their bodies did not run. |
| PASS | `make lint publication-check lab-check lab-verify` | Ruff/Black (84 files), publication scan, golden check, temporary fresh bundle rebuild and digest/schema verification passed; 54 snapshots/17,118 rows/18 cases unchanged. |
| PASS | `npm run typecheck`; `npm run lint`; `npm test` | **223 tests passed**, 17 files; typecheck and lint passed. |
| PASS | `NEXT_PUBLIC_API_URL=http://127.0.0.1:7861 npm run build` | Production build passed with the changed local API configuration. |
| PASS | Complete explicit local acceptance browser suite | **5 passed**. API-backed actual values/source origin, missing/partial/disqualifying/malformed/incompatible context and failed refresh exercised. |
| PASS | Complete existing e2e suite using system Chromium config | **7 passed**. Repaired inherited published-weekly case now exercises its original behavior. |
| PASS | v1 paired original evidence/receipt tests | All4 Python checks (included in full585) and3 TypeScript checks (included in223) passed. |
| PASS | `git diff --check`; protected-file diff + untracked scans | Clean whitespace; no changed/untracked models, evaluations, forecasts, published bundles, dbt SQL/pins or v1 policy/spec/canonical/goldens. |
| BLOCKED | Exact three dbt incremental nodes, baseline and modified | Configured proxy CONNECT403 for hub.getdbt.com; no bypass, pin alteration, skip or integrity disabling. Run in normal CI/network before merge. |

The initial new integration harness run had one harness-only assertion failure because it counted
Next.js `/predictions/` script assets as API requests. The predicate was corrected to exact API
route shape; every API request still must target127.0.0.1:7861. All five flows then passed. No
application behavior was changed or assertion about the API origin removed to hide that result.

Logs and traces: `/tmp/ffai-acceptance/modified-full-pytest.log`, `modified-python-checks.log`,
`modified-final-build.log`, `integration-browser-final.log`, `e2e-final.log` and their result
directories. Baseline/initial-modified failure evidence is retained beside them. Standard managed
Playwright installation/CI execution was not claimed; installed system Chromium154 was used.

Final branch `work`; full HEAD `3ddbb36dbf67f33c769c45645957aa92b690b457` (unchanged).
Final status: **17 modified tracked files, 40 untracked files, no staged changes**. All initial work is retained. The detached baseline remains
at the same full SHA and has no tracked or untracked changes. No Git publication action occurred.

Final `git status --porcelain=v1 --untracked-files=all`:

```text
 M ffai/pipeline/score.py
 M ffai/scoring.py
 M ffai/serve/app.py
 M ffai/serve/marts.py
 M ffai/serve/schemas.py
 M frontend-next/e2e/decision-lab.spec.ts
 M frontend-next/src/app/predictions/page.tsx
 M frontend-next/src/app/start-sit/page.tsx
 M frontend-next/src/components/decision-lab/ActionPanel.tsx
 M frontend-next/src/components/player/PlayerHistoryChart.tsx
 M frontend-next/src/components/player/PlayerProfile.tsx
 M frontend-next/src/components/portfolio/PortfolioHome.tsx
 M frontend-next/src/components/predictions/PredictionsBoard.tsx
 M frontend-next/src/components/start-sit/StartSitEngine.tsx
 M frontend-next/src/content/performance.ts
 M frontend-next/src/lib/api/types.ts
 M tests/test_decision_lab_exporter.py
?? docs/CONTEXT_DATA_CONTRACT.md
?? docs/SCORING_ACTUALS.md
?? docs/START_SIT_ACCEPTANCE.md
?? docs/START_SIT_FOUNDATIONS.md
?? ffai/context/__init__.py
?? ffai/context/probe.py
?? ffai/context/snapshot.py
?? ffai/serve/actuals.py
?? frontend-next/acceptance/acceptance.spec.ts
?? frontend-next/e2e/start-sit.spec.ts
?? frontend-next/playwright.acceptance.config.ts
?? frontend-next/src/components/player/__tests__/PlayerHistoryChart.test.tsx
?? frontend-next/src/components/predictions/__tests__/PredictionsBoard.test.tsx
?? frontend-next/src/components/start-sit/ContextEvidence.tsx
?? frontend-next/src/components/start-sit/__tests__/context-evidence.test.tsx
?? frontend-next/src/components/start-sit/__tests__/start-sit-readiness.test.tsx
?? frontend-next/src/components/start-sit/__tests__/start-sit.test.tsx
?? frontend-next/src/components/ui/ActualPoints.tsx
?? frontend-next/src/lib/decision-lab/__tests__/v1-compatibility.test.ts
?? frontend-next/src/lib/start-sit/__tests__/adapter.test.ts
?? frontend-next/src/lib/start-sit/__tests__/fixtures.ts
?? frontend-next/src/lib/start-sit/__tests__/readiness.test.ts
?? frontend-next/src/lib/start-sit/adapter.ts
?? frontend-next/src/lib/start-sit/readiness.ts
?? scripts/probe_context.py
?? tests/acceptance_api.py
?? tests/fixtures/context/normalized.json
?? tests/fixtures/context/sources.json
?? tests/fixtures/start_sit_compat/historical-inputs.json
?? tests/fixtures/start_sit_compat/historical-outcomes.json
?? tests/fixtures/start_sit_compat/historical-receipt-0.json
?? tests/fixtures/start_sit_compat/historical-receipt-1.json
?? tests/fixtures/start_sit_compat/index.json
?? tests/fixtures/start_sit_compat/synthetic-inputs.json
?? tests/fixtures/start_sit_compat/synthetic-outcomes.json
?? tests/fixtures/start_sit_compat/synthetic-receipt-0.json
?? tests/test_actual_resolution.py
?? tests/test_context_api.py
?? tests/test_context_snapshot.py
?? tests/test_start_sit_compatibility.py
```
