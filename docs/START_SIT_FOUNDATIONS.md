# Start/Sit foundations: implementation and review

This is the initial implementation record. The subsequent
[acceptance review](START_SIT_ACCEPTANCE.md) supersedes its readiness/counts and inherited-failure
claims with a detached-baseline comparison, bounded fixes and final review commands.

Initial checkout: branch `work`, HEAD `3ddbb36dbf67f33c769c45645957aa92b690b457`, clean working tree.
The supplied task branch was retained. No reset, worktree, commit, push, PR, deployment,
production job, model retraining, or MotherDuck write was performed.

## A. Scoring

See [SCORING_ACTUALS.md](SCORING_ACTUALS.md) for the resolver contract, revision agreement,
source precedence and format metadata. Actual browsing supports **standard, half-PPR and PPR**;
an absent/conflicting conversion is null with a reason. The hand fixture uses 60 receiving yards,
six actual receptions and two estimated receptions: actuals are **6 / 9 / 12**, independent of
the estimate. Zero and negative observations remain numbers. An old API without format metadata
can supply only legacy PPR actuals to the new UI; half/standard are withheld until declared.

Changed paths: `ffai/scoring.py`, `ffai/serve/{actuals,app,schemas,marts}.py`,
`ffai/pipeline/score.py`, frontend API types, PredictionsBoard/ActualPoints and player history.
Future attachment is additive; existing stored PPR meaning and frozen evidence are unchanged.
No SQL model or contract changed.

## B. One comparison path

`frontend-next/src/lib/start-sit/adapter.ts` verifies the same-origin Decision Lab manifest,
policy-spec digest and exact latest **published weekly** input file. It does not fall back to
an older week, API response, historical case or synthetic sample. Direct comparison calls also
validate snapshot shape/cardinality. It uses the existing `buildDecisionInputs` and `evaluate`
functions; there is no second baseline, threshold, ranking or tie-breaking algorithm.

The screen requires QB/RB/WR/TE/ordinary FLEX, two to eight distinct nominations and explicit
availability assumptions. FLEX excludes QB. Missing selected rows are retained visibly and block
reduced-set ranking. Weekly availability starts unknown. Nomination is only `user_nominated`
ownership, never an imported roster. Forecast generation, cutoff, rules, baseline agreement,
population limitations and expandable identities are visible. Decisions support **only the
implemented PPR rules**, including four-point passing TDs and the other listed coefficients;
arbitrary custom PPR leagues are unverified.

The pure v1 result is unchanged. Computing does not record an action. Existing receipt/storage
utilities record only intentional user actions. Paired original historical/synthetic evidence
and v1 receipts are under `tests/fixtures/start_sit_compat`; replay/import checks run in both
languages. Context is never added to v1 identity, events or receipts.

Focus/visibility refresh suspends a computed display during verification. Sequence checks prevent
an older request from replacing a newer snapshot. A changed content digest invalidates the shown
comparison without rewriting a saved record. Nominations persist in session storage; blocked
storage does not prevent use. The homepage floor console is an educational filter, not lineup
advice. The Decision Lab retains explicit historical and synthetic exploration.

## C. Context feasibility and readiness

See [CONTEXT_DATA_CONTRACT.md](CONTEXT_DATA_CONTRACT.md) for source facts, documentation versus
observed evidence, exact probe commands, cache/refresh rules, UTC conversion, rights limitations
and process statuses. The new `ffai/context` normalizer builds a roster/game universe independent
of statistical history. Rookies remain unscored, traded players retain current versus forecast
team, conflicting canonical identities remain unresolved, and outcome fields do not enter inputs.

The explicit live CLI run at **2026-10-31 18:26:15 UTC** observed 272 schedule rows, 3,063 roster
rows and 2,459 injury rows. All three sources remained **partial**. Injury rows covered weeks
1–8, not target week 9. Provider publication/update times and complete current-period availability
were not established. These results demonstrate accessible samples, not a game-day service.
Rendered documentation hosts were blocked; the installed client and official raw dictionaries
were inspected. Live raw payloads remain ignored/local; no live export was placed in public files.

The optional API route `GET /context` reads `artifacts/context/snapshot.json`, validates its
schema/digest and returns named 404/503 responses for missing/invalid companions. It performs no
downloads and does not affect application health or historical serving. For the static browser,
reviewed development companions can be exported separately to `frontend-next/public/context`.
The quick comparator loads `/context/snapshot.json` with no-store. Absent or corrupt context never
repairs or replaces a forecast and is not confused with corrupt forecast evidence.

`src/lib/start-sit/readiness.ts` separately matches exact forecast ID/content digest/period and
evaluates explicit time. Known byes, unavailability, passed kickoff, unresolved identity, missing
coverage, mismatched releases and stale/unverified sources suppress ordinary current advice.
Questionable is not out; an absent report is not active; an unknown kickoff is not unlocked.
Source fetch ages are assumed maxima: schedules 24h, roster/injuries 6h; companion 6h, forecast
168h. These are configurable code constants, not provider SLAs. Unknown content age remains
visible. League ownership, lineup and platform locks remain unverified even with usable sources.
The UI does not enable a validated current mode from the observed partial feeds.

Readiness is checked on snapshot digest change, focus/visibility return and each second for local
deadline passage (the timer makes no provider calls). Blocked/incompatible context preserves the
original dated policy result for inspection. Intentional historical hypothetical exploration is
required before recording under that state, and cannot record a new real action through that
flow. Saved old records remain unchanged. Context-specific persistence and receipt v2 are deferred.

## Local launch and disable

```bash
# Repository root, prepared Python 3.11 environment:
make api
# Another terminal:
cd frontend-next
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open the local `/start-sit`, `/predictions` and `/decision-lab` routes in your development browser.
The quick comparator works from same-origin committed evidence even when the prediction API is
unavailable. The optional context file is intentionally absent by default. Remove only a context
companion you exported, or stop exporting it, to disable context integration. Correct scoring and
the shared dated policy remain available; disabling context does not restore the arbitrary winner.
No new credentials or automatic ingestion schedules are needed for default tests/development.

Suggested commit boundaries (nothing has been committed):

1. Actual resolver, future actual attachment, API metadata, frontend format displays and tests.
2. Verified weekly adapter/StartSitEngine, educational copy and compatibility/browser fixtures.
3. Context probe/normalizer/contracts, optional artifact route and shared readiness UI/tests.

Suggested PR summary: “Make prediction actuals respect requested scoring and move Start/Sit to
the verified Decision Lab policy; add explicit roster/game capability probing and a conservative,
separately versioned context companion.” This does not claim better accuracy or production readiness.

## Validation and known failures

The full Python suite was attempted after the slices: **555 passed, three failed, three setup
errors**. The failures are the same three exporter expectations observed before editing (39
versus 44 real snapshots, three versus eight weekly snapshots, and an expectation that every
baseline is player history). Setup errors are the same three incremental-dbt tests blocked by
`hub.getdbt.com` network policy. Tests and artifacts were not weakened or rewritten to hide them.
Additional final targeted checks cover the later context/refresh review fixes.

Commands used for the relevant checks:

```bash
.venv/bin/python -m pytest tests
.venv/bin/python -m pytest tests/test_actual_resolution.py tests/test_scoring_reconciliation.py tests/test_weekly_policy.py tests/test_api.py tests/test_context_snapshot.py tests/test_context_api.py tests/test_start_sit_compatibility.py
make lint lab-check lab-verify
cd frontend-next
npm run typecheck
npm run lint
npm test
npm run build
```

New Start/Sit browser tests passed on installed Chromium 154, including mobile/keyboard operation,
API outage, unknown assumptions, no automatic action and corrupt evidence. Four existing Lab
browser tests passed; one failed because the hardcoded case `real-weekly-2026-w03-rb-1ed3fb` is
absent from the original committed bundle. The test was left unchanged. Playwright's Chromium
download is blocked by network policy; the browser run used a temporary supported config with
`launchOptions.executablePath='/usr/bin/chromium'`, leaving tracked configuration unchanged.

No dbt SQL changed, so no additional warehouse model build was required. Frozen models,
evaluation metrics, published forecasts, v1 policy/spec/canonicalization/golden fixtures and
committed Decision Lab bundles remain unchanged; `lab-check` rebuilt into a temporary directory
and confirmed the committed bundle/public copy still match. Final exact check counts and working
tree status are recorded in the task handoff.

### Final verification record

- Final targeted Python command above: **101 passed** (scoring/API56, context/API41, v1 compatibility4).
- `make lint`: passed Ruff and Black (83 Python files checked).
- `make lab-check lab-verify`: passed; 54 snapshots, 17,118 input rows, 18 cases; golden fixtures current.
- `npm run typecheck`, `npm run lint`: passed.
- `npm test`: **216 passed**, 17 files.
- `npm run build`: passed, 17 routes generated.
- Final browser command: `npm exec playwright test -- --config=/tmp/ffai-compat-playwright.config.cjs --output=/tmp/ffai-final-browser-results` (system Chromium; temporary config). **6 passed, 1 pre-existing absent-case failure**, documented above.
- `git diff --check`: passed. `git diff --name-only -- artifacts dbt ffai/decision_lab frontend-next/public frontend-next/src/lib/decision-lab/policy.ts frontend-next/src/lib/decision-lab/canonical.ts`: empty.
- HEAD remains `3ddbb36dbf67f33c769c45645957aa92b690b457` on `work`; nothing committed, pushed or deployed.

Final `git status --short` (directory entries include the new tests/helpers/fixtures):

```text
 M ffai/pipeline/score.py
 M ffai/scoring.py
 M ffai/serve/app.py
 M ffai/serve/marts.py
 M ffai/serve/schemas.py
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
?? docs/CONTEXT_DATA_CONTRACT.md
?? docs/SCORING_ACTUALS.md
?? docs/START_SIT_FOUNDATIONS.md
?? ffai/context/
?? ffai/serve/actuals.py
?? frontend-next/e2e/start-sit.spec.ts
?? frontend-next/src/components/player/__tests__/
?? frontend-next/src/components/predictions/__tests__/
?? frontend-next/src/components/start-sit/ContextEvidence.tsx
?? frontend-next/src/components/start-sit/__tests__/
?? frontend-next/src/components/ui/ActualPoints.tsx
?? frontend-next/src/lib/decision-lab/__tests__/v1-compatibility.test.ts
?? frontend-next/src/lib/start-sit/
?? scripts/probe_context.py
?? tests/fixtures/context/
?? tests/fixtures/start_sit_compat/
?? tests/test_actual_resolution.py
?? tests/test_context_api.py
?? tests/test_context_snapshot.py
?? tests/test_start_sit_compatibility.py
```
