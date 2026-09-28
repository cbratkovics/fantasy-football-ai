# ADR-0034 — Frontend test tooling: vitest + Testing Library for unit, component and parity tests; Playwright for the browser flow (2026-09-27)

**Context.** `frontend-next` (Next 14, React 18, Node 20) shipped with `npm run lint` and
`npm run build` as its only checks; it had no test runner. The Decision Lab (ADR-0032, ADR-0033)
puts real logic in the browser — canonical JSON, SHA-256 identities, the policy mirror, receipt
validation and merge, `localStorage` persistence — and that logic must agree byte-for-byte with
the Python reference. A unit runner alone cannot show that the page loads the bundle, compares,
records and reveals correctly; a browser runner alone is too slow and too coarse to pin a hash.

**Decision.** Two runners, both configured inside `frontend-next` and both run by CI's frontend
job and `make lab-test` / `make lab-e2e`:

* **vitest** with **Testing Library** for unit, component and parity tests (`npm test`). The
  parity tests read `tests/fixtures/decision_lab/golden.json` — the same file
  `python -m ffai.decision_lab.golden --check` pins — and must reproduce every `expected` block:
  canonical text, digests, policy status, recommended id, ordered reason codes, gap and gate
  values, baseline preference, limitations, explanation text, outcome metrics, and the receipt
  lifecycle. A change to the policy therefore fails in whichever language was not updated.
* **Playwright** (chromium) for the browser flow (`npm run test:e2e`): load the bundle, compare
  alternatives, change an assumption (a child decision appears; the original is unchanged),
  record the action, reveal the outcome, export and import receipts, reload and find the same
  receipts. The e2e config starts `next start` itself against the production build, with
  `NEXT_PUBLIC_API_URL` pointing at a local port so nothing reaches the deployed API.

Both were chosen because they fit the existing stack without a framework migration: vitest reads
the project's TypeScript and path aliases as-is and runs in a jsdom environment; Playwright ships
its own browser and needs no running dev server. Jest would have needed a separate transform
configuration for Next's ESM output; Cypress adds a heavier runtime for no gain over the one flow
we test. `npm run typecheck` (`tsc --noEmit`) runs before the tests so type errors are reported as
such rather than as failing assertions.

**Consequences.** The frontend CI job runs `lint → typecheck → test → build → playwright install →
test:e2e → publication check` and takes a few minutes longer. The golden fixture is a shared
contract: regenerate it with `make lab-golden` only when the policy specification changes, and
expect the TypeScript parity tests to fail until the mirror is updated. Playwright is a dev
dependency only; the Vercel build does not install browsers.
