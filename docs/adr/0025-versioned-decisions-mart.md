# ADR-0025 — `fct_decision_policy` is versioned; v1 keeps its relation name, the API pins a version (2026-09-15)

**Context.** Every gold mart already has an enforced contract, so a breaking change fails the
build instead of the API. Versions are the next step: an additive change ships without touching
what the API serves. `fct_decision_policy` is the public decisions mart (`/marts/decisions`, the
site's Decisions panel). `export_gold` named files by node *name*; both versions of a versioned
model share a name.

**Decision.** `fct_decision_policy` has `versions: [1, 2]`, `latest_version: 2`. v1 is the SQL
that was served, byte-identical, with `config.alias: fct_decision_policy` so its relation, its
exported parquet (`fct_decision_policy.parquet`), its `_export_manifest.json` row, and the API
response are unchanged. v2 adds two additive columns for recommended players with an outcome:
`recommendation_within_3_rate` and `recommendation_interval_coverage` (actual inside
floor..ceiling); it exports as `fct_decision_policy_v2.parquet`. `export_gold` now names files by
alias and skips models with `meta.export = false`. The API reads an explicit version
(`ffai.serve.marts.DECISIONS_MART_VERSION = 1`, resolved through `mart_table()`), loads both
tables when present, and a test asserts v1 carries no v2 column. Exposures pin
`ref('fct_decision_policy', v=1)`.

Deprecation policy (also in the model description): a version is served for at least one full
season after its successor ships; when the API moves to v2, v1 gets a `deprecation_date` one
season out, keeps building and exporting until then, and is deleted after. An unversioned
`ref('fct_decision_policy')` resolves to the latest version.

**Consequences.** Two decision tables build each week (2,336 rows each). The v2 parquet appears
in `artifacts/marts` after the next weekly export; the API ignores it until the pin moves.
