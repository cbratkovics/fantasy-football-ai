# ADR-0026 — Slim CI compares and defers to the cached `main` dev build, not a prod manifest (2026-09-15)

**Context.** The brief proposed committing the weekly job's prod `manifest.json` under
`artifacts/dbt/prod_manifest/` and running `dbt build --select state:modified+ --defer --state
<it> --target dev` in CI. Two facts argued against it. (1) `--defer` rewrites unmodified
`ref()`s to the *state's* relations: prod relations live in MotherDuck catalog `ffai`, while the
CI dev target is a fresh local DuckDB file (catalog `ffai_dev`), so every deferred ref would
point at a catalog CI cannot see unless it attached MotherDuck read-only with a token on every
push (compute on every CI run, no secrets on fork PRs). (2) The manifest is 1.7 MB (169 KB
gzipped, 943 macros) and its `metadata` block changes every run; committing it weekly is churn
in a repo that will later `copier update`. The owner also asked for one state source: selecting
against a prod manifest while deferring to a dev warehouse invites env-specific vars to make
nodes look modified in one and not the other.

**Decision.** Pushes to `main` run a full dev build and save `dbt/target/manifest.json`,
`.duckdb/ffai_dev.duckdb`, and the commit hash to the Actions cache (`actions/cache/save@v5`,
key `dbt-dev-state-<sha>`). Every other run (`pull_request`, other branches) restores the newest
`dbt-dev-state-*` entry, copies the DuckDB file into place, and runs `dbt build --select
state:modified+ --defer --state .dbt-state --target dev --full-refresh`: modified nodes and
their descendants build into the restored file, unmodified upstream refs resolve to the tables
already in it. If no state is restored (first run, or the cache was evicted: GitHub evicts
entries not accessed for seven days, so a quiet week means the next PR does a full build) the
job falls back to the full dev build. The prod manifest is not committed; nothing else needs it.
Locally, `make dbt-state` / `make dbt-slim` do the same against `.dbt-state/` (git-ignored).

**Consequences.** Prod builds stay in `weekly.yml` only (one per week on MotherDuck). CI needs no
secret. A pull request that changes one gold model builds that model and its tests, with docs
generate and the description check still running over the whole project. Cache misses are
silent-by-design and logged as "full dev build".
