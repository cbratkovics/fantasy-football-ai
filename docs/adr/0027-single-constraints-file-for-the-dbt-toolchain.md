# ADR-0027 — One `constraints.txt` pins dbt-core, dbt-duckdb, and DuckDB for CI and the weekly job (2026-09-15)

**Context.** `requirements-train.txt` floats (`dbt-core>=1.9`, `dbt-duckdb>=1.9`, `duckdb>=1.0`).
State comparison (ADR-0026), the incremental materialisation (ADR-0023), and the snapshot
(ADR-0024) all assume the dev build in CI, the state it was compared against, and the prod build
on MotherDuck run the same adapter and engine; a silent minor bump between a Tuesday prod build
and a Wednesday CI run would make `state:modified+` see macro changes that nobody made.

**Decision.** `constraints.txt` at the repo root pins `dbt-core==1.12.4`, `dbt-duckdb==1.11.0`,
`duckdb==1.5.5` (the versions every Phase B result was produced with). CI (both Python jobs),
`weekly.yml`, and `make install` pass `-c constraints.txt`; the requirements files keep their
ranges so a bump is a one-line change to the constraints file. The API image is unchanged: it
does not install dbt and pins nothing new (its `duckdb>=1.0` reads parquet only).

**Consequences.** Bumping dbt is a deliberate commit that CI validates before the weekly job
sees it. pip's cache key includes the constraints file.
