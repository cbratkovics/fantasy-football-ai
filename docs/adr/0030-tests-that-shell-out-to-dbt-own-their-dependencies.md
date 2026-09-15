# ADR-0030 — A test that shells out to dbt installs its own packages; CI job boundaries are not a dependency manager (2026-09-15)

**Context.** CI went red on `main` at `d3179e1`: the `python` job runs pytest, the incremental
equivalence tests (`tests/test_dbt_incremental.py`, ADR-0023) invoke `dbt run` in a subprocess,
and only the `dbt` job ran `dbt deps`. The tests passed locally because `dbt/dbt_packages/` was
already installed, so the dependency was invisible until a clean runner exposed it: "dbt expects
3 package(s) … found only 0 package(s) installed in dbt_packages".

**Decision.** The test module owns its dependency install: a session-scoped, autouse fixture
runs `dbt deps --project-dir dbt --profiles-dir dbt` once before the first dbt invocation, using
the dbt console script next to the current interpreter (`Path(sys.executable).parent / "dbt"`,
falling back to `python -m dbt`), applies the same empty-directory cleanup as `make dbt-deps`
(the macOS `<pkg> 2` quirk), and skips the install only when every package named in
`dbt/package-lock.yml` (transitive `dbt_date` included) is present and non-empty. If deps fails
or leaves a package missing the fixture fails the tests with dbt's stdout and stderr; it never
skips them, because a skip would hide exactly this regression. Belt and braces: the `python` CI
job also runs `make dbt-deps` behind an `actions/cache` entry keyed on `packages.yml` and
`package-lock.yml`, shared with the `dbt` job, so the download happens once. The same three
changes are in the template (`ds-dbt-stack-template` v0.2.1) and the rule is in the skill.

**Consequences.** `rm -rf dbt/dbt_packages && pytest tests/test_dbt_incremental.py` passes from
a clean tree. Any future test that shells out to a tool with its own package step must follow
the same pattern; CI jobs may pre-install for speed but tests may not depend on it. The
`python -m dbt.cli.main` invocation is gone, and with it the runpy "found in sys.modules"
warning.
