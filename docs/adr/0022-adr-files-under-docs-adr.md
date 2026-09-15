# ADR-0022 — New decision records are files under `docs/adr/`, indexed from `DECISIONS.md` (2026-09-15)

**Context.** ADR-0001 … ADR-0021 live as sections of `docs/DECISIONS.md`, and README,
ARCHITECTURE, model YAML, and code comments link to them by number. The second-pass brief asks
for one file per decision under `docs/adr/NNNN-*.md`, which is also the layout the
`ds-dbt-stack-template` ships (`docs/adr/0000-template.md`).

**Decision.** From ADR-0022 on, each decision is its own file `docs/adr/NNNN-<slug>.md` with the
same three-part shape (Context, Decision, Consequences). `docs/DECISIONS.md` keeps the first 21
records in place and gains an index line per new file, so every existing `ADR-00xx` reference
resolves either to a section or to an index entry. Numbering stays global.

**Consequences.** Two places to look, one numbering. The template instantiates only the
per-file layout; this repository is the one exception and says so at the top of the index.
