# ADR-0029 — The stack is extracted into a private copier template; this repository keeps its names (2026-09-15)

**Context.** The second pass added the template seams (ADR-0028) so the dbt + scikit-learn +
FastAPI + Next.js + GitHub Pages stack could be reused for a second project (`nba-ai-ml`) without
re-deriving it. The question was where the generic form lives and how far this repository bends
towards it.

**Decision.** The generic form lives in a separate private repository,
`cbratkovics/ds-dbt-stack-template` (copier, not cookiecutter, so generated projects can
`copier update` later; delimiters `[= =]` / `[% %]` so dbt's `{{ }}`, Python's `df[[...]]`
indexing, shell heredocs and TypeScript generics pass through). It was written from the
boundary audit (`docs/TEMPLATE_BOUNDARY.md`): Generic code copied, Parameterizable code copied
with variables, Domain code replaced by stubs behind `SourceLoader`, `TargetSpec` and
`FeatureModule`. This repository is *not* regenerated from the template and keeps every name its
API, contracts and artifacts already use (`within_3_rate`, `player_id`, `week_%02d.json`,
`fct_player_week`, the `positions` block in model metadata, XGBoost as the challenger). The
template's `docs/SECOND_USE_CHECKLIST.md` lists each such difference and why. Three smoke runs
(default answers; no frontend + tiers; a 120-period game-day grain with three cohorts) pass from
a clean `copier copy` with no manual edits (`SMOKE_RESULTS.md` in the template).

**Consequences.** Improvements flow from the template to *new* projects; this repository adopts
them by hand when they matter (the reproducibility knob and the versioned-mart reconciliation rule
already did, in the other direction). The Claude Code skill `evidence-first-ml-pipeline` now has
three modes (audit, instantiate, apply) and points at the template's schemas and checklists.
