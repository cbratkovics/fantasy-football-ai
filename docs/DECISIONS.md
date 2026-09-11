# Architecture decision records

Short, dated records of decisions that shape this repository. Newest last.

## ADR-0001 — `main` is fast-forwarded to the production branch (2026-09-10)

**Context.** The live site (www.winmyleague.ai) was built from
`origin/codex/polish-ui-components-including-header-and-footer-mlj0ni`, nine commits ahead of
`main`, containing `backend/evaluation/decision_evaluator.py`, `tests/test_decision_evaluator.py`,
`analytics/sql/risk_strategy.sql`, `docs/PORTFOLIO_CASE_STUDY.md`, and the "Decision Lab"
frontend (see `AUDIT.md` §6). `main` and production had diverged.

**Decision.** `main` was fast-forwarded (`git merge --ff-only`) to that branch head (`403d7ca`).
No conflicts existed because the branch already contained `main`. From this commit on, `main`
matches what production was built from, and the rebuild proceeds from that state.

**Consequences.** The branch's README, dashboard, and evaluator are the starting point for the
slim rebuild. The other four `codex/*` remote branches are ancestors of this one and are left
untouched.

## ADR-0002 — Rescue the git-ignored production trainer (2026-09-10)

**Context.** The only real-data models were trained by
`backend/scripts/training/train_production_ml_models.py`, which `backend/.gitignore` excluded via
a blanket `scripts/` rule. The artifacts were committed; their producer was not.

**Decision.** The trainer is copied verbatim to `ffai/_legacy/train_production_ml_models.py` and
committed as the reference for the as-of feature scheme. The `scripts/` and `archive/` ignore
lines are removed from `backend/.gitignore`. The remaining untracked `backend/scripts/**` files
are **not** added: several embed a database credential and all of them are superseded by the
rebuild; they are deleted with the rest of `backend/` in Phase 4.

## ADR-0003 — Remove committed credentials, leave rotation to the owner (2026-09-10)

**Context.** `scripts/run/start_backend.sh` and `scripts/run_with_env.sh` contained a Supabase
Postgres connection string with a password (committed 2025-07-31). Live-looking OpenAI and
Anthropic keys exist in the untracked `backend/.env.local`.

**Decision.** Both scripts are deleted from the working tree. History is **not** rewritten and no
credential is rotated by the rebuild; both actions require the owner's accounts and are listed in
`docs/REBUILD_REPORT.md`. `.env*` files stay ignored; `*.env.example` files with placeholder values
remain tracked.
