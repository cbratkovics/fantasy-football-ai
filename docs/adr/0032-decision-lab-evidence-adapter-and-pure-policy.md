# ADR-0032 — Decision Lab: an offline evidence adapter over the gold marts and a pure, versioned decision policy (2026-09-27)

**Context.** The site's existing Decisions panel selects a precomputed floor threshold and a
season over `fct_decision_policy` (v1, pinned by the API, ADR-0025). It is a policy-metrics
view, not a saved, user-specific start/sit workflow: nothing in it names the user's actual
alternatives for one slot, records what they chose, or attaches what happened afterwards. The
underlying marts also carry definitions that must not be presented as roster decisions:
`fct_player_decisions.regret` compares a player with the *best scorer in the whole position pool*
that week, and its replacement level is a rank-k projection (`var replacement_rank`), not what was
available on a roster; the weekly scoring targets are players with recent stat rows, not verified
rosters; the prediction floors are position-level residual quantiles (10th percentile), not
player-specific guarantees. A user-facing "what should I start" feature needs evidence that is
versioned, never leaks outcomes into the inputs, and can be recomputed by anyone from the
committed files, with the caveats above stated where the comparison is made.

**Decision.** `ffai/decision_lab` is an offline adapter over evidence the repository already
commits; it trains nothing, opens no socket, and introduces no new warehouse or hosted service.

*Snapshots.* `scripts/export_decision_lab.py` (`make lab-build`) reads the committed
`artifacts/marts/*.parquet` (`fct_player_week`, `dim_player`), the prediction files, and the
evaluation artifacts, and writes **inputs snapshots** — one per source family, season and week,
champion candidate only, containing projections, floors, ceilings, the baseline, availability
fields and provenance, and *never* target-week actuals, error metrics, hindsight ranks or regret
— and, separately, **outcome snapshots** with their own content identities that name the inputs
snapshot they belong to. Full-data aggregates are reconciled to the evaluation artifacts before a
snapshot is marked verified; a mismatch blocks it rather than publishing a weaker copy. Baseline
provenance (player history, position history, or the prediction itself) is derived by the
population rule plus reconciliation and recorded as `unknown` on a mismatch; the additive
`fct_player_week.baseline_source` column added with this record carries the same value exactly
from the next prod export onwards, and the adapter prefers it when the parquet has it.
`_build.json` and `_last_attempt.json` are volatile build metadata and never enter an identity.

*Policy.* `policy_spec.json` (`policy_version` 1.0.0) is the specification; `ffai.decision_lab.policy`
is the Python reference and `frontend-next/src/lib/decision-lab` the TypeScript mirror, both held
to the same golden fixtures (`tests/fixtures/decision_lab/golden.json`, `make lab-golden`,
checked by pytest and vitest). `evaluate(inputs)` is a pure function of a `decision_inputs/1.0`
document whose schema has no field for outcomes. Precedence: **HOLD** (invalid or incompatible
evidence: corrupt or blocked snapshot, mixed periods or model versions, bad parameters) >
**REVIEW** (choice set too small or unresolved, projection gap within tolerance or below
`min_projection_gap`, floor below `min_floor`) > **RECOMMEND** the leading option, conditional on
the recorded assumptions. The causal trailing-mean baseline is a separately labelled comparator:
its preferred player is reported next to the model's and never substituted for it. The
user-decision states (`not_recorded` / `recorded` / `declined`, outcome `attached`) are a
different state machine from the pipeline's PUBLISH / HOLD / PROMOTE and share no vocabulary
with it on purpose.

*Placement.* The bundle is committed under `artifacts/decision_lab` and duplicated into
`frontend-next/public/decision-lab` because the Vercel project root is `frontend-next` and
static files must live under it. `make lab-check` rebuilds into a temp directory and compares
both copies; `make lab-verify` re-reads them fail-closed. The `decision_lab` exposure in
`dbt/models/gold/_exposures.yml` records the lineage.

**Consequences.** No new warehouse, service or secret. A fixture dbt build cannot validate the
bundle (the 40-player fixture has no full-data aggregates to reconcile); CI's python job runs
the lab check against the committed full-data marts instead, and the dbt job's comment says so.
The weekly job exports the bundle after the gold export and stages
`frontend-next/public/decision-lab` with the artifacts; an export failure leaves the previous
bundle in place, writes `_last_attempt.json`, opens a `weekly-hold` issue and does not block the
artifact commit. Two copies of the bundle live in the tree and CI verifies they are equal. Every
recommendation the lab shows is reproducible from committed files by
`python -m ffai.decision_lab.replay`.
