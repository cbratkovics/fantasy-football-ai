# Portfolio case study

## Executive framing

Win My League is presented as a **decision-science system**, not as a collection of
advanced-algorithm claims. The loop that matters is:

> ingest events → validate data → estimate outcomes and uncertainty → apply a policy → explain the
> action → monitor outcomes by cohort.

Fantasy football is an authentic, safe domain for that loop. It does **not** prove fraud expertise,
scale, or commercial production experience; in an interview, distinguish the transferable methods
from domain experience that has not been earned.

## What the repository is today (after the 2026-09 rebuild)

The read-only audit in `AUDIT.md` found that the previous codebase served constants and seeded
random numbers, that its published accuracy figures had no computational source, and that every
tracked training path leaked. The rebuild kept two seeds — the git-ignored lagged-feature trainer
and the branch evaluator — and deleted everything else. The result:

| Loop stage | Implementation | Evidence |
|---|---|---|
| Ingest | nflverse only, dated parquet cache (`ffai/data/nflverse.py`) | `docs/DATA_SOURCES.md` |
| Validate | grain / range / null / freshness / row-count contracts (`ffai/data/contracts.py`); PSI drift (`ffai/eval/drift.py`) | `tests/test_contracts.py`, `tests/test_drift.py` |
| Estimate | one as-of feature module; RF champion + XGB challenger per position; residual-quantile intervals | `tests/test_asof_no_leakage.py` on real data; `artifacts/models/<version>/metadata.json` |
| Policy | deterministic publish / hold / promote rule in the weekly job (`ffai/pipeline/weekly.py`, `ffai/models/registry.py`) | `tests/test_weekly_policy.py`, `tests/test_registry.py` |
| Explain | every served record carries model version, feature version, interval method, and the reception-based derivation of half/standard points | `ffai/serve/schemas.py` |
| Monitor | forward holdout + causal trailing-mean baseline + rolling-origin folds, written as a hashed, commit-stamped artifact; weekly rolling evaluation appended by CI | `artifacts/eval/*.json`, `docs/MODEL_CARD.md` |

## What changed in the claims

Every number that used to appear in the README, the site, and the fixtures was removed. Figures
now exist only inside `artifacts/eval/<eval_id>.json` and `artifacts/models/<version>/metadata.json`,
and every place that shows one (model card, `/performance` API, the frontend performance page)
reads it from there and names the key. "Within ±3 points" is defined explicitly as
`mean(|actual − prediction| ≤ 3)` on the same rows as MAE, which was the audit's central finding:
the old headline accuracy and the old MAE could not have come from the same sample.

## Decision layer: designed, partially implemented

The evaluator supports a policy sweep over a `decision_score` (coverage, selected MAE, downside
rate against the prediction floor). No component yet *produces* a decision score, so
`policy_sweep` is empty in the current artifact and the frontend shows no threshold simulator.
The `analytics/sql/risk_strategy.sql` mart is a design sketch for a warehouse that does not exist
in this repository. Both are honest "proposed" items, not implemented ones.

## Metric tree (unchanged intent)

**North star:** incremental lineup points per eligible decision versus a declared baseline.

| Metric | Definition | Status |
|---|---|---|
| MAE, median AE, RMSE, within ±3 / ±5 | see `metric_definitions` in the eval artifact | implemented, per cohort |
| Causal trailing-mean baseline | player's earlier realised points, position fallback, outcomes appended only after the week is scored | implemented |
| Rolling-origin MAE | refit before each week of the test season, score that week | implemented (17 folds) |
| Interval coverage | share of actuals inside [floor, ceiling] | proposed (intervals are validation-residual quantiles; coverage not yet reported) |
| Recommendation rate, hit rate, regret, downside rate | policy metrics over a decision score | designed in the evaluator; no scorer yet |
| PSI per feature | vs training deciles; hold > 0.25 on top-10 features | implemented in the weekly job |

## Interview narrative (two minutes)

"I inherited a forecasting repository whose published accuracy could not be reproduced and whose
API served random numbers. Instead of patching it I audited it, kept the one leak-free feature
scheme and the one honest evaluator, and rebuilt a small system around them: a single as-of
feature module with a leakage test on real data, a champion/challenger trainer with a frozen
temporal split, an evaluator that writes a hashed artifact with a causal baseline and rolling-origin
folds, a stateless API that only reads committed artifacts, and a weekly CI job that scores,
shadow-evaluates, and publishes or holds by a deterministic rule. The resulting numbers are
modest — the model beats a trailing-mean baseline but weekly fantasy scores are noisy — and the
project's value is that every figure it shows can be traced to its data hash, split, baseline, and
commit."

## What is next

1. Interval coverage and calibration in the weekly rolling evaluation.
2. A decision scorer (probability of beating a replacement-level line) so the policy sweep and a
   threshold simulator become real.
3. Opponent and injury context as features — each requires a licence note, a contract check, and
   the same leakage test.
4. A frozen external ranking as a second baseline (only if a licence permits committing it).
