# Portfolio case study and codebase audit

## Executive framing

Win My League should be presented as a **decision-science system**, not as a collection of advanced-algorithm claims. The valuable story is the full loop:

> ingest events → validate data → estimate outcomes and uncertainty → apply a cost-aware policy → explain the action → monitor outcomes by cohort.

Fantasy football provides an authentic and safe domain for that loop. It does **not** prove fraud expertise, scale, or commercial production experience. In an interview, explicitly distinguish the transferable methods from domain experience you have not earned.

## Codebase audit

### What is strong

- The repository spans source adapters, feature engineering, modeling, API serving, a relational schema, async-job/cache scaffolding, infrastructure, and a user interface.
- Position-specific modeling captures real domain heterogeneity rather than assuming one population.
- Tiers and uncertainty can support a decision product that is more useful than a sorted point forecast.
- The web experience makes technical work inspectable by non-technical reviewers.

### What currently weakens credibility

1. **Claims outrun evidence.** Historical copy asserted accuracy, latency, infrastructure, coverage, A/B testing, and revenue outcomes without a reproducible artifact. The homepage and README now label demonstrations and avoid those claims; remaining legacy routes/data should receive the same treatment.
2. **Fixtures resemble evaluation results.** `predictions_2024.json` includes aggregate “accuracy” metadata but appears to be a small static demonstration. It must not power a measured-results headline.
3. **Broad surface area, uneven depth.** Multiple startup scripts, requirements files, deployment guides, prediction engines, and training modules create ambiguity about the canonical path.
4. **Potential leakage is not visibly controlled.** Sports data are temporal. Every rolling feature must use only information available before kickoff, and every evaluation must split by time—not random row.
5. **No durable analytical contract.** Metric names need grains, denominators, eligibility logic, windows, and ownership. A starter contract is included in `analytics/sql/risk_strategy.sql`.
6. **Operational claims need telemetry.** Caching, queues, and cloud manifests demonstrate design intent; only traces, load-test artifacts, deployments, or incident records demonstrate operation.

## Target product

### User and decision

- **User:** a fantasy manager choosing one player for a lineup slot before kickoff.
- **Decision:** recommend, review, or abstain for each eligible player.
- **Value:** incremental realized points versus a declared baseline ranking.
- **Error asymmetry:** a fragile high-upside selection can be more costly than missing modest upside.

### Policy

Keep prediction and policy separate. A model estimates a distribution; a configurable policy chooses an action.

```text
expected_utility = expected_points
                 - risk_tolerance × downside_shortfall
                 - replacement_level_opportunity_cost
```

This enables a client-style conversation: “At threshold 0.70, what recommendation coverage do we retain, and which cohorts absorb the added downside?” That is more useful than “Which model has the best RMSE?”

## Metric tree

### North-star

**Incremental lineup points per eligible decision**, versus a frozen expert-ranking or trailing-average baseline.

### Decision metrics

| Metric | Definition | Why it matters |
|---|---|---|
| Recommendation rate | recommended eligible decisions / eligible decisions | Measures coverage; prevents quality gains through universal abstention |
| Hit rate | recommendations beating replacement level / recommendations | Policy precision analogue |
| Regret | best eligible realized points − selected realized points | Measures the cost of the action |
| Downside rate | selected outcomes below defined floor / selections | Captures asymmetric harm |
| Review rate | review actions / eligible decisions | Estimates operational/user friction |

### Model diagnostics

- MAE and median absolute error by position, week, availability status, and projection decile.
- Prediction-interval empirical coverage and average width.
- Calibration curve for “beats replacement” probability.
- Population stability index or Jensen–Shannon distance for monitored inputs.
- Missingness, freshness, and out-of-range rates for every production feature.

Metrics should always expose time window, denominator, cohort, baseline version, and policy version.

## Evaluation design

1. Build features **as of** a declared decision timestamp.
2. Use rolling-origin folds: train on prior weeks/seasons, validate on the next block, and test once on the final untouched block.
3. Compare against at least three baselines: prior-week points, rolling mean, and a frozen external/expert rank available at decision time.
4. Tune models on validation folds; tune the decision threshold separately against a stated utility function.
5. Bootstrap games/weeks—not individual correlated rows—for uncertainty around policy deltas.
6. Report every metric by cohort and show failure cases, not only an aggregate.
7. Save an immutable manifest containing input hashes, code commit, feature version, model parameters, split dates, and outputs.

## Analytics engineering plan

Use a star schema with a point-in-time-safe fact table:

- `fct_player_decisions`: one row per league, week, slot, player, decision timestamp, policy action, score, and version.
- `fct_player_outcomes`: one row per player/week with realized outcome and post-event availability.
- `dim_player`, `dim_team`, `dim_week`, `dim_policy`, and `dim_model`.
- `mart_policy_daily`: cohort-level decision and outcome metrics used by the dashboard.

Add data tests for uniqueness at the declared grain, accepted action values, nonnegative prediction intervals, outcome joins, timestamp ordering, and source freshness.

## Production path

### Canonicalize first

1. Select one application entry point, one dependency strategy, and one training pipeline.
2. Archive or remove emergency/alternate server scripts after behavior is covered by tests.
3. Split runtime dependencies from training/dev dependencies and pin them with a lock strategy.
4. Put all configuration behind validated environment settings; keep auth/payment paths outside the core portfolio demo.

### Model lifecycle

- Register a champion and challenger with input schema, training window, metrics, and artifact hash.
- Shadow the challenger before any policy change.
- Log score, action, feature version, model version, policy version, latency, and freshness.
- Define rollback and abstention behavior before deployment.
- Alert on input validity, drift, interval coverage, and decision outcomes—not merely service uptime.

### Responsible low-cost AI

An LLM is not necessary for scoring. If included, use it only to draft a plain-language explanation from **structured, precomputed drivers**. Require a fixed JSON schema, prohibit invented features, cache output, cap tokens, and always retain a deterministic template fallback. This demonstrates restraint and product judgment better than an autonomous agent.

## Interview narrative

### Two-minute version

“I started with a forecasting repository and reframed it around the actual user decision. A good average prediction can still produce a poor action, so I separated the model score from a configurable policy and defined coverage, regret, downside, and interval calibration alongside MAE. Because the data are temporal, the evaluation uses as-of features and rolling-origin validation. I designed cohort marts and monitoring so a stakeholder can see where a policy helps and where it fails. The app’s threshold simulator makes that trade-off tangible. I also audited the repository’s claims: demos are labeled, and no result is promoted until its data, split, baseline, version, and reproduction command are recorded.”

### Be prepared to explain

- why random cross-validation leaks future information;
- why calibration and interval width both matter;
- how threshold changes trade coverage for downside;
- the difference between model performance and policy performance;
- how entity/time grains prevent double counting;
- how to investigate a cohort regression;
- how a prototype becomes a monitored, reversible production service;
- which parts are implemented, prototyped, or proposed.

## Sequenced roadmap

### Phase 1 — trustworthy evidence

- Freeze a licensed historical dataset and document the as-of timestamp semantics.
- Extend the implemented forward-time evaluation CLI to rolling-origin folds and add a frozen expert-ranking baseline.
- Generate a checked-in HTML/JSON model card from one command.
- Replace all remaining fixture-derived accuracy copy with generated metrics or explicit demo labels.

The first executable foundation now lives in `backend/evaluation/decision_evaluator.py`.
It validates the input contract, computes a same-week-leakage-safe trailing baseline,
holds out only future periods, reports position cohorts, sweeps policy thresholds,
and writes a JSON artifact with the input hash and code commit. This is intentionally
independent of training code so a candidate model cannot redefine its own evaluation.

### Phase 2 — analytical product

- Materialize the decision/outcome marts and add SQL/data-contract tests.
- Connect the dashboard to cohort, coverage, regret, and calibration outputs.
- Add failure-case slices and downloadable evidence manifests.

### Phase 3 — production proof

- Canonicalize the API/training path and remove redundant scripts.
- Add unit, integration, contract, and load tests in CI.
- Deploy a small public read-only demo with structured telemetry and a documented cost ceiling.
- Publish one postmortem or model-change memo showing how evidence changed a decision.

This order is intentional: one rigorous, reproducible baseline is more impressive than ten unverified advanced models.
