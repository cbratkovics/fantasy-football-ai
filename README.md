# Win My League — Applied Decision Science

An end-to-end fantasy-football decision system and data-science portfolio case study. The project asks a practical question: **how should a user act when forecasts are uncertain and mistakes have different costs?**

The sports setting is intentionally low-risk, but the workflow transfers to risk strategy: time-sensitive signals, entity-level scores, asymmetric errors, policy thresholds, cohort monitoring, and recommendations that must be explainable.

> **Evidence standard:** repository fixtures and UI examples are demo data. They are not presented as measured production performance. A number becomes a portfolio claim only when a reproducible evaluation artifact records its dataset, time split, baseline, model version, and metric definition.

## What is implemented

| Layer | Implementation | Status |
|---|---|---|
| Product | Next.js decision-lab experience, projections, tiers, and evaluation views | Implemented |
| API | FastAPI routes for players, predictions, tiers, auth, and subscriptions | Implemented; integration maturity varies |
| Modeling | Feature engineering, position models, ensembles, uncertainty, and GMM tiers | Prototype modules |
| Data | Sleeper/ESPN/weather connectors plus synthetic fixture generation | Mixed live connectors and demos |
| Operations | Docker, Railway/Vercel configuration, Alembic, Redis/Celery scaffolding | Deployment scaffolding |
| Decision analytics | Policy simulator and a documented SQL metric contract | Portfolio reference implementation |
| Evaluation | Forward-time holdout, causal baseline, cohort metrics, policy sweep, evidence manifest | Implemented CLI |

See [`docs/PORTFOLIO_CASE_STUDY.md`](docs/PORTFOLIO_CASE_STUDY.md) for the codebase audit, target architecture, evaluation design, metric tree, and interview narrative.

## Run the portfolio

```bash
cd frontend-next
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The landing-page simulator lets you change a recommendation threshold and see the coverage/downside trade-off.

## Repository map

```text
backend/
  api/          FastAPI route modules
  data/         source adapters and ingestion prototypes
  ml/           features, models, tiers, and training prototypes
  services/     inference, explanation, payments, and integrations
frontend-next/
  src/app/      Next.js routes
  src/components/portfolio/  portfolio case-study interface
analytics/sql/  decision-performance metric contract
docs/           architecture and portfolio documentation
```

## Recommended evaluation contract

Every reported experiment should include:

1. immutable input snapshot or content hash;
2. rolling-origin train, validation, and test windows;
3. naive and expert-ranking baselines;
4. MAE and interval coverage by position/week;
5. policy coverage, regret, and downside by cohort;
6. model/feature versions and one command that reproduces the report.

This prioritizes honest, decision-relevant evidence over an unsupported “accuracy” headline.

### Generate an evidence artifact

Provide a CSV containing `player_id`, `season`, `week`, `position`, `prediction`,
`actual`, and `decision_score`; `prediction_floor` is optional. Scores must be on
a 0–1 scale if you use the default policy thresholds.

```bash
python -m backend.evaluation.decision_evaluator predictions.csv \
  --test-start 2024-10 \
  --output artifacts/evaluation.json
```

The generated JSON records the input SHA-256, Git commit, forward-time split,
causal trailing-mean baseline, position cohorts, and threshold-policy metrics.

## Stack

Python, FastAPI, pandas, scikit-learn, XGBoost/LightGBM prototypes, PostgreSQL, Redis/Celery, Next.js, TypeScript, D3, Docker, and Terraform.

## License

[MIT](LICENSE) · Christopher Bratkovics
