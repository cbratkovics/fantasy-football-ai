# Model card — 20260911-asof_v1-d333de20

_Generated 2026-09-11T03:44:17+00:00 from committed artifacts; do not edit by hand. Regenerate with
`python scripts/evaluate.py`._

| Field | Value | Source key |
|---|---|---|
| Model version | `20260911-asof_v1-d333de20` | `models/20260911-asof_v1-d333de20/metadata.json: model_version` |
| Feature version | `asof_v1` | `metadata.json: feature_version` |
| Champion candidate | `rf` at every position | `metadata.json: positions[*].champion` |
| Target | fantasy_points_ppr (nflverse, regular season) | `metadata.json: target` |
| Data | nflverse weekly player stats via `nflreadpy==0.1.5`, 34293 rows, sha256 `d333de203dc4…` | `metadata.json: data_library, input_rows, input_sha256` |
| Data through (training pull) | season 2024, week 18 | `metadata.json: data_through` |
| Split | train 2019, 2020, 2021, 2022 · validation 2023 · test 2024 (whole seasons, forward in time) | `metadata.json: seasons` |
| Trained at | 2026-09-11T02:01:34+00:00 | `metadata.json: trained_at_utc` |
| Code commit (training) | `c0a47cc7e81af913a6c701286d8667d11375f7c9` | `metadata.json: code_commit` |
| Libraries | scikit-learn 1.9.1, xgboost 3.0.3 | `metadata.json: sklearn_version, xgboost_version` |
| Interval method | validation-season residual quantiles (10th/90th) per position and candidate, added to the point prediction; floor clipped at 0 | `metadata.json: interval_method` |

## What it predicts

Regular-season PPR fantasy points for the upcoming game of a QB/RB/WR/TE, using only that
player's stat rows from strictly earlier weeks (`ffai/features/asof.py`, 65 features,
per position QB 44 / RB 40 / WR 27 / TE 27).
Standard and half-PPR points are derived by rules, not multipliers (`ffai/scoring.py`).

## Frozen test-season evaluation (season 2024, champion per position)

Kind: `frozen_test` — the season held out when the model was trained and selected.
Metric definitions: mean(|actual - prediction|) (`mae`); mean(|actual - prediction| <= 3), same rows as mae
(`within_3_rate`). Baseline: causal trailing mean of the player's earlier realised points (position fallback); outcomes of a week are added only after that week is scored.
Source: `artifacts/eval/eval-20260911-20260911-asof_v1-d333de20-rf.json` (input `artifacts/models/20260911-asof_v1-d333de20/test_predictions.csv`, sha256 `30cf233f8cd8…`, evaluation commit `8e22a47cfc99ba85861cccbe63732da42feb27ab`).

| Cohort | n | MAE | Median AE | RMSE | Within ±3 | Within ±5 | Baseline MAE | Baseline within ±3 |
|---|---|---|---|---|---|---|---|---|
| All | 5747 | 4.5512 | 3.387 | 6.1875 | 44.9% | 66.9% | 4.8589 | 42.4% |
| QB | 653 | 6.1328 | 5.079 | 7.7107 | 30.8% | 49.2% | 6.6844 | 26.5% |
| RB | 1504 | 4.5025 | 3.5365 | 6.064 | 42.9% | 67.4% | 4.804 | 43.3% |
| TE | 1199 | 3.6439 | 2.75 | 4.933 | 54.5% | 76.3% | 3.6636 | 53.6% |
| WR | 2391 | 4.6049 | 3.331 | 6.36 | 45.2% | 66.8% | 4.9942 | 40.6% |

Keys: `metrics.*`, `cohorts[pos].*`, `baseline.*`, `cohorts[pos].baseline.*`.

Rolling origin (rf; refit on all rows strictly before (season, week); score that week): mean fold MAE
**4.5221** vs baseline **5.2697** over
17 folds (`rolling_origin.mean_mae`, `rolling_origin.mean_baseline_mae`).

| Week | n | MAE | Baseline MAE |
|---|---|---|---|
| 2 | 326 | 4.3364 | 4.9676 |
| 3 | 345 | 4.7543 | 5.1571 |
| 4 | 333 | 4.4777 | 5.181 |
| 5 | 288 | 4.882 | 5.5263 |
| 6 | 289 | 4.6743 | 5.0117 |
| 7 | 327 | 4.7017 | 5.3123 |
| 8 | 349 | 4.4494 | 5.0396 |
| 9 | 320 | 4.3689 | 5.0084 |
| 10 | 291 | 4.1324 | 4.7419 |
| 11 | 307 | 4.7986 | 5.7813 |
| 12 | 283 | 4.3435 | 5.2223 |
| 13 | 344 | 4.1611 | 5.1837 |
| 14 | 279 | 4.6844 | 5.5257 |
| 15 | 344 | 4.7313 | 5.6908 |
| 16 | 336 | 4.2957 | 5.048 |
| 17 | 350 | 4.4629 | 5.816 |
| 18 | 337 | 4.6209 | 5.3716 |


## Out-of-sample season evaluation (season 2025, champion per position)

Kind: `out_of_sample_season` — a complete season that no training, validation, selection, or tuning decision ever touched; scored with the same frozen artifact and feature builder.
Metric definitions: mean(|actual - prediction|) (`mae`); mean(|actual - prediction| <= 3), same rows as mae
(`within_3_rate`). Baseline: causal trailing mean of the player's earlier realised points (position fallback); outcomes of a week are added only after that week is scored.
Source: `artifacts/eval/eval-20260911-20260911-asof_v1-d333de20-rf-oos2025.json` (input `artifacts/models/20260911-asof_v1-d333de20/oos_predictions_2025.csv`, sha256 `6ba7b21b2772…`, evaluation commit `8e22a47cfc99ba85861cccbe63732da42feb27ab`).

| Cohort | n | MAE | Median AE | RMSE | Within ±3 | Within ±5 | Baseline MAE | Baseline within ±3 |
|---|---|---|---|---|---|---|---|---|
| All | 5914 | 4.4909 | 3.287 | 6.1377 | 46.0% | 67.3% | 4.8046 | 43.2% |
| QB | 650 | 6.635 | 5.648 | 8.2294 | 25.4% | 44.9% | 7.079 | 22.5% |
| RB | 1539 | 4.5439 | 3.295 | 6.2838 | 45.4% | 67.9% | 4.786 | 44.8% |
| TE | 1264 | 3.5882 | 2.548 | 5.0716 | 55.8% | 76.8% | 3.748 | 53.6% |
| WR | 2461 | 4.3551 | 3.239 | 5.8938 | 46.9% | 67.9% | 4.7582 | 42.3% |

Keys: `metrics.*`, `cohorts[pos].*`, `baseline.*`, `cohorts[pos].baseline.*`.

Rolling origin (rf; refit on all rows strictly before (season, week); score that week): mean fold MAE
**4.436** vs baseline **5.1794** over
17 folds (`rolling_origin.mean_mae`, `rolling_origin.mean_baseline_mae`).

| Week | n | MAE | Baseline MAE |
|---|---|---|---|
| 2 | 339 | 4.7398 | 4.9546 |
| 3 | 347 | 4.3121 | 4.9046 |
| 4 | 352 | 4.595 | 5.3034 |
| 5 | 311 | 4.2145 | 4.8791 |
| 6 | 317 | 4.4148 | 5.093 |
| 7 | 335 | 4.5545 | 5.313 |
| 8 | 287 | 4.5224 | 5.245 |
| 9 | 309 | 4.4479 | 5.153 |
| 10 | 310 | 4.4351 | 5.4668 |
| 11 | 324 | 4.4548 | 5.0689 |
| 12 | 308 | 4.3523 | 5.1687 |
| 13 | 349 | 4.2346 | 4.9626 |
| 14 | 325 | 4.3215 | 5.1075 |
| 15 | 341 | 4.6027 | 5.4234 |
| 16 | 362 | 4.4755 | 5.4675 |
| 17 | 353 | 4.3888 | 5.2878 |
| 18 | 344 | 4.3456 | 5.2512 |


## 2024 test season versus out-of-sample season

On the 2025 season (n = 5914) the frozen champion's MAE is 4.4909 versus 4.5512 on the 2024 test season (−0.060 points); within ±3 is 46.0% versus 44.9% (+1.140 pp). The causal trailing-mean baseline moved from MAE 4.8589 to 4.8046 (−0.054), so the model-minus-baseline gap is -0.308 in 2024 and -0.314 in 2025. Per position MAE: QB 6.1328 → 6.635 (+0.502); RB 4.5025 → 4.5439 (+0.041); TE 3.6439 → 3.5882 (−0.056); WR 4.6049 → 4.3551 (−0.250). Rolling-origin mean MAE: 4.5221 → 4.436 (−0.086); its baseline 5.2697 → 5.1794.

## Candidate comparison (validation season 2023 selects the champion)

| Position | Champion | RF val MAE | XGB val MAE | RF test MAE | XGB test MAE | Position-mean baseline test MAE |
|---|---|---|---|---|---|---|
| QB | rf | 6.050 | 6.312 | 6.133 | 6.403 | 7.553 |
| RB | rf | 4.586 | 4.703 | 4.502 | 4.652 | 6.549 |
| WR | rf | 4.578 | 4.613 | 4.605 | 4.700 | 6.294 |
| TE | rf | 3.552 | 3.705 | 3.644 | 3.614 | 4.706 |

Keys: `positions[pos].candidates[cand].val_mae / test_mae`, `positions[pos].baseline_position_mean.test_mae`.
Sample sizes (`positions[pos].n_train / n_val / n_test`): QB 2376/651/653, RB 5760/1445/1504, WR 9031/2429/2391, TE 4466/1162/1199.

## Draft tiers — 20260911-tiers_prevseason_v1-2024

Preseason GMM tiers from prior-season aggregates only (`ffai/models/tiers.py`; StandardScaler -> PCA (>=90% variance) -> GaussianMixture(full, n_init=5), n_components by BIC in [4, 12]; tiers ordered by component mean prior PPG).
Evaluated on the 2024 season (`artifacts/tiers/20260911-tiers_prevseason_v1-2024/metadata.json: evaluation`):

| Position | Components (BIC) | PCA dims | Players | Spearman(tier rank, realised PPR/game) | Share within tier rank band |
|---|---|---|---|---|---|
| QB | 10 | 3 | 60 | 0.607 | 29.6% |
| RB | 4 | 3 | 118 | 0.654 | 53.1% |
| WR | 4 | 3 | 190 | 0.617 | 47.8% |
| TE | 5 | 3 | 99 | 0.761 | 61.9% |

## Intended use and limitations

* Weekly start/sit and ranking context for season-long PPR/half/standard leagues. Not for betting.
* Features are the player's own recent production only: no opponent, injury, weather, depth
  chart, Vegas line, or snap data (`asof_v1`). Players with no prior stat row get no prediction.
* Errors are large relative to weekly scores (see MAE above); the 10th/90th residual quantiles
  give an 80% empirical interval on the validation season, not a guarantee.
* Trained on 2019, 2020, 2021, 2022; distribution drift is monitored weekly with PSI
  (`ffai/eval/drift.py`) but the model is not retrained automatically.