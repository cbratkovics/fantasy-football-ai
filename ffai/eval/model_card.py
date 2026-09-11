"""Render ``docs/MODEL_CARD.md`` from committed artifacts.

Every number in the card is read from ``artifacts/models/<version>/metadata.json``,
``artifacts/eval/<eval_id>.json`` (and, if present, ``artifacts/tiers/<version>/metadata.json``
and ``artifacts/eval/rolling_<season>.json``). The card lists the JSON key next to each figure
so a reader can trace it. Nothing is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment

from ffai.config import ARTIFACTS_DIR, REPO_ROOT

TEMPLATE = """# Model card — {{ meta.model_version }}

_Generated {{ eval.generated_at_utc }} from committed artifacts; do not edit by hand. Regenerate with
`python scripts/evaluate.py`._

| Field | Value | Source key |
|---|---|---|
| Model version | `{{ meta.model_version }}` | `models/{{ meta.model_version }}/metadata.json: model_version` |
| Feature version | `{{ meta.feature_version }}` | `metadata.json: feature_version` |
| Champion candidate | {{ champion_summary }} | `metadata.json: positions[*].champion` |
| Target | {{ meta.target }} | `metadata.json: target` |
| Data | nflverse weekly player stats via `{{ meta.data_library }}`, {{ meta.input_rows }} rows, sha256 `{{ meta.input_sha256[:12] }}…` | `metadata.json: data_library, input_rows, input_sha256` |
| Data through | season {{ meta.data_through.season }}, week {{ meta.data_through.week }} | `metadata.json: data_through` |
| Split | train {{ meta.seasons.train | join(', ') }} · validation {{ meta.seasons.val }} · test {{ meta.seasons.test }} (whole seasons, forward in time) | `metadata.json: seasons` |
| Trained at | {{ meta.trained_at_utc }} | `metadata.json: trained_at_utc` |
| Code commit (training) | `{{ meta.code_commit }}` | `metadata.json: code_commit` |
| Code commit (evaluation) | `{{ eval.code_commit }}` | `eval/{{ eval.eval_id }}.json: code_commit` |
| Libraries | scikit-learn {{ meta.sklearn_version }}{% if meta.xgboost_version %}, xgboost {{ meta.xgboost_version }}{% endif %} | `metadata.json: sklearn_version, xgboost_version` |
| Interval method | {{ meta.interval_method }} | `metadata.json: interval_method` |

## What it predicts

Regular-season PPR fantasy points for the upcoming game of a QB/RB/WR/TE, using only that
player's stat rows from strictly earlier weeks (`ffai/features/asof.py`, {{ n_features }} features,
per position QB {{ meta.positions.QB.n_features }} / RB {{ meta.positions.RB.n_features }} / WR {{ meta.positions.WR.n_features }} / TE {{ meta.positions.TE.n_features }}).
Standard and half-PPR points are derived by rules, not multipliers (`ffai/scoring.py`).

## Frozen test-season evaluation ({{ eval.split.test_start_season }}, champion per position)

Metric definitions: {{ eval.metric_definitions.mae }} (`mae`); {{ eval.metric_definitions.within_3_rate }}
(`within_3_rate`). Baseline: {{ eval.metric_definitions.baseline }}.
Source: `artifacts/eval/{{ eval.eval_id }}.json`.

| Cohort | n | MAE | Median AE | RMSE | Within ±3 | Within ±5 | Baseline MAE | Baseline within ±3 |
|---|---|---|---|---|---|---|---|---|
| All | {{ eval.metrics.n }} | {{ eval.metrics.mae }} | {{ eval.metrics.median_ae }} | {{ eval.metrics.rmse }} | {{ pct(eval.metrics.within_3_rate) }} | {{ pct(eval.metrics.within_5_rate) }} | {{ eval.baseline.mae }} | {{ pct(eval.baseline.within_3_rate) }} |
{% for pos, c in eval.cohorts.items() -%}
| {{ pos }} | {{ c.n }} | {{ c.mae }} | {{ c.median_ae }} | {{ c.rmse }} | {{ pct(c.within_3_rate) }} | {{ pct(c.within_5_rate) }} | {{ c.baseline.mae }} | {{ pct(c.baseline.within_3_rate) }} |
{% endfor %}
Keys: `metrics.*`, `cohorts[pos].*`, `baseline.*`, `cohorts[pos].baseline.*`.

## Candidate comparison (validation season {{ meta.seasons.val }} selects the champion)

| Position | Champion | RF val MAE | XGB val MAE | RF test MAE | XGB test MAE | Position-mean baseline test MAE |
|---|---|---|---|---|---|---|
{% for pos, p in meta.positions.items() -%}
| {{ pos }} | {{ p.champion }} | {{ '%.3f' | format(p.candidates.rf.val_mae) }} | {{ '%.3f' | format(p.candidates.xgb.val_mae) }} | {{ '%.3f' | format(p.candidates.rf.test_mae) }} | {{ '%.3f' | format(p.candidates.xgb.test_mae) }} | {{ '%.3f' | format(p.baseline_position_mean.test_mae) }} |
{% endfor %}
Keys: `positions[pos].candidates[cand].val_mae / test_mae`, `positions[pos].baseline_position_mean.test_mae`.
Sample sizes (`positions[pos].n_train / n_val / n_test`): {% for pos, p in meta.positions.items() %}{{ pos }} {{ p.n_train }}/{{ p.n_val }}/{{ p.n_test }}{{ ", " if not loop.last }}{% endfor %}.

{% if eval.rolling_origin -%}
## Rolling-origin evaluation ({{ eval.rolling_origin.candidate }}, season {{ eval.split.test_start_season }})

{{ eval.rolling_origin.strategy }}. Mean fold MAE **{{ eval.rolling_origin.mean_mae }}** vs baseline
**{{ eval.rolling_origin.mean_baseline_mae }}** over {{ eval.rolling_origin.folds | length }} folds
(`rolling_origin.mean_mae`, `rolling_origin.mean_baseline_mae`).

| Week | n | MAE | Baseline MAE |
|---|---|---|---|
{% for f in eval.rolling_origin.folds -%}
| {{ f.week }} | {{ f.n }} | {{ f.mae }} | {{ f.baseline_mae }} |
{% endfor %}
{% endif -%}

{% if tiers -%}
## Draft tiers — {{ tiers.tier_version }}

Preseason GMM tiers from prior-season aggregates only (`ffai/models/tiers.py`). Evaluated on the
{{ tiers.evaluation.season }} season (`artifacts/tiers/{{ tiers.tier_version }}/metadata.json: evaluation`):

| Position | Components (BIC) | PCA dims | Players | Spearman(tier rank, realised PPR/game) | Share within tier rank band |
|---|---|---|---|---|---|
{% for pos, t in tiers.positions.items() -%}
| {{ pos }} | {{ t.n_components }} | {{ t.pca_components }} | {{ t.n_players }} | {{ '%.3f' | format(tiers.evaluation.positions[pos].spearman) }} | {{ pct(tiers.evaluation.positions[pos].within_band_rate) }} |
{% endfor %}
{% endif -%}

## Intended use and limitations

* Weekly start/sit and ranking context for season-long PPR/half/standard leagues. Not for betting.
* Features are the player's own recent production only: no opponent, injury, weather, depth
  chart, Vegas line, or snap data (`asof_v1`). Players with no prior stat row get no prediction.
* Errors are large relative to weekly scores (see MAE above); the 10th/90th residual quantiles
  give an 80% empirical interval on the validation season, not a guarantee.
* Trained on {{ meta.seasons.train | join(', ') }}; distribution drift is monitored weekly with PSI
  (`ffai/eval/drift.py`) but the model is not retrained automatically.
"""


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def render(
    meta: dict[str, Any],
    eval_artifact: dict[str, Any],
    tiers_meta: dict[str, Any] | None = None,
) -> str:
    env = Environment(autoescape=False, trim_blocks=False, lstrip_blocks=False)
    env.globals["pct"] = _pct
    champions = {p: v["champion"] for p, v in meta["positions"].items()}
    if len(set(champions.values())) == 1:
        champion_summary = f"`{next(iter(champions.values()))}` at every position"
    else:
        champion_summary = ", ".join(f"{p}: `{c}`" for p, c in champions.items())
    n_features = len({f for p in meta["positions"].values() for f in p["features"]})
    return env.from_string(TEMPLATE).render(
        meta=meta,
        eval=eval_artifact,
        tiers=tiers_meta,
        champion_summary=champion_summary,
        n_features=n_features,
    )


def write_model_card(
    model_version: str,
    eval_id: str,
    *,
    tiers_version: str | None = None,
    artifacts: Path = ARTIFACTS_DIR,
    out_path: Path = REPO_ROOT / "docs" / "MODEL_CARD.md",
) -> Path:
    meta = json.loads((artifacts / "models" / model_version / "metadata.json").read_text())
    ev = json.loads((artifacts / "eval" / f"{eval_id}.json").read_text())
    tiers = None
    if tiers_version:
        tiers = json.loads((artifacts / "tiers" / tiers_version / "metadata.json").read_text())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(meta, ev, tiers), encoding="utf-8")
    return out_path
