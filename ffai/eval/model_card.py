"""Render ``docs/MODEL_CARD.md`` from committed artifacts.

Every number in the card is read from ``artifacts/models/<version>/metadata.json``, the
evaluation artifacts listed in ``manifest.evaluations`` (``artifacts/eval/<eval_id>.json``) and,
if present, ``artifacts/tiers/<version>/metadata.json``. The card prints the JSON key next to each
figure so a reader can trace it. Nothing is typed by hand; the plain-English comparison paragraph
is composed from artifact values only (deltas, no adjectives).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment

from ffai.config import ARTIFACTS_DIR, REPO_ROOT
from ffai.models import registry

TEMPLATE = """# Model card — {{ meta.model_version }}

_Generated {{ generated_at }} from committed artifacts; do not edit by hand. Regenerate with
`python scripts/evaluate.py`._

| Field | Value | Source key |
|---|---|---|
| Model version | `{{ meta.model_version }}` | `models/{{ meta.model_version }}/metadata.json: model_version` |
| Feature version | `{{ meta.feature_version }}` | `metadata.json: feature_version` |
| Champion candidate | {{ champion_summary }} | `metadata.json: positions[*].champion` |
| Target | {{ meta.target }} | `metadata.json: target` |
| Data | nflverse weekly player stats via `{{ meta.data_library }}`, {{ meta.input_rows }} rows, sha256 `{{ meta.input_sha256[:12] }}…` | `metadata.json: data_library, input_rows, input_sha256` |
| Data through (training pull) | season {{ meta.data_through.season }}, week {{ meta.data_through.week }} | `metadata.json: data_through` |
| Split | train {{ meta.seasons.train | join(', ') }} · validation {{ meta.seasons.val }} · test {{ meta.seasons.test }} (whole seasons, forward in time) | `metadata.json: seasons` |
| Trained at | {{ meta.trained_at_utc }} | `metadata.json: trained_at_utc` |
| Code commit (training) | `{{ meta.code_commit }}` | `metadata.json: code_commit` |
| Libraries | scikit-learn {{ meta.sklearn_version }}{% if meta.xgboost_version %}, xgboost {{ meta.xgboost_version }}{% endif %} | `metadata.json: sklearn_version, xgboost_version` |
| Interval method | {{ meta.interval_method }} | `metadata.json: interval_method` |

## What it predicts

Regular-season PPR fantasy points for the upcoming game of a QB/RB/WR/TE, using only that
player's stat rows from strictly earlier weeks (`ffai/features/asof.py`, {{ n_features }} features,
per position QB {{ meta.positions.QB.n_features }} / RB {{ meta.positions.RB.n_features }} / WR {{ meta.positions.WR.n_features }} / TE {{ meta.positions.TE.n_features }}).
Standard and half-PPR points are derived by rules, not multipliers (`ffai/scoring.py`).

{% for ev in evals -%}
## {{ ev.title }} (season {{ ev.season }}, champion per position)

Kind: `{{ ev.kind }}` — {{ ev.kind_note }}
Metric definitions: {{ ev.metric_definitions.mae }} (`mae`); {{ ev.metric_definitions.within_3_rate }}
(`within_3_rate`). Baseline: {{ ev.metric_definitions.baseline }}.
Source: `artifacts/eval/{{ ev.eval_id }}.json` (input `{{ ev.input.path }}`, sha256 `{{ ev.input.sha256[:12] }}…`, evaluation commit `{{ ev.code_commit }}`).

| Cohort | n | MAE | Median AE | RMSE | Within ±3 | Within ±5 | Baseline MAE | Baseline within ±3 |
|---|---|---|---|---|---|---|---|---|
| All | {{ ev.metrics.n }} | {{ ev.metrics.mae }} | {{ ev.metrics.median_ae }} | {{ ev.metrics.rmse }} | {{ pct(ev.metrics.within_3_rate) }} | {{ pct(ev.metrics.within_5_rate) }} | {{ ev.baseline.mae }} | {{ pct(ev.baseline.within_3_rate) }} |
{% for pos, c in ev.cohorts.items() -%}
| {{ pos }} | {{ c.n }} | {{ c.mae }} | {{ c.median_ae }} | {{ c.rmse }} | {{ pct(c.within_3_rate) }} | {{ pct(c.within_5_rate) }} | {{ c.baseline.mae }} | {{ pct(c.baseline.within_3_rate) }} |
{% endfor %}
Keys: `metrics.*`, `cohorts[pos].*`, `baseline.*`, `cohorts[pos].baseline.*`.
{% if ev.rolling_origin %}
Rolling origin ({{ ev.rolling_origin.candidate }}; {{ ev.rolling_origin.strategy }}): mean fold MAE
**{{ ev.rolling_origin.mean_mae }}** vs baseline **{{ ev.rolling_origin.mean_baseline_mae }}** over
{{ ev.rolling_origin.folds | length }} folds (`rolling_origin.mean_mae`, `rolling_origin.mean_baseline_mae`).

| Week | n | MAE | Baseline MAE |
|---|---|---|---|
{% for f in ev.rolling_origin.folds -%}
| {{ f.week }} | {{ f.n }} | {{ f.mae }} | {{ f.baseline_mae }} |
{% endfor %}
{% endif %}
{% endfor -%}

{% if comparison -%}
## {{ comparison.title }}

{{ comparison.text }}

{% endif -%}
## Candidate comparison (validation season {{ meta.seasons.val }} selects the champion)

| Position | Champion | RF val MAE | XGB val MAE | RF test MAE | XGB test MAE | Position-mean baseline test MAE |
|---|---|---|---|---|---|---|
{% for pos, p in meta.positions.items() -%}
| {{ pos }} | {{ p.champion }} | {{ '%.3f' | format(p.candidates.rf.val_mae) }} | {{ '%.3f' | format(p.candidates.xgb.val_mae) }} | {{ '%.3f' | format(p.candidates.rf.test_mae) }} | {{ '%.3f' | format(p.candidates.xgb.test_mae) }} | {{ '%.3f' | format(p.baseline_position_mean.test_mae) }} |
{% endfor %}
Keys: `positions[pos].candidates[cand].val_mae / test_mae`, `positions[pos].baseline_position_mean.test_mae`.
Sample sizes (`positions[pos].n_train / n_val / n_test`): {% for pos, p in meta.positions.items() %}{{ pos }} {{ p.n_train }}/{{ p.n_val }}/{{ p.n_test }}{{ ", " if not loop.last }}{% endfor %}.

{% if tiers -%}
## Draft tiers — {{ tiers.tier_version }}

Preseason GMM tiers from prior-season aggregates only (`ffai/models/tiers.py`; {{ tiers.method }}).
Evaluated on the {{ tiers.evaluation.season }} season (`artifacts/tiers/{{ tiers.tier_version }}/metadata.json: evaluation`):

| Position | Components (BIC) | PCA dims | Players | Spearman(tier rank, realised PPR/game) | Share within tier rank band |
|---|---|---|---|---|---|
{% for pos, t in tiers.positions.items() -%}
| {{ pos }} | {{ t.n_components }} | {{ t.pca_components }} | {{ t.n_players }} | {{ '%.3f' | format(tiers.evaluation.positions[pos].spearman) }} | {{ pct(tiers.evaluation.positions[pos].within_band_rate) }} |
{% endfor %}
{% if tiers.comparison_to_previous -%}
Change versus `{{ tiers.comparison_to_previous.previous_tier_version }}` (`metadata.json: comparison_to_previous`):

| Position | Components before → after | Spearman before → after | Within band before → after | Kept |
|---|---|---|---|---|
{% for pos, c in tiers.comparison_to_previous.positions.items() -%}
| {{ pos }} | {{ c.before.n_components }} → {{ c.after.n_components }} | {{ '%.3f' | format(c.before.spearman) }} → {{ '%.3f' | format(c.after.spearman) }} | {{ pct(c.before.within_band_rate) }} → {{ pct(c.after.within_band_rate) }} | {{ c.kept }} |
{% endfor %}
{% endif -%}
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

KIND_TITLES = {
    "frozen_test": "Frozen test-season evaluation",
    "out_of_sample_season": "Out-of-sample season evaluation",
}
KIND_NOTES = {
    "frozen_test": "the season held out when the model was trained and selected.",
    "out_of_sample_season": (
        "a complete season that no training, validation, selection, or tuning decision ever "
        "touched; scored with the same frozen artifact and feature builder."
    ),
}


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _delta(a: float, b: float, unit: str = "") -> str:
    d = b - a
    sign = "+" if d >= 0 else "−"
    return f"{sign}{abs(d):.3f}{unit}"


def comparison_paragraph(evals: list[dict[str, Any]]) -> dict[str, str] | None:
    """Plain-English deltas between the frozen test and each out-of-sample season (no adjectives)."""
    frozen = next((e for e in evals if e["kind"] == "frozen_test"), None)
    oos = [e for e in evals if e["kind"] == "out_of_sample_season"]
    if frozen is None or not oos:
        return None
    parts = []
    for e in oos:
        fm, om = frozen["metrics"], e["metrics"]
        fb, ob = frozen["baseline"], e["baseline"]
        s = (
            f"On the {e['season']} season (n = {om['n']}) the frozen champion's MAE is {om['mae']} "
            f"versus {fm['mae']} on the {frozen['season']} test season ({_delta(fm['mae'], om['mae'])} points); "
            f"within ±3 is {_pct(om['within_3_rate'])} versus {_pct(fm['within_3_rate'])} "
            f"({_delta(100 * fm['within_3_rate'], 100 * om['within_3_rate'], ' pp')}). "
            f"The causal trailing-mean baseline moved from MAE {fb['mae']} to {ob['mae']} "
            f"({_delta(fb['mae'], ob['mae'])}), so the model-minus-baseline gap is "
            f"{fm['mae'] - fb['mae']:+.3f} in {frozen['season']} and {om['mae'] - ob['mae']:+.3f} in {e['season']}. "
        )
        per = []
        for pos in frozen["cohorts"]:
            if pos in e["cohorts"]:
                per.append(
                    f"{pos} {frozen['cohorts'][pos]['mae']} → {e['cohorts'][pos]['mae']} "
                    f"({_delta(frozen['cohorts'][pos]['mae'], e['cohorts'][pos]['mae'])})"
                )
        s += "Per position MAE: " + "; ".join(per) + ". "
        if frozen.get("rolling_origin") and e.get("rolling_origin"):
            fr, orr = frozen["rolling_origin"], e["rolling_origin"]
            s += (
                f"Rolling-origin mean MAE: {fr['mean_mae']} → {orr['mean_mae']} "
                f"({_delta(fr['mean_mae'], orr['mean_mae'])}); its baseline {fr['mean_baseline_mae']} → "
                f"{orr['mean_baseline_mae']}."
            )
        parts.append(s)
    return {
        "title": f"{frozen['season']} test season versus out-of-sample season"
        + ("s" if len(oos) > 1 else ""),
        "text": "\n\n".join(parts),
    }


def render(
    meta: dict[str, Any],
    eval_artifacts: list[dict[str, Any]],
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
    evals = []
    for ev in sorted(eval_artifacts, key=lambda e: (e.get("season") or 0, e["eval_id"])):
        kind = ev.get("kind", "frozen_test")
        evals.append(
            {
                **ev,
                "kind": kind,
                "title": KIND_TITLES.get(kind, kind),
                "kind_note": KIND_NOTES.get(kind, ""),
                "season": ev.get("season") or ev["split"]["test_start_season"],
            }
        )
    generated_at = max(e["generated_at_utc"] for e in evals) if evals else ""
    return env.from_string(TEMPLATE).render(
        meta=meta,
        evals=evals,
        comparison=comparison_paragraph(evals),
        tiers=tiers_meta,
        champion_summary=champion_summary,
        n_features=n_features,
        generated_at=generated_at,
    )


def write_model_card(
    manifest: dict[str, Any],
    *,
    artifacts: Path = ARTIFACTS_DIR,
    out_path: Path = REPO_ROOT / "docs" / "MODEL_CARD.md",
) -> Path:
    """Render the card for the manifest's champion, every registered evaluation, and tiers."""
    model_version, _ = registry.slot(manifest, "champion")
    meta = json.loads((artifacts / "models" / model_version / "metadata.json").read_text())
    evs = [
        json.loads((artifacts / e["path"]).read_text(encoding="utf-8"))
        for e in registry.evaluations(manifest)
    ]
    tiers = None
    if manifest.get("tiers_version"):
        tiers = json.loads(
            (artifacts / "tiers" / manifest["tiers_version"] / "metadata.json").read_text()
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(meta, evs, tiers), encoding="utf-8")
    return out_path
