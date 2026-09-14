-- Grain: one row per (season, week): in-season champion vs challenger shadow evaluation written by
-- the weekly job (artifacts/eval/rolling_<season>.json).
select
    season,
    week,
    season * 100 + week as period_key,
    n,
    champion_mae,
    champion_within_3_rate,
    champion_baseline_mae,
    challenger_mae,
    challenger_within_3_rate,
    challenger_baseline_mae,
    scored_at_utc
from {{ ref('brz_eval_rolling') }}
