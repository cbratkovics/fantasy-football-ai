-- Grain: one row per (eval_id, season, week): the rolling-origin folds of each evaluation
-- artifact (a model refit on everything strictly before the week, scored on that week).
with artifacts as (
    select * from {{ ref('brz_eval_artifacts') }}
)

select
    a.eval_id,
    a.kind,
    a.model.version as model_version,
    a.rolling_origin.candidate as candidate,
    a.rolling_origin.strategy as strategy,
    cast(f.season as integer) as season,
    cast(f.week as integer) as week,
    cast(f.n as integer) as n,
    f.mae,
    f.baseline_mae,
    a.rolling_origin.mean_mae as window_mean_mae,
    a.rolling_origin.mean_baseline_mae as window_mean_baseline_mae
from artifacts as a, unnest(a.rolling_origin.folds) as u (f)
