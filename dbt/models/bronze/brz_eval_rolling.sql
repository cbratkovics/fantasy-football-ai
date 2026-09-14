{{ config(materialized='table') }}

-- One row per scored in-season week from artifacts/eval/rolling_<season>.json (zero rows while a
-- season has no scored week yet).
with files as (
    select * from {{ source('repo_files', 'eval_rolling') }}
)

select
    f.filename as source_file,
    cast(f.season as integer) as season,
    cast(w.week as integer) as week,
    cast(w.n as integer) as n,
    cast(w.champion.mae as double) as champion_mae,
    cast(w.champion.within_3_rate as double) as champion_within_3_rate,
    cast(w.champion.baseline_mae as double) as champion_baseline_mae,
    cast(w.challenger.mae as double) as challenger_mae,
    cast(w.challenger.within_3_rate as double) as challenger_within_3_rate,
    cast(w.challenger.baseline_mae as double) as challenger_baseline_mae,
    cast(w.scored_at_utc as timestamp) as scored_at_utc
from files as f, unnest(f.weeks) as u (w)
