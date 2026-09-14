-- Grain: one row per (player_id, season, week, model_version, candidate). Every prediction the
-- project has ever recorded, from three artifact families:
--   frozen_test           models/<v>/test_predictions.csv   (scored at training time)
--   out_of_sample_season  models/<v>/oos_predictions_*.csv  (frozen artifact, later season)
--   weekly                predictions/<season>/week_*.json  (the live weekly job)
--
-- Deduplication rule: the families should not overlap, but if the same key appears twice the
-- weekly file wins, then out-of-sample, then frozen test; within a family the latest
-- generated_at_utc wins, then the greatest source_file path. Deterministic and documented so a
-- re-scored week can never silently double-count.
--
-- eval_window / window_start_key: the causal baseline (fct_player_week) treats each frozen-test
-- and out-of-sample file as one evaluation window starting at week 1 of its season, and each
-- weekly file as its own one-week window — exactly how ffai.eval.evaluator was run.
with unioned as (
    select
        'frozen_test' as source,
        source_file,
        model_version,
        candidate,
        player_id,
        season,
        week,
        position,
        player_display_name,
        team,
        prediction,
        prediction_floor,
        prediction_ceiling,
        cast(null as double) as receptions_estimate,
        actual as actual_recorded,
        cast(null as timestamp) as generated_at_utc,
        3 as source_priority
    from {{ ref('brz_predictions_test') }}

    union all

    select
        'out_of_sample_season' as source,
        source_file,
        model_version,
        candidate,
        player_id,
        season,
        week,
        position,
        player_display_name,
        team,
        prediction,
        prediction_floor,
        prediction_ceiling,
        cast(null as double) as receptions_estimate,
        actual as actual_recorded,
        cast(null as timestamp) as generated_at_utc,
        2 as source_priority
    from {{ ref('brz_predictions_oos') }}

    union all

    select
        'weekly' as source,
        source_file,
        model_version,
        candidate,
        player_id,
        season,
        week,
        position,
        player_display_name,
        team,
        prediction,
        prediction_floor,
        prediction_ceiling,
        receptions_estimate,
        actual as actual_recorded,
        generated_at_utc,
        1 as source_priority
    from {{ ref('brz_predictions_weekly') }}
),

ranked as (
    select
        *,
        row_number() over (
            partition by player_id, season, week, model_version, candidate
            order by source_priority asc, generated_at_utc desc nulls last, source_file desc
        ) as dedup_rank
    from unioned
)

select
    player_id,
    season,
    week,
    season * 100 + week as period_key,
    model_version,
    candidate,
    position,
    player_display_name,
    team,
    prediction,
    prediction_floor,
    prediction_ceiling,
    receptions_estimate,
    actual_recorded,
    source,
    source_file,
    generated_at_utc,
    case
        when source = 'weekly' then 'weekly:' || season || '-' || lpad(cast(week as varchar), 2, '0')
        else source || ':' || season
    end as eval_window,
    case when source = 'weekly' then season * 100 + week else season * 100 + 1 end as window_start_key
from ranked
where dedup_rank = 1
