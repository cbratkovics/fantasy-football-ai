-- Grain: one row per (player_id, season, week, model_version, candidate): the floor policy of
-- analytics/sql/risk_strategy.sql (now deleted, ADR-0016) applied to every scored player-week.
--
--   policy_action        recommend when prediction_floor >= min_floor (var), else review
--   replacement_level    the prediction of the rank-k player at the position that week
--                        (k = var replacement_rank: 12-team starters), i.e. what a free
--                        replacement was expected to score
--   best_eligible_points max realised points among the position's scored players that week
--   regret               best_eligible_points - actual (what the best alternative earned more)
--   hit                  actual >= replacement_level_points
--   downside             actual < prediction_floor (the floor was breached)
-- Outcome columns are null until the week has an actual.
{% set min_floor = var('min_floor') %}
{% set replacement_rank = var('replacement_rank') %}

with scored as (
    select
        *,
        row_number() over (
            partition by season, week, position, model_version, candidate
            order by prediction desc, player_id asc
        ) as prediction_rank,
        max(actual) over (partition by season, week, position, model_version, candidate) as best_eligible_points,
        case position
            {% for pos, k in replacement_rank.items() %}
            when '{{ pos }}' then {{ k }}
            {% endfor %}
        end as replacement_rank
    from {{ ref('fct_player_week') }}
),

replacement as (
    select
        season,
        week,
        position,
        model_version,
        candidate,
        prediction as replacement_level_points
    from scored
    where prediction_rank = replacement_rank
)

select
    cast(s.player_id as varchar) as player_id,
    cast(s.season as integer) as season,
    cast(s.week as integer) as week,
    cast(s.period_key as integer) as period_key,
    cast(s.model_version as varchar) as model_version,
    cast(s.candidate as varchar) as candidate,
    cast(s.position as varchar) as position,
    cast(s.eval_window as varchar) as eval_window,
    cast(s.prediction as double) as prediction,
    cast(s.prediction_floor as double) as prediction_floor,
    cast(s.prediction_ceiling as double) as prediction_ceiling,
    cast(s.prediction_rank as integer) as prediction_rank,
    cast({{ min_floor }} as double) as min_floor,
    cast(case when s.prediction_floor >= {{ min_floor }} then 'recommend' else 'review' end as varchar) as policy_action,
    cast(s.replacement_rank as integer) as replacement_rank,
    cast(r.replacement_level_points as double) as replacement_level_points,
    cast(s.actual as double) as actual,
    cast(s.best_eligible_points as double) as best_eligible_points,
    cast(s.best_eligible_points - s.actual as double) as regret,
    cast(case when s.actual is not null and r.replacement_level_points is not null then s.actual >= r.replacement_level_points end as boolean) as hit,
    cast(case when s.actual is not null then s.actual < s.prediction_floor end as boolean) as downside,
    cast(s.abs_error as double) as abs_error
from scored as s
left join replacement as r
    on
        s.season = r.season
        and s.week = r.week
        and s.position = r.position
        and s.model_version = r.model_version
        and s.candidate = r.candidate
