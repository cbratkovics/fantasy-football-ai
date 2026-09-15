-- Grain: one row per (season, week, position, model_version, candidate, min_floor): the
-- policy-metrics mart of analytics/sql/risk_strategy.sql swept over var min_floor_grid, so the
-- API can answer "what if the floor threshold were X" without a rebuild. Only weeks with
-- realised outcomes contribute to the outcome columns; counts cover every scored decision.
--
-- v2 (ADR-0025): v1 plus two additive interval-outcome columns for recommended players:
-- recommendation_within_3_rate and recommendation_interval_coverage (actual inside floor..ceiling).
-- v1 (fct_decision_policy_v1.sql, relation alias fct_decision_policy) is what the API serves.
{% set grid = var('min_floor_grid') %}

with grid as (
    select unnest([{{ grid | join(', ') }}]) as min_floor
),

decisions as (
    select
        d.season,
        d.week,
        d.position,
        d.model_version,
        d.candidate,
        g.min_floor,
        d.prediction,
        d.prediction_floor,
        d.prediction_ceiling,
        d.replacement_level_points,
        d.actual,
        d.best_eligible_points,
        case when d.prediction_floor >= g.min_floor then 'recommend' else 'review' end as policy_action
    from {{ ref('fct_player_decisions') }} as d
    cross join grid as g
),

metrics as (
    select
        season,
        week,
        position,
        model_version,
        candidate,
        min_floor,
        count(*) as eligible_decisions,
        count(*) filter (where policy_action = 'recommend') as recommendations,
        count(*) filter (where policy_action = 'review') as reviews,
        count(actual) filter (where policy_action = 'recommend') as recommendations_with_outcome,
        avg(abs(prediction - actual)) filter (where policy_action = 'recommend') as recommendation_mae,
        avg(abs(prediction - actual)) filter (where policy_action = 'review') as review_mae,
        avg(best_eligible_points - actual) filter (where policy_action = 'recommend') as mean_regret,
        avg(case when actual >= replacement_level_points then 1.0 else 0.0 end)
        filter (where policy_action = 'recommend' and actual is not null and replacement_level_points is not null) as hit_rate,
        avg(case when actual < prediction_floor then 1.0 else 0.0 end)
        filter (where policy_action = 'recommend' and actual is not null) as downside_rate,
        avg(case when abs(prediction - actual) <= 3 then 1.0 else 0.0 end)
        filter (where policy_action = 'recommend' and actual is not null) as recommendation_within_3_rate,
        avg(case when prediction_floor <= actual and actual <= prediction_ceiling then 1.0 else 0.0 end)
        filter (where policy_action = 'recommend' and actual is not null) as recommendation_interval_coverage
    from decisions
    group by 1, 2, 3, 4, 5, 6
)

select
    cast(season as integer) as season,
    cast(week as integer) as week,
    cast(season * 100 + week as integer) as period_key,
    cast(position as varchar) as position,
    cast(model_version as varchar) as model_version,
    cast(candidate as varchar) as candidate,
    cast(min_floor as double) as min_floor,
    cast(eligible_decisions as integer) as eligible_decisions,
    cast(recommendations as integer) as recommendations,
    cast(reviews as integer) as reviews,
    cast(recommendations * 1.0 / nullif(eligible_decisions, 0) as double) as recommendation_rate,
    cast(reviews * 1.0 / nullif(eligible_decisions, 0) as double) as review_rate,
    cast(recommendations_with_outcome as integer) as recommendations_with_outcome,
    cast(recommendation_mae as double) as recommendation_mae,
    cast(review_mae as double) as review_mae,
    cast(mean_regret as double) as mean_regret,
    cast(hit_rate as double) as hit_rate,
    cast(downside_rate as double) as downside_rate,
    cast(recommendation_within_3_rate as double) as recommendation_within_3_rate,
    cast(recommendation_interval_coverage as double) as recommendation_interval_coverage
from metrics
