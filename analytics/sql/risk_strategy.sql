-- Decision-performance mart for the portfolio dashboard.
-- Grain: one row per UTC decision date, position, model version, and policy version.
-- Assumes point-in-time-safe decision and post-event outcome facts.

with eligible_decisions as (
    select
        d.decision_id,
        cast(d.decision_at as date) as decision_date,
        d.position,
        d.model_version,
        d.policy_version,
        d.policy_action,
        d.predicted_points,
        d.prediction_floor,
        d.replacement_level_points,
        o.realized_points,
        max(o.realized_points) over (
            partition by d.league_id, d.season, d.week, d.lineup_slot
        ) as best_eligible_points
    from fct_player_decisions d
    left join fct_player_outcomes o
      on d.player_id = o.player_id
     and d.season = o.season
     and d.week = o.week
    where d.is_eligible
      and d.decision_at < d.event_start_at
),

policy_metrics as (
    select
        decision_date,
        position,
        model_version,
        policy_version,
        count(*) as eligible_decisions,
        sum(case when policy_action = 'recommend' then 1 else 0 end) as recommendations,
        sum(case when policy_action = 'review' then 1 else 0 end) as reviews,
        avg(case
            when policy_action = 'recommend' and realized_points is not null
            then abs(predicted_points - realized_points)
        end) as recommendation_mae,
        avg(case
            when policy_action = 'recommend' and realized_points is not null
            then best_eligible_points - realized_points
        end) as mean_regret,
        avg(case
            when policy_action = 'recommend' and realized_points is not null
            then case when realized_points >= replacement_level_points then 1.0 else 0.0 end
        end) as hit_rate,
        avg(case
            when policy_action = 'recommend' and realized_points is not null
            then case when realized_points < prediction_floor then 1.0 else 0.0 end
        end) as downside_rate
    from eligible_decisions
    group by 1, 2, 3, 4
)

select
    *,
    recommendations * 1.0 / nullif(eligible_decisions, 0) as recommendation_rate,
    reviews * 1.0 / nullif(eligible_decisions, 0) as review_rate
from policy_metrics;
