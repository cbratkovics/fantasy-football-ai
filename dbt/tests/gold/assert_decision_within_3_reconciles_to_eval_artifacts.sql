-- Claim discipline for the v2 decisions mart (ADR-0025): recommendation_within_3_rate is the same
-- "within ±3" metric the evaluation artifacts publish, so it must trace to them. At min_floor = 0
-- every scored player is recommended (floors are clipped at 0), so for every committed artifact
-- the outcome-weighted aggregate of fct_decision_policy v2 over the artifact's window, model
-- version, and per-position candidate must equal the published n and within_3_rate to 1e-4, per
-- position and overall. At stricter thresholds the column is a subset statistic of these same
-- reconciled rows. Any row returned is a disagreement and fails the build; the API may not move
-- to v2 while this test is absent or failing.
{% set tol = 0.0001 %}

with published as (
    select
        eval_id,
        kind || ':' || season as eval_window,
        model_version,
        cohort,
        candidate,
        n,
        within_3_rate
    from {{ ref('slv_eval_metrics') }}
),

windows as (
    select distinct
        eval_window,
        season,
        week,
        model_version,
        candidate
    from {{ ref('fct_weekly_eval') }}
),

inclusive as (
    select
        d.season,
        d.week,
        d.position,
        d.model_version,
        d.candidate,
        d.recommendations_with_outcome,
        d.recommendation_within_3_rate,
        w.eval_window
    from {{ ref('fct_decision_policy', v=2) }} as d
    inner join windows as w
        on
            d.season = w.season
            and d.week = w.week
            and d.model_version = w.model_version
            and d.candidate = w.candidate
    where d.min_floor = 0
),

by_position as (
    select
        p.eval_id,
        p.cohort,
        sum(i.recommendations_with_outcome) as n,
        sum(i.recommendation_within_3_rate * i.recommendations_with_outcome)
        / sum(i.recommendations_with_outcome) as within_3_rate
    from published as p
    inner join inclusive as i
        on
            p.eval_window = i.eval_window
            and p.model_version = i.model_version
            and p.cohort = i.position
            and p.candidate = i.candidate
    where p.cohort <> 'ALL'
    group by 1, 2
),

overall as (
    select
        eval_id,
        'ALL' as cohort,
        sum(n) as n,
        sum(within_3_rate * n) / sum(n) as within_3_rate
    from by_position
    group by 1
),

computed as (
    select * from by_position
    union all
    select * from overall
)

select
    p.eval_id,
    p.cohort,
    p.n as published_n,
    c.n as computed_n,
    p.within_3_rate as published_within_3,
    c.within_3_rate as computed_within_3
from published as p
left join computed as c on p.eval_id = c.eval_id and p.cohort = c.cohort
where
    c.n is null
    or p.n <> c.n
    or abs(p.within_3_rate - c.within_3_rate) > {{ tol }}
