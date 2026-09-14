-- The claim-discipline rule in dbt: for every committed evaluation artifact, the n-weighted
-- aggregate of fct_weekly_eval over the artifact's window (its season, model version, and the
-- candidate it evaluated per position) must match the published n, MAE, within-3 and within-5
-- to 1e-4 (the artifacts are rounded to 4 dp). Any row returned is a disagreement and fails
-- the build. Baseline MAE is reconciled separately (needs the full stats history).
{% set tol = 0.0001 %}

with published as (
    select
        eval_id,
        kind || ':' || season as eval_window,
        model_version,
        cohort,
        candidate,
        n,
        mae,
        within_3_rate,
        within_5_rate
    from {{ ref('slv_eval_metrics') }}
),

-- the candidate evaluated for each position comes from the artifact itself
by_position as (
    select
        p.eval_id,
        p.cohort,
        sum(w.n) as n,
        sum(w.mae * w.n) / sum(w.n) as mae,
        sum(w.within_3_rate * w.n) / sum(w.n) as within_3_rate,
        sum(w.within_5_rate * w.n) / sum(w.n) as within_5_rate
    from published as p
    inner join {{ ref('fct_weekly_eval') }} as w
        on
            p.eval_window = w.eval_window
            and p.model_version = w.model_version
            and p.cohort = w.cohort
            and p.candidate = w.candidate
    where p.cohort <> 'ALL'
    group by 1, 2
),

overall as (
    select
        eval_id,
        'ALL' as cohort,
        sum(n) as n,
        sum(mae * n) / sum(n) as mae,
        sum(within_3_rate * n) / sum(n) as within_3_rate,
        sum(within_5_rate * n) / sum(n) as within_5_rate
    from by_position
    group by 1
),

computed as (
    select * from by_position
    union all
    select * from overall
),

compared as (
    select
        p.eval_id,
        p.cohort,
        p.n as published_n,
        c.n as computed_n,
        p.mae as published_mae,
        c.mae as computed_mae,
        p.within_3_rate as published_within_3,
        c.within_3_rate as computed_within_3,
        p.within_5_rate as published_within_5,
        c.within_5_rate as computed_within_5
    from published as p
    left join computed as c on p.eval_id = c.eval_id and p.cohort = c.cohort
)

select *
from compared
where
    computed_n is null
    or published_n <> computed_n
    or abs(published_mae - computed_mae) > {{ tol }}
    or abs(published_within_3 - computed_within_3) > {{ tol }}
    or abs(published_within_5 - computed_within_5) > {{ tol }}
