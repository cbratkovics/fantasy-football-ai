-- Grain: one row per (eval_window, season, week, model_version, candidate, cohort), cohort =
-- 'ALL' or a position. Weekly evaluation computed in SQL from fct_player_week over rows with a
-- realised actual; metric definitions match the evaluation artifacts (mae = mean |actual -
-- prediction|, within_k = share with |error| <= k on the same rows). Weeks with no actual yet
-- produce no row.
with rows_with_actual as (
    select * from {{ ref('fct_player_week') }}
    where actual is not null
),

cohorts as (
    select
        'ALL' as cohort,
        *
    from rows_with_actual
    union all
    select
        position as cohort,
        *
    from rows_with_actual
)

select
    cast(eval_window as varchar) as eval_window,
    cast(season as integer) as season,
    cast(week as integer) as week,
    cast(period_key as integer) as period_key,
    cast(model_version as varchar) as model_version,
    cast(candidate as varchar) as candidate,
    cast(cohort as varchar) as cohort,
    cast(count(*) as integer) as n,
    cast(avg(abs_error) as double) as mae,
    cast(median(abs_error) as double) as median_ae,
    cast(sqrt(avg(abs_error * abs_error)) as double) as rmse,
    cast(avg(case when within_3 then 1.0 else 0.0 end) as double) as within_3_rate,
    cast(avg(case when within_5 then 1.0 else 0.0 end) as double) as within_5_rate,
    cast(avg(case when interval_hit then 1.0 else 0.0 end) as double) as interval_coverage,
    cast(avg(baseline_abs_error) as double) as baseline_mae,
    cast(avg(case when baseline_within_3 then 1.0 else 0.0 end) as double) as baseline_within_3_rate,
    cast(avg(abs_error) - avg(baseline_abs_error) as double) as mae_minus_baseline
from cohorts
group by 1, 2, 3, 4, 5, 6, 7
