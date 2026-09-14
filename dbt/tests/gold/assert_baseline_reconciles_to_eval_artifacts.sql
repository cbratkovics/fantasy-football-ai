-- The SQL causal baseline must reproduce the evaluator's baseline MAE per artifact and cohort
-- to 1e-4, and its within-3 rate to within three rows. Rows, not 1e-4, because a trailing mean
-- whose error is exactly 3.0 is decided by floating-point rounding: the evaluator averages
-- exact rationals (statistics.mean) while DuckDB rounds fsum(x) / n, and the two can differ by
-- one ulp on such ties. Observed: one WR row of 2,391 in the 2024 artifact on local DuckDB, two
-- on MotherDuck (a different summation order). Any definitional error moves the rate by far
-- more than that. Needs the full stats history (the seed of every player's trailing mean), so
-- it is disabled when the warehouse is built from the CI fixture (var full_stats = false).
{{ config(enabled=var('full_stats')) }}
{% set tol = 0.0001 %}
{% set max_rows_off = 3 %}

with published as (
    select
        eval_id,
        kind || ':' || season as eval_window,
        model_version,
        cohort,
        candidate,
        baseline_mae,
        baseline_within_3_rate
    from {{ ref('slv_eval_metrics') }}
),

by_position as (
    select
        p.eval_id,
        p.cohort,
        sum(w.n) as n,
        sum(w.baseline_mae * w.n) / sum(w.n) as baseline_mae,
        sum(w.baseline_within_3_rate * w.n) / sum(w.n) as baseline_within_3_rate
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
        sum(baseline_mae * n) / sum(n) as baseline_mae,
        sum(baseline_within_3_rate * n) / sum(n) as baseline_within_3_rate
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
    c.n,
    p.baseline_mae as published_baseline_mae,
    c.baseline_mae as computed_baseline_mae,
    p.baseline_within_3_rate as published_baseline_within_3,
    c.baseline_within_3_rate as computed_baseline_within_3,
    abs(p.baseline_within_3_rate - c.baseline_within_3_rate) * c.n as within_3_rows_off
from published as p
left join computed as c on p.eval_id = c.eval_id and p.cohort = c.cohort
where
    c.baseline_mae is null
    or abs(p.baseline_mae - c.baseline_mae) > {{ tol }}
    or abs(p.baseline_within_3_rate - c.baseline_within_3_rate) * c.n > {{ max_rows_off }} + 0.5
