-- Grain: one row per (eval_id, cohort). cohort = 'ALL' plus one row per position. The numbers
-- here are the committed evaluation artifacts verbatim (rounded to 4 dp as published); the gold
-- reconciliation tests compare fct_player_week / fct_weekly_eval against this table.
with artifacts as (
    select * from {{ ref('brz_eval_artifacts') }}
),

cohorts as (
    select
        a.eval_id,
        'ALL' as cohort,
        cast(null as varchar) as candidate,
        a.metrics.n as n,
        a.metrics.mae as mae,
        a.metrics.median_ae as median_ae,
        a.metrics.rmse as rmse,
        a.metrics.within_3_rate as within_3_rate,
        a.metrics.within_5_rate as within_5_rate,
        a.baseline.n as baseline_n,
        a.baseline.mae as baseline_mae,
        a.baseline.median_ae as baseline_median_ae,
        a.baseline.rmse as baseline_rmse,
        a.baseline.within_3_rate as baseline_within_3_rate,
        a.baseline.within_5_rate as baseline_within_5_rate
    from artifacts as a
    {% for pos in ['QB', 'RB', 'WR', 'TE'] %}
    union all
    select
        a.eval_id,
        '{{ pos }}' as cohort,
        a.model.candidate.{{ pos }} as candidate,
        a.cohorts.{{ pos }}.n as n,
        a.cohorts.{{ pos }}.mae as mae,
        a.cohorts.{{ pos }}.median_ae as median_ae,
        a.cohorts.{{ pos }}.rmse as rmse,
        a.cohorts.{{ pos }}.within_3_rate as within_3_rate,
        a.cohorts.{{ pos }}.within_5_rate as within_5_rate,
        a.cohorts.{{ pos }}.baseline.n as baseline_n,
        a.cohorts.{{ pos }}.baseline.mae as baseline_mae,
        a.cohorts.{{ pos }}.baseline.median_ae as baseline_median_ae,
        a.cohorts.{{ pos }}.baseline.rmse as baseline_rmse,
        a.cohorts.{{ pos }}.baseline.within_3_rate as baseline_within_3_rate,
        a.cohorts.{{ pos }}.baseline.within_5_rate as baseline_within_5_rate
    from artifacts as a
    {% endfor %}
)

select
    a.eval_id,
    a.kind,
    a.season,
    a.model.version as model_version,
    a.model.feature_version,
    a.code_commit,
    a.generated_at_utc,
    a.input.path as input_path,
    a.input.sha256 as input_sha256,
    a.baseline.name as baseline_name,
    c.cohort,
    c.candidate,
    cast(c.n as integer) as n,
    c.mae,
    c.median_ae,
    c.rmse,
    c.within_3_rate,
    c.within_5_rate,
    cast(c.baseline_n as integer) as baseline_n,
    c.baseline_mae,
    c.baseline_median_ae,
    c.baseline_rmse,
    c.baseline_within_3_rate,
    c.baseline_within_5_rate
from cohorts as c
inner join artifacts as a on c.eval_id = a.eval_id
