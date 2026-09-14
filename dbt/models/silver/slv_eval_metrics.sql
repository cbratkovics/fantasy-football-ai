-- Grain: one row per (eval_id, cohort). cohort = 'ALL' plus one row per position. The numbers
-- here are the committed evaluation artifacts verbatim (rounded to 4 dp as published); the gold
-- reconciliation tests compare fct_player_week / fct_weekly_eval against this table.
with artifacts as (
    select * from {{ ref('brz_eval_artifacts') }}
),

cohorts as (
    select
        eval_id,
        'ALL' as cohort,
        cast(null as varchar) as candidate,
        metrics.n as n,
        metrics.mae as mae,
        metrics.median_ae as median_ae,
        metrics.rmse as rmse,
        metrics.within_3_rate as within_3_rate,
        metrics.within_5_rate as within_5_rate,
        baseline.n as baseline_n,
        baseline.mae as baseline_mae,
        baseline.median_ae as baseline_median_ae,
        baseline.rmse as baseline_rmse,
        baseline.within_3_rate as baseline_within_3_rate,
        baseline.within_5_rate as baseline_within_5_rate
    from artifacts
    {% for pos in ['QB', 'RB', 'WR', 'TE'] %}
    union all
    select
        eval_id,
        '{{ pos }}' as cohort,
        model.candidate.{{ pos }} as candidate,
        cohorts.{{ pos }}.n,
        cohorts.{{ pos }}.mae,
        cohorts.{{ pos }}.median_ae,
        cohorts.{{ pos }}.rmse,
        cohorts.{{ pos }}.within_3_rate,
        cohorts.{{ pos }}.within_5_rate,
        cohorts.{{ pos }}.baseline.n,
        cohorts.{{ pos }}.baseline.mae,
        cohorts.{{ pos }}.baseline.median_ae,
        cohorts.{{ pos }}.baseline.rmse,
        cohorts.{{ pos }}.baseline.within_3_rate,
        cohorts.{{ pos }}.baseline.within_5_rate
    from artifacts
    {% endfor %}
)

select
    a.eval_id,
    a.kind,
    a.season,
    a.model.version as model_version,
    a.model.feature_version as feature_version,
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
