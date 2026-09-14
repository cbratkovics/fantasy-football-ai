{#- Contract: the newest period's row count is within `tolerance_pct` of the period before it.
    Replaces the Python newest_week_row_count band (200-600 rows) with a relative rule that also
    works on the small CI fixture. Passes trivially when fewer than two periods exist. The largest
    consecutive-week swing in nflverse 2019-2025 is 23 % (bye-heavy weeks), hence a 35 % default. -#}
{% test row_count_within_pct_of_prior_period(model, period_columns, tolerance_pct=0.35) %}

{%- set cols = period_columns | join(', ') -%}

with per_period as (
    select
        {{ cols }},
        count(*) as row_count
    from {{ model }}
    group by {{ cols }}
),

ordered as (
    select
        {{ cols }},
        row_count,
        lag(row_count) over (order by {{ cols }}) as prior_row_count,
        row_number() over (order by {{ cols }} desc) as recency
    from per_period
)

select
    {{ cols }},
    row_count,
    prior_row_count,
    abs(row_count - prior_row_count) * 1.0 / prior_row_count as change_pct,
    {{ tolerance_pct }} as tolerance_pct
from ordered
where
    recency = 1
    and prior_row_count is not null
    and abs(row_count - prior_row_count) * 1.0 / prior_row_count > {{ tolerance_pct }}

{% endtest %}
