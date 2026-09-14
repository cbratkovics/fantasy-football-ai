-- Freshness contract (HOLD-grade): the stats must reach the expected complete (season, week)
-- the weekly job passes as vars (expected_season / expected_week). When the vars are unset the
-- test still refs the model (so it stays in the silver selection) and passes.
{% set season = var('expected_season') %}
{% set week = var('expected_week') %}

with latest as (
    select max(period_key) as latest_period_key from {{ ref('slv_player_stats') }}
)

select
    latest_period_key,
    {% if season is none or week is none %}
    cast(null as integer) as expected_period_key
from latest
where false
    {% else %}
    {{ season }} * 100 + {{ week }} as expected_period_key
from latest
where latest_period_key is null or latest_period_key < {{ season }} * 100 + {{ week }}
    {% endif %}
