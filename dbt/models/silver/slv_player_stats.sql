-- Grain: one row per (player_id, season, week). Regular season, fantasy positions only.
--
-- Materialisation: incremental (delete+insert on the grain key). The largest table in the
-- warehouse and the only one whose transformation is local to its own grain (typing, the dedup
-- window below, and the rules macro all operate inside one (player_id, season, week) partition),
-- so appending by period is safe. fct_player_week is *not* incremental: its causal baseline is a
-- running window over a whole evaluation window. Bronze is a one-to-one copy of the newest
-- snapshot file, whose name changes with every pull. See ADR-0023.
--
-- Lookback: nflverse restates stats for earlier weeks (stat corrections), so an incremental run
-- reprocesses the newest `stats_lookback_periods` distinct periods already in the table (default
-- 4) plus anything newer, and delete+insert replaces those rows. A correction older than the
-- lookback is only picked up by a full refresh.
--
-- Full-refresh policy: `dbt build --full-refresh` (weekly.yml input full_refresh=true, or
-- FFAI_DBT_FULL_REFRESH=1 for the contracts wrapper) at the start of a season, after any change to
-- this model's SQL or columns (on_schema_change = fail makes a silent column change impossible),
-- or after a source restatement older than the lookback. CI always builds a fresh DuckDB file, so
-- every CI build is a full refresh; tests/test_dbt_incremental.py proves that a full refresh and
-- an incremental run over the same input produce identical rows.
--
-- Deduplication rule: nflverse publishes one row per player-week, but if a snapshot ever carried
-- two (a mid-week trade duplicate, a re-published row), the row with the higher PPR points is
-- kept, ties broken by team name, so the result is deterministic across builds.
--
-- Rows of the week being scored (var target_season / target_week) and anything later are dropped:
-- they are incomplete while the week is being played (ffai.pipeline.weekly step 2).
{{
    config(
        materialized='incremental',
        incremental_strategy='delete+insert',
        unique_key=['player_id', 'season', 'week'],
        on_schema_change='fail'
    )
}}

{% set target_season = var('target_season') %}
{% set target_week = var('target_week') %}

with in_scope as (
    select *
    from {{ ref('brz_player_stats') }}
    where
        season_type = 'REG'
        and position in ('QB', 'RB', 'WR', 'TE')
        {% if target_season is not none and target_week is not none %}
        and not (season = {{ target_season }} and week >= {{ target_week }})
        and season <= {{ target_season }}
        {% endif %}
        {% if is_incremental() %}
        -- reprocess the newest `stats_lookback_periods` periods already loaded, plus anything newer
        and season * 100 + week >= (
            select coalesce(min(loaded.period_key), 0)
            from (
                select distinct t.period_key
                from {{ this }} as t
                order by t.period_key desc
                limit {{ var('stats_lookback_periods') }}
            ) as loaded
        )
        {% endif %}
),

ranked as (
    select
        *,
        row_number() over (
            partition by player_id, season, week
            order by fantasy_points_ppr desc nulls last, team asc
        ) as dedup_rank
    from in_scope
)

select
    player_id,
    player_display_name,
    position,
    season,
    week,
    season * 100 + week as period_key,
    team,
    opponent_team,
    {% for col in stat_columns() -%}
    {{ col }},
    {% endfor -%}
    {{ fantasy_points('standard') }} as points_standard,
    {{ fantasy_points('half') }} as points_half,
    {{ fantasy_points('ppr') }} as points_ppr
from ranked
where dedup_rank = 1
