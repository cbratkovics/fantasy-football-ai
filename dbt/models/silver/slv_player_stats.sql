-- Grain: one row per (player_id, season, week). Regular season, fantasy positions only.
--
-- Deduplication rule: nflverse publishes one row per player-week, but if a snapshot ever carried
-- two (a mid-week trade duplicate, a re-published row), the row with the higher PPR points is
-- kept, ties broken by team name, so the result is deterministic across builds.
--
-- Rows of the week being scored (var target_season / target_week) and anything later are dropped:
-- they are incomplete while the week is being played (ffai.pipeline.weekly step 2).
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
