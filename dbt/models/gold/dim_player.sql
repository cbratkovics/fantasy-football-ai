-- Grain: one row per player_id. Current (latest-seen) attributes, no history: the newest stats
-- row wins; players known only from a prediction file or a tiers file fall back to those.
with from_stats as (
    select
        player_id,
        player_display_name,
        position,
        team,
        min(season) over (partition by player_id) as first_season,
        season as last_season,
        week as last_week,
        count(*) over (partition by player_id) as games_played,
        row_number() over (partition by player_id order by period_key desc) as rn
    from {{ ref('slv_player_stats') }}
),

from_predictions as (
    select
        player_id,
        player_display_name,
        position,
        team,
        row_number() over (partition by player_id order by period_key desc, candidate asc) as rn
    from {{ ref('slv_predictions') }}
),

from_tiers as (
    select
        player_id,
        player_display_name,
        position,
        team,
        row_number() over (partition by player_id order by tiers_version desc) as rn
    from {{ ref('slv_tiers') }}
),

ids as (
    select player_id from from_stats
    union
    select player_id from from_predictions
    union
    select player_id from from_tiers
)

select
    cast(i.player_id as varchar) as player_id,
    cast(coalesce(s.player_display_name, p.player_display_name, t.player_display_name) as varchar) as player_display_name,
    cast(coalesce(s.position, p.position, t.position) as varchar) as position,
    cast(coalesce(s.team, p.team, t.team) as varchar) as team,
    cast(s.first_season as integer) as first_stats_season,
    cast(s.last_season as integer) as last_stats_season,
    cast(s.last_week as integer) as last_stats_week,
    cast(coalesce(s.games_played, 0) as integer) as games_played,
    cast(
        case
            when s.player_id is not null then 'stats'
            when p.player_id is not null then 'predictions'
            else 'tiers'
        end as varchar
    ) as attribute_source
from ids as i
left join from_stats as s on i.player_id = s.player_id and s.rn = 1
left join from_predictions as p on i.player_id = p.player_id and p.rn = 1
left join from_tiers as t on i.player_id = t.player_id and t.rn = 1
