-- Grain: one row per (tiers_version, position, player_id): the preseason tier and rank next to
-- what the player then did in the tier's season (PPR per game from the rules macro), and the
-- realised rank among the tiered players of that position. Players with no game that season
-- keep null outcomes.
with tiers as (
    select * from {{ ref('slv_tiers') }}
),

season_totals as (
    select
        player_id,
        season,
        count(*) as games,
        sum(points_ppr) as ppr_total,
        avg(points_ppr) as ppr_per_game
    from {{ ref('slv_player_stats') }}
    group by 1, 2
),

joined as (
    select
        t.tiers_version,
        t.season,
        t.position,
        t.player_id,
        t.player_display_name,
        t.team,
        t.tier,
        t.tier_rank,
        t.tier_probability,
        t.ppg_prev,
        t.games_prev,
        s.games,
        s.ppr_total,
        s.ppr_per_game,
        case when s.games is not null then
            rank() over (
                partition by t.tiers_version, t.position, (s.games is not null)
                order by s.ppr_per_game desc, t.player_id
            )
        end as realized_rank
    from tiers as t
    left join season_totals as s on t.player_id = s.player_id and t.season = s.season
),

tier_sizes as (
    select tiers_version, position, count(*) as n_tiered from tiers group by 1, 2
)

select
    cast(j.tiers_version as varchar) as tiers_version,
    cast(j.season as integer) as season,
    cast(j.position as varchar) as position,
    cast(j.player_id as varchar) as player_id,
    cast(j.player_display_name as varchar) as player_display_name,
    cast(j.team as varchar) as team,
    cast(j.tier as integer) as tier,
    cast(j.tier_rank as integer) as tier_rank,
    cast(j.tier_probability as double) as tier_probability,
    cast(j.ppg_prev as double) as ppg_prev,
    cast(j.games_prev as integer) as games_prev,
    cast(j.games as integer) as games,
    cast(j.ppr_total as double) as ppr_total,
    cast(j.ppr_per_game as double) as ppr_per_game,
    cast(j.realized_rank as integer) as realized_rank,
    cast(j.tier_rank - j.realized_rank as integer) as rank_delta,
    cast(ts.n_tiered as integer) as n_tiered
from joined as j
inner join tier_sizes as ts on j.tiers_version = ts.tiers_version and j.position = ts.position
