-- Grain: one row per (tiers_version, position, player_id). Deduplication rule: a player listed
-- in two tiers of the same version keeps the tier with the higher membership probability, then
-- the lower tier number.
with ranked as (
    select
        *,
        row_number() over (
            partition by tiers_version, position, player_id
            order by tier_probability desc, tier asc
        ) as dedup_rank
    from {{ ref('brz_tiers') }}
)

select
    tiers_version,
    season,
    position,
    player_id,
    player_display_name,
    team,
    tier,
    tier_probability,
    ppg_prev,
    games_prev,
    row_number() over (partition by tiers_version, position order by tier asc, ppg_prev desc, player_id asc) as tier_rank
from ranked
where dedup_rank = 1
