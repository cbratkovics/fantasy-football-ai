-- Grain: one row per (player_id, period_key) for every period a prediction exists for. Resolves
-- the player's attributes as they were when that period was scored, from the snp_player SCD2
-- history (ADR-0024): a snapshot version captured with data through period K (as_of_period_key)
-- is the state used to score periods K+1 .. K' where K' is the next version's capture period.
-- Periods before the first capture, and players captured later than the period, cannot be
-- resolved from history; they fall back to the current version and is_exact_asof is false.
-- tests/gold/assert_asof_exact_after_first_capture.sql proves is_exact_asof is true for every
-- period after the snapshot's first capture. Not exported to artifacts/marts (meta export=false).
{{ config(materialized='view', meta={'export': false}) }}

with periods as (
    select distinct
        player_id,
        period_key
    from {{ ref('slv_predictions') }}
),

history as (
    select
        player_id,
        player_display_name,
        position,
        team,
        as_of_period_key as valid_from_period_key,
        lead(as_of_period_key) over (partition by player_id order by dbt_valid_from) as valid_to_period_key,
        dbt_valid_from
    from {{ ref('snp_player') }}
),

matched as (
    select
        p.player_id,
        p.period_key,
        h.player_display_name,
        h.position,
        h.team,
        h.valid_from_period_key,
        h.dbt_valid_from
    from periods as p
    left join history as h
        on
            p.player_id = h.player_id
            and p.period_key > h.valid_from_period_key
            and (h.valid_to_period_key is null or p.period_key <= h.valid_to_period_key)
)

select
    cast(m.player_id as varchar) as player_id,
    cast(m.period_key as integer) as period_key,
    cast(coalesce(m.player_display_name, c.player_display_name) as varchar) as player_display_name,
    cast(coalesce(m.position, c.position) as varchar) as position,
    cast(coalesce(m.team, c.team) as varchar) as team,
    cast(m.valid_from_period_key is not null as boolean) as is_exact_asof,
    cast(coalesce(m.valid_from_period_key, c.version_from_period_key) as integer) as version_from_period_key,
    cast(coalesce(m.dbt_valid_from, c.version_valid_from_utc) as timestamp) as version_valid_from_utc
from matched as m
left join {{ ref('dim_player_current') }} as c on m.player_id = c.player_id
