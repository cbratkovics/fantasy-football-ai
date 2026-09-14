{{ config(materialized='table') }}

-- One row per (tiers_version, position, tier, player) from every artifacts/tiers/*/tiers.json.
with files as (
    select * from {{ source('repo_files', 'tiers') }}
),

by_position as (
    {% for pos in ['QB', 'RB', 'WR', 'TE'] %}
    select
        f.filename,
        f.tier_version,
        f.season,
        '{{ pos }}' as position,
        t.tier,
        t.players
    from files as f, unnest(f.positions.{{ pos }}) as u (t)
    {{ "union all" if not loop.last }}
    {% endfor %}
)

select
    b.filename as source_file,
    cast(b.tier_version as varchar) as tiers_version,
    cast(b.season as integer) as season,
    b.position,
    cast(b.tier as integer) as tier,
    cast(p.player_id as varchar) as player_id,
    cast(p.name as varchar) as player_display_name,
    cast(p.team as varchar) as team,
    cast(p.ppg_prev as double) as ppg_prev,
    cast(p.games_prev as integer) as games_prev,
    cast(p.tier_probability as double) as tier_probability
from by_position as b, unnest(b.players) as u (p)
