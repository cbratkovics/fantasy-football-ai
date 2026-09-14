{{ config(materialized='table') }}

-- Newest nflverse snapshot only. Cache files are named
-- player_stats_<first>-<last>_<load-date>.parquet, so the lexicographically greatest filename is
-- the widest season range at the latest load date. Columns: the id + raw stat columns the
-- package keeps (ffai.data.nflverse.ID_COLUMNS / STAT_COLUMNS), typed; nothing filtered.
with snapshots as (
    select * from {{ source('repo_files', 'player_stats') }}
),

newest as (
    select max(filename) as filename from snapshots
)

select
    s.filename as source_file,
    cast(s.player_id as varchar) as player_id,
    cast(s.player_display_name as varchar) as player_display_name,
    cast(s.position as varchar) as position,
    cast(s.season as integer) as season,
    cast(s.week as integer) as week,
    cast(s.season_type as varchar) as season_type,
    cast(s.team as varchar) as team,
    cast(s.opponent_team as varchar) as opponent_team,
{% for col in stat_columns() -%}
cast(s.{{ col }} as double) as {{ col }}{{ "," if not loop.last }}  -- noqa: LT02

{% endfor %}
from snapshots as s
inner join newest as n on s.filename = n.filename
