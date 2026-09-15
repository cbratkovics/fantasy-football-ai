-- Grain: one row per player_id — the current version from the snp_player snapshot (SCD2,
-- ADR-0024). Same attributes as dim_player plus when this version was first captured, so a
-- downstream mart can choose between "attributes now" (this view) and "attributes as they were
-- when the period was scored" (dim_player_asof). Not exported to artifacts/marts (meta export=false).
{{ config(materialized='view', meta={'export': false}) }}

select
    cast(player_id as varchar) as player_id,
    cast(player_display_name as varchar) as player_display_name,
    cast(position as varchar) as position,
    cast(team as varchar) as team,
    cast(as_of_period_key as integer) as version_from_period_key,
    cast(dbt_valid_from as timestamp) as version_valid_from_utc
from {{ ref('snp_player') }}
where dbt_valid_to is null
