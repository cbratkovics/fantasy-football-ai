{#- SCD Type 2 history of the player dimension (ADR-0024). `check` strategy on the attributes
    that actually change during a season (team on a trade, position on a role change, display
    name on a rename); a change closes the current row (dbt_valid_to) and opens a new one.
    as_of_period_key is the warehouse's data-through period at the time of the capture (the newest
    complete (season, week) in slv_player_stats); it is not a check column, so it records when each
    version was first seen. dim_player_current and dim_player_asof (gold views) read this table.
    History starts at the first prod run of this snapshot: earlier periods resolve to the current
    record with is_exact_asof = false. -#}
{% snapshot snp_player %}

{{
    config(
        schema='snapshots',
        unique_key='player_id',
        strategy='check',
        check_cols=['player_display_name', 'position', 'team'],
        hard_deletes='ignore'
    )
}}

select
    d.player_id,
    d.player_display_name,
    d.position,
    d.team,
    (select max(s.period_key) from {{ ref('slv_player_stats') }} as s) as as_of_period_key
from {{ ref('dim_player') }} as d

{% endsnapshot %}
