-- Grain: one row per (player_id, season, week, scoring_format). Realised points under each
-- scoring format, computed by the same rules macro (never a multiplier of PPR).
{% for fmt in ['standard', 'half', 'ppr'] %}
select
    player_id,
    season,
    week,
    period_key,
    position,
    '{{ fmt }}' as scoring_format,
    points_{{ fmt }} as actual
from {{ ref('slv_player_stats') }}
{{ "union all" if not loop.last }}
{% endfor %}
