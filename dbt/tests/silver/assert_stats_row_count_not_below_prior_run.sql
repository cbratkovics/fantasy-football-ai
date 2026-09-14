-- Row-count monotonicity: the stats frame may only grow between weekly runs. The prior run's
-- row count comes from manifest.last_run.stats_rows via the prior_row_count var. Unset = pass.
{% set prior = var('prior_row_count') %}

select
    count(*) as row_count,
{% if prior is none %}
cast(null as integer) as prior_row_count
from {{ ref('slv_player_stats') }}
having false
{% else %}
    {{ prior }} as prior_row_count
from {{ ref('slv_player_stats') }}
having count(*) < {{ prior }}
    {% endif %}
