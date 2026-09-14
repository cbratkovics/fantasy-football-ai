-- Where a prediction artifact recorded an actual and the stats row is in the warehouse, the two
-- must agree to 0.01 (both are nflverse PPR points: one attached at scoring time, one computed
-- by the rules macro now). A mismatch would mean the artifact and the warehouse describe
-- different games.
select
    p.player_id,
    p.season,
    p.week,
    p.model_version,
    p.candidate,
    p.actual_recorded,
    a.actual as actual_from_stats
from {{ ref('slv_predictions') }} as p
inner join {{ ref('slv_actuals') }} as a
    on p.player_id = a.player_id
    and p.period_key = a.period_key
    and a.scoring_format = 'ppr'
where
    p.actual_recorded is not null
    and abs(p.actual_recorded - a.actual) > 0.01
