-- dim_player_asof must resolve every period after the snapshot's first capture exactly from
-- history (ADR-0024): for any player present in snp_player, a prediction period later than the
-- earliest as_of_period_key in the snapshot has a matching version, so is_exact_asof is true.
-- Rows returned are periods that should have resolved from history but fell back to the current
-- record.
with first_capture as (
    select min(as_of_period_key) as first_period_key from {{ ref('snp_player') }}
)

select
    a.player_id,
    a.period_key,
    a.is_exact_asof,
    f.first_period_key
from {{ ref('dim_player_asof') }} as a
cross join first_capture as f
where
    a.period_key > f.first_period_key
    and not a.is_exact_asof
    and a.player_id in (select s.player_id from {{ ref('snp_player') }} as s)
