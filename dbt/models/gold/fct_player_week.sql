-- Grain: one row per (player_id, season, week, model_version, candidate): every recorded
-- prediction next to what happened, with the error metrics and the causal baseline.
--
-- actual: realised PPR points from slv_actuals (rules macro over nflverse stats); when the
--   stats row is not in the warehouse (CI fixture, or a player scored but absent from stats)
--   the value recorded in the prediction artifact is used and actual_source says so. Where
--   both exist they agree (tests/gold/assert_recorded_actuals_match_stats.sql).
-- baseline: the evaluator's causal trailing mean (ffai.eval.evaluator.add_causal_baseline),
--   reproduced in SQL (sums use fsum, DuckDB's exact-summation aggregate, because the evaluator
--   takes means as exact rationals and an error of exactly 3.0 must land on the same side): the mean of the player's realised points over (a) stats rows strictly
--   before the evaluation window started and (b) evaluated rows of the same window and
--   candidate from strictly earlier periods; falling back to the position's history under the
--   same rule; falling back to the prediction itself. Frozen-test and out-of-sample files are
--   one window each (from week 1 of their season); each weekly file is its own window.
{% set target_scoring_format = var('target_scoring_format') %}

with predictions as (
    select * from {{ ref('slv_predictions') }}
),

actual_by_format as (
    select
        player_id,
        period_key,
        position,
        max(case when scoring_format = 'ppr' then actual end) as actual_ppr,
        max(case when scoring_format = 'half' then actual end) as actual_half,
        max(case when scoring_format = 'standard' then actual end) as actual_standard
    from {{ ref('slv_actuals') }}
    group by 1, 2, 3
),

windows as (
    select distinct
        eval_window,
        window_start_key
    from predictions
),

-- (a) seeds: everything realised strictly before each window began, per player and per position
seed_player as (
    select
        w.eval_window,
        a.player_id,
        fsum(a.actual_ppr) as seed_sum,
        count(a.actual_ppr) as seed_cnt
    from windows as w
    inner join actual_by_format as a on w.window_start_key > a.period_key
    group by 1, 2
),

seed_position as (
    select
        w.eval_window,
        a.position,
        fsum(a.actual_ppr) as seed_sum,
        count(a.actual_ppr) as seed_cnt
    from windows as w
    inner join actual_by_format as a on w.window_start_key > a.period_key
    group by 1, 2
),

joined as (
    select
        p.*,
        a.actual_ppr as actual_from_stats,
        a.actual_half,
        a.actual_standard,
        coalesce(a.actual_ppr, p.actual_recorded) as actual
    from predictions as p
    left join actual_by_format as a on p.player_id = a.player_id and p.period_key = a.period_key
),

-- (b) running totals of evaluated rows inside the window, strictly earlier periods only
running as (
    select
        *,
        fsum(actual) over (
            partition by eval_window, candidate, player_id
            order by period_key
            range between unbounded preceding and 1 preceding
        ) as run_player_sum,
        count(actual) over (
            partition by eval_window, candidate, player_id
            order by period_key
            range between unbounded preceding and 1 preceding
        ) as run_player_cnt,
        fsum(actual) over (
            partition by eval_window, candidate, position
            order by period_key
            range between unbounded preceding and 1 preceding
        ) as run_position_sum,
        count(actual) over (
            partition by eval_window, candidate, position
            order by period_key
            range between unbounded preceding and 1 preceding
        ) as run_position_cnt
    from joined
),

with_baseline as (
    select
        r.*,
        coalesce(sp.seed_sum, 0) + coalesce(r.run_player_sum, 0) as player_hist_sum,
        coalesce(sp.seed_cnt, 0) + coalesce(r.run_player_cnt, 0) as player_hist_cnt,
        coalesce(so.seed_sum, 0) + coalesce(r.run_position_sum, 0) as position_hist_sum,
        coalesce(so.seed_cnt, 0) + coalesce(r.run_position_cnt, 0) as position_hist_cnt
    from running as r
    left join seed_player as sp on r.eval_window = sp.eval_window and r.player_id = sp.player_id
    left join seed_position as so on r.eval_window = so.eval_window and r.position = so.position
),

final as (
    select
        *,
        case
            when player_hist_cnt > 0 then player_hist_sum / player_hist_cnt
            when position_hist_cnt > 0 then position_hist_sum / position_hist_cnt
            else prediction
        end as baseline
    from with_baseline
)

select
    cast(player_id as varchar) as player_id,
    cast(season as integer) as season,
    cast(week as integer) as week,
    cast(period_key as integer) as period_key,
    cast(model_version as varchar) as model_version,
    cast(candidate as varchar) as candidate,
    cast(position as varchar) as position,
    cast(source as varchar) as source,
    cast(eval_window as varchar) as eval_window,
    cast('{{ target_scoring_format }}' as varchar) as target_scoring_format,
    cast(prediction as double) as prediction,
    cast(prediction_floor as double) as prediction_floor,
    cast(prediction_ceiling as double) as prediction_ceiling,
    cast(receptions_estimate as double) as receptions_estimate,
    cast(prediction - 0.5 * receptions_estimate as double) as prediction_half,
    cast(prediction - 1.0 * receptions_estimate as double) as prediction_standard,
    cast(actual as double) as actual,
    cast(
        case
            when actual_from_stats is not null then 'stats'
            when actual_recorded is not null then 'artifact'
        end as varchar
    ) as actual_source,
    cast(actual_half as double) as actual_half,
    cast(actual_standard as double) as actual_standard,
    cast(abs(actual - prediction) as double) as abs_error,
    cast(case when actual is not null then abs(actual - prediction) <= 3 end as boolean) as within_3,
    cast(case when actual is not null then abs(actual - prediction) <= 5 end as boolean) as within_5,
    cast(
        case when actual is not null then prediction_floor <= actual and actual <= prediction_ceiling end as boolean
    ) as interval_hit,
    cast(baseline as double) as baseline,
    cast(abs(actual - baseline) as double) as baseline_abs_error,
    cast(case when actual is not null then abs(actual - baseline) <= 3 end as boolean) as baseline_within_3
from final
