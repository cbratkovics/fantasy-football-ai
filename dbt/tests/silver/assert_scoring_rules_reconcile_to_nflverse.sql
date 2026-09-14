-- The rules macro must reproduce nflverse's own fantasy_points (standard) and fantasy_points_ppr
-- on every row to within 0.01 (the Python twin: tests/test_scoring_reconciliation.py). Rows
-- where nflverse's column is null are ignored.
select
    player_id,
    season,
    week,
    points_standard,
    fantasy_points,
    points_ppr,
    fantasy_points_ppr,
    greatest(
        abs(points_standard - fantasy_points),
        abs(points_ppr - fantasy_points_ppr)
    ) as abs_diff
from {{ ref('slv_player_stats') }}
where
    (fantasy_points is not null and abs(points_standard - fantasy_points) > 0.01)
    or (fantasy_points_ppr is not null and abs(points_ppr - fantasy_points_ppr) > 0.01)
