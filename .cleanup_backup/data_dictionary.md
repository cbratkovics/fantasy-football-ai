## Fantasy Football Analysis Data Dictionary

- #### This data dictionary consolidates all metrics used in the dataset for fantasy football analysis. The columns represent player-specific statistics for performance evaluation, consistency tracking, and predictive modeling.
- #### This unified dictionary provides a **comprehensive reference** for all relevant **fantasy football player statistics**, enabling **draft strategy optimization, weekly performance evaluation, and predictive modeling**.

---

| **Column Name**                    | **Definition** |
|-------------------------------------|--------------------------------------------------|
| **`season`**                        | The year of the NFL season. |
| **`week`**                          | The specific week of the NFL season. |
| **`player_id`**                     | Unique identifier for each player. |
| **`player_name`**                   | The name of the player. |
| **`team`**                          | The team the player is currently playing for. |
| **`position`**                      | The position the player plays (e.g., QB, RB, WR, TE). |
| **`fantasy_points`**                | The number of fantasy points scored by the player in a given week. |
| **`total_fantasy_points`**          | Total fantasy points accumulated by the player over the season. |
| **`ppg`**                           | **Points Per Game (PPG)**: Average fantasy points scored per game. |
| **`projected_points`**              | The number of fantasy points the player was projected to score in a given week. |
| **`total_projected_points`**        | Sum of all **projected fantasy points** for the player throughout the season. |
| **`projected_ppg`**                 | **Projected Points Per Game**: Average projected fantasy points per game. |
| **`weeks_played`**                  | Total number of weeks the player played in the season. |
| **`prev_week_points`**              | Fantasy points the player scored in the **previous week** (used for momentum analysis). |
| **`last_3_avg`**                    | Rolling **3-week average** of the player's fantasy points (measures **recent performance trends**). |
| **`season_avg`**                    | The player's average fantasy points per game over the entire season. |
| **`fantasy_stdev`**                 | Standard deviation of the player’s weekly fantasy points, indicating **consistency** (higher values mean more variability). |
| **`fantasy_var`**                   | Variance of the player’s fantasy points (higher values indicate more unpredictable performance). |
| **`boom_weeks`**                    | Number of weeks where the player **exceeded 20 fantasy points**, indicating **explosive games**. |
| **`bust_weeks`**                    | Number of weeks where the player **scored less than 7 fantasy points**, indicating **poor performance**. |
| **`high_ceiling_weeks`**            | Number of weeks where the player's fantasy points were in the **top 25%** of their weekly performances. |
| **`low_floor_weeks`**               | Number of weeks where the player's fantasy points were in the **bottom 25%** of their weekly performances. |
| **`fantasy_z_score`**               | Standardized fantasy performance metric calculated using Z-score normalization. |
| **`momentum`**                      | Difference between the player's current week's fantasy points and their **3-week rolling average**, indicating **positive or negative momentum**. |
| **`positional_avg_fantasy_points`** | The average fantasy points scored by all players at the same position. |
| **`availability_score`**            | **Availability Metric**: Measures how much of the season a player was active (`weeks_played / 17`). |
| **`weekly_consistency_score`**      | **Reliability Score**: How consistent a player is week to week. Higher values indicate more **consistent** performances. |
| **`efficiency_ratio`**              | **Performance Efficiency**: Measures how well the player **performed relative to projections** (`fantasy_points / projected_points`). A higher value means the player **outperformed expectations**. |

---