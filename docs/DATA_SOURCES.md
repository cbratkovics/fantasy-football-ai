# Data sources

This repository uses **one** external data source: the nflverse data releases, read through the
`nflreadpy` Python client. No other API, scrape, paid feed, or database is used anywhere in
training, evaluation, or serving. The former Sleeper, ESPN, sportsdata.io, OpenWeather,
Open-Meteo, collegefootballdata, and Pro-Football-Reference code paths were deleted in the
2026-09 rebuild (see `AUDIT.md` §2 for what they were).

## Client

| Item | Value |
|---|---|
| Library | `nflreadpy` (recorded in each training run's `metadata.json` as `data_library`, e.g. `nflreadpy==0.1.5`) |
| Fallback | `nfl_data_py` is imported only if `nflreadpy` is unavailable; column renames are applied so the rest of the code is unchanged |
| Module | `ffai/data/nflverse.py` |
| Cache | `data/cache/<dataset>_<first>-<last>_<YYYY-MM-DD>.parquet` (git-ignored); a same-day cache hit needs no network |

## Datasets and columns used

| Loader | nflverse release | Used for | Columns kept |
|---|---|---|---|
| `load_weekly_stats(seasons)` | `player_stats` (weekly summary level), regular season, positions QB/RB/WR/TE | features, targets, scoring reconciliation | `player_id, player_display_name, position, season, week, season_type, team, opponent_team` + `completions, attempts, passing_yards, passing_tds, passing_interceptions, passing_air_yards, passing_2pt_conversions, sack_fumbles_lost, carries, rushing_yards, rushing_tds, rushing_fumbles_lost, rushing_2pt_conversions, receptions, targets, receiving_yards, receiving_tds, receiving_air_yards, receiving_fumbles_lost, receiving_2pt_conversions, special_teams_tds, fantasy_points, fantasy_points_ppr` |
| `load_schedules(seasons)` | `schedules` | determining the current season / next week in the weekly job | `season, game_type, week, gameday` |
| `load_rosters(seasons)` | `rosters` | preseason context for draft tiers (age / experience where available) | `season, gsis_id, position, team, birth_date, years_exp` |
| `load_injuries(seasons)` | `injuries` | loaded for completeness; **not** a feature in v1 | — |

`fantasy_points` and `fantasy_points_ppr` are nflverse's own values. `ffai/scoring.py` recomputes
them from the raw stat columns with explicit rules and `tests/test_scoring_reconciliation.py`
asserts equality (tolerance 0.01) on every row; half-PPR is produced by the same rules with a 0.5
reception weight.

## Grain, seasons, refresh

* Grain: one row per `(player_id, season, week)`; `ffai/data/contracts.py` checks uniqueness and
  value ranges on every load in the weekly job.
* Historical window for the frozen evaluation: 2019–2024 regular seasons (train 2019–2022,
  validation 2023, test 2024). The weekly job loads through the current season.
* Refresh cadence: nflverse republishes player stats after each game day; the weekly job
  (`.github/workflows/weekly.yml`) runs on Tuesdays and loads the latest release. There is no
  intra-week or real-time ingestion.

## Licence

nflverse data releases are published by the nflverse project under the
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) licence as stated on the
[nflverse-data](https://github.com/nflverse/nflverse-data) repository; the `nflreadpy` client is
MIT-licensed. Attribution: *Data from nflverse (https://github.com/nflverse).* Users of this
repository are responsible for checking the current terms at the source before redistributing
derived data. NFL statistics themselves are facts; team and player names are used descriptively.

## What is deliberately not used

* No injury, weather, opponent, Vegas line, or depth-chart features in `asof_v1`
  (listed as future work in `docs/MODEL_CARD.md`).
* No fantasy-platform APIs (Sleeper, ESPN, Yahoo) and no scraping.
* No paid APIs and no LLM calls.
