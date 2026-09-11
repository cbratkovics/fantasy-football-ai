# Data licensing and use

This note replaces the former `backend/docs/COMMERCIAL_USE_COMPLIANCE.md`, which described seven
sources (Sleeper, ESPN authenticated and public, sportsdata.io, OpenWeather, Open-Meteo,
collegefootballdata, Pro-Football-Reference scraping). None of those are used any more; the code
paths were deleted in the 2026-09 rebuild (`AUDIT.md` §2, ADR-0007).

## The single source: nflverse

| Item | Status |
|---|---|
| Data | nflverse data releases (`player_stats`, `schedules`, `rosters`, `injuries`), read through `nflreadpy` |
| Licence | CC-BY-4.0 as stated by the nflverse project; `nflreadpy` is MIT |
| Attribution | "Data from nflverse (https://github.com/nflverse)" — present in `docs/DATA_SOURCES.md` and the model card |
| Rate limiting | not applicable: releases are static files on GitHub; the loader caches parquet locally per day |
| Terms check | owner's responsibility to re-check the nflverse repository terms before redistributing derived data |

## What this project does with the data

* Trains and evaluates models on regular-season weekly stats for QB/RB/WR/TE (2019–2024 frozen
  window; the weekly job extends through the current season).
* Publishes derived numbers (predictions, tiers, evaluation metrics) as committed JSON.
* Does **not** redistribute the raw nflverse files (the parquet cache is git-ignored), scrape any
  site, call any authenticated fantasy-platform API, or use any paid feed.

## Things that would require a new review

Adding injury reports as features (nflverse `injuries`, same licence), opponent/team context
(nflverse `team_stats`), or any non-nflverse source. Each addition needs a row in
`docs/DATA_SOURCES.md`, a contract check in `ffai/data/contracts.py`, and a note here.

## Names and marks

Player and team names are used descriptively as facts. The project is not affiliated with the
NFL, any team, or any fantasy platform.
