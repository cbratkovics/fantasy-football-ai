# Context source contract (v1)

This is a bounded development probe and artifact contract, not a game-day service. Roster membership, injury designation, forecast coverage, source readiness and permission to change a fantasy lineup are different facts. No raw live feed samples are committed by this work. Test fixtures contain invented identities and explicitly describe synthetic observation times.

## Evidence inspected on 2026-10-31

The installed client is `nflreadpy==0.1.5`. Its Python source was inspected before making calls. The following observations used explicit network calls through the environment proxy, TLS verification enabled, client cache disabled and a 15-second request timeout. Observation time denotes our first fetch during this task, not the provider's publication time.

| Source | Observation (UTC) | Observed 2026 coverage | Limits |
| --- | --- | --- | --- |
| `load_schedules(2026)` | 2026-10-31T18:24:22Z | 272 unique games; REG weeks 1–18; gameday/gametime nonnull; week 8 has 14 games | No per-row publication, revision or effective timestamp. Full-looking coverage does not prove provider completeness or exclude later changes. |
| `load_rosters(2026)` | 2026-10-31T18:24:22Z | 3,063 rows, 32 team codes, 3,062 distinct nonnull GSIS IDs and one null ID. Row week values span 1–8. | This season snapshot is not a complete week-by-week historical roster. Each nonnull GSIS ID appeared once. Week 8 alone has 2,259 rows, so exact-week filtering cannot establish the full target universe. Current team requires qualified season-snapshot evidence; no source update time was present. |
| `load_injuries(2026)` | 2026-10-31T18:24:22Z | 2,459 rows, weeks 1–8, 980 distinct GSIS IDs; 320 week-8 rows. Report designations: Out, Doubtful, Questionable, null (1,403 rows). | A null/absent designation is not Active. No row publication, update or effective timestamp was present. A working injury loader proves this observation only, not a timely current feed or future coverage. |

A second bounded observation confirmed the same aggregate coverage. The roster week distribution was 366 / 8 / 11 / 16 / 19 / 25 / 359 / 2,259 for weeks 1–8. It is recorded to expose the ambiguity of the `week` field rather than infer historical membership.

The requested rendered primary documentation hosts returned HTTP 403 under this environment's destination policy:

- <https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html>
- <https://nflreadpy.nflverse.com/>

The official source dictionaries were readable:

- <https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_schedules.csv>
- <https://raw.githubusercontent.com/nflverse/nflreadr/main/data-raw/dictionary_rosters.csv>

The schedule dictionary explicitly defines `gametime` as 24-hour **Eastern time**, regardless of stadium timezone. The roster dictionary describes `team` and primary `position` as NFL.com values; `status` is roster status (Active, Inactive, reserve, practice squad, etc.). Those descriptions do not validate fantasy-platform lock rules or player game eligibility.

## Identity, grain and coverage

Use GSIS IDs (`gsis_id`, normalized to `player_id`), never fuzzy name joins. Duplicate display names are harmless. Missing IDs and conflicting team/position rows remain unresolved. The roster universe is constructed independently of statistical history and then left-joined to explicitly linked forecast records. Rookies remain identities with missing forecast coverage. Forecast-only identities remain diagnostics rather than verified roster members. Preserve the forecast-era team alongside the current roster-team evidence.

Schedules identify a game by `game_id` and teams by provider team codes. The expected grain is one row per game. Multiple games for a team in a requested period, incompatible duplicates and unknown kickoff times must not silently select a row. A missing game establishes a bye only when the input explicitly supplies a complete valid schedule assertion. Merely downloading a file does not supply that assertion. Source absence, partial coverage and a proven bye produce different reason codes.

Injury rows join by GSIS ID, season/type/week and compatible team. `Out` is clear unavailability; `Questionable` and `Doubtful` remain named uncertainty. Absence from a report, null `report_status` and `ACT` roster membership do not establish game availability. Conflicting injury rows remain unresolved. Underlying schedules may include scores, betting lines and outcomes; normalization allowlists only pre-decision game identity, teams, kickoff and schedule state, discarding those later-known fields.

## Seasons and timestamps

The installed schedule loader downloads `schedules/games.parquet` and filters the season. The installed roster loader constructs `rosters/roster_<season>.parquet` and accepts seasons from 1920 through the client's current roster year. The installed injury loader constructs `injuries/injuries_<season>.parquet` and accepts seasons from 2009 through its current season. These are client-supported ranges, not demonstrated coverage for every year. Only the 2026 coverage above was observed in this task.

Normalize provider Eastern local date/time using `America/New_York`, with aware UTC comparisons. Reject ambiguous/nonexistent local times and missing times as unknown rather than guessing a deadline. Early international games and daylight-saving changes use the same named timezone. Postponements retain their schedule state/revision; a later kickoff cannot be assumed without a known applicable revision. Regular season targets stop at the repository's supported final week. No next supported game produces `no_target`; do not manufacture week 19 or treat postseason as regular-season advice.

`provider_updated_at_utc` is null when not supplied. `first_observed_at_utc` is the first observed identical content digest in this probe's retained cache; `fetched_at_utc` is the last actual fetch, and a cache reread preserves both. Unknown effective/publication times stay null with a reason. A new fetch does not prove recent content and does not establish historical pregame availability. As-of use excludes records or source observations first available after the requested decision time. Synthetic historical fixtures test these rules but do not prove historical provider availability.

The public release URLs are mutable snapshots. No historical release-version lookup, guaranteed revision archive or timestamped pregame capture was verified. Previously saved probe artifacts provide only this application's own first-observed history. They must not be backdated.

## Cache and request bounds

Existing `ffai.data.nflverse` statistical caching is unchanged: dated parquet keyed by season range and local date; same-day read avoids a fetch. Context probing has separate provenance-aware cache controls and explicit refresh.

The installed client's default is a 24-hour memory cache. Its downloader checks this cache before any HTTP request. Probe network calls set `cache_mode='off'` and a bounded timeout in a fresh subprocess, leaving the caller's configuration unchanged, so explicit refresh cannot be defeated by that inner cache. Its `clear_cache(pattern)` examples are not a reliable targeted refresh mechanism: actual keys are hashes of URL and arguments. No destructive cache clearing is required. No automatic live request is made during API startup, serving, or default tests.

Freshness thresholds in the context/readiness layer are explicit local policy assumptions, not measured provider SLAs. Fetch recency, provider content time, target-period coverage and source verification remain separately reported. There is no high-frequency ingestion job or last-minute coverage promise.

## Rights and operational use

The installed open-source client license and a data dictionary do not grant blanket rights to redistribute every downstream feed. Per-feed redistribution, NFL.com/other upstream terms and suitability for commercial use were not established in this task. Keep live raw payloads in ignored local output/cache paths; do not publish them by default. The probe's normalized snapshot should be reviewed for permitted fields and rights before production publication. Only synthetic fixtures are suitable for checked-in test samples here.

A successful live fetch means the requested public URL was accessible at the observation time. It does not mean a source is complete, timely, injury-verified, licensed for a deployment, or adequate for automatic lineup action. The comparator can always remain a clearly dated conditional comparison while this optional context foundation is disabled or unavailable.

## Executable probe and companion contract

From the repository root, run the default offline path:

```bash
.venv/bin/python scripts/probe_context.py --season 2026 --week 9 \
  --as-of 2026-10-31T13:00:00Z \
  --fixtures tests/fixtures/context/sources.json --out /tmp/ffai-context-fixture
.venv/bin/python -m pytest tests/test_context_snapshot.py
```

For an explicit bounded live observation:

```bash
.venv/bin/python scripts/probe_context.py --season 2026 --week 9 \
  --as-of now --live --refresh --out /tmp/ffai-context-live
```

`--as-of now` chooses the UTC decision time after the explicit fetches complete. An explicit earlier UTC as-of excludes sources first observed after that time, even if their rows describe earlier games. Omit `--week` to look for the next supported regular-season kickoff; incomplete or unknown kickoff coverage produces `target_unknown`. A complete valid schedule with no later supported games produces `no_target`.

The delivered CLI was run successfully at `2026-10-31T18:26:15+00:00`: 272 schedule rows, 3,063 roster rows and 2,459 injury rows observed. Normalization retained 955 fantasy-position roster identities and one unresolved missing-ID diagnostic. No forecast was linked, so all 955 remained unscored. All three live source states were `partial`: completeness and provider update timestamps were not independently established. No live normalized payload was committed or placed in the public frontend.

`--out` always writes the current attempt's `capabilities.json` and `snapshot.json`, even when a required source fails. Exit **0** means schedule and roster rows were usable at the requested as-of; exit **2** means a required source was unavailable. Injury failure is optional and does not discard other results. Neither zero exit nor `live_verified` means complete/current coverage or actionable eligibility. A fixture report always has `live_verified: false`.

Each source request uses a fresh subprocess, the installed official loader, cache OFF, a 15-second socket timeout and a 30-second process deadline, with no retry. Context cache files live separately under ignored `data/cache/context/`. Same-day reads are additionally bounded by source-specific assumed ages: schedules 24 hours, rosters 6 hours, injuries 6 hours. Explicit `--refresh` bypasses both this cache and the underlying client cache. An unchanged digest retains its first observation; stale content never gains a provider update timestamp. A failed refresh returns failure rather than reusing the previous payload as a fresh success. The cache retains only the most recent raw version per source/season, not a historical archive.

To connect a context export, supply `--forecast artifacts/decision_lab/inputs/<snapshot-id>.json`. The companion links the exact `snapshot_id`, canonical content digest, season and week. `--out frontend-next/public/context` is available for reviewed **synthetic/development** exports; use `--out artifacts/context` for the optional API artifact route. The output filename is `snapshot.json`. Do not publish live normalized data before establishing its redistribution rights. Removing this optional companion disables context integration while leaving correct actual scoring and the verified dated comparator available.

The snapshot schema is `context_snapshot/1.0`. Its `context_snapshot_id` is the existing canonical JSON SHA-256 of every field except the ID itself; this is an integrity check, not an authenticity signature. `ffai.context.load_snapshot` verifies schema, structure and digest without any network calls. Context is a separate companion: it does not change policy-v1 canonical inputs, decision IDs, receipts or replay semantics. Forecast coverage is `present`, `missing_history`, or `unresolved_reference`; `missing_history` means no matched model record, not a proven explanation of the model's omission. Source/forecast linkage and per-player context remain separately inspectable. Historical hypothetical exploration never consults today's context.
