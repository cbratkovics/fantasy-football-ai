# ADR-0031 — Drift reference matched to the window's week positions when the window crosses a season boundary; every run writes a drift artifact (2026-09-15)

**Context.** The weekly job held 2026 week 2 twice (`run-20260915T121613Z`, `run-20260915T123508Z`)
on drift: QB median monitored PSI 0.329 with seven flagged lag features, RB `rushing_yards_L1` at
0.32. Contracts had passed. Recomputing per-feature PSI from the run's own cache file showed
nothing unusual about 2026: the same construction at week 2 of 2022, 2023 and 2024 gives QB
medians 0.350, 0.380 and 0.345, all holds. The mechanism is the window, not the reference or the
lag features: at target week 2 the last four *played* weeks are the previous season's weeks
16–18 plus week 1, and ADR-0010 compares that window with the training-time bucket of the newest
week (weeks 1–4). Three quarters of the rows are late-season QB production held against an
early-season reference. Lag windows are not truncated at the boundary (L1/L3/L5 cross seasons by
design; only `*_season_avg` resets, and it is already excluded), and the reference was never
pooled across the season. ADR-0010's calibration scanned end-weeks 4–17 (target weeks 5–18) of
2021–2024 and therefore never saw a boundary-crossing window; it would have held three of the
last five seasons' week-2 runs. The obvious repair, comparing only the current season's week-1
rows with week-1 training rows, is worse: ten-bin PSI on ≈35 QB rows gives medians of 0.4–1.5.

**Decision.** Keep the four-week window and the ADR-0010 thresholds. `ffai.eval.drift.reference_for_window`
chooses the reference: while the window sits inside one season, the training-time week-of-season
bucket of the newest week (unchanged); when the window spans more than one season, decile edges
recomputed from the training seasons' rows at the same week positions the window contains (for
2026 week 2: training-season rows at weeks 16, 17, 18 and 1, ≈560 QB rows), falling back to the
bucket when fewer than `MIN_REFERENCE_ROWS` (100) matched rows exist. The training rows are the
feature rows of `metadata.json → seasons.train` rebuilt at run time from the current stats frame
by the same feature module; `metadata.json` and the frozen pipelines are untouched. Every run
whose contracts pass writes `artifacts/drift/<run_id>.json` (`drift_report_version 1.1`,
`artifacts/schemas/drift_report.schema.json`): per position the PSI per feature, status, median,
flagged and severe features, rows, and a `reference` block naming the mode, the window's seasons
and weeks, the reference weeks, the training seasons and the matched row count. The workflow
already commits `artifacts/`, so the file lands on HOLD, WARN and OK alike. The run log keeps the
per-position summary plus `reference_mode` and points at the file.

Calibration, `scripts/calibrate_drift.py` (`make calibrate-drift`), target weeks 2–18 of
2021–2024, four positions each, bucket (ADR-0010) versus hybrid:

| Scan | Bucket reference holds | Hybrid reference holds |
|---|---|---|
| Target weeks 2–4 (12 windows; never scanned before) | 7, all QB: 2022 wk2, 2023 wk2–4, 2024 wk2–4 | 1: 2024 wk4 QB (median 0.333) |
| Target weeks 5–18 (56 windows; the ADR-0010 scan) | 2: 2023 wk5, 2024 wk5 (QB) | 2: the same two |
| All (68 windows) | 9 | 3 |

The weeks 5–18 result is unchanged: on those windows the hybrid selects the bucket reference and
the reports are identical (`tests/test_drift.py::test_mid_season_window_is_identical_under_both_references`).
2024 week 4 QB stays a hold: with a matched reference (window weeks 18, 1, 2, 3 against the same
training weeks) its median is 0.333, above the threshold on its own merits, and it sits next to
2024 week 5, which ADR-0010 already held; I cannot tell from the data whether early 2024 was a
real QB shift or noise, and the threshold is not tuned to remove one window. On the 2026 week-2
run itself the hybrid verdict is QB warn (median 0.107, 0 flagged), RB warn (0.038), WR ok, TE ok;
the reproduction file `artifacts/drift/run-20260915T123508Z.json` carries the run's bucket verdict
and the hybrid verdict side by side.

**Consequences.** Early-season runs no longer hold on the calendar. The next weekly run will
score 2026 week 2 (or week 3, if it runs after Monday night) under the hybrid rule and write its
own drift artifact. Anyone revisiting the threshold has per-feature PSI for every run in the
repository, not only in Actions logs. The calibration script is committed so the next rule change
reruns the same scan. Lesson recorded in the model card: the monitor was calibrated on weeks
5–18 only and held week 2 in production; calibrate over every seasonal position the job will
ever run at.
