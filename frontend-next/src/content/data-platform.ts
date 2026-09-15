import { PROJECT } from '@/lib/project'

export const repoFile = (path: string) => `${PROJECT.repoUrl}/blob/main/${path}`
export const dbtDoc = (uniqueId: string) => `${PROJECT.pagesUrl}#!/model/${uniqueId}`

export const PLATFORM_STORIES = [
  {
    id: 'corrections', title: 'Handle corrections without rebuilding every transformation', model: 'model.ffai_dbt.slv_player_stats',
    problem: 'Published statistics can be restated after a game.',
    choice: 'Incremental delete+insert at player × season × week reprocesses the newest four distinct loaded periods plus new periods.',
    tradeoff: 'A correction outside that conservative, uncalibrated window waits for a full refresh. The fixture tests use a two-period window; they do not prove universal equivalence.',
    proof: 'tests/test_dbt_incremental.py', adr: 'docs/adr/0023-incremental-silver-stats-with-lookback.md',
    effect: 'Recent corrected scores flow downstream; fct_player_week stays a full table because its causal baseline depends on earlier observations.',
  },
  {
    id: 'history', title: 'Preserve captured history without inventing the past', model: 'model.ffai_dbt.dim_player_asof',
    problem: 'Latest-seen player attributes cannot truthfully describe every earlier scoring period.',
    choice: 'snp_player checks name, position and team; the as-of view uses a capture when period > start and period ≤ next capture.',
    tradeoff: 'History begins at first capture. Earlier periods fall back to current attributes with is_exact_asof=false; this is warehouse knowledge, not a transaction ledger.',
    proof: 'dbt/tests/gold/assert_asof_exact_after_first_capture.sql', adr: 'docs/adr/0024-player-snapshot-and-asof-views.md',
    effect: 'The snapshot-backed views are not exported, and the current player API still reads latest-seen dim_player.',
  },
  {
    id: 'versioning', title: 'Evolve a mart without breaking its consumer', model: 'model.ffai_dbt.fct_decision_policy.v2',
    problem: 'New interval metrics should not silently change an existing API contract.',
    choice: 'v1 retains alias fct_decision_policy; latest v2 exports as fct_decision_policy_v2 and adds within-3 rate and interval coverage.',
    tradeoff: 'An unversioned dbt ref resolves to latest, while /marts/decisions deliberately pins v1. Exported does not mean served.',
    proof: 'dbt/tests/gold/assert_decision_within_3_reconciles_to_eval_artifacts.sql', adr: 'docs/adr/0025-versioned-decisions-mart.md',
    effect: 'The response now reports mart_version from the serving constant; it remains a backwards-compatible v1 API.',
  },
  {
    id: 'rollups', title: 'Keep metrics stable across sources and rollups', model: 'model.ffai_dbt.slv_predictions',
    problem: 'Overlapping prediction families and incomplete outcomes can distort a headline rate.',
    choice: 'Weekly predictions win deterministic source precedence; model version and candidate stay in the key, and unplayed outcomes remain null.',
    tradeoff: 'Rates require their actual denominators. A narrowed population with a higher hit rate is not automatically a better policy.',
    proof: 'dbt/models/silver/_silver.yml', adr: 'docs/DECISIONS.md',
    effect: 'Evaluation artifacts and exported marts remain independent provenance paths that are reconciled under aligned scope.',
  },
] as const

export const WEEKLY_EXAMPLE = [
  { error: 1, within3: true, within5: true }, { error: 3, within3: true, within5: true },
  { error: 4.5, within3: false, within5: true }, { error: 6, within3: false, within5: false },
] as const
