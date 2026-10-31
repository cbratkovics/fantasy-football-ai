// The only sport-specific sentences on the evaluation pages. Everything else on /performance,
// the decisions panel, and the player history is a label from lib/project.ts or a value from the
// API. A second domain replaces this file and nothing else on those pages.
import { ENTITY_PERIOD, PROJECT, UNITS } from '@/lib/project'

export const PERFORMANCE_COPY = {
  heroEyebrow: 'Model evaluation · committed artifact',
  heroTitle: ['Evidence, not', 'just output.'] as const,
  heroLede: `All figures use the implemented PPR rules. The champion model is scored on a forward-time holdout against a causal baseline that only ever sees a ${PROJECT.entity.name}'s earlier realised ${UNITS}. Read every figure with its sample size and cohort in view.`,
  baselineReadout: `The baseline is the honest yardstick: a trailing mean of the ${PROJECT.entity.name}'s own earlier realised ${UNITS} that never sees the ${PROJECT.period.name} it is scored on. A model earns its place only where it beats that column for the cohort you care about.`,
  cohortReadout: `${capitalise(PROJECT.cohort.name)} cohorts share one model version and one split, so their figures can be compared directly. Sample sizes differ by ${PROJECT.cohort.name}; use them when weighing the differences.`,
  decisions: {
    title: 'Explore a historical floor filter',
    intro: `This educational retrospective filter selects a ${PROJECT.entity.name} when the prediction floor (10th residual percentile) clears a threshold. It does not establish current lineup eligibility or replace the shared Start/Sit comparator. This panel reads`,
    hitRate: `filtered ${PROJECT.entity.name}s who beat replacement level`,
    downside: `filtered ${PROJECT.entity.name}s who scored below their floor`,
    regret: `vs the ${PROJECT.period.name}'s best at the ${PROJECT.cohort.name}`,
    scoredRows: `scored ${ENTITY_PERIOD}s at floor ≥`,
  },
  player: {
    eyebrow: `${capitalise(PROJECT.entity.name)} history · read from gold marts exported by the weekly build`,
    lede: `PPR scoring only. Each row is one scored ${ENTITY_PERIOD} from fct_player_week: the prediction with its floor–ceiling band, the realised ${UNITS} once the ${PROJECT.period.name} is played, the error, and the causal baseline (a trailing mean of the ${PROJECT.entity.name}'s own earlier ${UNITS} that never sees the ${PROJECT.period.name} it is scored on). Frozen-test rows are the held-out test ${PROJECT.period.season}, out-of-sample rows a later complete ${PROJECT.period.season} the same frozen model scored, weekly rows the batch publication files. The residual-offset ranges are not guaranteed bounds.`,
  },
} as const

function capitalise(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1)
}
