import { apiGet } from './client'
import type {
  EvalKind,
  Health,
  ManifestResponse,
  PerformanceArtifact,
  PerformanceResponse,
  PlayerResponse,
  Position,
  PredictionsResponse,
  Root,
  ScoringFormat,
} from './types'

export interface WeekRef {
  season: number
  week: number
}

/** Parse "predictions/<season>/week_<ww>.json" from the manifest into a season/week pair. */
export function parseLatestPredictions(latest: string | undefined): WeekRef | null {
  if (!latest) return null
  const match = latest.match(/predictions\/(\d{4})\/week_(\d{1,2})\.json$/)
  if (!match) return null
  return { season: Number(match[1]), week: Number(match[2]) }
}

export const getRoot = () => apiGet<Root>('/')

export const getHealth = () => apiGet<Health>('/health')

export const getManifest = () => apiGet<ManifestResponse>('/manifest')

export async function getLatestWeek(): Promise<WeekRef> {
  const { manifest } = await getManifest()
  const ref = parseLatestPredictions(manifest.predictions?.latest)
  if (!ref) throw new Error('manifest has no predictions.latest entry')
  return ref
}

export const getPredictions = (
  season: number,
  week: number,
  scoring: ScoringFormat = 'ppr',
  position?: Position,
) =>
  apiGet<PredictionsResponse>(`/predictions/${season}/${week}`, { scoring, position })

export const getPlayer = (playerId: string) =>
  apiGet<PlayerResponse>(`/players/${encodeURIComponent(playerId)}`)

/** Every registered evaluation artifact (frozen test season plus any out-of-sample seasons). */
export const getPerformance = () => apiGet<PerformanceResponse>('/performance')

/** One evaluation artifact by id. Ids come from /performance or /health, never from the UI. */
export const getPerformanceById = (evalId: string) =>
  apiGet<PerformanceArtifact>(`/performance/${encodeURIComponent(evalId)}`)

/** Short noun for each evaluation kind, used in labels. */
export const EVAL_KIND_LABEL: Record<EvalKind, string> = {
  frozen_test: 'frozen test season',
  out_of_sample_season: 'out-of-sample season',
}

/** What each evaluation kind means, worded from the artifact's role in model selection. */
export const EVAL_KIND_EXPLANATION: Record<EvalKind, string> = {
  frozen_test: 'the season held out when the model was trained and selected',
  out_of_sample_season:
    'a complete season no training, validation, selection or tuning decision touched, scored with the same frozen artifact',
}

/** "2025 · out-of-sample season" — derived from the artifact alone. */
export const evaluationLabel = (a: Pick<PerformanceArtifact, 'season' | 'kind'>) =>
  `${a.season} · ${EVAL_KIND_LABEL[a.kind] ?? a.kind}`

/** Evaluations ordered newest season first, ties broken so the frozen test sorts after out-of-sample. */
export function sortEvaluations(evaluations: PerformanceArtifact[]): PerformanceArtifact[] {
  return [...evaluations].sort((x, y) => y.season - x.season || x.eval_id.localeCompare(y.eval_id))
}

/** The artifact with the highest season, or undefined when the list is empty. */
export const latestEvaluation = (evaluations: PerformanceArtifact[] | undefined) =>
  evaluations && evaluations.length > 0 ? sortEvaluations(evaluations)[0] : undefined
