import { apiGet } from './client'
import type {
  Health,
  ManifestResponse,
  PerformanceArtifact,
  PlayerResponse,
  Position,
  PredictionsResponse,
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

export const getPerformance = () => apiGet<PerformanceArtifact>('/performance')
