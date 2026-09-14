// Gold marts: parquet tables exported by the weekly dbt build and read by the API in-process.
// Every number these return is labelled on the page as coming from the marts, never from the
// evaluation artifacts (which /performance still serves verbatim).
import { apiGet } from './client'
import type { DecisionsResponse, PlayerWeekResponse, Position, WeeklyEvalResponse } from './types'

export const getWeeklyEval = (params?: { cohort?: string; season?: number; candidate?: string; model_version?: string }) =>
  apiGet<WeeklyEvalResponse>('/marts/weekly_eval', params)

export const getPlayerWeek = (playerId: string, candidate?: 'rf' | 'xgb' | 'all') =>
  apiGet<PlayerWeekResponse>(`/marts/player_week/${encodeURIComponent(playerId)}`, { candidate })

export const getDecisions = (params: { min_floor: number; season?: number; position?: Position; candidate?: string }) =>
  apiGet<DecisionsResponse>('/marts/decisions', params)
