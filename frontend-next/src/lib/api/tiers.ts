import { apiGet } from './client'
import type { Position, TiersResponse } from './types'

export const getTiers = (position: Position) => apiGet<TiersResponse>(`/tiers/${position}`)
