import { apiClient } from './client'

export interface Player {
  id: string
  name: string
  team: string
  tier_confidence: number
  projected_points: number
  consistency_score: number
  floor: number
  ceiling: number
  adp: number
  injury_status?: string | null
}

export interface Tier {
  tier: number
  label: string
  color: string
  players: Player[]
  avg_points: number
  point_range: {
    min: number
    max: number
  }
}

export interface TierBreak {
  between_tiers: number[]
  point_gap: number
  significance: string
  recommendation: string
}

export interface PositionTiersResponse {
  position: string
  scoring_type: string
  updated_at: string
  tiers: Tier[]
  tier_breaks: TierBreak[]
  total_players: number
}

export interface AllTiersResponse {
  scoring_type: string
  updated_at: string
  positions: {
    [key: string]: PositionTiersResponse
  }
}

export const tiersApi = {
  // Get tiers for a specific position
  getPositionTiers: async (
    position: string,
    scoringType: string = 'ppr'
  ): Promise<PositionTiersResponse> => {
    try {
      const response = await apiClient.get(`/tiers/positions/${position}`, {
        params: { scoring_type: scoringType }
      })
      return response.data
    } catch (error) {
      console.error('Error fetching position tiers:', error)
      throw error
    }
  },

  // Get tiers for all positions
  getAllTiers: async (
    scoringType: string = 'ppr'
  ): Promise<AllTiersResponse> => {
    try {
      const response = await apiClient.get('/tiers/all', {
        params: { scoring_type: scoringType }
      })
      return response.data
    } catch (error) {
      console.error('Error fetching all tiers:', error)
      throw error
    }
  }
}