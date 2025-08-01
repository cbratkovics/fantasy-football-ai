import { apiClient } from './client'

export interface PlayerRanking {
  player_id: string
  name: string
  position: string
  team: string
  age: number
  years_exp: number
  tier: number
  tier_label: string
  predicted_points: number
  confidence_interval: {
    low: number
    high: number
  }
  trend: string
  injury_status?: string | null
  status: string
}

export interface PlayerDetail {
  player_id: string
  name: string
  first_name: string
  last_name: string
  position: string
  team: string
  age: number
  years_exp: number
  status: string
  injury_status?: string | null
  fantasy_positions: string[]
  additional_info: {
    height?: string
    weight?: string
    college?: string
    birth_date?: string
  }
  last_updated: string
  season_stats: {
    games_played: number
    total_points_ppr: number
    avg_points_ppr: number
  }
}

export interface WeeklyPrediction {
  player_id: string
  player_name: string
  position: string
  team: string
  week: number
  predicted_points: number
  confidence: number
}

export const playersApi = {
  // Get player rankings
  getRankings: async (
    position?: string,
    limit: number = 100
  ): Promise<PlayerRanking[]> => {
    try {
      const response = await apiClient.get('/players/rankings', {
        params: { position, limit }
      })
      return response.data
    } catch (error) {
      console.error('Error fetching player rankings:', error)
      throw error
    }
  },

  // Get player details
  getPlayerDetail: async (playerId: string): Promise<PlayerDetail> => {
    try {
      const response = await apiClient.get(`/players/${playerId}`)
      return response.data
    } catch (error) {
      console.error('Error fetching player detail:', error)
      throw error
    }
  },

  // Get weekly predictions
  getWeeklyPredictions: async (
    week: number,
    position?: string,
    limit: number = 50
  ): Promise<WeeklyPrediction[]> => {
    try {
      const response = await apiClient.get(`/predictions/week/${week}`, {
        params: { position, limit }
      })
      return response.data
    } catch (error) {
      console.error('Error fetching weekly predictions:', error)
      throw error
    }
  },

  // Search players
  searchPlayers: async (query: string): Promise<PlayerRanking[]> => {
    try {
      const response = await apiClient.get('/players/search', {
        params: { q: query }
      })
      return response.data
    } catch (error) {
      console.error('Error searching players:', error)
      throw error
    }
  }
}