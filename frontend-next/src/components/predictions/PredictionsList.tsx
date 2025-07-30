'use client'

import { useQuery } from '@tanstack/react-query'
import { PredictionCard } from './PredictionCard'
import axios from 'axios'

interface PredictionsListProps {
  week: number
  playerId?: string | null
}

export function PredictionsList({ week, playerId }: PredictionsListProps) {
  const { data: predictions, isLoading, error } = useQuery({
    queryKey: ['predictions', week, playerId],
    queryFn: async () => {
      const params = new URLSearchParams({
        week: week.toString(),
        ...(playerId && { player_id: playerId })
      })
      const response = await axios.get(`/api/predictions?${params}`)
      return response.data
    }
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600">Error loading predictions</p>
      </div>
    )
  }

  if (!predictions || predictions.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">No predictions available for this week</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {predictions.map((prediction: any) => (
        <PredictionCard key={prediction.id} prediction={prediction} />
      ))}
    </div>
  )
}