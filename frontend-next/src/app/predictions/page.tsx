'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  CalendarIcon, 
  ArrowPathIcon,
  BoltIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  FunnelIcon,
  MagnifyingGlassIcon
} from '@heroicons/react/24/outline'
import { playersApi, type WeeklyPrediction } from '@/lib/api/players'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

type Position = 'All' | 'QB' | 'RB' | 'WR' | 'TE'
type SortBy = 'predicted_points' | 'name' | 'position'

export default function PredictionsPage() {
  const [predictions, setPredictions] = useState<WeeklyPrediction[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedWeek, setSelectedWeek] = useState(1)
  const [selectedPosition, setSelectedPosition] = useState<Position>('All')
  const [sortBy, setSortBy] = useState<SortBy>('predicted_points')
  const [searchQuery, setSearchQuery] = useState('')

  const positions: Position[] = ['All', 'QB', 'RB', 'WR', 'TE']
  const weeks = Array.from({ length: 18 }, (_, i) => i + 1)

  // Fetch predictions
  const fetchPredictions = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const data = await playersApi.getWeeklyPredictions(
        selectedWeek,
        selectedPosition === 'All' ? undefined : selectedPosition,
        100
      )
      setPredictions(data)
    } catch (err) {
      console.error('Error fetching predictions:', err)
      setError('Failed to load predictions. Please try again.')
      // Use mock data as fallback
      setPredictions(generateMockPredictions())
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPredictions()
  }, [selectedWeek, selectedPosition])

  // Filter and sort predictions
  const filteredPredictions = predictions
    .filter(pred => 
      pred.player_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      pred.team.toLowerCase().includes(searchQuery.toLowerCase())
    )
    .sort((a, b) => {
      switch (sortBy) {
        case 'predicted_points':
          return b.predicted_points - a.predicted_points
        case 'name':
          return a.player_name.localeCompare(b.player_name)
        case 'position':
          return a.position.localeCompare(b.position)
        default:
          return 0
      }
    })

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-500'
    if (confidence >= 0.65) return 'text-yellow-500'
    return 'text-red-500'
  }

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High'
    if (confidence >= 0.65) return 'Medium'
    return 'Low'
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="pt-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="mb-8">
            <Breadcrumb items={[{ name: 'Predictions', current: true }]} />
            
            <div className="mt-4">
              <h1 className="text-4xl font-bold text-gray-900">
                Weekly Predictions
              </h1>
              <p className="mt-2 text-lg text-gray-600">
                AI-powered fantasy point predictions updated in real-time with injury reports and matchup analysis
              </p>
            </div>
          </div>

          {/* Controls */}
          <div className="mb-8 space-y-4">
            {/* Week Selector */}
            <div className="flex items-center gap-4 overflow-x-auto pb-2">
              <CalendarIcon className="h-5 w-5 text-gray-500 flex-shrink-0" />
              <div className="flex gap-2">
                {weeks.map(week => (
                  <button
                    key={week}
                    onClick={() => setSelectedWeek(week)}
                    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                      selectedWeek === week
                        ? 'bg-indigo-600 text-white'
                        : 'bg-white text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    Week {week}
                  </button>
                ))}
              </div>
            </div>

            {/* Filters Row */}
            <div className="flex flex-col sm:flex-row gap-4">
              {/* Position Filter */}
              <div className="flex items-center gap-2">
                <FunnelIcon className="h-5 w-5 text-gray-500" />
                <select
                  value={selectedPosition}
                  onChange={(e) => setSelectedPosition(e.target.value as Position)}
                  className="rounded-md border-gray-300 py-2 pl-3 pr-10 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  {positions.map(pos => (
                    <option key={pos} value={pos}>{pos}</option>
                  ))}
                </select>
              </div>

              {/* Sort By */}
              <div className="flex items-center gap-2">
                <ChartBarIcon className="h-5 w-5 text-gray-500" />
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as SortBy)}
                  className="rounded-md border-gray-300 py-2 pl-3 pr-10 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  <option value="predicted_points">Points (High to Low)</option>
                  <option value="name">Name (A-Z)</option>
                  <option value="position">Position</option>
                </select>
              </div>

              {/* Search */}
              <div className="flex-1">
                <div className="relative">
                  <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search players or teams..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full rounded-md border-gray-300 pl-10 pr-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  />
                </div>
              </div>

              {/* Refresh */}
              <button
                onClick={fetchPredictions}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50"
              >
                <ArrowPathIcon className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
            </div>
          </div>

          {/* Predictions Table */}
          {loading ? (
            <div className="flex justify-center items-center py-20">
              <ArrowPathIcon className="h-8 w-8 animate-spin text-indigo-600" />
            </div>
          ) : error ? (
            <div className="text-center py-20">
              <ExclamationTriangleIcon className="mx-auto h-12 w-12 text-red-500" />
              <p className="mt-2 text-gray-600">{error}</p>
              <button
                onClick={fetchPredictions}
                className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
              >
                Try Again
              </button>
            </div>
          ) : (
            <div className="bg-white shadow-sm rounded-lg overflow-hidden">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Player
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Position
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Team
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Predicted Points
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Confidence
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredPredictions.map((prediction, idx) => (
                    <motion.tr
                      key={prediction.player_id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, delay: idx * 0.02 }}
                      className="hover:bg-gray-50"
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">
                          {prediction.player_name}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          prediction.position === 'QB' ? 'bg-red-100 text-red-800' :
                          prediction.position === 'RB' ? 'bg-blue-100 text-blue-800' :
                          prediction.position === 'WR' ? 'bg-green-100 text-green-800' :
                          'bg-yellow-100 text-yellow-800'
                        }`}>
                          {prediction.position}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {prediction.team}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <div className="flex items-center justify-center gap-1">
                          <BoltIcon className="h-4 w-4 text-indigo-500" />
                          <span className="text-lg font-semibold text-gray-900">
                            {prediction.predicted_points.toFixed(1)}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <span className={`text-sm font-medium ${getConfidenceColor(prediction.confidence)}`}>
                          {getConfidenceLabel(prediction.confidence)}
                          <span className="text-xs text-gray-500 ml-1">
                            ({Math.round(prediction.confidence * 100)}%)
                          </span>
                        </span>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Info Box */}
          <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-blue-900 mb-2">
              How Our Predictions Work
            </h3>
            <ul className="space-y-2 text-sm text-blue-800">
              <li className="flex items-start gap-2">
                <CheckCircleIcon className="h-5 w-5 flex-shrink-0 mt-0.5" />
                <span>Updated hourly with latest injury reports and weather data</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircleIcon className="h-5 w-5 flex-shrink-0 mt-0.5" />
                <span>ML models analyze 100+ factors including matchups, trends, and historical performance</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircleIcon className="h-5 w-5 flex-shrink-0 mt-0.5" />
                <span>Confidence scores help you identify the most reliable predictions</span>
              </li>
            </ul>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}

// Mock data generator for fallback
function generateMockPredictions(): WeeklyPrediction[] {
  const players = [
    { name: 'Patrick Mahomes', position: 'QB', team: 'KC' },
    { name: 'Josh Allen', position: 'QB', team: 'BUF' },
    { name: 'Christian McCaffrey', position: 'RB', team: 'SF' },
    { name: 'Austin Ekeler', position: 'RB', team: 'LAC' },
    { name: 'Tyreek Hill', position: 'WR', team: 'MIA' },
    { name: 'Justin Jefferson', position: 'WR', team: 'MIN' },
    { name: 'Travis Kelce', position: 'TE', team: 'KC' },
    { name: 'Mark Andrews', position: 'TE', team: 'BAL' },
  ]

  return players.map((player, idx) => ({
    player_id: `${player.name.toLowerCase().replace(' ', '_')}`,
    player_name: player.name,
    position: player.position,
    team: player.team,
    week: 1,
    predicted_points: 25 - idx * 1.5 + Math.random() * 5,
    confidence: 0.95 - idx * 0.05
  }))
}

// Add this import if not already present
import { CheckCircleIcon } from '@heroicons/react/24/outline'