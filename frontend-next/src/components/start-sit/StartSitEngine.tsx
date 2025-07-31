'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  PlusIcon, 
  XMarkIcon, 
  CheckIcon,
  ExclamationTriangleIcon,
  ChartBarIcon,
  BoltIcon,
  ShieldCheckIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon
} from '@heroicons/react/24/outline'
import tiersData from '@/data/tiers_2024.json'
import predictionsData from '@/data/predictions_2024.json'

interface Player {
  id: string
  name: string
  team: string
  position: string
  tier: number
  tier_confidence: number
  projected_points: number
  consistency_score: number
  weekly_prediction?: {
    projected: number
    floor: number
    ceiling: number
    confidence: number
    matchup_rating: number
    factors: {
      opponent: string
      home_away: string
      weather_impact: number
      injury_status: string
    }
  }
}

interface Comparison {
  id: string
  players: Player[]
  position: string
}

export function StartSitEngine() {
  const [activeComparisons, setActiveComparisons] = useState<Comparison[]>([])
  const [availablePlayers, setAvailablePlayers] = useState<Player[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedPosition, setSelectedPosition] = useState<'ALL' | 'QB' | 'RB' | 'WR' | 'TE'>('ALL')

  // Initialize available players with weekly predictions
  useEffect(() => {
    const allPlayers: Player[] = []
    
    Object.entries(tiersData.tiers).forEach(([position, positionTiers]) => {
      positionTiers.forEach((tier: any) => {
        tier.players.forEach((player: any) => {
          const weeklyPrediction = predictionsData.weekly_predictions.week_1[player.id as keyof typeof predictionsData.weekly_predictions.week_1]
          
          allPlayers.push({
            ...player,
            position,
            tier: tier.tier,
            weekly_prediction: weeklyPrediction ? {
              projected: weeklyPrediction.projected,
              floor: weeklyPrediction.floor,
              ceiling: weeklyPrediction.ceiling,
              confidence: weeklyPrediction.confidence,
              matchup_rating: weeklyPrediction.matchup_rating,
              factors: weeklyPrediction.factors
            } : undefined
          })
        })
      })
    })
    
    setAvailablePlayers(allPlayers)
  }, [])

  const createComparison = (position: string = 'ALL') => {
    const newComparison: Comparison = {
      id: Math.random().toString(36).substr(2, 9),
      players: [],
      position
    }
    setActiveComparisons(prev => [...prev, newComparison])
  }

  const addPlayerToComparison = (comparisonId: string, player: Player) => {
    setActiveComparisons(prev => prev.map(comp => 
      comp.id === comparisonId 
        ? { 
            ...comp, 
            players: [...comp.players, player],
            position: comp.position === 'ALL' ? player.position : comp.position
          }
        : comp
    ))
  }

  const removePlayerFromComparison = (comparisonId: string, playerId: string) => {
    setActiveComparisons(prev => prev.map(comp => 
      comp.id === comparisonId 
        ? { ...comp, players: comp.players.filter(p => p.id !== playerId) }
        : comp
    ))
  }

  const removeComparison = (comparisonId: string) => {
    setActiveComparisons(prev => prev.filter(comp => comp.id !== comparisonId))
  }

  const getRecommendation = (players: Player[]) => {
    if (players.length < 2) return null
    
    const bestPlayer = players.reduce((best, current) => {
      const bestScore = (best.weekly_prediction?.projected || best.projected_points) * 
                        (best.weekly_prediction?.confidence || best.tier_confidence)
      const currentScore = (current.weekly_prediction?.projected || current.projected_points) * 
                          (current.weekly_prediction?.confidence || current.tier_confidence)
      return currentScore > bestScore ? current : best
    })

    return bestPlayer
  }

  const getPlayerScore = (player: Player) => {
    return (player.weekly_prediction?.projected || player.projected_points) * 
           (player.weekly_prediction?.confidence || player.tier_confidence)
  }

  const filteredPlayers = availablePlayers.filter(player => {
    const matchesSearch = player.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         player.team.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesPosition = selectedPosition === 'ALL' || player.position === selectedPosition
    const notInComparisons = !activeComparisons.some(comp => 
      comp.players.some(p => p.id === player.id)
    )
    
    return matchesSearch && matchesPosition && notInComparisons
  }).slice(0, 20)

  return (
    <div className="space-y-8">
      {/* Add Comparison Button */}
      <div className="flex justify-center">
        <button
          onClick={() => createComparison()}
          className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors font-semibold"
        >
          <PlusIcon className="w-5 h-5" />
          New Comparison
        </button>
      </div>

      {/* Active Comparisons */}
      <div className="space-y-6">
        <AnimatePresence>
          {activeComparisons.map((comparison) => {
            const recommendation = getRecommendation(comparison.players)
            
            return (
              <motion.div
                key={comparison.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="bg-white/10 backdrop-blur-sm rounded-lg border border-white/20 overflow-hidden"
              >
                {/* Comparison Header */}
                <div className="flex justify-between items-center p-4 border-b border-white/10">
                  <div className="flex items-center gap-3">
                    <h3 className="text-lg font-semibold text-white">
                      {comparison.position !== 'ALL' ? `${comparison.position} ` : ''}Comparison
                    </h3>
                    {comparison.players.length > 0 && (
                      <span className="text-sm text-gray-400">
                        ({comparison.players.length} player{comparison.players.length !== 1 ? 's' : ''})
                      </span>
                    )}
                  </div>
                  <button
                    onClick={() => removeComparison(comparison.id)}
                    className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
                  >
                    <XMarkIcon className="w-5 h-5" />
                  </button>
                </div>

                {/* Recommendation */}
                {recommendation && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="p-4 bg-green-500/20 border-b border-green-400/30"
                  >
                    <div className="flex items-center gap-3 mb-2">
                      <CheckIcon className="w-5 h-5 text-green-400" />
                      <span className="font-semibold text-green-300">Recommendation: Start {recommendation.name}</span>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <div className="text-gray-400">Projected Points</div>
                        <div className="font-semibold text-white">
                          {recommendation.weekly_prediction?.projected || Math.round(recommendation.projected_points)}
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400">Confidence</div>
                        <div className="font-semibold text-white">
                          {Math.round((recommendation.weekly_prediction?.confidence || recommendation.tier_confidence) * 100)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400">Matchup</div>
                        <div className={`font-semibold ${
                          (recommendation.weekly_prediction?.matchup_rating || 5) >= 7 ? 'text-green-400' :
                          (recommendation.weekly_prediction?.matchup_rating || 5) >= 5 ? 'text-yellow-400' : 'text-red-400'
                        }`}>
                          {recommendation.weekly_prediction?.matchup_rating || 'N/A'}/10
                        </div>
                      </div>
                      <div>
                        <div className="text-gray-400">vs {recommendation.weekly_prediction?.factors.opponent || 'TBD'}</div>
                        <div className="font-semibold text-white capitalize">
                          {recommendation.weekly_prediction?.factors.home_away || 'N/A'}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Players Grid */}
                <div className="p-4">
                  {comparison.players.length === 0 ? (
                    <div className="text-center py-8">
                      <p className="text-gray-400 mb-4">Add players to compare their weekly outlook</p>
                      <p className="text-sm text-gray-500">Search for players below to get started</p>
                    </div>
                  ) : (
                    <div className="grid gap-4">
                      {comparison.players.map((player, index) => {
                        const isRecommended = recommendation?.id === player.id
                        const playerScore = getPlayerScore(player)
                        const maxScore = Math.max(...comparison.players.map(getPlayerScore))
                        const scorePercentage = (playerScore / maxScore) * 100

                        return (
                          <motion.div
                            key={player.id}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                            className={`relative p-4 rounded-lg border transition-all duration-200 ${
                              isRecommended 
                                ? 'bg-green-600/20 border-green-400/40 shadow-lg' 
                                : 'bg-white/5 border-white/10'
                            }`}
                          >
                            {isRecommended && (
                              <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                                <CheckIcon className="w-4 h-4 text-white" />
                              </div>
                            )}

                            <div className="flex justify-between items-start mb-3">
                              <div className="flex items-center gap-3">
                                <div>
                                  <h4 className="font-semibold text-white">{player.name}</h4>
                                  <p className="text-sm text-gray-400">{player.team} • {player.position}</p>
                                </div>
                              </div>
                              <button
                                onClick={() => removePlayerFromComparison(comparison.id, player.id)}
                                className="p-1 text-gray-400 hover:text-white transition-colors"
                              >
                                <XMarkIcon className="w-4 h-4" />
                              </button>
                            </div>

                            {/* Player Stats Grid */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                              <div className="text-center">
                                <div className="flex items-center justify-center gap-1 mb-1">
                                  <BoltIcon className="w-4 h-4 text-blue-400" />
                                  <span className="text-xs text-gray-400">Projected</span>
                                </div>
                                <div className="font-semibold text-white">
                                  {player.weekly_prediction?.projected || Math.round(player.projected_points)}
                                </div>
                              </div>

                              <div className="text-center">
                                <div className="flex items-center justify-center gap-1 mb-1">
                                  <ShieldCheckIcon className="w-4 h-4 text-green-400" />
                                  <span className="text-xs text-gray-400">Floor</span>
                                </div>
                                <div className="font-semibold text-green-400">
                                  {player.weekly_prediction?.floor || Math.round(player.projected_points * 0.7)}
                                </div>
                              </div>

                              <div className="text-center">
                                <div className="flex items-center justify-center gap-1 mb-1">
                                  <ArrowTrendingUpIcon className="w-4 h-4 text-purple-400" />
                                  <span className="text-xs text-gray-400">Ceiling</span>
                                </div>
                                <div className="font-semibold text-purple-400">
                                  {player.weekly_prediction?.ceiling || Math.round(player.projected_points * 1.3)}
                                </div>
                              </div>

                              <div className="text-center">
                                <div className="flex items-center justify-center gap-1 mb-1">
                                  <ChartBarIcon className="w-4 h-4 text-yellow-400" />
                                  <span className="text-xs text-gray-400">Confidence</span>
                                </div>
                                <div className="font-semibold text-white">
                                  {Math.round((player.weekly_prediction?.confidence || player.tier_confidence) * 100)}%
                                </div>
                              </div>
                            </div>

                            {/* Matchup Info */}
                            {player.weekly_prediction && (
                              <div className="flex items-center justify-between text-sm border-t border-white/10 pt-3">
                                <div className="flex items-center gap-2">
                                  <span className="text-gray-400">vs {player.weekly_prediction.factors.opponent}</span>
                                  <span className="capitalize text-gray-400">
                                    ({player.weekly_prediction.factors.home_away})
                                  </span>
                                </div>
                                <div className="flex items-center gap-2">
                                  <span className="text-gray-400">Matchup:</span>
                                  <span className={`font-semibold ${
                                    player.weekly_prediction.matchup_rating >= 7 ? 'text-green-400' :
                                    player.weekly_prediction.matchup_rating >= 5 ? 'text-yellow-400' : 'text-red-400'
                                  }`}>
                                    {player.weekly_prediction.matchup_rating}/10
                                  </span>
                                </div>
                              </div>
                            )}

                            {/* Score Bar */}
                            <div className="mt-3">
                              <div className="flex justify-between items-center text-xs mb-1">
                                <span className="text-gray-400">Overall Score</span>
                                <span className="text-white font-semibold">{Math.round(scorePercentage)}%</span>
                              </div>
                              <div className="w-full bg-gray-700 rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full transition-all duration-500 ${
                                    isRecommended ? 'bg-green-500' : 'bg-blue-500'
                                  }`}
                                  style={{ width: `${scorePercentage}%` }}
                                />
                              </div>
                            </div>
                          </motion.div>
                        )
                      })}
                    </div>
                  )}
                </div>
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>

      {/* Player Search & Add */}
      {activeComparisons.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
        >
          <h3 className="text-lg font-semibold text-white mb-4">Add Players to Compare</h3>
          
          {/* Search Controls */}
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <div className="flex-1">
              <input
                type="text"
                placeholder="Search players..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            {/* Position Filter */}
            <div className="inline-flex rounded-lg bg-white/10 backdrop-blur-sm border border-white/20 p-1">
              {(['ALL', 'QB', 'RB', 'WR', 'TE'] as const).map((position) => (
                <button
                  key={position}
                  onClick={() => setSelectedPosition(position)}
                  className={`px-3 py-1 text-sm font-semibold rounded-md transition-all duration-200 ${
                    selectedPosition === position
                      ? 'bg-blue-600 text-white shadow-lg'
                      : 'text-gray-300 hover:text-white hover:bg-white/10'
                  }`}
                >
                  {position}
                </button>
              ))}
            </div>
          </div>

          {/* Available Players */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-96 overflow-y-auto">
            {filteredPlayers.map((player) => (
              <div
                key={player.id}
                className="p-3 bg-white/5 hover:bg-white/10 rounded-lg border border-white/10 transition-colors"
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="flex-1 min-w-0">
                    <h4 className="font-semibold text-white text-sm truncate">{player.name}</h4>
                    <p className="text-xs text-gray-400">{player.team} • {player.position}</p>
                  </div>
                  <div className="text-right ml-2">
                    <div className="text-xs font-medium text-white">
                      {player.weekly_prediction?.projected || Math.round(player.projected_points)} pts
                    </div>
                    <div className="text-xs text-gray-400">Tier {player.tier}</div>
                  </div>
                </div>

                <div className="flex flex-wrap gap-1">
                  {activeComparisons.map((comparison) => (
                    <button
                      key={comparison.id}
                      onClick={() => addPlayerToComparison(comparison.id, player)}
                      disabled={comparison.players.length >= 5 || 
                               (comparison.position !== 'ALL' && comparison.position !== player.position)}
                      className="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded transition-colors"
                    >
                      Add to Comparison {activeComparisons.indexOf(comparison) + 1}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {filteredPlayers.length === 0 && (
            <div className="text-center py-8">
              <p className="text-gray-400">No players found matching your search</p>
            </div>
          )}
        </motion.div>
      )}

      {/* Getting Started */}
      {activeComparisons.length === 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white/10 backdrop-blur-sm rounded-lg p-8 border border-white/20 text-center"
        >
          <ExclamationTriangleIcon className="w-12 h-12 text-blue-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">Ready to Make Smart Decisions?</h3>
          <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
            Create a comparison to get AI-powered start/sit recommendations based on weekly predictions, 
            matchup analysis, and player tier data.
          </p>
          <button
            onClick={() => createComparison()}
            className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors font-semibold"
          >
            <PlusIcon className="w-5 h-5" />
            Create Your First Comparison
          </button>
        </motion.div>
      )}
    </div>
  )
}