'use client'

import { motion } from 'framer-motion'
import { 
  TrophyIcon, 
  ChartBarIcon, 
  BoltIcon, 
  ShieldCheckIcon,
  ArrowTrendingUpIcon,
  CalendarDaysIcon,
  MapPinIcon,
  HeartIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import predictionsData from '@/data/predictions_2024.json'
import tiersData from '@/data/tiers_2024.json'

interface PlayerProfileProps {
  player: {
    id: string
    name: string
    team: string
    position: string
    tier: number
    tierLabel: string
    tierColor: string
    tier_confidence: number
    projected_points: number
    consistency_score: number
  }
}

export function PlayerProfile({ player }: PlayerProfileProps) {
  // Get weekly prediction
  const weeklyPrediction = predictionsData.weekly_predictions.week_1[player.id as keyof typeof predictionsData.weekly_predictions.week_1]
  
  // Get season projection
  const seasonProjection = predictionsData.season_projections[player.id as keyof typeof predictionsData.season_projections]

  // Get tier breaks to show value gaps
  const tierBreaks = tiersData.tier_breaks[player.position as keyof typeof tiersData.tier_breaks]
  const relevantTierBreak = tierBreaks?.find(tb => tb.between_tiers[0] === player.tier)

  // Get other players in same tier for comparison
  const sameTierPlayers = tiersData.tiers[player.position as keyof typeof tiersData.tiers].find((t: any) => t.tier === player.tier)?.players || []
  const tierRankInTier = sameTierPlayers.findIndex((p: any) => p.id === player.id) + 1

  // Calculate overall rank
  let overallRank = 1
  const allTiers = tiersData.tiers[player.position as keyof typeof tiersData.tiers]
  for (const tier of allTiers) {
    for (const p of tier.players) {
      if (p.id === player.id) break
      overallRank++
    }
    if (tier.players.some((p: any) => p.id === player.id)) break
  }

  return (
    <div className="mx-auto max-w-7xl px-6 lg:px-8">
      <div className="grid lg:grid-cols-3 gap-8">
        {/* Main Profile */}
        <div className="lg:col-span-2 space-y-6">
          {/* Player Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
          >
            <div className="flex items-start justify-between mb-6">
              <div>
                <h1 className="text-4xl font-bold text-white mb-2">{player.name}</h1>
                <div className="flex items-center gap-4 text-lg">
                  <span className="text-gray-300">{player.team}</span>
                  <span className="text-gray-400">•</span>
                  <span className="text-blue-400 font-semibold">{player.position}</span>
                  <span className="text-gray-400">•</span>
                  <span className="text-gray-300">Overall #{overallRank}</span>
                </div>
              </div>
              
              {/* Tier Badge */}
              <div 
                className="px-4 py-2 rounded-lg border-2 text-center min-w-[120px]"
                style={{ 
                  backgroundColor: `${player.tierColor}20`,
                  borderColor: `${player.tierColor}60`
                }}
              >
                <div className="text-sm text-gray-300">Tier {player.tier}</div>
                <div className="font-semibold text-white">{player.tierLabel}</div>
                <div className="text-xs text-gray-400">#{tierRankInTier} in tier</div>
              </div>
            </div>

            {/* Key Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-white/5 rounded-lg">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <BoltIcon className="w-5 h-5 text-blue-400" />
                  <span className="text-sm text-gray-400">Projected</span>
                </div>
                <div className="text-2xl font-bold text-white">{Math.round(player.projected_points)}</div>
                <div className="text-xs text-gray-400">Season Points</div>
              </div>

              <div className="text-center p-4 bg-white/5 rounded-lg">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <ShieldCheckIcon className="w-5 h-5 text-green-400" />
                  <span className="text-sm text-gray-400">Consistency</span>
                </div>
                <div className="text-2xl font-bold text-white">{Math.round(player.consistency_score * 100)}%</div>
                <div className="text-xs text-gray-400">Week-to-Week</div>
              </div>

              <div className="text-center p-4 bg-white/5 rounded-lg">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <TrophyIcon className="w-5 h-5 text-yellow-400" />
                  <span className="text-sm text-gray-400">Confidence</span>
                </div>
                <div className="text-2xl font-bold text-white">{Math.round(player.tier_confidence * 100)}%</div>
                <div className="text-xs text-gray-400">Tier Placement</div>
              </div>

              <div className="text-center p-4 bg-white/5 rounded-lg">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <ChartBarIcon className="w-5 h-5 text-purple-400" />
                  <span className="text-sm text-gray-400">PPG</span>
                </div>
                <div className="text-2xl font-bold text-white">
                  {seasonProjection ? seasonProjection.ppg : Math.round(player.projected_points / 17)}
                </div>
                <div className="text-xs text-gray-400">Per Game</div>
              </div>
            </div>
          </motion.div>

          {/* Weekly Prediction */}
          {weeklyPrediction && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
            >
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <CalendarDaysIcon className="w-5 h-5" />
                Week 1 Prediction
              </h2>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Projected Points</span>
                    <span className="text-2xl font-bold text-blue-400">{weeklyPrediction.projected}</span>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-400">Floor</span>
                      <span className="text-green-400 font-semibold">{weeklyPrediction.floor}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-400">Ceiling</span>
                      <span className="text-purple-400 font-semibold">{weeklyPrediction.ceiling}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-400">Confidence</span>
                      <span className="text-white font-semibold">{Math.round(weeklyPrediction.confidence * 100)}%</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center gap-2 mb-2">
                    <MapPinIcon className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-300">Matchup Details</span>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-400">Opponent</span>
                      <span className="text-white font-semibold">{weeklyPrediction.factors.opponent}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-400">Location</span>
                      <span className="text-white font-semibold capitalize">{weeklyPrediction.factors.home_away}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-400">Matchup Rating</span>
                      <span className={`font-semibold ${
                        weeklyPrediction.matchup_rating >= 7 ? 'text-green-400' :
                        weeklyPrediction.matchup_rating >= 5 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {weeklyPrediction.matchup_rating}/10
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-400">Health Status</span>
                      <span className={`font-semibold capitalize ${
                        weeklyPrediction.factors.injury_status === 'healthy' ? 'text-green-400' : 'text-yellow-400'
                      }`}>
                        {weeklyPrediction.factors.injury_status}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Season Projection */}
          {seasonProjection && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
            >
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <ArrowTrendingUpIcon className="w-5 h-5" />
                2024 Season Outlook
              </h2>

              <div className="grid md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-400 mb-1">{seasonProjection.total_points}</div>
                  <div className="text-sm text-gray-400">Total Points</div>
                  <div className="text-xs text-gray-500 mt-1">{seasonProjection.games_played} games</div>
                </div>

                <div className="text-center">
                  <div className="text-3xl font-bold text-green-400 mb-1">{seasonProjection.boom_weeks}</div>
                  <div className="text-sm text-gray-400">Boom Weeks</div>
                  <div className="text-xs text-gray-500 mt-1">High scoring games</div>
                </div>

                <div className="text-center">
                  <div className="text-3xl font-bold text-red-400 mb-1">{seasonProjection.bust_weeks}</div>
                  <div className="text-sm text-gray-400">Bust Weeks</div>
                  <div className="text-xs text-gray-500 mt-1">Low scoring games</div>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-white/10">
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">Weekly Consistency</span>
                  <span className="text-white font-semibold">{Math.round(seasonProjection.weekly_consistency * 100)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${seasonProjection.weekly_consistency * 100}%` }}
                  />
                </div>
              </div>
            </motion.div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Tier Analysis */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
          >
            <h3 className="text-lg font-semibold text-white mb-4">Tier Analysis</h3>
            
            <div className="space-y-4">
              <div className="p-3 bg-white/5 rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-400">Tier Placement</span>
                  <span className="font-semibold" style={{ color: player.tierColor }}>
                    Tier {player.tier}
                  </span>
                </div>
                <div className="text-xs text-gray-400">
                  {player.tierLabel} • {Math.round(player.tier_confidence * 100)}% confidence
                </div>
              </div>

              {relevantTierBreak && (
                <div className="p-3 bg-red-500/10 border border-red-400/20 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <ExclamationTriangleIcon className="w-4 h-4 text-red-400" />
                    <span className="text-sm text-red-400 font-semibold">Tier Break Below</span>
                  </div>
                  <div className="text-xs text-gray-400">
                    {relevantTierBreak.point_gap} point drop to next tier • {relevantTierBreak.significance}
                  </div>
                </div>
              )}

              <div className="p-3 bg-white/5 rounded-lg">
                <div className="text-sm text-gray-400 mb-2">Same Tier Players</div>
                <div className="space-y-1">
                  {sameTierPlayers.slice(0, 5).map((p: any, index: number) => (
                    <div key={p.id} className={`text-xs flex justify-between ${
                      p.id === player.id ? 'text-white font-semibold' : 'text-gray-400'
                    }`}>
                      <span>{index + 1}. {p.name}</span>
                      <span>{Math.round(p.projected_points)} pts</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>

          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
          >
            <h3 className="text-lg font-semibold text-white mb-4">Quick Actions</h3>
            
            <div className="space-y-3">
              <button className="w-full flex items-center gap-3 p-3 bg-blue-600/20 hover:bg-blue-600/30 border border-blue-400/30 rounded-lg transition-colors text-left">
                <HeartIcon className="w-4 h-4 text-blue-400" />
                <span className="text-white">Add to Favorites</span>
              </button>
              
              <button className="w-full flex items-center gap-3 p-3 bg-green-600/20 hover:bg-green-600/30 border border-green-400/30 rounded-lg transition-colors text-left">
                <ChartBarIcon className="w-4 h-4 text-green-400" />
                <span className="text-white">Compare Players</span>
              </button>
              
              <button className="w-full flex items-center gap-3 p-3 bg-purple-600/20 hover:bg-purple-600/30 border border-purple-400/30 rounded-lg transition-colors text-left">
                <TrophyIcon className="w-4 h-4 text-purple-400" />
                <span className="text-white">View Tier Chart</span>
              </button>
            </div>
          </motion.div>

          {/* Draft Strategy */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
          >
            <h3 className="text-lg font-semibold text-white mb-4">Draft Strategy</h3>
            
            <div className="space-y-3 text-sm">
              <div className="p-3 bg-white/5 rounded-lg">
                <div className="font-semibold text-white mb-1">Target Round</div>
                <div className="text-gray-400">
                  Rounds {Math.max(1, Math.ceil(overallRank / 12) - 1)}-{Math.ceil(overallRank / 12) + 1}
                </div>
              </div>

              <div className="p-3 bg-white/5 rounded-lg">
                <div className="font-semibold text-white mb-1">Value Assessment</div>
                <div className="text-gray-400">
                  {player.tier <= 2 ? 'Premium pick - high floor/ceiling' :
                   player.tier <= 4 ? 'Solid value - reliable production' :
                   'Later round target - upside play'}
                </div>
              </div>

              {relevantTierBreak && (
                <div className="p-3 bg-yellow-500/10 border border-yellow-400/20 rounded-lg">
                  <div className="font-semibold text-yellow-400 mb-1">Tier Break</div>
                  <div className="text-gray-400 text-xs">
                    Consider waiting - significant drop-off after this tier
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}