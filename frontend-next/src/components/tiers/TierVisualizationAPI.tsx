'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  ChartBarIcon, 
  ArrowDownIcon, 
  InformationCircleIcon, 
  TableCellsIcon, 
  PresentationChartLineIcon,
  ArrowPathIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { TierChart } from './TierChart'
import { tiersApi, type Tier, type TierBreak, type Player as TierPlayer } from '@/lib/api/tiers'

type Position = 'QB' | 'RB' | 'WR' | 'TE'
type ScoringType = 'standard' | 'half' | 'ppr'

export function TierVisualizationAPI() {
  const [selectedPosition, setSelectedPosition] = useState<Position>('QB')
  const [scoringType, setScoringType] = useState<ScoringType>('ppr')
  const [hoveredTier, setHoveredTier] = useState<number | null>(null)
  const [showBreaks, setShowBreaks] = useState(true)
  const [viewMode, setViewMode] = useState<'grid' | 'chart'>('grid')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [tiers, setTiers] = useState<Tier[]>([])
  const [tierBreaks, setTierBreaks] = useState<TierBreak[]>([])
  const [lastUpdated, setLastUpdated] = useState<string>('')

  const positions: Position[] = ['QB', 'RB', 'WR', 'TE']
  const scoringTypes: { value: ScoringType; label: string }[] = [
    { value: 'standard', label: 'Standard' },
    { value: 'half', label: 'Half PPR' },
    { value: 'ppr', label: 'Full PPR' }
  ]

  // Fetch tier data
  const fetchTierData = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const data = await tiersApi.getPositionTiers(selectedPosition, scoringType)
      setTiers(data.tiers)
      setTierBreaks(data.tier_breaks)
      setLastUpdated(data.updated_at)
    } catch (err) {
      console.error('Error fetching tier data:', err)
      setError('Using demo data - API temporarily unavailable.')
      
      // Use fallback mock data
      const mockData = generateMockTierData(selectedPosition)
      setTiers(mockData.tiers)
      setTierBreaks(mockData.tier_breaks)
      setLastUpdated(new Date().toISOString())
    } finally {
      setLoading(false)
    }
  }

  // Fetch data on component mount and when position/scoring changes
  useEffect(() => {
    fetchTierData()
  }, [selectedPosition, scoringType])

  // Auto-refresh every 5 minutes
  useEffect(() => {
    const interval = setInterval(fetchTierData, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [selectedPosition, scoringType])

  const getPlayerRank = (playerId: string): number => {
    let rank = 1
    for (const tier of tiers) {
      for (const player of tier.players) {
        if (player.id === playerId) return rank
        rank++
      }
    }
    return rank
  }

  const getTierBreak = (tierNumber: number) => {
    return tierBreaks?.find(tb => tb.between_tiers[0] === tierNumber)
  }

  const formatLastUpdated = () => {
    if (!lastUpdated) return ''
    const date = new Date(lastUpdated)
    return date.toLocaleString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      hour: '2-digit', 
      minute: '2-digit' 
    })
  }

  return (
    <div className="space-y-8">
      {/* Controls Row */}
      <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
        {/* Position Selector */}
        <div className="inline-flex rounded-lg bg-white/10 backdrop-blur-sm border border-white/20 p-1">
          {positions.map((position) => (
            <button
              key={position}
              onClick={() => setSelectedPosition(position)}
              className={`px-6 py-3 text-sm font-semibold rounded-md transition-all duration-200 ${
                selectedPosition === position
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'text-gray-300 hover:text-white hover:bg-white/10'
              }`}
            >
              {position}
            </button>
          ))}
        </div>

        {/* Scoring Type Selector */}
        <div className="inline-flex rounded-lg bg-white/10 backdrop-blur-sm border border-white/20 p-1">
          {scoringTypes.map((type) => (
            <button
              key={type.value}
              onClick={() => setScoringType(type.value)}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                scoringType === type.value
                  ? 'bg-indigo-600 text-white shadow-lg'
                  : 'text-gray-300 hover:text-white hover:bg-white/10'
              }`}
            >
              {type.label}
            </button>
          ))}
        </div>
      </div>

      {/* View Mode and Options */}
      <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
        {/* View Mode Toggle */}
        <div className="inline-flex rounded-lg bg-white/10 backdrop-blur-sm border border-white/20 p-1">
          <button
            onClick={() => setViewMode('grid')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 flex items-center gap-2 ${
              viewMode === 'grid'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'text-gray-300 hover:text-white hover:bg-white/10'
            }`}
          >
            <TableCellsIcon className="w-4 h-4" />
            Grid View
          </button>
          <button
            onClick={() => setViewMode('chart')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 flex items-center gap-2 ${
              viewMode === 'chart'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'text-gray-300 hover:text-white hover:bg-white/10'
            }`}
          >
            <PresentationChartLineIcon className="w-4 h-4" />
            Chart View
          </button>
        </div>

        {/* Options and Refresh */}
        <div className="flex items-center gap-4">
          {viewMode === 'grid' && (
            <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
              <input
                type="checkbox"
                checked={showBreaks}
                onChange={(e) => setShowBreaks(e.target.checked)}
                className="rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
              />
              Show value drops
            </label>
          )}
          
          <button
            onClick={fetchTierData}
            disabled={loading}
            className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-300 hover:text-white bg-white/10 hover:bg-white/20 rounded-md transition-all duration-200 disabled:opacity-50"
          >
            <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Last Updated */}
      {lastUpdated && (
        <div className="text-sm text-gray-400 text-center">
          Last updated: {formatLastUpdated()}
        </div>
      )}

      {/* Main Content */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-20">
          <ArrowPathIcon className="w-12 h-12 text-blue-500 animate-spin mb-4" />
          <p className="text-gray-300">Loading tier data...</p>
        </div>
      ) : error ? (
        <div className="flex flex-col items-center justify-center py-20">
          <ExclamationTriangleIcon className="w-12 h-12 text-red-500 mb-4" />
          <p className="text-red-400 mb-4">{error}</p>
          <button
            onClick={fetchTierData}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-500 transition-colors"
          >
            Try Again
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          <AnimatePresence mode="wait">
            {viewMode === 'chart' ? (
              <TierChart key={selectedPosition} position={selectedPosition} />
            ) : (
              tiers.map((tier, tierIndex) => {
                const tierBreak = getTierBreak(tier.tier)
                const isHovered = hoveredTier === tier.tier
                
                return (
                  <motion.div
                    key={`${selectedPosition}-${tier.tier}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3, delay: tierIndex * 0.05 }}
                    className="space-y-2"
                  >
                    {/* Tier Break Indicator */}
                    {showBreaks && tierBreak && tierIndex > 0 && (
                      <motion.div
                        initial={{ opacity: 0, scaleY: 0 }}
                        animate={{ opacity: 1, scaleY: 1 }}
                        className="flex items-center justify-center py-2"
                      >
                        <div className="flex items-center gap-3 bg-red-500/20 border border-red-400/30 rounded-lg px-4 py-2">
                          <ArrowDownIcon className="w-4 h-4 text-red-400" />
                          <span className="text-sm font-medium text-red-400">
                            {tierBreak.point_gap} point drop • {tierBreak.significance}
                          </span>
                          <span className="text-xs text-red-300">
                            {tierBreak.recommendation}
                          </span>
                        </div>
                      </motion.div>
                    )}

                    {/* Tier Header */}
                    <div
                      className={`flex items-center justify-between p-4 rounded-t-lg transition-all duration-200 ${
                        isHovered ? 'shadow-lg scale-[1.01]' : ''
                      }`}
                      style={{ 
                        backgroundColor: `${tier.color}20`,
                        borderColor: `${tier.color}40`,
                        borderWidth: '1px'
                      }}
                      onMouseEnter={() => setHoveredTier(tier.tier)}
                      onMouseLeave={() => setHoveredTier(null)}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className="w-4 h-4 rounded-full"
                          style={{ backgroundColor: tier.color }}
                        />
                        <h3 className="text-lg font-semibold text-white">
                          Tier {tier.tier}: {tier.label}
                        </h3>
                        <span className="text-sm text-gray-400">
                          ({tier.players.length} player{tier.players.length !== 1 ? 's' : ''})
                        </span>
                      </div>
                      <div className="flex items-center gap-4 text-sm text-gray-400">
                        <div className="flex items-center gap-2">
                          <ChartBarIcon className="w-4 h-4" />
                          <span>{tier.avg_points} avg pts</span>
                        </div>
                        <span className="text-xs">
                          ({tier.point_range.min} - {tier.point_range.max})
                        </span>
                      </div>
                    </div>

                    {/* Players Grid */}
                    <motion.div
                      className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 p-4 bg-white/5 rounded-b-lg border border-white/10"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      transition={{ duration: 0.3, delay: tierIndex * 0.05 + 0.05 }}
                    >
                      {tier.players.map((player, playerIndex) => {
                        const rank = getPlayerRank(player.id)
                        
                        return (
                          <motion.div
                            key={player.id}
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.2, delay: tierIndex * 0.05 + playerIndex * 0.02 }}
                            className="group relative bg-white/10 backdrop-blur-sm rounded-lg p-4 hover:bg-white/20 transition-all duration-200 cursor-pointer"
                          >
                            {/* Player Rank Badge */}
                            <div className="absolute -top-2 -left-2 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center text-xs font-bold text-white">
                              {rank}
                            </div>

                            {/* Injury Indicator */}
                            {player.injury_status && (
                              <div className="absolute -top-2 -right-2 px-2 py-0.5 bg-red-500 rounded text-xs font-medium text-white">
                                {player.injury_status}
                              </div>
                            )}

                            <div className="space-y-2">
                              <div className="flex justify-between items-start">
                                <div className="flex-1">
                                  <h4 className="font-semibold text-white text-sm group-hover:text-blue-300 transition-colors">
                                    {player.name}
                                  </h4>
                                  <p className="text-xs text-gray-400">{player.team}</p>
                                </div>
                                <div className="text-right">
                                  <p className="text-sm font-medium text-white">
                                    {player.projected_points}
                                  </p>
                                  <p className="text-xs text-gray-400">
                                    pts/week
                                  </p>
                                </div>
                              </div>

                              {/* Stats Row */}
                              <div className="flex justify-between items-center text-xs">
                                <span className="text-gray-400">
                                  {Math.round(player.consistency_score * 100)}% reliable
                                </span>
                                <span className="text-gray-400">
                                  ADP: {player.adp}
                                </span>
                              </div>

                              {/* Confidence Bar */}
                              <div className="space-y-1">
                                <div className="flex justify-between items-center text-xs">
                                  <span className="text-gray-400">Tier Fit</span>
                                  <span className="text-white">{Math.round(player.tier_confidence * 100)}%</span>
                                </div>
                                <div className="w-full bg-gray-700 rounded-full h-1.5">
                                  <div
                                    className="h-1.5 rounded-full transition-all duration-500"
                                    style={{
                                      width: `${player.tier_confidence * 100}%`,
                                      backgroundColor: tier.color
                                    }}
                                  />
                                </div>
                              </div>
                            </div>

                            {/* Hover Tooltip */}
                            <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10 pointer-events-none">
                              <div className="bg-gray-900 text-white text-xs rounded-lg p-3 whitespace-nowrap shadow-lg border border-gray-700 space-y-1">
                                <div className="font-semibold text-center">{player.name}</div>
                                <div className="text-gray-400 text-center">Overall: #{rank} • ADP: {player.adp}</div>
                                <div className="grid grid-cols-3 gap-2 pt-1 border-t border-gray-700">
                                  <div className="text-center">
                                    <div className="text-gray-400">Floor</div>
                                    <div className="text-red-400">{player.floor}</div>
                                  </div>
                                  <div className="text-center">
                                    <div className="text-gray-400">Proj</div>
                                    <div className="text-blue-400">{player.projected_points}</div>
                                  </div>
                                  <div className="text-center">
                                    <div className="text-gray-400">Ceiling</div>
                                    <div className="text-green-400">{player.ceiling}</div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </motion.div>
                        )
                      })}
                    </motion.div>
                  </motion.div>
                )
              })
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Legend */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
        className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
      >
        <div className="flex items-center gap-2 mb-4">
          <InformationCircleIcon className="w-5 h-5 text-blue-400" />
          <h3 className="text-lg font-semibold text-white">Understanding AI-Powered Tiers</h3>
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 text-sm text-gray-300">
          <div>
            <h4 className="font-medium text-white mb-2">Tier Fit %</h4>
            <p>ML confidence that this player belongs in their assigned tier based on projected performance.</p>
          </div>
          <div>
            <h4 className="font-medium text-white mb-2">Value Drops</h4>
            <p>Red indicators show significant fantasy point drops. Don't reach past these natural breakpoints.</p>
          </div>
          <div>
            <h4 className="font-medium text-white mb-2">Reliability Score</h4>
            <p>Consistency of weekly performance. 90%+ = very steady, 70%- = boom or bust player.</p>
          </div>
          <div>
            <h4 className="font-medium text-white mb-2">Floor/Ceiling</h4>
            <p>AI-predicted range of likely outcomes. Floor = bad week, Ceiling = best case scenario.</p>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

// Mock data generator for fallback when API is unavailable
function generateMockTierData(position: Position) {
  const playersByPosition = {
    QB: [
      'Josh Allen', 'Lamar Jackson', 'Jalen Hurts', 'Joe Burrow', 'Justin Herbert',
      'Tua Tagovailoa', 'Dak Prescott', 'Russell Wilson', 'Aaron Rodgers', 'Kirk Cousins',
      'Geno Smith', 'Daniel Jones', 'Trevor Lawrence', 'Deshaun Watson', 'Kenny Pickett',
      'Mac Jones', 'Jordan Love', 'Brock Purdy', 'Anthony Richardson', 'C.J. Stroud'
    ],
    RB: [
      'Christian McCaffrey', 'Austin Ekeler', 'Derrick Henry', 'Saquon Barkley', 'Nick Chubb',
      'Dalvin Cook', 'Jonathan Taylor', 'Josh Jacobs', 'Aaron Jones', 'Alvin Kamara',
      'Joe Mixon', 'Najee Harris', 'Kenneth Walker III', 'Tony Pollard', 'Miles Sanders',
      'Dameon Pierce', 'Breece Hall', 'Travis Etienne', 'Cam Akers', 'D\'Andre Swift'
    ],
    WR: [
      'Cooper Kupp', 'Stefon Diggs', 'Tyreek Hill', 'Davante Adams', 'DeAndre Hopkins',
      'A.J. Brown', 'Ja\'Marr Chase', 'Justin Jefferson', 'CeeDee Lamb', 'Mike Evans',
      'Keenan Allen', 'DK Metcalf', 'Tyler Lockett', 'Amari Cooper', 'Terry McLaurin',
      'Michael Pittman Jr.', 'Jaylen Waddle', 'Amon-Ra St. Brown', 'Chris Godwin', 'Tee Higgins'
    ],
    TE: [
      'Travis Kelce', 'Mark Andrews', 'T.J. Hockenson', 'George Kittle', 'Kyle Pitts',
      'Darren Waller', 'Dallas Goedert', 'Pat Freiermuth', 'Tyler Higbee', 'Evan Engram',
      'David Njoku', 'Gerald Everett', 'Hayden Hurst', 'Logan Thomas', 'Robert Tonyan',
      'Mike Gesicki', 'Noah Fant', 'Hunter Henry', 'Zach Ertz', 'Cole Kmet'
    ]
  }

  const teams = ['KC', 'BUF', 'PHI', 'CIN', 'LAC', 'MIA', 'DAL', 'DEN', 'GB', 'MIN', 'SEA', 'NYG', 'JAX', 'CLE', 'PIT', 'NE', 'GB', 'SF', 'LAR', 'TB']
  const players = playersByPosition[position] || playersByPosition.QB
  
  const tierConfig = {
    QB: { sizes: [3, 3, 3, 3, 8], labels: ['Elite', 'High QB1', 'Mid QB1', 'Low QB1', 'Streaming'], basePoints: 22 },
    RB: { sizes: [5, 5, 6, 4], labels: ['Elite RB1', 'High RB1', 'RB2', 'Flex'], basePoints: 16 },
    WR: { sizes: [5, 5, 6, 4], labels: ['Elite WR1', 'High WR1', 'WR2', 'Flex'], basePoints: 15 },
    TE: { sizes: [3, 3, 4, 10], labels: ['Elite TE1', 'High TE1', 'Streaming', 'Waiver'], basePoints: 11 }
  }

  const config = tierConfig[position]
  const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#95A5A6']
  
  let playerIndex = 0
  const tiers = []
  
  config.sizes.forEach((size, tierIndex) => {
    const tierPlayers = []
    const pointDecline = tierIndex * 3
    
    for (let i = 0; i < size && playerIndex < players.length; i++) {
      tierPlayers.push({
        id: `${players[playerIndex].toLowerCase().replace(/\s+/g, '_')}_${playerIndex}`,
        name: players[playerIndex],
        team: teams[playerIndex % teams.length],
        tier_confidence: 0.95 - tierIndex * 0.05 - Math.random() * 0.1,
        projected_points: config.basePoints - pointDecline - Math.random() * 2,
        consistency_score: 0.85 - tierIndex * 0.05 - Math.random() * 0.1,
        floor: config.basePoints - pointDecline - 4 - Math.random() * 2,
        ceiling: config.basePoints - pointDecline + 6 + Math.random() * 3,
        adp: playerIndex + 1,
        injury_status: Math.random() < 0.05 ? 'Questionable' : null
      })
      playerIndex++
    }
    
    if (tierPlayers.length > 0) {
      tiers.push({
        tier: tierIndex + 1,
        label: config.labels[tierIndex],
        color: colors[tierIndex],
        players: tierPlayers,
        avg_points: tierPlayers.reduce((sum, p) => sum + p.projected_points, 0) / tierPlayers.length,
        point_range: {
          min: Math.min(...tierPlayers.map(p => p.projected_points)),
          max: Math.max(...tierPlayers.map(p => p.projected_points))
        }
      })
    }
  })
  
  const tier_breaks = []
  for (let i = 0; i < tiers.length - 1; i++) {
    const gap = tiers[i].point_range.min - tiers[i + 1].point_range.max
    if (gap > 1) {
      tier_breaks.push({
        between_tiers: [tiers[i].tier, tiers[i + 1].tier],
        point_gap: gap,
        significance: gap > 3 ? 'Major drop-off' : 'Moderate drop',
        recommendation: `Target ${tiers[i].label} players before this break`
      })
    }
  }
  
  return { tiers, tier_breaks }
}