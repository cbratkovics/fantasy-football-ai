'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChartBarIcon, ArrowDownIcon, InformationCircleIcon, TableCellsIcon, PresentationChartLineIcon } from '@heroicons/react/24/outline'
import tiersData from '@/data/tiers_2024.json'
import { TierChart } from './TierChart'

interface Player {
  id: string
  name: string
  team: string
  tier_confidence: number
  projected_points: number
  consistency_score: number
}

interface Tier {
  tier: number
  label: string
  color: string
  players: Player[]
}

type Position = 'QB' | 'RB' | 'WR' | 'TE'

export function TierVisualization() {
  const [selectedPosition, setSelectedPosition] = useState<Position>('QB')
  const [hoveredTier, setHoveredTier] = useState<number | null>(null)
  const [showBreaks, setShowBreaks] = useState(true)
  const [viewMode, setViewMode] = useState<'grid' | 'chart'>('grid')

  const positions: Position[] = ['QB', 'RB', 'WR', 'TE']
  const tiers = tiersData.tiers[selectedPosition] as Tier[]
  const tierBreaks = tiersData.tier_breaks[selectedPosition]

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

  return (
    <div className="space-y-8">
      {/* Position Selector */}
      <div className="flex justify-center">
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
      </div>

      {/* Controls */}
      <div className="flex justify-center items-center gap-6">
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

        {/* Show Breaks Toggle (only for grid view) */}
        {viewMode === 'grid' && (
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
            <input
              type="checkbox"
              checked={showBreaks}
              onChange={(e) => setShowBreaks(e.target.checked)}
              className="rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
            />
            Show tier breaks
          </label>
        )}
      </div>

      {/* Tier Visualization */}
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
                transition={{ duration: 0.3, delay: tierIndex * 0.1 }}
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
                        {tierBreak.point_gap} point drop-off • {tierBreak.significance}
                      </span>
                    </div>
                  </motion.div>
                )}

                {/* Tier Header */}
                <div
                  className={`flex items-center justify-between p-4 rounded-t-lg transition-all duration-200 ${
                    isHovered ? 'shadow-lg scale-[1.02]' : ''
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
                  <div className="flex items-center gap-2 text-sm text-gray-400">
                    <ChartBarIcon className="w-4 h-4" />
                    <span>
                      {Math.round(tier.players.reduce((acc, p) => acc + p.projected_points, 0) / tier.players.length)} avg pts
                    </span>
                  </div>
                </div>

                {/* Players Grid */}
                <motion.div
                  className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 p-4 bg-white/5 rounded-b-lg border border-white/10"
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  transition={{ duration: 0.3, delay: tierIndex * 0.1 + 0.1 }}
                >
                  {tier.players.map((player, playerIndex) => {
                    const rank = getPlayerRank(player.id)
                    
                    return (
                      <motion.div
                        key={player.id}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.2, delay: tierIndex * 0.1 + playerIndex * 0.05 }}
                        className="group relative bg-white/10 backdrop-blur-sm rounded-lg p-4 hover:bg-white/20 transition-all duration-200 cursor-pointer"
                      >
                        {/* Player Rank Badge */}
                        <div className="absolute -top-2 -left-2 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center text-xs font-bold text-white">
                          {rank}
                        </div>

                        <div className="space-y-2">
                          <div className="flex justify-between items-start">
                            <div>
                              <h4 className="font-semibold text-white text-sm group-hover:text-blue-300 transition-colors">
                                {player.name}
                              </h4>
                              <p className="text-xs text-gray-400">{player.team}</p>
                            </div>
                            <div className="text-right">
                              <p className="text-sm font-medium text-white">
                                {Math.round(player.projected_points)} pts
                              </p>
                              <p className="text-xs text-gray-400">
                                {Math.round(player.consistency_score * 100)}% consistent
                              </p>
                            </div>
                          </div>

                          {/* Confidence Bar */}
                          <div className="space-y-1">
                            <div className="flex justify-between items-center text-xs">
                              <span className="text-gray-400">Tier Confidence</span>
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
                        <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10">
                          <div className="bg-gray-900 text-white text-xs rounded-lg p-2 whitespace-nowrap shadow-lg border border-gray-700">
                            <div className="text-center">
                              <div className="font-semibold">{player.name}</div>
                              <div className="text-gray-400">Overall Rank #{rank}</div>
                              <div className="text-blue-400">{player.projected_points} projected points</div>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )
                  })}
                </motion.div>
              </motion.div>
            )
          }))}
        </AnimatePresence>
      </div>

      {/* Legend */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
        className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
      >
        <div className="flex items-center gap-2 mb-4">
          <InformationCircleIcon className="w-5 h-5 text-blue-400" />
          <h3 className="text-lg font-semibold text-white">How to Read the Tiers</h3>
        </div>
        <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>
            <h4 className="font-medium text-white mb-2">Tier Confidence</h4>
            <p>How certain our GMM algorithm is about a player's tier placement. Higher confidence means more reliable tier assignment.</p>
          </div>
          <div>
            <h4 className="font-medium text-white mb-2">Tier Breaks</h4>
            <p>Significant point gaps between tiers indicate natural draft breakpoints. Major drops suggest tier boundaries.</p>
          </div>
          <div>
            <h4 className="font-medium text-white mb-2">Consistency Score</h4>
            <p>Measures week-to-week performance reliability. Higher scores indicate more predictable fantasy production.</p>
          </div>
          <div>
            <h4 className="font-medium text-white mb-2">Draft Strategy</h4>
            <p>Target players within the same tier interchangeably. Wait for tier breaks to maximize value in your draft.</p>
          </div>
        </div>
      </motion.div>
    </div>
  )
}