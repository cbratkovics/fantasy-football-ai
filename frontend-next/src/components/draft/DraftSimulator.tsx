'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  UserIcon, 
  ComputerDesktopIcon, 
  PlayIcon, 
  PauseIcon,
  CheckCircleIcon,
  ClockIcon,
  TrophyIcon,
  ChartBarIcon
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
  adp?: number
}

interface DraftPick {
  round: number
  pick: number
  teamId: string
  teamName: string
  player: Player
  isUser: boolean
  timestamp: number
}

interface Team {
  id: string
  name: string
  isUser: boolean
  roster: Player[]
  picks: DraftPick[]
}

export function DraftSimulator() {
  const [draftStarted, setDraftStarted] = useState(false)
  const [currentPick, setCurrentPick] = useState(1)
  const [currentTeam, setCurrentTeam] = useState(0)
  const [timeRemaining, setTimeRemaining] = useState(90)
  const [isAutoPick, setIsAutoPick] = useState(false)
  const [draftComplete, setDraftComplete] = useState(false)
  const [selectedPosition, setSelectedPosition] = useState<'ALL' | 'QB' | 'RB' | 'WR' | 'TE'>('ALL')
  
  const [teams, setTeams] = useState<Team[]>([
    { id: '1', name: 'Your Team', isUser: true, roster: [], picks: [] },
    { id: '2', name: 'AI Team 1', isUser: false, roster: [], picks: [] },
    { id: '3', name: 'AI Team 2', isUser: false, roster: [], picks: [] },
    { id: '4', name: 'AI Team 3', isUser: false, roster: [], picks: [] },
    { id: '5', name: 'AI Team 4', isUser: false, roster: [], picks: [] },
    { id: '6', name: 'AI Team 5', isUser: false, roster: [], picks: [] },
    { id: '7', name: 'AI Team 6', isUser: false, roster: [], picks: [] },
    { id: '8', name: 'AI Team 7', isUser: false, roster: [], picks: [] },
    { id: '9', name: 'AI Team 8', isUser: false, roster: [], picks: [] },
    { id: '10', name: 'AI Team 9', isUser: false, roster: [], picks: [] },
    { id: '11', name: 'AI Team 10', isUser: false, roster: [], picks: [] },
    { id: '12', name: 'AI Team 11', isUser: false, roster: [], picks: [] },
  ])
  
  const [availablePlayers, setAvailablePlayers] = useState<Player[]>([])
  const [draftPicks, setDraftPicks] = useState<DraftPick[]>([])
  
  const totalRounds = 16
  const totalPicks = teams.length * totalRounds
  const pickTimeLimit = 90

  // Initialize available players
  useEffect(() => {
    const allPlayers: Player[] = []
    let adpCounter = 1
    
    Object.entries(tiersData.tiers).forEach(([position, positionTiers]) => {
      positionTiers.forEach((tier: any) => {
        tier.players.forEach((player: any) => {
          allPlayers.push({
            ...player,
            position,
            tier: tier.tier,
            adp: adpCounter++
          })
        })
      })
    })
    
    // Sort by projected points (descending) to simulate realistic ADP
    allPlayers.sort((a, b) => b.projected_points - a.projected_points)
    allPlayers.forEach((player, index) => {
      player.adp = index + 1
    })
    
    setAvailablePlayers(allPlayers)
  }, [])

  // Timer countdown
  useEffect(() => {
    if (!draftStarted || draftComplete || timeRemaining <= 0) return

    const timer = setInterval(() => {
      setTimeRemaining(prev => {
        if (prev <= 1) {
          handleAutoPick()
          return pickTimeLimit
        }
        return prev - 1
      })
    }, 1000)

    return () => clearInterval(timer)
  }, [draftStarted, draftComplete, timeRemaining, currentPick])

  const getCurrentRound = () => Math.ceil(currentPick / teams.length)
  const getCurrentPickInRound = () => ((currentPick - 1) % teams.length) + 1
  const isSnakeDraft = getCurrentRound() % 2 === 0
  const actualTeamIndex = isSnakeDraft ? teams.length - getCurrentPickInRound() : getCurrentPickInRound() - 1
  const currentDraftingTeam = teams[actualTeamIndex]

  const getRecommendedPlayers = () => {
    const userTeam = teams.find(t => t.isUser)
    if (!userTeam) return []

    const positionCounts = userTeam.roster.reduce((acc, player) => {
      acc[player.position] = (acc[player.position] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    const needsPosition = (pos: string) => {
      const maxNeeded = pos === 'QB' ? 2 : pos === 'TE' ? 2 : 4
      return (positionCounts[pos] || 0) < maxNeeded
    }

    return availablePlayers
      .filter(player => needsPosition(player.position))
      .slice(0, 8)
  }

  const handlePlayerSelect = (player: Player) => {
    if (!currentDraftingTeam.isUser) return

    const pick: DraftPick = {
      round: getCurrentRound(),
      pick: currentPick,
      teamId: currentDraftingTeam.id,
      teamName: currentDraftingTeam.name,
      player,
      isUser: true,
      timestamp: Date.now()
    }

    setDraftPicks(prev => [...prev, pick])
    setAvailablePlayers(prev => prev.filter(p => p.id !== player.id))
    setTeams(prev => prev.map(team => 
      team.id === currentDraftingTeam.id 
        ? { ...team, roster: [...team.roster, player], picks: [...team.picks, pick] }
        : team
    ))

    nextPick()
  }

  const handleAutoPick = () => {
    if (draftComplete) return

    const team = currentDraftingTeam
    let selectedPlayer: Player

    if (team.isUser) {
      // Auto-pick best available for user
      selectedPlayer = getRecommendedPlayers()[0] || availablePlayers[0]
    } else {
      // AI team logic - pick based on positional needs and tiers
      const positionCounts = team.roster.reduce((acc, player) => {
        acc[player.position] = (acc[player.position] || 0) + 1
        return acc
      }, {} as Record<string, number>)

      const getPositionPriority = () => {
        if ((positionCounts.QB || 0) === 0) return 'QB'
        if ((positionCounts.RB || 0) < 2) return 'RB'
        if ((positionCounts.WR || 0) < 2) return 'WR'
        if ((positionCounts.TE || 0) === 0) return 'TE'
        if ((positionCounts.RB || 0) < 3) return 'RB'
        if ((positionCounts.WR || 0) < 3) return 'WR'
        return 'ALL'
      }

      const priorityPosition = getPositionPriority()
      const candidates = priorityPosition === 'ALL' 
        ? availablePlayers.slice(0, 10)
        : availablePlayers.filter(p => p.position === priorityPosition).slice(0, 5)

      selectedPlayer = candidates[0] || availablePlayers[0]
    }

    if (selectedPlayer) {
      const pick: DraftPick = {
        round: getCurrentRound(),
        pick: currentPick,
        teamId: team.id,
        teamName: team.name,
        player: selectedPlayer,
        isUser: team.isUser,
        timestamp: Date.now()
      }

      setDraftPicks(prev => [...prev, pick])
      setAvailablePlayers(prev => prev.filter(p => p.id !== selectedPlayer.id))
      setTeams(prev => prev.map(t => 
        t.id === team.id 
          ? { ...t, roster: [...t.roster, selectedPlayer], picks: [...t.picks, pick] }
          : t
      ))

      nextPick()
    }
  }

  const nextPick = () => {
    setTimeRemaining(pickTimeLimit)
    
    if (currentPick >= totalPicks) {
      setDraftComplete(true)
      setDraftStarted(false)
    } else {
      setCurrentPick(prev => prev + 1)
    }
  }

  const startDraft = () => {
    setDraftStarted(true)
    setTimeRemaining(pickTimeLimit)
  }

  const filteredAvailablePlayers = selectedPosition === 'ALL' 
    ? availablePlayers.slice(0, 50)
    : availablePlayers.filter(p => p.position === selectedPosition).slice(0, 25)

  const userTeam = teams.find(t => t.isUser)
  const isUserTurn = currentDraftingTeam?.isUser

  // Auto-pick for AI teams
  useEffect(() => {
    if (draftStarted && !draftComplete && !isUserTurn && currentDraftingTeam) {
      const timeout = setTimeout(() => {
        handleAutoPick()
      }, 2000 + Math.random() * 3000) // 2-5 second delay for AI picks

      return () => clearTimeout(timeout)
    }
  }, [currentPick, draftStarted, isUserTurn])

  return (
    <div className="space-y-8">
      {/* Draft Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
      >
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div className="flex items-center gap-4">
            {!draftStarted && !draftComplete ? (
              <button
                onClick={startDraft}
                className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors font-semibold"
              >
                <PlayIcon className="w-5 h-5" />
                Start Draft
              </button>
            ) : draftComplete ? (
              <div className="flex items-center gap-2 text-green-400">
                <CheckCircleIcon className="w-6 h-6" />
                <span className="font-semibold">Draft Complete!</span>
              </div>
            ) : (
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <ClockIcon className="w-5 h-5 text-blue-400" />
                  <span className="font-semibold text-white">
                    {Math.floor(timeRemaining / 60)}:{(timeRemaining % 60).toString().padStart(2, '0')}
                  </span>
                </div>
                {isUserTurn && (
                  <button
                    onClick={handleAutoPick}
                    className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-500 transition-colors text-sm"
                  >
                    Auto Pick
                  </button>
                )}
              </div>
            )}
          </div>

          <div className="flex items-center gap-6 text-sm">
            <div className="text-gray-300">
              Round <span className="font-semibold text-white">{getCurrentRound()}</span> of {totalRounds}
            </div>
            <div className="text-gray-300">
              Pick <span className="font-semibold text-white">{currentPick}</span> of {totalPicks}
            </div>
            {draftStarted && !draftComplete && (
              <div className="flex items-center gap-2">
                {currentDraftingTeam?.isUser ? (
                  <UserIcon className="w-5 h-5 text-blue-400" />
                ) : (
                  <ComputerDesktopIcon className="w-5 h-5 text-orange-400" />
                )}
                <span className="font-semibold text-white">
                  {currentDraftingTeam?.name}
                </span>
              </div>
            )}
          </div>
        </div>

        {isUserTurn && draftStarted && !draftComplete && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mt-4 p-4 bg-blue-600/20 border border-blue-400/30 rounded-lg"
          >
            <p className="text-blue-300 font-medium">Your turn to pick! Select a player below.</p>
          </motion.div>
        )}
      </motion.div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Available Players */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-white">Available Players</h2>
            
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

          {/* Recommended Players (User Turn Only) */}
          {isUserTurn && draftStarted && !draftComplete && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-green-500/20 border border-green-400/30 rounded-lg p-4"
            >
              <h3 className="text-lg font-semibold text-green-300 mb-3 flex items-center gap-2">
                <TrophyIcon className="w-5 h-5" />
                AI Recommendations
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {getRecommendedPlayers().slice(0, 4).map((player, index) => (
                  <button
                    key={player.id}
                    onClick={() => handlePlayerSelect(player)}
                    className="text-left p-3 bg-green-600/20 hover:bg-green-600/30 rounded-lg transition-colors border border-green-400/20"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-semibold text-white text-sm">{player.name}</div>
                        <div className="text-xs text-green-300">{player.team} • {player.position}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs text-green-300">#{index + 1} Rec</div>
                        <div className="text-xs text-gray-400">Tier {player.tier}</div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          {/* Player List */}
          <div className="space-y-2">
            <AnimatePresence>
              {filteredAvailablePlayers.map((player) => (
                <motion.div
                  key={player.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -100 }}
                  className={`p-4 rounded-lg border transition-all duration-200 ${
                    isUserTurn && draftStarted && !draftComplete
                      ? 'bg-white/10 border-white/20 hover:bg-white/20 cursor-pointer'
                      : 'bg-white/5 border-white/10'
                  }`}
                  onClick={() => isUserTurn && draftStarted && !draftComplete && handlePlayerSelect(player)}
                >
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-3">
                      <div className="text-sm font-medium text-gray-400">#{player.adp}</div>
                      <div>
                        <div className="font-semibold text-white">{player.name}</div>
                        <div className="text-sm text-gray-400">{player.team} • {player.position}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium text-white">{Math.round(player.projected_points)} pts</div>
                      <div className="text-xs text-gray-400">Tier {player.tier}</div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* Draft Board & Your Team */}
        <div className="space-y-6">
          {/* Your Team */}
          {userTeam && (
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4 border border-white/20">
              <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                <UserIcon className="w-5 h-5" />
                Your Team
              </h3>
              
              {userTeam.roster.length === 0 ? (
                <p className="text-gray-400 text-sm">No players drafted yet</p>
              ) : (
                <div className="space-y-2">
                  {userTeam.roster.map((player, index) => (
                    <div key={player.id} className="flex justify-between items-center text-sm">
                      <div>
                        <div className="font-medium text-white">{player.name}</div>
                        <div className="text-xs text-gray-400">{player.position} • {player.team}</div>
                      </div>
                      <div className="text-xs text-blue-400">R{Math.ceil((index + 1) / 1)}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Team Composition */}
              <div className="mt-4 pt-3 border-t border-white/10">
                <div className="text-xs text-gray-400 mb-2">Roster Composition</div>
                <div className="grid grid-cols-4 gap-2 text-xs">
                  {(['QB', 'RB', 'WR', 'TE'] as const).map(pos => {
                    const count = userTeam.roster.filter(p => p.position === pos).length
                    return (
                      <div key={pos} className="text-center">
                        <div className="text-white font-medium">{count}</div>
                        <div className="text-gray-400">{pos}</div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Recent Picks */}
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4 border border-white/20">
            <h3 className="text-lg font-semibold text-white mb-3">Recent Picks</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {draftPicks.slice(-8).reverse().map((pick) => (
                <div key={`${pick.pick}`} className="flex justify-between items-center text-sm">
                  <div className="flex items-center gap-2">
                    {pick.isUser ? (
                      <UserIcon className="w-4 h-4 text-blue-400" />
                    ) : (
                      <ComputerDesktopIcon className="w-4 h-4 text-orange-400" />
                    )}
                    <div>
                      <div className="font-medium text-white">{pick.player.name}</div>
                      <div className="text-xs text-gray-400">{pick.teamName}</div>
                    </div>
                  </div>
                  <div className="text-xs text-gray-400">
                    {pick.round}.{pick.pick - (pick.round - 1) * teams.length}
                  </div>
                </div>
              )) || <p className="text-gray-400 text-sm">No picks yet</p>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}