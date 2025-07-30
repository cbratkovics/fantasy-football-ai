'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'

interface PlayerSearchProps {
  onPlayerSelect: (playerId: string | null) => void
}

export function PlayerSearch({ onPlayerSelect }: PlayerSearchProps) {
  const [search, setSearch] = useState('')
  const [selectedPlayer, setSelectedPlayer] = useState<any>(null)

  const { data: players } = useQuery({
    queryKey: ['players', search],
    queryFn: async () => {
      if (!search || search.length < 2) return []
      const response = await axios.get(`/api/players/search?q=${search}`)
      return response.data
    },
    enabled: search.length >= 2
  })

  const handleSelect = (player: any) => {
    setSelectedPlayer(player)
    onPlayerSelect(player.id)
    setSearch('')
  }

  const handleClear = () => {
    setSelectedPlayer(null)
    onPlayerSelect(null)
    setSearch('')
  }

  return (
    <div className="relative">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        Player Filter
      </label>
      
      {selectedPlayer ? (
        <div className="flex items-center justify-between bg-blue-50 p-3 rounded-lg">
          <div>
            <p className="font-medium text-gray-900">{selectedPlayer.name}</p>
            <p className="text-sm text-gray-600">{selectedPlayer.team} - {selectedPlayer.position}</p>
          </div>
          <button
            onClick={handleClear}
            className="text-gray-400 hover:text-gray-600"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      ) : (
        <div>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search players..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
          />
          
          {players && players.length > 0 && (
            <div className="absolute z-10 mt-1 w-full bg-white shadow-lg rounded-md border border-gray-200">
              {players.map((player: any) => (
                <button
                  key={player.id}
                  onClick={() => handleSelect(player)}
                  className="w-full text-left px-4 py-2 hover:bg-gray-50 border-b border-gray-100 last:border-b-0"
                >
                  <p className="font-medium text-gray-900">{player.name}</p>
                  <p className="text-sm text-gray-600">{player.team} - {player.position}</p>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}