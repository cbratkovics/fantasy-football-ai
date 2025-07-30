'use client'

import { useAuth } from '@clerk/nextjs'
import { useState } from 'react'
import { DashboardLayout } from '@/components/dashboard/DashboardLayout'
import { PredictionsList } from '@/components/predictions/PredictionsList'
import { PlayerSearch } from '@/components/predictions/PlayerSearch'
import { WeekSelector } from '@/components/predictions/WeekSelector'
import { SubscriptionBanner } from '@/components/dashboard/SubscriptionBanner'

export default function Dashboard() {
  const { isLoaded, userId } = useAuth()
  const [selectedWeek, setSelectedWeek] = useState(getCurrentWeek())
  const [selectedPlayer, setSelectedPlayer] = useState<string | null>(null)

  if (!isLoaded) {
    return <div>Loading...</div>
  }

  if (!userId) {
    window.location.href = '/signin'
    return null
  }

  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="mt-2 text-gray-600">
            Get AI-powered predictions with transparent explanations
          </p>
        </div>

        <SubscriptionBanner />

        <div className="grid gap-6 md:grid-cols-12">
          <div className="md:col-span-3">
            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Settings</h2>
              <WeekSelector 
                selectedWeek={selectedWeek} 
                onWeekChange={setSelectedWeek} 
              />
              <div className="mt-4">
                <PlayerSearch 
                  onPlayerSelect={setSelectedPlayer}
                />
              </div>
            </div>
          </div>

          <div className="md:col-span-9">
            <PredictionsList 
              week={selectedWeek}
              playerId={selectedPlayer}
            />
          </div>
        </div>
      </div>
    </DashboardLayout>
  )
}

function getCurrentWeek(): number {
  // Simple calculation - would be more complex in production
  const now = new Date()
  const seasonStart = new Date(2024, 8, 5) // Sept 5, 2024
  const weeksSinceStart = Math.floor((now.getTime() - seasonStart.getTime()) / (7 * 24 * 60 * 60 * 1000))
  return Math.min(Math.max(1, weeksSinceStart + 1), 18)
}