import { PlayerProfile } from '@/components/player/PlayerProfile'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { notFound } from 'next/navigation'
import tiersData from '@/data/tiers_2024.json'

interface PlayerPageProps {
  params: {
    id: string
  }
}

export default function PlayerPage({ params }: PlayerPageProps) {
  // Find player in tiers data
  let foundPlayer = null
  let playerPosition = ''
  
  Object.entries(tiersData.tiers).forEach(([position, positionTiers]) => {
    positionTiers.forEach((tier: any) => {
      const player = tier.players.find((p: any) => p.id === params.id)
      if (player) {
        foundPlayer = { ...player, position, tier: tier.tier, tierLabel: tier.label, tierColor: tier.color }
        playerPosition = position
      }
    })
  })

  if (!foundPlayer) {
    notFound()
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      <Navigation />
      <main className="pt-24">
        <PlayerProfile player={foundPlayer} />
      </main>
      <Footer />
    </div>
  )
}

export async function generateStaticParams() {
  const playerIds: string[] = []
  
  Object.entries(tiersData.tiers).forEach(([position, positionTiers]) => {
    positionTiers.forEach((tier: any) => {
      tier.players.forEach((player: any) => {
        playerIds.push(player.id)
      })
    })
  })

  return playerIds.map((id) => ({ id }))
}