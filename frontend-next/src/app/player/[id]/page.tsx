import type { Metadata } from 'next'
import { PlayerProfile } from '@/components/player/PlayerProfile'

export const metadata: Metadata = {
  title: 'Player history | Win My League',
  description: 'Prediction history for a single player: frozen test rows and weekly published rows.',
}

interface PlayerPageProps {
  params: { id: string }
}

export default function PlayerPage({ params }: PlayerPageProps) {
  return <PlayerProfile playerId={decodeURIComponent(params.id)} />
}
