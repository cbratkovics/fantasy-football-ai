import type { Metadata } from 'next'
import { TierBoard } from '@/components/tiers/TierBoard'

export const metadata: Metadata = {
  title: 'Historical player groups | Win My League',
  description: 'Explore historical player groups from prior-season inputs, with their season and evidence clearly identified.',
}

export default function TiersPage() {
  return <TierBoard />
}
