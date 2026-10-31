import type { Metadata } from 'next'
import { PageShell } from '@/components/ui/PageShell'
import { PredictionsBoard } from '@/components/predictions/PredictionsBoard'

export const metadata: Metadata = {
  title: 'Player projections | Win My League',
  description: 'Browse published player projections and choose players for a PPR comparison.',
}

export default function PredictionsPage() {
  return (
    <PageShell
      wide
      crumb="Player projections"
      title="Player projections"
      lede="Find players in a published week and add your alternatives to a comparison."
    >
      <PredictionsBoard />
    </PageShell>
  )
}
