import type { Metadata } from 'next'
import { PageShell } from '@/components/ui/PageShell'
import { PredictionsBoard } from '@/components/predictions/PredictionsBoard'

export const metadata: Metadata = {
  title: 'Weekly Predictions | Win My League',
  description: 'Weekly point projections with floor and ceiling, read from the latest committed predictions artifact.',
}

export default function PredictionsPage() {
  return (
    <PageShell
      wide
      crumb="Predictions"
      eyebrow="Weekly projections · committed artifact"
      title={
        <>
          Point, floor, <em className="font-serif font-normal text-moss">ceiling.</em>
        </>
      }
      lede="Every row below comes from the predictions file the weekly job last published. The scoring format is derived from the PPR target on the server; nothing is re-estimated in the browser."
    >
      <PredictionsBoard />
    </PageShell>
  )
}
