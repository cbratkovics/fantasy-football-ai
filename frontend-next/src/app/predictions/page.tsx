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
      lede="Browse published projections, not verified available players. Forecasts are trained on PPR; half-PPR and standard forecasts are derived estimates. Actuals use realized evidence under the requested scoring rules."
    >
      <PredictionsBoard />
    </PageShell>
  )
}
