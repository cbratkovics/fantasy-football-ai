import type { Metadata } from 'next'
import { PerformanceDashboard } from '@/components/dashboard/PerformanceDashboard'
import { PageShell } from '@/components/ui/PageShell'

export const metadata: Metadata = {
  title: 'Model results | Win My League Decision Lab',
  description: 'Held-out evaluation of the champion model against a causal baseline, read from the committed evaluation artifact.',
}

export default function PerformancePage() {
  return (
    <PageShell title="How well did the projections perform?" crumb="Model results" lede="Compare historical model errors with a baseline built only from earlier results." wide>
      <PerformanceDashboard />
    </PageShell>
  )
}
