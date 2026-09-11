import type { Metadata } from 'next'
import { PerformanceDashboard } from '@/components/dashboard/PerformanceDashboard'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'

export const metadata: Metadata = {
  title: 'Model Evaluation | Win My League Decision Lab',
  description: 'Held-out evaluation of the champion model against a causal baseline, read from the committed evaluation artifact.',
}

export default function PerformancePage() {
  return (
    <div className="evaluation-shell">
      <Navigation />
      <main>
        <PerformanceDashboard />
      </main>
      <Footer />
    </div>
  )
}
