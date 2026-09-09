import type { Metadata } from 'next'
import { PerformanceDashboard } from '@/components/dashboard/PerformanceDashboard'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'

export const metadata: Metadata = {
  title: 'Model Evaluation | Win My League Decision Lab',
  description: 'Explore model error, positional cohorts, and the methodology behind the Win My League decision system.',
}

export default function PerformancePage() {
  return (
    <div className="evaluation-shell">
      <Navigation />
      <main>
        <header className="evaluation-hero">
          <div className="eyebrow"><span /> Model evaluation · historical sample</div>
          <div className="evaluation-title-row">
            <h1>Evidence, not<br /><em>just output.</em></h1>
            <div><p>Inspect model error and reliability across positional cohorts. Every result should be read with its sample, time window, and methodology in view.</p><span>2019—2024 · PRE-GAME FEATURES</span></div>
          </div>
        </header>
        <PerformanceDashboard />
      </main>
      <Footer />
    </div>
  )
}
