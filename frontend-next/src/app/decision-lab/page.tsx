import type { Metadata } from 'next'
import { Suspense } from 'react'
import { PageShell } from '@/components/ui/PageShell'
import { DecisionLab } from '@/components/decision-lab/DecisionLab'
import { DECISION_LAB_COPY } from '@/content/decision-lab'

export const metadata: Metadata = {
  title: 'Explore a decision | Win My League',
  description: 'Compare a published week, replay a past player decision, or learn with clearly labeled examples.',
}

export default function DecisionLabPage() {
  return (
    <PageShell
      wide
      crumb={DECISION_LAB_COPY.page.crumb}
      title="Explore a decision"
      lede="Compare a published week, replay a past decision, or learn with a clearly labeled synthetic example."
    >
      <Suspense fallback={<div role="status" className="border border-[#c5c7c0] bg-[#f8f7f2] p-5 text-base text-[#61706c]">{DECISION_LAB_COPY.load.loading}</div>}>
        <DecisionLab />
      </Suspense>
    </PageShell>
  )
}
