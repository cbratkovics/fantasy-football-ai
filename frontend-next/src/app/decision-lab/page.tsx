import type { Metadata } from 'next'
import { Suspense } from 'react'
import { PageShell } from '@/components/ui/PageShell'
import { DecisionLab } from '@/components/decision-lab/DecisionLab'
import { DECISION_LAB_COPY } from '@/content/decision-lab'

export const metadata: Metadata = {
  title: 'Decision Lab | Win My League',
  description: 'A one-slot start/sit comparison over the committed evidence bundle, computed offline in the browser with recorded assumptions and receipts.',
}

export default function DecisionLabPage() {
  return (
    <PageShell
      wide
      crumb={DECISION_LAB_COPY.page.crumb}
      eyebrow={DECISION_LAB_COPY.page.eyebrow}
      title={
        <>
          One slot, <em className="font-serif font-normal text-moss">recorded assumptions.</em>
        </>
      }
      lede={DECISION_LAB_COPY.page.lede}
    >
      <Suspense fallback={<div role="status" className="border border-[#c5c7c0] bg-[#f8f7f2] p-8 font-mono text-xs uppercase tracking-widest text-[#61706c]">{DECISION_LAB_COPY.load.loading}</div>}>
        <DecisionLab />
      </Suspense>
    </PageShell>
  )
}
