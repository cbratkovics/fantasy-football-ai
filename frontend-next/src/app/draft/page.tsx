import type { Metadata } from 'next'
import { PageShell } from '@/components/ui/PageShell'
import { DraftSimulator } from '@/components/draft/DraftSimulator'

export const metadata: Metadata = {
  title: 'Draft Board | Win My League',
  description: 'A snake mock draft whose board is built from the committed preseason tiers artifact.',
}

export default function DraftPage() {
  return (
    <PageShell
      wide
      crumb="Draft"
      eyebrow="Draft board built from the current tiers artifact"
      title={
        <>
          Draft by tier, <em className="font-serif font-normal text-moss">then by prior PPG.</em>
        </>
      }
      lede="Opponents follow one transparent rule: take the best available player by tier, breaking ties on prior-season points per game, subject to simple roster caps. There is no hidden strategy engine."
    >
      <DraftSimulator />
    </PageShell>
  )
}
