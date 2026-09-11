import type { Metadata } from 'next'
import { PageShell } from '@/components/ui/PageShell'
import { TierBoard } from '@/components/tiers/TierBoard'

export const metadata: Metadata = {
  title: 'Preseason Tiers | Win My League',
  description: 'Gaussian-mixture preseason tiers built from prior-season aggregates, read from the committed tiers artifact.',
}

export default function TiersPage() {
  return (
    <PageShell
      wide
      crumb="Tiers"
      eyebrow="Preseason tiers · Gaussian mixture"
      title={
        <>
          Groups, not <em className="font-serif font-normal text-moss">rankings.</em>
        </>
      }
      lede="Tiers are fitted once per preseason from prior-season aggregates. Each player carries the probability the mixture assigned to their tier, and the artifact records how well those tiers matched the season that followed."
    >
      <TierBoard />
    </PageShell>
  )
}
