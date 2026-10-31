import type { Metadata } from 'next'
import { PageShell } from '@/components/ui/PageShell'
import { DraftSimulator } from '@/components/draft/DraftSimulator'

export const metadata: Metadata = {
  title: 'Historical draft practice | Win My League',
  description: 'Practice a snake draft using a clearly dated historical grouping artifact, not current draft rankings.',
}

export default function DraftPage() {
  return <PageShell wide crumb="Draft practice" title="Historical draft practice"
    lede="Try a snake-draft exercise with the stored historical player groups. This is a learning exercise, not current-season draft advice.">
    <DraftSimulator />
  </PageShell>
}
