import type { Metadata } from 'next'
import { PageShell } from '@/components/ui/PageShell'
import { StartSitEngine } from '@/components/start-sit/StartSitEngine'

export const metadata: Metadata = {
  title: 'Start/Sit | Win My League',
  description: 'Compare two to eight alternatives for one slot using verified PPR evidence and the Decision Lab policy.',
}

export default function StartSitPage() {
  return (
    <PageShell
      wide
      crumb="Start/Sit"
      title="Who should I start?"
      lede="Compare your options for one slot, then check what matters before choosing."
    >
      <StartSitEngine />
    </PageShell>
  )
}
