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
      eyebrow="Start/Sit · latest published week"
      title={
        <>
          Compare one <em className="font-serif font-normal text-moss">starting slot.</em>
        </>
      }
      lede="Nominate alternatives from one published weekly snapshot. The shared Decision Lab policy checks eligibility, availability assumptions, baseline evidence and the projection gap. Dated evidence does not establish game-day readiness."
    >
      <StartSitEngine />
    </PageShell>
  )
}
