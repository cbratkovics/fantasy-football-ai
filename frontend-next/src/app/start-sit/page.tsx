import type { Metadata } from 'next'
import { PageShell } from '@/components/ui/PageShell'
import { StartSitEngine } from '@/components/start-sit/StartSitEngine'

export const metadata: Metadata = {
  title: 'Start/Sit | Win My League',
  description: 'Compare two or more players from the latest published predictions: point estimate, floor, and ceiling side by side.',
}

export default function StartSitPage() {
  return (
    <PageShell
      wide
      crumb="Start/Sit"
      eyebrow="Start/Sit · latest published week"
      title={
        <>
          Compare the <em className="font-serif font-normal text-moss">whole interval.</em>
        </>
      }
      lede="Pick any players from the latest predictions file. The recommendation is simply the higher point estimate; the floor and ceiling are shown so you can weigh the downside yourself."
    >
      <StartSitEngine />
    </PageShell>
  )
}
