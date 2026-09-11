import type { Metadata } from 'next'
import { PageShell } from '@/components/ui/PageShell'
import { Section } from '@/components/ui/Prose'
import { SITE } from '@/lib/constants'

export const metadata: Metadata = {
  title: 'Terms | Win My League',
  description: 'Terms of use for a free, read-only portfolio project.',
}

export default function TermsPage() {
  return (
    <PageShell
      crumb="Terms"
      eyebrow="Terms of use"
      title={
        <>
          Free, as-is, <em className="font-serif font-normal text-moss">for your own judgement.</em>
        </>
      }
      lede="Short terms for a small project. Using the site means you accept them."
    >
      <Section number="01" title="What the projections are">
        <p>
          The projections are statistical estimates produced by models trained on historical nflverse data. They are not statements of fact and carry
          no guarantee of accuracy beyond what the evaluation page reports for the held-out season. Football outcomes depend on injuries, coaching
          decisions, and chance that the models do not observe.
        </p>
        <p>
          They are information to weigh, not advice to follow. You are responsible for your own lineup, draft, and any other decisions. Do not use the
          site as the basis for wagering.
        </p>
      </Section>
      <Section number="02" title="Availability">
        <p>
          This is an independent portfolio project. Features may change or be removed, the API may be unavailable, and published weeks may be held when
          the weekly job&apos;s policy says so. Everything is provided as-is, without warranties of any kind.
        </p>
      </Section>
      <Section number="03" title="Acceptable use">
        <p>
          Do not attempt to break, overload, or gain unauthorised access to the site or the API, and do not scrape it at a volume that degrades it for
          other people. The code and artifacts are public in the repository; use that instead of bulk-extracting the API.
        </p>
      </Section>
      <Section number="04" title="Liability">
        <p>
          To the extent permitted by law, {SITE.name} and its operator are not liable for losses arising from your use of the site, including decisions
          made in reliance on its projections.
        </p>
      </Section>
      <Section number="05" title="Contact">
        <p>
          Questions about these terms go to <a href={`mailto:${SITE.supportEmail}`} className="underline">{SITE.supportEmail}</a>.
        </p>
      </Section>
    </PageShell>
  )
}
