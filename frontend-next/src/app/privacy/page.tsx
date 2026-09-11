import type { Metadata } from 'next'
import { PageShell } from '@/components/ui/PageShell'
import { Section } from '@/components/ui/Prose'
import { SITE } from '@/lib/constants'

export const metadata: Metadata = {
  title: 'Privacy | Win My League',
  description: 'What this site collects (almost nothing) and how to reach the operator.',
}

export default function PrivacyPage() {
  return (
    <PageShell
      crumb="Privacy"
      eyebrow="Privacy"
      title={
        <>
          No accounts, <em className="font-serif font-normal text-moss">no tracking.</em>
        </>
      }
      lede="This site is a read-only view of committed artifacts. It has no sign-up, no forms, and no analytics."
    >
      <Section number="01" title="What is collected">
        <p>
          The site itself stores nothing about you. There are no accounts, cookies set by the application, or analytics scripts. Your browser fetches
          data from the API directly; the API keeps no user records.
        </p>
        <p>
          The hosting providers for the site and the API may keep ordinary server logs (request path, timestamp, IP address) for operational purposes,
          under their own retention policies.
        </p>
      </Section>
      <Section number="02" title="If you email us">
        <p>
          If you write to <a href={`mailto:${SITE.supportEmail}`} className="underline">{SITE.supportEmail}</a>, your message and address are kept only as long
          as needed to reply. Ask in the same thread and the conversation will be deleted.
        </p>
      </Section>
      <Section number="03" title="Third parties">
        <p>
          The site links to GitHub and LinkedIn. Those sites have their own privacy policies once you leave this one. No data is shared with advertisers
          or data brokers because none is collected.
        </p>
      </Section>
      <Section number="04" title="Changes">
        <p>This page will be updated if any of the above changes. The current version is always the one published here.</p>
      </Section>
    </PageShell>
  )
}
