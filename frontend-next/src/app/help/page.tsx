import type { Metadata } from 'next'
import Link from 'next/link'
import { PageShell } from '@/components/ui/PageShell'
import { Faq, Section } from '@/components/ui/Prose'
import { ROUTES, SITE } from '@/lib/constants'

export const metadata: Metadata = {
  title: 'Help | Win My League',
  description: 'Answers to practical questions about the site and how to get in touch.',
}

export default function HelpPage() {
  return (
    <PageShell
      crumb="Help"
      eyebrow="Help · practical questions"
      title={
        <>
          Stuck? <em className="font-serif font-normal text-moss">Start here.</em>
        </>
      }
      lede={
        <>
          If the answer is not below, email <a href={`mailto:${SITE.supportEmail}`} className="underline">{SITE.supportEmail}</a>.
        </>
      }
    >
      <Section number="01" title="Using the site">
        <Faq
          items={[
            {
              q: 'A page says the request failed. What does that mean?',
              a: 'The site could not reach the API, or the API had no artifact for what was asked (for example a week that has not been published). Nothing is substituted in place of missing data; try again later or pick a published week.',
            },
            {
              q: 'Why is a week missing from the predictions?',
              a: 'Only weeks the scheduled job actually published exist. If the job held a week, there is no file for it and the API returns a not-found error for that week.',
            },
            {
              q: 'Why does a player have no page?',
              a: 'Player pages exist only for players that appear in the champion’s held-out test rows or in a published predictions file.',
            },
            {
              q: 'Can I import my league?',
              a: 'No. There is no league import, account, or saved roster. The draft board and start/sit tools work entirely from the published artifacts.',
            },
            {
              q: 'Where do the numbers on the evaluation page come from?',
              a: (
                <>
                  From the evaluation artifact named at the top of that page. The same file is committed in the{' '}
                  <a href={SITE.repoUrl} target="_blank" rel="noreferrer" className="underline">repository</a>.
                </>
              ),
            },
          ]}
        />
      </Section>

      <Section number="02" title="Running it yourself">
        <p>
          The API and this frontend are both in the repository. Start the API with <code className="bg-sand px-1 font-mono text-xs">uvicorn ffai.serve.app:app --port 7860</code>,
          set <code className="bg-sand px-1 font-mono text-xs">NEXT_PUBLIC_API_URL</code> for the frontend, and every page will read from your local artifacts.
        </p>
      </Section>

      <Section number="03" title="More reading">
        <p>
          <Link href={ROUTES.howItWorks} className="underline">How it works</Link> describes the pipeline; <Link href={ROUTES.learn} className="underline">Learn</Link>{' '}
          explains how to read the outputs; <Link href={ROUTES.about} className="underline">About</Link> covers who built it and why.
        </p>
      </Section>
    </PageShell>
  )
}
