import type { Metadata } from 'next'
import Link from 'next/link'
import { PageShell } from '@/components/ui/PageShell'
import { Cards, Section } from '@/components/ui/Prose'
import { ROUTES, SITE } from '@/lib/constants'

export const metadata: Metadata = {
  title: 'About | Win My League',
  description: 'An independent, artifact-backed fantasy football projection project built by Christopher Bratkovics.',
}

export default function AboutPage() {
  return (
    <PageShell
      crumb="About"
      eyebrow="About · a portfolio project"
      title={
        <>
          Built in the open, <em className="font-serif font-normal text-moss">measured before published.</em>
        </>
      }
      lede="Win My League is a one-person applied data science project. It exists to show an honest, end-to-end forecasting system rather than to sell advice."
    >
      <Section number="01" title="What this is">
        <p>
          A weekly projection system for fantasy football built entirely on public nflverse data. Models are trained on a player&apos;s own history,
          evaluated on a forward-time holdout against a causal baseline, and published by a scheduled job that can refuse to publish. The site reads
          everything it shows from the artifacts that job commits.
        </p>
      </Section>

      <Section number="02" title="Who builds it">
        <p>
          <b>Christopher Bratkovics</b> builds and maintains the models, the API, and this site. He holds an M.S. in Applied Data Science and a B.S.
          in Computer Science and works in enterprise analytics and data engineering. The project started as a way to apply that work to a problem he
          cares about.
        </p>
        <div className="flex flex-wrap gap-3 pt-2">
          <a href={SITE.repoUrl} target="_blank" rel="noreferrer" className="portfolio-button primary">Repository</a>
          <a href={SITE.authorLinkedin} target="_blank" rel="noreferrer" className="portfolio-button secondary">LinkedIn</a>
          <a href={`mailto:${SITE.supportEmail}`} className="portfolio-button secondary">Email</a>
        </div>
      </Section>

      <Section number="03" title="How it is built">
        <Cards
          items={[
            { title: 'Data-driven', text: 'Projections come from models trained on nflverse history, not from opinion or hand-tuned rankings.' },
            { title: 'Transparent', text: 'Every projection ships with a floor and ceiling and names the model and feature version that produced it.' },
            { title: 'Open', text: 'The modelling code, the API, and the committed artifacts are public in the repository.' },
            { title: 'Evaluated first', text: 'Nothing is published until it has been scored against a baseline on a held-out season.' },
          ]}
        />
      </Section>

      <Section number="04" title="Where to start">
        <p>
          Open the <Link href={ROUTES.predictions} className="underline">predictions</Link> for the latest published week, read the{' '}
          <Link href={ROUTES.performance} className="underline">evaluation</Link> to see how the model compares with its baseline, or read{' '}
          <Link href={ROUTES.howItWorks} className="underline">how it works</Link> for the full pipeline.
        </p>
      </Section>
    </PageShell>
  )
}
