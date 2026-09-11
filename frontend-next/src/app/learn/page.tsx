import type { Metadata } from 'next'
import Link from 'next/link'
import { PageShell } from '@/components/ui/PageShell'
import { Faq, Section } from '@/components/ui/Prose'
import { ROUTES } from '@/lib/constants'

export const metadata: Metadata = {
  title: 'Learn | Win My League',
  description: 'How to read a projection with a floor and ceiling, what a tier means, and how to use the evaluation page.',
}

export default function LearnPage() {
  return (
    <PageShell
      crumb="Learn"
      eyebrow="Learn · reading the outputs"
      title={
        <>
          Read the interval, <em className="font-serif font-normal text-moss">not just the number.</em>
        </>
      }
      lede="Short guides to the three things the site publishes: weekly projections, preseason tiers, and the evaluation behind them."
    >
      <Section number="01" title="Projections">
        <p>
          Each projection is three numbers. The <b>prediction</b> is the model&apos;s point estimate of PPR points for that week. The <b>floor</b> and{' '}
          <b>ceiling</b> are the point estimate shifted by the lower and upper residual quantiles observed on a validation season for that position.
          A wide interval means the model has historically been less precise for players like this one; a narrow one means the opposite.
        </p>
        <p>
          Half-PPR and standard totals are derived on the server by subtracting a share of the player&apos;s estimated receptions, so switching the
          scoring control never changes the underlying model output.
        </p>
      </Section>

      <Section number="02" title="Tiers">
        <p>
          Tiers group players whose prior-season profile looks alike. They are fitted once before the season and do not update week to week. The
          probability next to each player is the mixture&apos;s confidence that they belong to that tier rather than a neighbouring one; a low value
          means the player sits near a boundary. Tier order follows mean prior-season points per game, so a lower tier number is the stronger group.
        </p>
      </Section>

      <Section number="03" title="Evaluation">
        <p>
          The <Link href={ROUTES.performance} className="underline">evaluation page</Link> compares the champion to a baseline that only knows a
          player&apos;s own earlier points. The gap between the two columns is the honest measure of what the model adds. Positional cohorts show where
          the model is stronger or weaker; the rolling-origin folds show how error moves through a season when the model is refit before every scored
          week.
        </p>
      </Section>

      <Section number="04" title="Common questions">
        <Faq
          items={[
            {
              q: 'How accurate are the projections?',
              a: (
                <>
                  Exactly as accurate as the evaluation page says for the cohort you care about. No accuracy figure is written into this site by hand;
                  every number is read from the committed evaluation artifact.
                </>
              ),
            },
            {
              q: 'Should I always start the higher prediction?',
              a: 'Not necessarily. If you need a safe result, a higher floor can matter more than a higher point estimate. The start/sit tool shows both so you can decide.',
            },
            {
              q: 'How often do predictions change?',
              a: 'A scheduled job runs once per week during the season. It publishes a new predictions file only when its policy allows it, and the manifest records what it decided and why.',
            },
            {
              q: 'Does the model know about injuries, weather, or news?',
              a: 'No. The only input is nflverse weekly data. Missed games show up in a player’s history after the fact, but there is no injury report or weather feed.',
            },
          ]}
        />
      </Section>
    </PageShell>
  )
}
