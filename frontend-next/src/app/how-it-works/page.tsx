import type { Metadata } from 'next'
import Link from 'next/link'
import { PageShell } from '@/components/ui/PageShell'
import { Cards, Section } from '@/components/ui/Prose'
import { ROUTES } from '@/lib/constants'

export const metadata: Metadata = {
  title: 'How It Works | Win My League',
  description: 'nflverse data, as-of features, a RandomForest champion with an XGBoost challenger, GMM preseason tiers, and a weekly publish/hold/promote job.',
}

export default function HowItWorksPage() {
  return (
    <PageShell
      crumb="How it works"
      eyebrow="Pipeline · from nflverse to a published artifact"
      title={
        <>
          What actually <em className="font-serif font-normal text-moss">runs.</em>
        </>
      }
      lede="This page describes the system as built. If a capability is not listed here, the site does not have it."
    >
      <Section number="01" title="Data">
        <p>
          The only data source is <b>nflverse</b>: weekly player statistics and roster files. There are no injury feeds, weather feeds, news
          scrapers, or third-party projections. Every input file is hashed and the hash is recorded with the model that consumed it.
        </p>
      </Section>

      <Section number="02" title="Features">
        <p>
          Features are computed <b>as-of</b> each week: a player&apos;s row for a given week uses only that player&apos;s own earlier weeks (trailing
          averages, variance, opportunity share, games played) plus static context such as age. Outcomes of a week are only added to the history after
          that week has been scored, so nothing from the future leaks into the past.
        </p>
        <p>
          The feature set carries a version string that every API response echoes, so a projection can always be tied to the exact feature
          definition that produced it.
        </p>
      </Section>

      <Section number="03" title="Models">
        <p>
          One model per position (QB, RB, WR, TE), trained on a PPR points target. The <b>champion</b> is a RandomForest; an <b>XGBoost challenger</b> is
          trained beside it on the same rows. Floors and ceilings are residual quantiles from a validation season, added to the point prediction per
          position and candidate; the floor is clipped at zero. Half-PPR and standard scoring are derived on the server from the PPR prediction using
          the player&apos;s as-of receptions estimate.
        </p>
      </Section>

      <Section number="04" title="Evaluation">
        <p>
          The holdout is a forward-time split: everything before the test start is training data, everything after is scored once. A{' '}
          <b>causal baseline</b> (a trailing mean of the player&apos;s own earlier realised points, with a position fallback) is scored on the same rows.
          Rolling-origin folds refit the model before each scored week. All of it is written to an evaluation artifact with the metric formulas
          embedded, and that artifact is what the <Link href={ROUTES.performance} className="underline">evaluation page</Link> renders.
        </p>
      </Section>

      <Section number="05" title="Tiers">
        <p>
          Preseason tiers come from a Gaussian mixture fitted on prior-season aggregates: points per game, its variance, games played, opportunity
          share, and age, scaled and reduced with PCA before clustering. The number of components is chosen by BIC. Tiers are ordered by their
          component&apos;s mean prior PPG, each player carries the membership probability the mixture assigned, and the artifact records how the tiers
          matched the realised season.
        </p>
      </Section>

      <Section number="06" title="Weekly job">
        <p>
          A scheduled GitHub Actions workflow pulls the latest nflverse data, rebuilds features, scores the champion and challenger, and applies a
          policy: <b>publish</b> the new predictions file, <b>hold</b> if the evaluation regressed or the data is incomplete, or <b>promote</b> the
          challenger when it beats the champion on the agreed metrics. The decision, its reasons, and the resulting versions are written to a
          manifest that the API reads at startup.
        </p>
      </Section>

      <Section number="07" title="What the site does">
        <Cards
          items={[
            { title: 'Predictions', text: 'The latest published week with point, floor, ceiling, and the scoring derivation.' },
            { title: 'Tiers', text: 'Preseason tiers per position with membership probabilities and the tier evaluation.' },
            { title: 'Draft board', text: 'A snake mock draft where opponents pick by tier, then prior PPG.' },
            { title: 'Start/Sit', text: 'Side-by-side intervals for any players in the latest predictions file.' },
            { title: 'Player history', text: 'Frozen test rows and weekly rows for one player, prediction against actual.' },
            { title: 'Evaluation', text: 'The full evaluation artifact: metrics, baseline, cohorts, folds, definitions, provenance.' },
          ]}
        />
        <p className="text-sm text-[#63706d]">
          Not included: league imports, trade analysis, injury or weather modelling, accounts, or paid plans.
        </p>
      </Section>
    </PageShell>
  )
}
