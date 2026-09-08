import { Metadata } from 'next'
import Link from 'next/link'
import { CheckIcon } from '@heroicons/react/24/outline'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'

export const metadata: Metadata = {
  title: 'Pricing | WinMyLeague.ai',
  description:
    'WinMyLeague.ai is free while in beta. No payment details required, no paid plans today.',
}

const included = [
  'Weekly point projections for all skill positions',
  'Confidence intervals on every projection',
  'Player tier visualizations',
  'Start/sit comparisons',
  'Mock draft simulator',
  'Player profiles with projection history',
]

const faqs = [
  {
    question: 'What does it cost?',
    answer:
      'Nothing. WinMyLeague.ai is free while in beta. There are no paid plans, and no payment details are collected anywhere on the site.',
  },
  {
    question: 'Will it stay free?',
    answer:
      'I have not decided. If paid plans are ever introduced, existing accounts will be told well before anything changes, and nothing will start charging automatically.',
  },
  {
    question: 'What do I need to sign up?',
    answer:
      'An email address. Accounts are handled through Clerk, a third-party authentication provider. See the privacy policy for what that means for your data.',
  },
  {
    question: 'How accurate are the projections?',
    answer:
      'Every projection ships with a confidence interval rather than a single headline accuracy number, because accuracy varies a great deal by position, by week, and by how much a player’s role has recently changed. Treat the projections as one input, not as a guarantee.',
  },
  {
    question: 'Can I delete my account?',
    answer:
      'Yes. Email support@winmyleague.ai and your account and associated data will be deleted.',
  },
]

export default function PricingPage() {
  return (
    <div className="min-h-screen bg-white">
      <Navigation />

      <main className="pt-24 pb-24">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          {/* Header */}
          <div className="mx-auto max-w-3xl text-center">
            <h1 className="text-base font-semibold leading-7 text-indigo-600">Pricing</h1>
            <p className="mt-2 text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl">
              Free while in beta
            </p>
            <p className="mt-6 text-lg leading-8 text-gray-600">
              WinMyLeague.ai is an independent project and is not monetized. Every feature is
              available at no cost, and the site does not collect payment details.
            </p>
          </div>

          {/* Single plan */}
          <div className="mx-auto mt-16 max-w-2xl">
            <div className="rounded-3xl p-8 ring-2 ring-indigo-600 sm:p-10">
              <h2 className="text-2xl font-bold tracking-tight text-gray-900">Beta access</h2>
              <p className="mt-4 text-sm leading-6 text-gray-600">
                Full access to everything currently built.
              </p>
              <p className="mt-6 flex items-baseline gap-x-2">
                <span className="text-5xl font-bold tracking-tight text-gray-900">$0</span>
                <span className="text-base text-gray-500">while in beta</span>
              </p>

              <ul role="list" className="mt-8 space-y-3 text-sm leading-6 text-gray-600">
                {included.map((feature) => (
                  <li key={feature} className="flex gap-x-3">
                    <CheckIcon
                      className="h-6 w-5 flex-none text-indigo-600"
                      aria-hidden="true"
                    />
                    {feature}
                  </li>
                ))}
              </ul>

              <Link
                href="/auth/signup"
                className="mt-10 block rounded-md bg-indigo-600 px-3 py-2.5 text-center text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
              >
                Create an account
              </Link>
              <p className="mt-4 text-center text-xs text-gray-500">
                No payment details required
              </p>
            </div>
          </div>

          {/* FAQ */}
          <div className="mx-auto mt-24 max-w-3xl">
            <h2 className="text-2xl font-bold tracking-tight text-gray-900 text-center">
              Common questions
            </h2>
            <dl className="mt-12 space-y-8">
              {faqs.map((faq) => (
                <div key={faq.question}>
                  <dt className="text-base font-semibold leading-7 text-gray-900">
                    {faq.question}
                  </dt>
                  <dd className="mt-2 text-base leading-7 text-gray-600">{faq.answer}</dd>
                </div>
              ))}
            </dl>
          </div>

          {/* Contact */}
          <div className="mx-auto mt-24 max-w-3xl text-center">
            <h2 className="text-2xl font-bold tracking-tight text-gray-900">
              Still have a question?
            </h2>
            <p className="mt-4 text-base leading-7 text-gray-600">
              Send a message and it comes straight to me.
            </p>
            <Link
              href="/contact"
              className="mt-6 inline-block rounded-md border border-gray-300 px-6 py-2.5 text-sm font-semibold text-gray-900 hover:bg-gray-50"
            >
              Contact
            </Link>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}
