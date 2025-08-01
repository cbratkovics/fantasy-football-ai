'use client'

import { useState } from 'react'
import { Metadata } from 'next'
import Link from 'next/link'
import { CheckIcon, XMarkIcon } from '@heroicons/react/24/outline'
import { ShieldCheckIcon, CreditCardIcon, ArrowPathIcon } from '@heroicons/react/24/solid'

// export const metadata: Metadata = {
//   title: 'Pricing | WinMyLeague.ai - Choose Your Plan',
//   description: 'Simple, transparent pricing for WinMyLeague.ai. Start free, upgrade anytime. Win more fantasy football championships.',
// }

const frequencies = [
  { value: 'monthly', label: 'Monthly', priceSuffix: '/month' },
  { value: 'annually', label: 'Annually', priceSuffix: '/year' },
]

const tiers = [
  {
    name: 'Free',
    id: 'tier-free',
    href: '/auth/signup',
    price: { monthly: '$0', annually: '$0' },
    description: 'Perfect for trying out our platform and casual players.',
    features: [
      'Basic player projections',
      'Weekly top 50 rankings',
      'Simple start/sit recommendations',
      '1 team',
      'Email support',
    ],
    notIncluded: [
      'Advanced AI predictions',
      'Player tier visualizations',
      'Mock draft simulator',
      'Trade analyzer',
      'Real-time updates',
      'Multiple teams',
      'API access',
      'Priority support'
    ],
    mostPopular: false,
    cta: 'Start free',
    emphasis: false,
  },
  {
    name: 'Pro',
    id: 'tier-pro',
    href: '/auth/signup?plan=pro',
    price: { monthly: '$19', annually: '$190' },
    description: 'Everything you need to dominate your fantasy league.',
    features: [
      'Advanced AI predictions (93.1% accuracy)',
      'Complete player tier system',
      'Weekly lineup optimizer',
      'Mock draft simulator with AI',
      'Trade analyzer & recommendations',
      'Real-time injury updates',
      'Weather impact analysis',
      '5 teams',
      'Priority email support',
      'Mobile app access',
    ],
    notIncluded: [
      'API access',
      'Custom scoring systems',
      'Unlimited teams',
      'Phone support',
    ],
    mostPopular: true,
    cta: 'Start 7-day trial',
    emphasis: true,
  },
  {
    name: 'League',
    id: 'tier-league',
    href: '/auth/signup?plan=league',
    price: { monthly: '$49', annually: '$490' },
    description: 'For serious players and league commissioners.',
    features: [
      'Everything in Pro',
      'Unlimited teams',
      'Custom scoring systems',
      'League-wide analytics',
      'API access',
      'CSV exports',
      'Advanced trade finder',
      'Dynasty league tools',
      'Keeper recommendations',
      'Phone & email support',
      'Custom integrations',
    ],
    notIncluded: [],
    mostPopular: false,
    cta: 'Contact sales',
    emphasis: false,
  },
]

const faqs = [
  {
    question: 'Can I change plans anytime?',
    answer: 'Yes! You can upgrade or downgrade your plan at any time. Changes take effect immediately, and we\'ll prorate any payments.',
  },
  {
    question: 'What payment methods do you accept?',
    answer: 'We accept all major credit cards, debit cards, and PayPal. All payments are processed securely through Stripe.',
  },
  {
    question: 'Is there a free trial?',
    answer: 'Yes! Pro and League plans include a 7-day free trial. No credit card required to start.',
  },
  {
    question: 'What happens to my data if I cancel?',
    answer: 'Your data remains accessible for 30 days after cancellation. You can export everything or reactivate anytime within that period.',
  },
  {
    question: 'Do you offer refunds?',
    answer: 'We offer a 30-day money-back guarantee. If you\'re not satisfied, contact us for a full refund.',
  },
  {
    question: 'Can I share my account?',
    answer: 'Each account is for individual use. For multiple users, consider our League plan or contact us for team pricing.',
  },
]

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(' ')
}

export default function PricingPage() {
  const [frequency, setFrequency] = useState(frequencies[0])

  return (
    <div className="bg-white py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        {/* Header */}
        <div className="mx-auto max-w-4xl text-center">
          <h1 className="text-base font-semibold leading-7 text-indigo-600">Pricing</h1>
          <p className="mt-2 text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl">
            Choose the plan that wins championships
          </p>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            Start free, upgrade when you need more. Cancel anytime. 
            Join 2,500+ players already winning with WinMyLeague.ai.
          </p>
        </div>

        {/* Billing frequency toggle */}
        <div className="mt-16 flex justify-center">
          <div className="relative">
            <div className="grid grid-cols-2 gap-x-1 rounded-full p-1 text-center text-xs font-semibold leading-5 bg-gray-100">
              {frequencies.map((option) => (
                <button
                  key={option.value}
                  onClick={() => setFrequency(option)}
                  className={classNames(
                    frequency.value === option.value
                      ? 'bg-indigo-600 text-white'
                      : 'text-gray-500 hover:text-gray-700',
                    'cursor-pointer rounded-full px-6 py-2 transition-colors duration-200'
                  )}
                >
                  {option.label}
                </button>
              ))}
            </div>
            {frequency.value === 'annually' && (
              <div className="absolute -top-8 left-1/2 -translate-x-1/2">
                <span className="inline-flex items-center rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-medium text-green-800">
                  Save 2 months
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Trust signals */}
        <div className="mx-auto mt-10 grid max-w-lg grid-cols-3 gap-8 text-center lg:max-w-none lg:grid-cols-3">
          <div className="flex flex-col items-center">
            <ShieldCheckIcon className="h-8 w-8 text-green-600" />
            <p className="mt-2 text-sm font-medium text-gray-900">30-day guarantee</p>
            <p className="text-xs text-gray-500">Full refund if not satisfied</p>
          </div>
          <div className="flex flex-col items-center">
            <CreditCardIcon className="h-8 w-8 text-blue-600" />
            <p className="mt-2 text-sm font-medium text-gray-900">Secure payments</p>
            <p className="text-xs text-gray-500">Powered by Stripe</p>
          </div>
          <div className="flex flex-col items-center">
            <ArrowPathIcon className="h-8 w-8 text-purple-600" />
            <p className="mt-2 text-sm font-medium text-gray-900">Cancel anytime</p>
            <p className="text-xs text-gray-500">No long-term contracts</p>
          </div>
        </div>

        {/* Pricing cards */}
        <div className="isolate mx-auto mt-16 grid max-w-md grid-cols-1 gap-y-8 lg:mx-0 lg:max-w-none lg:grid-cols-3 lg:gap-x-8 lg:gap-y-0">
          {tiers.map((tier, tierIdx) => (
            <div
              key={tier.id}
              className={classNames(
                tier.mostPopular ? 'ring-2 ring-indigo-600' : 'ring-1 ring-gray-200',
                tier.emphasis ? 'relative shadow-2xl' : 'shadow-sm',
                'rounded-3xl p-8 lg:p-10'
              )}
            >
              {tier.mostPopular && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                  <span className="inline-flex rounded-full bg-indigo-600 px-4 py-1 text-xs font-semibold text-white">
                    Most popular
                  </span>
                </div>
              )}
              
              <div>
                <h3 className="text-2xl font-bold tracking-tight text-gray-900">{tier.name}</h3>
                <p className="mt-4 text-sm leading-6 text-gray-600">{tier.description}</p>
                <p className="mt-6 flex items-baseline gap-x-1">
                  <span className="text-4xl font-bold tracking-tight text-gray-900">
                    {tier.price[frequency.value as keyof typeof tier.price]}
                  </span>
                  <span className="text-sm font-semibold leading-6 text-gray-600">
                    {frequency.priceSuffix}
                  </span>
                </p>
                <Link
                  href={tier.href}
                  className={classNames(
                    tier.emphasis
                      ? 'bg-indigo-600 text-white shadow-sm hover:bg-indigo-500'
                      : 'bg-white text-indigo-600 ring-1 ring-inset ring-indigo-200 hover:ring-indigo-300',
                    'mt-8 block rounded-md px-3.5 py-2.5 text-center text-sm font-semibold focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 transition-all duration-200'
                  )}
                >
                  {tier.cta}
                </Link>
              </div>
              
              <div className="mt-8">
                <h4 className="text-sm font-semibold leading-6 text-gray-900">What's included</h4>
                <ul role="list" className="mt-6 space-y-3 text-sm leading-6 text-gray-600">
                  {tier.features.map((feature) => (
                    <li key={feature} className="flex gap-x-3">
                      <CheckIcon className="h-6 w-5 flex-none text-indigo-600" aria-hidden="true" />
                      {feature}
                    </li>
                  ))}
                  {tier.notIncluded.map((feature) => (
                    <li key={feature} className="flex gap-x-3 text-gray-400">
                      <XMarkIcon className="h-6 w-5 flex-none" aria-hidden="true" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>

        {/* Enterprise section */}
        <div className="mt-24 rounded-3xl bg-gray-50 px-6 py-8 sm:px-10 sm:py-10 lg:px-20">
          <div className="lg:flex lg:items-center lg:justify-between">
            <div>
              <h3 className="text-2xl font-bold tracking-tight text-gray-900">
                Need a custom solution?
              </h3>
              <p className="mt-3 max-w-3xl text-base leading-6 text-gray-600">
                For large leagues, multiple organizations, or custom integrations, 
                we offer tailored enterprise solutions with dedicated support.
              </p>
            </div>
            <div className="mt-5 lg:ml-10 lg:mt-0 lg:flex-shrink-0">
              <Link
                href="/contact"
                className="inline-flex items-center rounded-md bg-indigo-600 px-6 py-3 text-base font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
              >
                Contact sales
              </Link>
            </div>
          </div>
        </div>

        {/* FAQ section */}
        <div className="mx-auto mt-24 max-w-4xl">
          <h2 className="text-2xl font-bold leading-10 tracking-tight text-gray-900 text-center">
            Frequently asked questions
          </h2>
          <dl className="mt-10 space-y-6 divide-y divide-gray-900/10">
            {faqs.map((faq) => (
              <div key={faq.question} className="pt-6 first:pt-0">
                <dt className="text-base font-semibold leading-7 text-gray-900">
                  {faq.question}
                </dt>
                <dd className="mt-2 text-base leading-7 text-gray-600">
                  {faq.answer}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </div>
  )
}