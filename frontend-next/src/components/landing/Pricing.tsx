'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { CheckIcon } from '@heroicons/react/24/outline'
import { loadStripe } from '@stripe/stripe-js'
import { useAuth } from '@clerk/nextjs'
import toast from 'react-hot-toast'

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!)

const tiers = [
  {
    name: 'Free',
    id: 'free',
    price: 0,
    description: 'Perfect for trying out our predictions',
    features: [
      '5 predictions per week',
      'Basic player rankings',
      'Public accuracy reports',
      'Email support',
    ],
    limitations: [
      'No lineup optimizer',
      'No API access',
      'Limited to 1 league',
    ],
    cta: 'Get Started',
    featured: false,
  },
  {
    name: 'Season Pass',
    id: 'pro',
    price: 20,
    description: 'Everything you need to win your league',
    features: [
      'Unlimited predictions all season',
      'Advanced lineup optimizer',
      'Real-time injury updates',
      'Custom scoring support',
      'Email/SMS alerts',
      'API access',
      'Priority support',
      '7-day free trial (August only)',
    ],
    limitations: [],
    cta: 'Start Free Trial',
    featured: true,
  },
]

export function Pricing() {
  const { isSignedIn } = useAuth()
  const [loading, setLoading] = useState(false)

  const handleSubscribe = async (tierId: string) => {
    if (tierId === 'free') {
      window.location.href = '/signup'
      return
    }

    if (!isSignedIn) {
      window.location.href = '/signup?plan=pro'
      return
    }

    setLoading(true)
    try {
      const response = await fetch('/api/create-checkout-session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          priceId: process.env.NEXT_PUBLIC_STRIPE_PRICE_ID,
        }),
      })

      const { sessionId } = await response.json()
      const stripe = await stripePromise
      
      if (stripe) {
        const { error } = await stripe.redirectToCheckout({ sessionId })
        if (error) {
          toast.error(error.message || 'Something went wrong')
        }
      }
    } catch (error) {
      toast.error('Failed to create checkout session')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="py-20 bg-gray-50">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            Simple, Transparent Pricing
          </h2>
          <p className="mt-4 text-lg text-gray-600">
            One price for the entire season. No hidden fees.
          </p>
        </div>

        <div className="mx-auto mt-16 grid max-w-lg grid-cols-1 gap-8 lg:max-w-4xl lg:grid-cols-2">
          {tiers.map((tier, index) => (
            <motion.div
              key={tier.id}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              className={`relative rounded-2xl ${
                tier.featured
                  ? 'bg-primary-600 shadow-xl ring-2 ring-primary-600'
                  : 'bg-white shadow-lg'
              } p-8`}
            >
              {tier.featured && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                  <span className="inline-flex items-center rounded-full bg-warning-100 px-4 py-1 text-sm font-semibold text-warning-800">
                    Most Popular
                  </span>
                </div>
              )}

              <div className="text-center">
                <h3 className={`text-2xl font-bold ${tier.featured ? 'text-white' : 'text-gray-900'}`}>
                  {tier.name}
                </h3>
                <p className={`mt-2 text-sm ${tier.featured ? 'text-primary-100' : 'text-gray-600'}`}>
                  {tier.description}
                </p>
                <div className={`mt-6 flex items-baseline justify-center gap-x-2 ${tier.featured ? 'text-white' : 'text-gray-900'}`}>
                  <span className="text-5xl font-bold tracking-tight">${tier.price}</span>
                  <span className={`text-sm ${tier.featured ? 'text-primary-100' : 'text-gray-600'}`}>
                    {tier.price > 0 ? '/season' : '/forever'}
                  </span>
                </div>
              </div>

              <ul className={`mt-8 space-y-3 ${tier.featured ? 'text-white' : 'text-gray-600'}`}>
                {tier.features.map((feature) => (
                  <li key={feature} className="flex items-start gap-3">
                    <CheckIcon className={`h-6 w-6 flex-shrink-0 ${tier.featured ? 'text-primary-200' : 'text-success-600'}`} />
                    <span className="text-sm">{feature}</span>
                  </li>
                ))}
                {tier.limitations.map((limitation) => (
                  <li key={limitation} className="flex items-start gap-3 opacity-60">
                    <span className="h-6 w-6 flex-shrink-0 text-center">✕</span>
                    <span className="text-sm">{limitation}</span>
                  </li>
                ))}
              </ul>

              <button
                onClick={() => handleSubscribe(tier.id)}
                disabled={loading}
                className={`mt-8 w-full rounded-lg px-4 py-3 text-sm font-semibold transition-colors duration-200 ${
                  tier.featured
                    ? 'bg-white text-primary-600 hover:bg-gray-100'
                    : 'bg-primary-600 text-white hover:bg-primary-700'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {loading ? 'Loading...' : tier.cta}
              </button>
            </motion.div>
          ))}
        </div>

        {/* FAQ */}
        <div className="mt-20 mx-auto max-w-2xl">
          <h3 className="text-2xl font-bold text-center text-gray-900 mb-8">
            Frequently Asked Questions
          </h3>
          <div className="space-y-6">
            {[
              {
                q: 'How long is a season?',
                a: 'A season runs from August through the Super Bowl in February. Your subscription covers the entire NFL season including playoffs.',
              },
              {
                q: 'Can I cancel anytime?',
                a: 'Yes! You can cancel your subscription at any time. You\'ll continue to have access until the end of your billing period.',
              },
              {
                q: 'Do you offer refunds?',
                a: 'We offer a 7-day free trial so you can test everything risk-free. After that, we offer refunds on a case-by-case basis.',
              },
              {
                q: 'What payment methods do you accept?',
                a: 'We accept all major credit cards, debit cards, and digital wallets through our secure payment processor, Stripe.',
              },
            ].map((faq, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white rounded-lg p-6 shadow-sm"
              >
                <h4 className="font-semibold text-gray-900">{faq.q}</h4>
                <p className="mt-2 text-gray-600">{faq.a}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}