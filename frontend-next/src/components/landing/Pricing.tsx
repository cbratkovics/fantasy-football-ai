'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { CheckIcon } from '@heroicons/react/24/outline'

const included = [
  'Weekly point projections for all skill positions',
  'Confidence intervals on every projection',
  'Player tier visualizations',
  'Start/sit comparisons',
  'Mock draft simulator',
]

export function Pricing() {
  return (
    <section className="py-24 sm:py-32 bg-gray-50">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            Free while in beta
          </h2>
          <p className="mt-4 text-lg leading-8 text-gray-600">
            WinMyLeague.ai is an independent project and is not monetized. Every feature is
            available at no cost, and no payment details are collected.
          </p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mx-auto mt-16 max-w-lg"
        >
          <div className="rounded-3xl bg-white p-8 ring-2 ring-primary-600 sm:p-10">
            <h3 className="text-2xl font-bold tracking-tight text-gray-900">Beta access</h3>
            <p className="mt-2 text-sm leading-6 text-gray-600">
              Full access to everything currently built.
            </p>

            <div className="mt-6 flex items-baseline gap-x-2">
              <span className="text-5xl font-bold tracking-tight text-gray-900">$0</span>
              <span className="text-base text-gray-500">while in beta</span>
            </div>

            <ul role="list" className="mt-8 space-y-3 text-sm leading-6 text-gray-600">
              {included.map((feature) => (
                <li key={feature} className="flex gap-x-3">
                  <CheckIcon
                    className="h-6 w-5 flex-none text-primary-600"
                    aria-hidden="true"
                  />
                  {feature}
                </li>
              ))}
            </ul>

            <Link
              href="/auth/signup"
              className="mt-10 block rounded-md bg-primary-600 px-3 py-2.5 text-center text-sm font-semibold text-white shadow-sm hover:bg-primary-500"
            >
              Create an account
            </Link>
            <p className="mt-4 text-center text-xs text-gray-500">
              No payment details required
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
