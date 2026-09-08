'use client'

import { motion } from 'framer-motion'
import { CheckCircleIcon, ShieldCheckIcon, ClockIcon, ArrowTrendingUpIcon } from '@heroicons/react/24/outline'
import { METRICS, FEATURES } from '@/lib/constants'
import Link from 'next/link'

const stats = [
  { label: 'Projections', value: 'Weekly', description: METRICS.accuracy.description },
  { label: 'Coverage', value: 'All skill positions', description: 'QB, RB, WR, TE, and K' },
  { label: 'Models', value: 'Ensemble', description: METRICS.models.architecture },
  { label: 'Explanations', value: 'Per prediction', description: 'Feature attributions included' },
]

export function Accuracy() {
  return (
    <section className="py-20 bg-white">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            Proven Accuracy You Can Trust
          </h2>
          <p className="mt-4 text-lg text-gray-600">
            Our AI models are continuously trained on the latest data and independently verified weekly
          </p>
        </div>

        <div className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-6 sm:grid-cols-2 lg:mx-0 lg:max-w-none lg:grid-cols-4">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="bg-gray-50 rounded-2xl p-8 text-center"
            >
              <div className="text-4xl font-bold text-primary-600">{stat.value}</div>
              <div className="mt-2 text-sm font-semibold text-gray-900">{stat.label}</div>
              <div className="mt-1 text-sm text-gray-600">{stat.description}</div>
            </motion.div>
          ))}
        </div>

        {/* Trust Indicators */}
        <div className="mt-16 bg-primary-50 rounded-2xl p-8">
          <div className="mx-auto max-w-3xl">
            <h3 className="text-center text-2xl font-semibold text-gray-900 mb-8">
              Why Our Predictions Are Different
            </h3>
            <div className="grid gap-4 sm:grid-cols-2">
              {FEATURES.core.slice(1, 7).map((feature, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.05 }}
                  viewport={{ once: true }}
                  className="flex items-start gap-3"
                >
                  <CheckCircleIcon className="h-6 w-6 text-success-600 flex-shrink-0" />
                  <span className="text-gray-700">{feature}</span>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {/* Trust Badges */}
        <div className="mt-16">
          <div className="grid grid-cols-1 gap-8 sm:grid-cols-3">
            <div className="text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-indigo-100">
                <ShieldCheckIcon className="h-6 w-6 text-indigo-600" />
              </div>
              <h3 className="mt-4 text-sm font-semibold text-gray-900">Independently Verified</h3>
              <p className="mt-2 text-sm text-gray-600">
                Accuracy numbers validated weekly against actual NFL results
              </p>
            </div>
            <div className="text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-indigo-100">
                <ClockIcon className="h-6 w-6 text-indigo-600" />
              </div>
              <h3 className="mt-4 text-sm font-semibold text-gray-900">Real-time Updates</h3>
              <p className="mt-2 text-sm text-gray-600">
                Predictions update within minutes of injury reports and news
              </p>
            </div>
            <div className="text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-indigo-100">
                <ArrowTrendingUpIcon className="h-6 w-6 text-indigo-600" />
              </div>
              <h3 className="mt-4 text-sm font-semibold text-gray-900">Proven Track Record</h3>
              <p className="mt-2 text-sm text-gray-600">
                Models trained on multiple seasons of historical NFL data
              </p>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="mt-12 text-center">
          <Link
            href="/how-it-works"
            className="inline-flex items-center text-sm font-semibold text-indigo-600 hover:text-indigo-500"
          >
            See how the projections are built
            <span aria-hidden="true" className="ml-2">→</span>
          </Link>
        </div>
      </div>
    </section>
  )
}