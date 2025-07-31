'use client'

import { motion } from 'framer-motion'
import { CheckCircleIcon } from '@heroicons/react/24/outline'
import { METRICS, FEATURES } from '@/lib/constants'

const stats = [
  { label: 'Prediction Accuracy', value: METRICS.accuracy.percentage, description: METRICS.accuracy.description },
  { label: 'Players Analyzed', value: METRICS.users.playersAnalyzed, description: 'All skill positions covered' },
  { label: 'Weekly Predictions', value: METRICS.users.predictions, description: 'Generated every week' },
  { label: 'Active Users', value: METRICS.users.active, description: 'Growing community' },
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
            Our ML models are continuously trained on the latest data and rigorously tested against actual results
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
      </div>
    </section>
  )
}