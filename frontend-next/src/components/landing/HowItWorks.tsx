'use client'

import { motion } from 'framer-motion'

const steps = [
  {
    number: '1',
    title: 'Connect Your League',
    description: 'Import your roster from ESPN, Yahoo, or Sleeper. Or use our manual entry.',
  },
  {
    number: '2',
    title: 'Get AI Predictions',
    description: 'Our models analyze thousands of data points to generate predictions with confidence scores.',
  },
  {
    number: '3',
    title: 'Understand the Why',
    description: 'See transparent explanations for every prediction - no black box algorithms.',
  },
  {
    number: '4',
    title: 'Optimize & Win',
    description: 'Use our lineup optimizer and weekly insights to maximize your score.',
  },
]

export function HowItWorks() {
  return (
    <section className="py-20 bg-white">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            How It Works
          </h2>
          <p className="mt-4 text-lg text-gray-600">
            Get started in minutes and see predictions instantly
          </p>
        </div>

        <div className="mx-auto mt-16 max-w-2xl lg:max-w-none">
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-4">
            {steps.map((step, index) => (
              <motion.div
                key={step.number}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="relative"
              >
                {/* Connector Line */}
                {index < steps.length - 1 && (
                  <div className="hidden lg:block absolute top-12 left-1/2 w-full h-0.5 bg-gradient-to-r from-primary-300 to-primary-600" />
                )}

                <div className="relative text-center">
                  <div className="mx-auto h-20 w-20 rounded-full bg-primary-600 flex items-center justify-center text-white text-2xl font-bold shadow-lg">
                    {step.number}
                  </div>
                  <h3 className="mt-6 text-xl font-semibold text-gray-900">
                    {step.title}
                  </h3>
                  <p className="mt-2 text-gray-600">
                    {step.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="mt-16 text-center"
        >
          <a
            href="/signup"
            className="inline-flex items-center justify-center px-8 py-3 text-lg font-semibold text-white bg-primary-600 rounded-lg hover:bg-primary-700 transition-colors duration-200"
          >
            Start Your Free Trial
          </a>
        </motion.div>
      </div>
    </section>
  )
}