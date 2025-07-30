'use client'

import { motion } from 'framer-motion'
import { 
  ChartBarIcon, 
  LightBulbIcon, 
  ShieldCheckIcon,
  BoltIcon,
  UserGroupIcon,
  CurrencyDollarIcon 
} from '@heroicons/react/24/outline'

const features = [
  {
    name: 'AI-Powered Predictions',
    description: 'Neural networks and ML models analyze player performance, matchups, and trends to generate accurate predictions.',
    icon: ChartBarIcon,
    color: 'text-primary-600',
    bgColor: 'bg-primary-100',
  },
  {
    name: 'Transparent Explanations',
    description: 'Understand WHY each prediction was made with clear, plain-English explanations of key factors.',
    icon: LightBulbIcon,
    color: 'text-warning-600',
    bgColor: 'bg-warning-100',
  },
  {
    name: 'Confidence Scores',
    description: 'Every prediction includes confidence levels and risk assessments to help you make informed decisions.',
    icon: ShieldCheckIcon,
    color: 'text-success-600',
    bgColor: 'bg-success-100',
  },
  {
    name: 'Real-Time Updates',
    description: 'Predictions adjust automatically based on injuries, weather, and breaking news throughout the week.',
    icon: BoltIcon,
    color: 'text-danger-600',
    bgColor: 'bg-danger-100',
  },
  {
    name: 'Draft Tier System',
    description: '16-tier player rankings aligned with draft rounds help you identify value picks and avoid reaches.',
    icon: UserGroupIcon,
    color: 'text-indigo-600',
    bgColor: 'bg-indigo-100',
  },
  {
    name: 'Lineup Optimizer',
    description: 'Maximize your weekly score with AI-optimized lineups based on matchups and projections.',
    icon: CurrencyDollarIcon,
    color: 'text-purple-600',
    bgColor: 'bg-purple-100',
  },
]

export function Features() {
  return (
    <section className="py-20 bg-gray-50">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
            Everything You Need to Dominate
          </h2>
          <p className="mt-4 text-lg text-gray-600">
            Powerful features designed for serious fantasy football players
          </p>
        </div>

        <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
          <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-10 lg:max-w-none lg:grid-cols-3">
            {features.map((feature, index) => (
              <motion.div
                key={feature.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="relative bg-white rounded-2xl p-8 shadow-sm hover:shadow-lg transition-shadow duration-200"
              >
                <dt className="flex items-center gap-x-3">
                  <div className={`${feature.bgColor} rounded-lg p-3`}>
                    <feature.icon className={`h-6 w-6 ${feature.color}`} aria-hidden="true" />
                  </div>
                  <span className="text-xl font-semibold text-gray-900">{feature.name}</span>
                </dt>
                <dd className="mt-4 text-base leading-7 text-gray-600">
                  {feature.description}
                </dd>
              </motion.div>
            ))}
          </dl>
        </div>

        {/* Feature Preview */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="mt-20"
        >
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
            <div className="bg-gradient-to-r from-primary-600 to-primary-700 px-8 py-6">
              <h3 className="text-2xl font-semibold text-white">Example Prediction</h3>
            </div>
            <div className="p-8">
              <div className="space-y-6">
                {/* Player Info */}
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-xl font-semibold text-gray-900">Josh Allen</h4>
                    <p className="text-gray-600">QB - Buffalo Bills</p>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold text-primary-600">26.5</div>
                    <div className="text-sm text-gray-600">Projected Points</div>
                  </div>
                </div>

                {/* Confidence */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700">Confidence</span>
                    <span className="text-sm font-semibold text-success-600">High (85%)</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-success-600 h-2 rounded-full" style={{ width: '85%' }} />
                  </div>
                </div>

                {/* Key Factors */}
                <div>
                  <h5 className="font-semibold text-gray-900 mb-3">Key Factors</h5>
                  <div className="space-y-2">
                    <div className="flex items-start gap-2">
                      <span className="text-success-600">↑</span>
                      <span className="text-sm text-gray-700">
                        Recent performance trending upward with 28.5 point average over last 3 games
                      </span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-success-600">↑</span>
                      <span className="text-sm text-gray-700">
                        Facing 28th ranked defense - favorable matchup
                      </span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-gray-400">→</span>
                      <span className="text-sm text-gray-700">
                        Consistent performer with 15% scoring variation
                      </span>
                    </div>
                  </div>
                </div>

                {/* Recommendation */}
                <div className="bg-success-50 border border-success-200 rounded-lg p-4">
                  <p className="text-success-800 font-medium">
                    💡 Strong start recommendation - high upside with minimal risk
                  </p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}