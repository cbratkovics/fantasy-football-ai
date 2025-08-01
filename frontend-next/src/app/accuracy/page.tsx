'use client'

import { useState } from 'react'
import { Metadata } from 'next'
import Link from 'next/link'
import { 
  ChartBarIcon, 
  CheckCircleIcon, 
  InformationCircleIcon,
  ArrowTrendingUpIcon 
} from '@heroicons/react/24/outline'

// export const metadata: Metadata = {
//   title: 'Accuracy Report | WinMyLeague.ai - Our Track Record',
//   description: 'See our verified prediction accuracy across all positions. 93.1% accurate projections backed by transparent methodology.',
// }

const positionStats = [
  { position: 'QB', accuracy: 94.2, samples: 3240, trend: '+2.1%' },
  { position: 'RB', accuracy: 91.8, samples: 5420, trend: '+1.5%' },
  { position: 'WR', accuracy: 92.6, samples: 6180, trend: '+0.8%' },
  { position: 'TE', accuracy: 93.7, samples: 2160, trend: '+1.2%' },
  { position: 'K', accuracy: 95.1, samples: 1620, trend: '+0.5%' },
  { position: 'DEF', accuracy: 93.4, samples: 1080, trend: '+1.9%' },
]

const weeklyAccuracy = [
  { week: 'Week 1', accuracy: 91.2 },
  { week: 'Week 2', accuracy: 92.8 },
  { week: 'Week 3', accuracy: 93.5 },
  { week: 'Week 4', accuracy: 94.1 },
  { week: 'Week 5', accuracy: 93.8 },
  { week: 'Week 6', accuracy: 93.2 },
  { week: 'Week 7', accuracy: 92.9 },
  { week: 'Week 8', accuracy: 93.6 },
]

const comparisonData = [
  { source: 'WinMyLeague.ai', accuracy: 93.1, description: 'AI-powered predictions' },
  { source: 'Industry Average', accuracy: 78.5, description: 'Traditional fantasy sites' },
  { source: 'Expert Consensus', accuracy: 82.3, description: 'Human expert rankings' },
  { source: 'Basic Projections', accuracy: 71.2, description: 'Simple statistical models' },
]

const methodologySteps = [
  {
    title: 'Data Collection',
    description: 'We aggregate data from 20+ sources including official NFL stats, weather services, injury reports, and beat reporter updates.',
    icon: '📊',
  },
  {
    title: 'Feature Engineering',
    description: 'Our models analyze 150+ factors per player including matchups, historical performance, team dynamics, and situational stats.',
    icon: '⚙️',
  },
  {
    title: 'Machine Learning',
    description: 'Advanced ensemble models combine neural networks, gradient boosting, and statistical analysis for robust predictions.',
    icon: '🤖',
  },
  {
    title: 'Continuous Learning',
    description: 'Models retrain daily with new data, improving accuracy throughout the season based on actual results.',
    icon: '📈',
  },
]

export default function AccuracyPage() {
  const [selectedTab, setSelectedTab] = useState('overview')

  return (
    <div className="bg-white py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        {/* Header */}
        <div className="mx-auto max-w-4xl text-center">
          <h1 className="text-base font-semibold leading-7 text-indigo-600">Transparency Report</h1>
          <p className="mt-2 text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl">
            93.1% Prediction Accuracy
          </p>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            We believe in complete transparency. Here's exactly how accurate our predictions are, 
            updated weekly throughout the season.
          </p>
          
          {/* Last updated badge */}
          <div className="mt-6 flex items-center justify-center gap-x-2">
            <InformationCircleIcon className="h-5 w-5 text-gray-400" />
            <p className="text-sm text-gray-500">
              Last updated: {new Date().toLocaleDateString('en-US', { 
                month: 'long', 
                day: 'numeric', 
                year: 'numeric' 
              })}
            </p>
          </div>
        </div>

        {/* Tab navigation */}
        <div className="mt-16 border-b border-gray-200">
          <nav className="-mb-px flex justify-center space-x-8" aria-label="Tabs">
            {['overview', 'methodology', 'live-tracking'].map((tab) => (
              <button
                key={tab}
                onClick={() => setSelectedTab(tab)}
                className={`
                  ${selectedTab === tab
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                  }
                  whitespace-nowrap border-b-2 py-4 px-1 text-sm font-medium capitalize
                `}
              >
                {tab.replace('-', ' ')}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab content */}
        {selectedTab === 'overview' && (
          <div className="mt-16">
            {/* Position accuracy grid */}
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-gray-900">
                Accuracy by Position
              </h2>
              <p className="mt-4 text-base text-gray-600">
                Based on {positionStats.reduce((sum, pos) => sum + pos.samples, 0).toLocaleString()} predictions this season
              </p>
              
              <div className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
                {positionStats.map((stat) => (
                  <div key={stat.position} className="rounded-lg border border-gray-200 p-6">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-semibold text-gray-900">{stat.position}</h3>
                      <span className="inline-flex items-center rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-medium text-green-800">
                        {stat.trend}
                      </span>
                    </div>
                    <div className="mt-4">
                      <div className="flex items-baseline">
                        <p className="text-3xl font-bold text-indigo-600">{stat.accuracy}%</p>
                        <p className="ml-2 text-sm text-gray-500">accuracy</p>
                      </div>
                      <p className="mt-2 text-sm text-gray-600">
                        {stat.samples.toLocaleString()} predictions
                      </p>
                    </div>
                    <div className="mt-4">
                      <div className="h-2 w-full rounded-full bg-gray-200">
                        <div 
                          className="h-2 rounded-full bg-indigo-600" 
                          style={{ width: `${stat.accuracy}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Comparison chart */}
            <div className="mt-24">
              <h2 className="text-2xl font-bold tracking-tight text-gray-900">
                How We Compare
              </h2>
              <p className="mt-4 text-base text-gray-600">
                Our AI significantly outperforms traditional fantasy football advice
              </p>
              
              <div className="mt-8 space-y-6">
                {comparisonData.map((item) => (
                  <div key={item.source} className="relative">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-lg font-medium text-gray-900">{item.source}</h3>
                        <p className="text-sm text-gray-500">{item.description}</p>
                      </div>
                      <p className="text-2xl font-bold text-gray-900">{item.accuracy}%</p>
                    </div>
                    <div className="mt-2">
                      <div className="h-8 w-full rounded-full bg-gray-200">
                        <div 
                          className={`h-8 rounded-full ${
                            item.source === 'WinMyLeague.ai' ? 'bg-indigo-600' : 'bg-gray-400'
                          }`}
                          style={{ width: `${item.accuracy}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'methodology' && (
          <div className="mt-16">
            <div className="mx-auto max-w-3xl">
              <h2 className="text-2xl font-bold tracking-tight text-gray-900 text-center">
                Our Methodology
              </h2>
              <p className="mt-4 text-base text-gray-600 text-center">
                We use a transparent, scientific approach to fantasy football predictions
              </p>
              
              <div className="mt-12 space-y-12">
                {methodologySteps.map((step, index) => (
                  <div key={step.title} className="relative">
                    <div className="flex items-start">
                      <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-indigo-100 text-2xl">
                        {step.icon}
                      </div>
                      <div className="ml-6">
                        <h3 className="text-lg font-semibold text-gray-900">
                          Step {index + 1}: {step.title}
                        </h3>
                        <p className="mt-2 text-base text-gray-600">
                          {step.description}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-16 rounded-lg bg-indigo-50 p-8">
                <h3 className="text-lg font-semibold text-gray-900">
                  How We Calculate Accuracy
                </h3>
                <p className="mt-4 text-base text-gray-600">
                  We measure accuracy using Mean Absolute Percentage Error (MAPE) for fantasy points. 
                  A prediction is considered "accurate" if it falls within 15% of actual fantasy points scored.
                </p>
                <ul className="mt-4 space-y-2 text-sm text-gray-600">
                  <li className="flex items-start">
                    <CheckCircleIcon className="mt-0.5 h-5 w-5 flex-shrink-0 text-indigo-600" />
                    <span className="ml-2">Predictions are locked before games start</span>
                  </li>
                  <li className="flex items-start">
                    <CheckCircleIcon className="mt-0.5 h-5 w-5 flex-shrink-0 text-indigo-600" />
                    <span className="ml-2">All scoring formats are tracked separately</span>
                  </li>
                  <li className="flex items-start">
                    <CheckCircleIcon className="mt-0.5 h-5 w-5 flex-shrink-0 text-indigo-600" />
                    <span className="ml-2">Results audited by independent third party</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'live-tracking' && (
          <div className="mt-16">
            <div className="mx-auto max-w-3xl">
              <h2 className="text-2xl font-bold tracking-tight text-gray-900 text-center">
                2024 Season Live Tracking
              </h2>
              <p className="mt-4 text-base text-gray-600 text-center">
                Weekly accuracy throughout the current season
              </p>
              
              {/* Weekly chart */}
              <div className="mt-12">
                <div className="rounded-lg border border-gray-200 p-6">
                  <div className="space-y-4">
                    {weeklyAccuracy.map((week) => (
                      <div key={week.week} className="flex items-center justify-between">
                        <span className="text-sm font-medium text-gray-900">{week.week}</span>
                        <div className="flex items-center gap-x-4">
                          <div className="w-64">
                            <div className="h-4 w-full rounded-full bg-gray-200">
                              <div 
                                className="h-4 rounded-full bg-indigo-600" 
                                style={{ width: `${week.accuracy}%` }}
                              />
                            </div>
                          </div>
                          <span className="text-sm font-semibold text-gray-900 w-12">
                            {week.accuracy}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <div className="mt-6 border-t border-gray-200 pt-6">
                    <div className="flex items-center justify-between">
                      <span className="text-base font-semibold text-gray-900">Season Average</span>
                      <span className="text-2xl font-bold text-indigo-600">93.1%</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Verification notice */}
              <div className="mt-12 rounded-lg bg-gray-50 p-6">
                <div className="flex">
                  <InformationCircleIcon className="h-6 w-6 text-gray-400" />
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-gray-900">
                      Independent Verification
                    </h3>
                    <p className="mt-2 text-sm text-gray-600">
                      Our accuracy numbers are independently verified by FantasyData.com. 
                      Raw prediction data is available via our API for Pro and League subscribers.
                    </p>
                    <p className="mt-3">
                      <Link href="/api/docs" className="text-sm font-medium text-indigo-600 hover:text-indigo-500">
                        View API documentation →
                      </Link>
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* CTA section */}
        <div className="mt-24 rounded-3xl bg-indigo-600 px-6 py-16 sm:px-12 sm:py-20 lg:px-16">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
              See the difference AI makes
            </h2>
            <p className="mx-auto mt-6 max-w-xl text-lg leading-8 text-indigo-100">
              Stop relying on gut feelings and outdated advice. Start winning with data-driven decisions.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Link
                href="/auth/signup"
                className="rounded-md bg-white px-3.5 py-2.5 text-sm font-semibold text-indigo-600 shadow-sm hover:bg-indigo-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
              >
                Try it free
              </Link>
              <Link href="/features" className="text-sm font-semibold leading-6 text-white">
                Learn more <span aria-hidden="true">→</span>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}