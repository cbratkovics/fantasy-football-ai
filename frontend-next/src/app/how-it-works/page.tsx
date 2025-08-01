import { Metadata } from 'next'
import Link from 'next/link'
import Image from 'next/image'
import { 
  UserPlusIcon, 
  CpuChipIcon, 
  ChartBarIcon, 
  TrophyIcon,
  PlayCircleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

export const metadata: Metadata = {
  title: 'How It Works | WinMyLeague.ai - Your Path to Fantasy Football Success',
  description: 'Learn how WinMyLeague.ai uses AI to analyze players, optimize lineups, and help you win your fantasy football league in 4 simple steps.',
}

const steps = [
  {
    id: 1,
    name: 'Connect Your League',
    description: 'Import your team from ESPN, Yahoo, or Sleeper in seconds. Or create a new team from scratch. We support all major scoring formats.',
    icon: UserPlusIcon,
    features: [
      'One-click import from major platforms',
      'Custom scoring settings support',
      'Multiple team management',
      'Secure OAuth connection'
    ],
    screenshot: '/screenshots/connect-league.png', // Placeholder
    time: '< 30 seconds'
  },
  {
    id: 2,
    name: 'AI Analyzes Everything',
    description: 'Our models process player stats, matchups, weather, injuries, and historical trends. Get insights that would take hours to compile manually.',
    icon: CpuChipIcon,
    features: [
      'Real-time data from 20+ sources',
      'Machine learning predictions',
      'Injury impact analysis',
      'Weather and venue factors'
    ],
    screenshot: '/screenshots/ai-analysis.png', // Placeholder
    time: 'Continuous updates'
  },
  {
    id: 3,
    name: 'Get Clear Recommendations',
    description: 'Receive easy-to-understand advice for drafts, weekly lineups, and trades. Every recommendation includes confidence scores and explanations.',
    icon: ChartBarIcon,
    features: [
      'Weekly start/sit decisions',
      'Trade value calculator',
      'Waiver wire targets',
      'Draft strategy guidance'
    ],
    screenshot: '/screenshots/recommendations.png', // Placeholder
    time: 'Updated hourly'
  },
  {
    id: 4,
    name: 'Dominate Your League',
    description: 'Make confident decisions backed by data. Track your improvement and celebrate as you climb the standings.',
    icon: TrophyIcon,
    features: [
      'Performance tracking dashboard',
      'Win probability calculations',
      'Season-long insights',
      'Championship strategies'
    ],
    screenshot: '/screenshots/dominate.png', // Placeholder
    time: 'All season long'
  }
]

const faqs = [
  {
    question: 'How accurate are the predictions?',
    answer: 'Our AI models achieve 93.1% accuracy in weekly player projections, significantly outperforming traditional fantasy football advice.'
  },
  {
    question: 'Which platforms do you support?',
    answer: 'We integrate with ESPN, Yahoo Fantasy, and Sleeper. You can also manage teams manually for any other platform.'
  },
  {
    question: 'How often is data updated?',
    answer: 'Player data, injuries, and news are updated in real-time. Projections refresh hourly during the season.'
  },
  {
    question: 'Is my league data secure?',
    answer: 'Absolutely. We use bank-level encryption and never share your data. Read-only access means we can\'t modify your leagues.'
  }
]

export default function HowItWorksPage() {
  return (
    <div className="bg-white">
      {/* Hero section */}
      <div className="relative isolate overflow-hidden bg-gradient-to-b from-indigo-100/20 pt-14">
        <div className="mx-auto max-w-7xl px-6 pt-10 pb-24 sm:pt-16 lg:px-8 lg:pt-24">
          {/* Breadcrumb */}
          <div className="mb-8">
            <Breadcrumb items={[{ name: 'How It Works', current: true }]} />
          </div>
          
          <div className="mx-auto max-w-2xl text-center">
            <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
              Win your league in 4 simple steps
            </h1>
            <p className="mt-6 text-lg leading-8 text-gray-600">
              WinMyLeague.ai makes fantasy football success simple. No spreadsheets, no guesswork—just 
              data-driven decisions that help you win.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <button className="inline-flex items-center gap-x-2 rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600">
                <PlayCircleIcon className="h-5 w-5" />
                Watch demo
              </button>
              <Link href="/auth/signup" className="text-sm font-semibold leading-6 text-gray-900">
                Get started free <span aria-hidden="true">→</span>
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Steps section */}
      <div className="mx-auto max-w-7xl px-6 lg:px-8 py-24 sm:py-32">
        <div className="mx-auto max-w-2xl lg:max-w-none">
          {steps.map((step, stepIdx) => (
            <div
              key={step.id}
              className={`${
                stepIdx % 2 === 0 ? 'lg:flex-row' : 'lg:flex-row-reverse'
              } flex flex-col lg:flex items-center gap-x-16 gap-y-10 ${
                stepIdx !== 0 ? 'mt-32' : ''
              }`}
            >
              {/* Content */}
              <div className="flex-1">
                <div className="lg:max-w-lg">
                  <p className="text-base font-semibold leading-7 text-indigo-600">
                    Step {step.id}
                  </p>
                  <h2 className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
                    {step.name}
                  </h2>
                  <p className="mt-6 text-lg leading-8 text-gray-600">
                    {step.description}
                  </p>
                  
                  <ul className="mt-8 space-y-3">
                    {step.features.map((feature) => (
                      <li key={feature} className="flex gap-x-3">
                        <CheckCircleIcon className="mt-1 h-5 w-5 flex-none text-indigo-600" />
                        <span className="text-base leading-7 text-gray-600">{feature}</span>
                      </li>
                    ))}
                  </ul>
                  
                  <div className="mt-8 flex items-center gap-x-4">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-600">
                      <step.icon className="h-6 w-6 text-white" />
                    </div>
                    <p className="text-sm font-medium text-gray-900">
                      Setup time: <span className="text-indigo-600">{step.time}</span>
                    </p>
                  </div>
                </div>
              </div>
              
              {/* Screenshot placeholder */}
              <div className="flex-1 lg:flex-none lg:w-[48rem]">
                <div className="aspect-[16/9] rounded-2xl bg-gray-100 shadow-xl ring-1 ring-gray-900/10">
                  <div className="flex h-full items-center justify-center">
                    <div className="text-center">
                      <step.icon className="mx-auto h-12 w-12 text-gray-400" />
                      <p className="mt-4 text-sm text-gray-500">
                        Screenshot coming soon
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Video section */}
      <div className="bg-gray-50">
        <div className="mx-auto max-w-7xl px-6 py-24 sm:py-32 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
              See WinMyLeague.ai in action
            </h2>
            <p className="mt-6 text-lg leading-8 text-gray-600">
              Watch a 2-minute demo showing how to set up your team and get your first recommendations.
            </p>
          </div>
          
          <div className="mx-auto mt-16 max-w-4xl">
            <div className="aspect-video rounded-2xl bg-gray-900 shadow-xl">
              <div className="flex h-full items-center justify-center">
                <button className="group relative inline-flex items-center justify-center">
                  <div className="absolute -inset-4 rounded-full bg-white/20 blur-lg transition group-hover:bg-white/30" />
                  <PlayCircleIcon className="relative h-20 w-20 text-white" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* FAQ section */}
      <div className="mx-auto max-w-7xl px-6 py-24 sm:py-32 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-2xl font-bold leading-10 tracking-tight text-gray-900">
            Frequently asked questions
          </h2>
        </div>
        <dl className="mx-auto mt-16 max-w-2xl space-y-8 divide-y divide-gray-900/10">
          {faqs.map((faq) => (
            <div key={faq.question} className="pt-8 first:pt-0">
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

      {/* CTA section */}
      <div className="bg-indigo-600">
        <div className="mx-auto max-w-7xl px-6 py-24 sm:py-32 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
              Ready to start winning?
            </h2>
            <p className="mx-auto mt-6 max-w-xl text-lg leading-8 text-indigo-100">
              Join thousands of fantasy players who are already dominating their leagues with WinMyLeague.ai.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Link
                href="/auth/signup"
                className="rounded-md bg-white px-3.5 py-2.5 text-sm font-semibold text-indigo-600 shadow-sm hover:bg-indigo-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
              >
                Start free trial
              </Link>
              <Link href="/pricing" className="text-sm font-semibold leading-6 text-white">
                View pricing <span aria-hidden="true">→</span>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}