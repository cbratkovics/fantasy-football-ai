import { Metadata } from 'next'
import Link from 'next/link'
import { 
  ChartBarIcon, 
  BoltIcon, 
  UsersIcon, 
  ArrowPathIcon, 
  ChartPieIcon,
  PresentationChartLineIcon 
} from '@heroicons/react/24/outline'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

export const metadata: Metadata = {
  title: 'Features | WinMyLeague.ai - AI-Powered Fantasy Football',
  description: 'Discover how WinMyLeague.ai helps you dominate your fantasy football league with AI-powered predictions, smart player tiers, and real-time insights.',
}

const features = [
  {
    name: 'AI-Powered Predictions',
    description: 'Our advanced machine learning models analyze thousands of data points to deliver 93.1% accurate weekly projections for every NFL player.',
    icon: BoltIcon,
    stats: {
      accuracy: '93.1%',
      dataPoints: '10,000+',
      updateFrequency: 'Real-time'
    },
    href: '/accuracy'
  },
  {
    name: 'Smart Player Tiers',
    description: 'Visual tier system groups players by projected value, making draft decisions and weekly starts crystal clear. No more guesswork.',
    icon: ChartBarIcon,
    stats: {
      tiers: 'Position-specific',
      confidence: 'Score included',
      updates: 'Weekly'
    },
    href: '/tiers'
  },
  {
    name: 'Weekly Lineup Optimizer',
    description: 'Automated lineup recommendations based on matchups, weather, injuries, and historical performance. Set your optimal lineup in seconds.',
    icon: UsersIcon,
    stats: {
      factors: '15+ analyzed',
      winRate: '+23% improvement',
      time: '< 30 seconds'
    },
    href: '/start-sit'
  },
  {
    name: 'Mock Draft Simulator',
    description: 'Practice with AI opponents that mimic real draft behavior. Test strategies and discover sleepers before your actual draft.',
    icon: ChartPieIcon,
    stats: {
      aiPlayers: '11 unique styles',
      scenarios: 'Unlimited',
      analysis: 'Post-draft grades'
    },
    href: '/draft'
  },
  {
    name: 'Real-time Updates',
    description: 'Instant notifications for injuries, weather changes, and lineup news. Never miss critical information that impacts your decisions.',
    icon: ArrowPathIcon,
    stats: {
      sources: '20+ integrated',
      latency: '< 1 minute',
      coverage: 'All NFL games'
    },
    href: '/dashboard'
  },
  {
    name: 'Performance Dashboard',
    description: 'Track your team\'s performance with detailed analytics. Understand what\'s working and optimize your strategy throughout the season.',
    icon: PresentationChartLineIcon,
    stats: {
      metrics: '25+ tracked',
      history: 'Full season',
      export: 'PDF reports'
    },
    href: '/performance'
  }
]

export default function FeaturesPage() {
  return (
    <div className="bg-white">
      {/* Hero section */}
      <div className="relative isolate overflow-hidden bg-gradient-to-b from-indigo-100/20 pt-14">
        <div className="mx-auto max-w-7xl px-6 pt-10 pb-24 sm:pt-16 lg:px-8 lg:pt-24">
          {/* Breadcrumb */}
          <div className="mb-8">
            <Breadcrumb items={[{ name: 'Features', current: true }]} />
          </div>
          
          <div className="mx-auto max-w-2xl text-center">
            <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
              Features that win championships
            </h1>
            <p className="mt-6 text-lg leading-8 text-gray-600">
              WinMyLeague.ai combines cutting-edge AI with deep fantasy football expertise. 
              Every feature is designed to give you an unfair advantage.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Link
                href="/auth/signup"
                className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
              >
                Start free trial
              </Link>
              <Link href="/how-it-works" className="text-sm font-semibold leading-6 text-gray-900">
                See how it works <span aria-hidden="true">→</span>
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Feature sections */}
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl lg:max-w-none">
          <dl className="grid grid-cols-1 gap-16 lg:grid-cols-2 lg:gap-24 py-24">
            {features.map((feature, index) => (
              <div key={feature.name} className="relative">
                <dt>
                  <div className="absolute left-0 top-0 flex h-16 w-16 items-center justify-center rounded-lg bg-indigo-600">
                    <feature.icon className="h-8 w-8 text-white" aria-hidden="true" />
                  </div>
                  <p className="ml-20 text-2xl font-bold leading-7 text-gray-900">
                    {feature.name}
                  </p>
                </dt>
                <dd className="mt-4 ml-20">
                  <p className="text-base leading-7 text-gray-600">
                    {feature.description}
                  </p>
                  
                  {/* Stats grid */}
                  <div className="mt-6 grid grid-cols-3 gap-4">
                    {Object.entries(feature.stats).map(([key, value]) => (
                      <div key={key} className="border-l-2 border-gray-200 pl-4">
                        <p className="text-2xl font-semibold text-indigo-600">{value}</p>
                        <p className="text-sm text-gray-500 capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</p>
                      </div>
                    ))}
                  </div>
                  
                  <div className="mt-6">
                    <Link
                      href={feature.href}
                      className="text-sm font-semibold leading-6 text-indigo-600 hover:text-indigo-500"
                    >
                      Learn more <span aria-hidden="true">→</span>
                    </Link>
                  </div>
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      {/* CTA section */}
      <div className="bg-indigo-50">
        <div className="mx-auto max-w-7xl px-6 py-24 sm:py-32 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
              Ready to dominate your league?
            </h2>
            <p className="mx-auto mt-6 max-w-xl text-lg leading-8 text-gray-600">
              Join thousands of fantasy players who are already winning with WinMyLeague.ai. 
              Start your free trial today.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Link
                href="/pricing"
                className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
              >
                View pricing
              </Link>
              <Link href="/accuracy" className="text-sm font-semibold leading-6 text-gray-900">
                Check our accuracy <span aria-hidden="true">→</span>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}