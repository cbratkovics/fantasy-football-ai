import { Metadata } from 'next'
import Link from 'next/link'
import {
  ChartBarIcon,
  LightBulbIcon,
  CpuChipIcon,
  RocketLaunchIcon
} from '@heroicons/react/24/outline'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

export const metadata: Metadata = {
  title: 'About | WinMyLeague.ai',
  description:
    'WinMyLeague.ai is an independent fantasy football projection tool built by Christopher Bratkovics.',
}

const values = [
  {
    title: 'Data-Driven',
    description:
      'Projections come from models trained on historical NFL data, not opinion or hand-tuned rankings.',
    icon: ChartBarIcon,
    color: 'blue'
  },
  {
    title: 'Transparent',
    description:
      'Every projection ships with a confidence interval and the factors that drove it, so you can judge it yourself.',
    icon: LightBulbIcon,
    color: 'yellow'
  },
  {
    title: 'Built in the Open',
    description:
      'The modeling code and API are public on GitHub. You can read exactly how the projections are produced.',
    icon: CpuChipIcon,
    color: 'green'
  },
  {
    title: 'Still Improving',
    description:
      'This is an active project. Models, features, and tooling change as the approach is refined.',
    icon: RocketLaunchIcon,
    color: 'purple'
  }
]

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />

      <main className="pt-24 pb-16">
        <div className="mx-auto max-w-4xl px-6 lg:px-8">
          <div className="mb-8">
            <Breadcrumb items={[{ name: 'About', current: true }]} />
          </div>

          {/* Header */}
          <section className="mb-16 text-center">
            <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl">
              About WinMyLeague.ai
            </h1>
            <p className="mt-6 text-xl leading-8 text-gray-600">
              An independent fantasy football projection tool, built and run by one person.
            </p>
          </section>

          {/* Mission */}
          <section className="mb-16">
            <div className="bg-gradient-to-r from-green-600 to-blue-600 rounded-2xl p-8 text-white">
              <h2 className="text-3xl font-bold mb-4">What This Is</h2>
              <p className="text-lg leading-8 text-green-50">
                Most fantasy advice is someone&apos;s opinion presented as a ranking. WinMyLeague.ai
                takes a different approach: ensemble machine learning models trained on historical
                NFL data produce weekly point projections, each with a confidence interval and an
                explanation of the factors behind it. You get the reasoning, not just a number, so
                you can decide how much weight to give it.
              </p>
            </div>
          </section>

          {/* Founder */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">Who Builds It</h2>
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
              <h3 className="text-2xl font-semibold text-gray-900">Christopher Bratkovics</h3>
              <p className="text-sm font-medium text-indigo-600 mt-1">Solo founder and builder</p>

              <div className="mt-6 space-y-4 text-gray-600 leading-relaxed">
                <p>
                  I build and maintain all of WinMyLeague.ai: the forecasting models, the API that
                  serves them, and this site.
                </p>
                <p>
                  I hold an M.S. in Applied Data Science and a B.S. in Computer Science, and I have
                  spent seven years working in enterprise analytics and data engineering. This
                  project started as a way to apply that work to a problem I actually care about.
                </p>
                <p>
                  The modeling work is public. If you want to see how the projections are produced
                  rather than take my word for it, the repository is the best place to start.
                </p>
              </div>

              <div className="mt-8 flex flex-col sm:flex-row gap-4">
                <a
                  href="https://github.com/cbratkovics"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-6 py-2.5 bg-gray-900 text-white rounded-lg font-semibold text-center hover:bg-gray-800 transition-colors"
                >
                  GitHub
                </a>
                <a
                  href="https://linkedin.com/in/cbratkovics"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-6 py-2.5 border border-gray-300 text-gray-900 rounded-lg font-semibold text-center hover:bg-gray-50 transition-colors"
                >
                  LinkedIn
                </a>
              </div>
            </div>
          </section>

          {/* Values */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">How It Is Built</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {values.map((value) => (
                <div
                  key={value.title}
                  className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
                >
                  <div
                    className={`inline-flex p-3 rounded-lg mb-4 ${
                      value.color === 'blue'
                        ? 'bg-blue-50 text-blue-600'
                        : value.color === 'yellow'
                        ? 'bg-yellow-50 text-yellow-600'
                        : value.color === 'green'
                        ? 'bg-green-50 text-green-600'
                        : 'bg-purple-50 text-purple-600'
                    }`}
                  >
                    <value.icon className="h-6 w-6" />
                  </div>
                  <h3 className="font-semibold text-gray-900 mb-2">{value.title}</h3>
                  <p className="text-sm text-gray-600 leading-relaxed">{value.description}</p>
                </div>
              ))}
            </div>
          </section>

          {/* CTA */}
          <section className="text-center bg-white rounded-2xl border border-gray-200 p-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Try It Out</h2>
            <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto">
              WinMyLeague.ai is free while in beta. Create an account to get weekly projections,
              player tiers, and start/sit comparisons.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/auth/signup"
                className="px-8 py-3 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-500 transition-colors"
              >
                Create an account
              </Link>
              <Link
                href="/features"
                className="px-8 py-3 border border-gray-300 text-gray-900 rounded-lg font-semibold hover:bg-gray-50 transition-colors"
              >
                Explore features
              </Link>
            </div>
          </section>
        </div>
      </main>

      <Footer />
    </div>
  )
}
