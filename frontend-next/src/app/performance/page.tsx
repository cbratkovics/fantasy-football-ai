import { PerformanceDashboard } from '@/components/dashboard/PerformanceDashboard'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'

export default function PerformancePage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      <Navigation />
      <main className="pt-24">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-4xl text-center mb-12">
            <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
              Performance Dashboard
            </h1>
            <p className="mt-6 text-xl leading-8 text-gray-300">
              Track model accuracy, analyze prediction performance, and monitor tier effectiveness 
              across all positions and time periods.
            </p>
          </div>
          <PerformanceDashboard />
        </div>
      </main>
      <Footer />
    </div>
  )
}