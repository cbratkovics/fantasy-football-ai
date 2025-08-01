import { TierVisualizationAPI } from '@/components/tiers/TierVisualizationAPI'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'

export default function TiersPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      <Navigation />
      <main className="pt-24">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-4xl text-center mb-12">
            <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
              AI-Powered Player Tiers
            </h1>
            <p className="mt-6 text-xl leading-8 text-gray-300">
              Real-time player tiers powered by machine learning. Our AI analyzes thousands of data points 
              to group players by their true fantasy value, revealing clear tiers and draft strategies.
            </p>
          </div>
          <TierVisualizationAPI />
        </div>
      </main>
      <Footer />
    </div>
  )
}