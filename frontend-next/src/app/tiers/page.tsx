import { TierVisualization } from '@/components/tiers/TierVisualization'
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
              Player Tiers
            </h1>
            <p className="mt-6 text-xl leading-8 text-gray-300">
              Our AI groups players by their true fantasy value, revealing clear tiers and value drops. 
              Know exactly when to draft each position and avoid reaching for overvalued players.
            </p>
          </div>
          <TierVisualization />
        </div>
      </main>
      <Footer />
    </div>
  )
}