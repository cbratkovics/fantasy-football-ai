import { DraftSimulator } from '@/components/draft/DraftSimulator'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'

export default function DraftPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      <Navigation />
      <main className="pt-24">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-4xl text-center mb-12">
            <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
              Mock Draft Simulator
            </h1>
            <p className="mt-6 text-xl leading-8 text-gray-300">
              Practice your draft strategy with AI-powered opponents. Get tier-based recommendations 
              and see how your draft compares to optimal strategies.
            </p>
          </div>
          <DraftSimulator />
        </div>
      </main>
      <Footer />
    </div>
  )
}