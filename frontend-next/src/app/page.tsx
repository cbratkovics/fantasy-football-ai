import { Hero } from '@/components/landing/Hero'
import { Features } from '@/components/landing/Features'
import { Accuracy } from '@/components/landing/Accuracy'
import { Pricing } from '@/components/landing/Pricing'
import { HowItWorks } from '@/components/landing/HowItWorks'
import { Footer } from '@/components/layout/Footer'
import { Navigation } from '@/components/layout/Navigation'

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <Navigation />
      <main>
        <Hero />
        <Accuracy />
        <Features />
        <HowItWorks />
        <Pricing />
      </main>
      <Footer />
    </div>
  )
}