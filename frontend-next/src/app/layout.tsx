import type { Metadata } from 'next'
import './globals.css'
import { Providers } from '@/components/providers'

export const metadata: Metadata = {
  title: 'Win My League · Fantasy Football Decision Lab',
  description:
    'Artifact-backed weekly fantasy football projections: as-of features, a versioned model, and an evaluation every figure traces back to.',
  keywords: 'fantasy football, player projections, start sit, decision lab',
  openGraph: {
    title: 'Win My League · Fantasy Football Decision Lab',
    description: 'Every figure is read from a committed evaluation artifact.',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full bg-gray-50 font-sans">
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
