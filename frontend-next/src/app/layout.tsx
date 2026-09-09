import type { Metadata } from 'next'
import './globals.css'
import { ClerkProvider } from '@clerk/nextjs'
import { Providers } from '@/components/providers'
import { Toaster } from 'react-hot-toast'

export const metadata: Metadata = {
  title: 'Win My League · Applied Data Science Decision Lab',
  description: 'An end-to-end data science case study: reliable forecasts, explicit decision policies, cohort evaluation, and production-minded delivery.',
  keywords: 'data science portfolio, decision science, machine learning, analytics engineering, fantasy football',
  openGraph: {
    title: 'Win My League · Applied Data Science Decision Lab',
    description: 'From noisy signals to measurable, explainable decisions.',
    images: ['/og-image.png'],
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <ClerkProvider>
      <html lang="en" className="h-full">
        <body className="h-full bg-gray-50 font-sans">
          <Providers>
            {children}
            <Toaster position="top-right" />
          </Providers>
        </body>
      </html>
    </ClerkProvider>
  )
}
