import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { ClerkProvider } from '@clerk/nextjs'
import { Providers } from '@/components/providers'
import { Toaster } from 'react-hot-toast'
import { METRICS } from '@/lib/constants'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: `Fantasy Football AI - ${METRICS.accuracy.percentage} Accurate Predictions`,
  description: 'AI-powered fantasy football predictions with transparent explanations. Get data-driven insights for your lineup decisions.',
  keywords: 'fantasy football, AI predictions, NFL, lineup optimizer, player rankings',
  openGraph: {
    title: `Fantasy Football AI - ${METRICS.accuracy.percentage} Accurate Predictions`,
    description: 'Make winning lineup decisions with AI-powered predictions',
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
        <body className={`${inter.className} h-full bg-gray-50`}>
          <Providers>
            {children}
            <Toaster position="top-right" />
          </Providers>
        </body>
      </html>
    </ClerkProvider>
  )
}