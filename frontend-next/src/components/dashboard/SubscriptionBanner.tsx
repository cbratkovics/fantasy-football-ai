'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import axios from 'axios'

export function SubscriptionBanner() {
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()

  const handleSubscribe = async () => {
    setIsLoading(true)
    try {
      const response = await axios.post('/api/create-checkout-session')
      if (response.data.url) {
        window.location.href = response.data.url
      }
    } catch (error) {
      console.error('Subscription error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg p-6 text-white">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Upgrade to Premium</h3>
          <p className="mt-1 text-sm text-blue-100">
            Get unlimited predictions, advanced analytics, and priority support
          </p>
        </div>
        <button
          onClick={handleSubscribe}
          disabled={isLoading}
          className="bg-white text-blue-600 px-4 py-2 rounded-md font-medium hover:bg-blue-50 transition-colors disabled:opacity-50"
        >
          {isLoading ? 'Loading...' : 'Upgrade Now'}
        </button>
      </div>
    </div>
  )
}