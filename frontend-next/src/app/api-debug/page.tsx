'use client'

import { useState } from 'react'
import { apiClient } from '@/lib/api/client'

export default function ApiDebugPage() {
  const [results, setResults] = useState<any>({})
  const [loading, setLoading] = useState(false)

  const testEndpoint = async (endpoint: string, name: string) => {
    setLoading(true)
    try {
      console.log(`Testing ${name}:`, `${process.env.NEXT_PUBLIC_API_URL}${endpoint}`)
      const response = await apiClient.get(endpoint)
      setResults(prev => ({
        ...prev,
        [name]: {
          status: 'success',
          data: response.data,
          url: `${process.env.NEXT_PUBLIC_API_URL}${endpoint}`
        }
      }))
    } catch (error: any) {
      console.error(`Error testing ${name}:`, error)
      setResults(prev => ({
        ...prev,
        [name]: {
          status: 'error',
          error: error.message,
          details: error.response?.data || error.response?.status || 'Network error',
          url: `${process.env.NEXT_PUBLIC_API_URL}${endpoint}`
        }
      }))
    }
    setLoading(false)
  }

  const testAll = async () => {
    setResults({})
    await testEndpoint('/health', 'Health Check')
    await testEndpoint('/tiers/positions/QB', 'QB Tiers')
    await testEndpoint('/tiers/positions/QB?scoring_type=ppr', 'QB Tiers (PPR)')
    await testEndpoint('/players/rankings', 'Player Rankings')
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">API Debug Tool</h1>
        
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Environment Info</h2>
          <div className="space-y-2 text-sm">
            <p><strong>API URL:</strong> {process.env.NEXT_PUBLIC_API_URL || 'Not set'}</p>
            <p><strong>Environment:</strong> {process.env.NODE_ENV}</p>
            <p><strong>Build Time:</strong> {new Date().toISOString()}</p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <button
            onClick={testAll}
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Testing...' : 'Test All Endpoints'}
          </button>
        </div>

        <div className="space-y-4">
          {Object.entries(results).map(([name, result]: [string, any]) => (
            <div key={name} className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold mb-2">{name}</h3>
              <p className="text-sm text-gray-600 mb-2"><strong>URL:</strong> {result.url}</p>
              <div className={`p-4 rounded ${
                result.status === 'success' ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
              }`}>
                <p className={`font-semibold ${
                  result.status === 'success' ? 'text-green-800' : 'text-red-800'
                }`}>
                  Status: {result.status}
                </p>
                {result.status === 'success' ? (
                  <pre className="mt-2 text-xs overflow-auto">
                    {JSON.stringify(result.data, null, 2)}
                  </pre>
                ) : (
                  <div className="mt-2 text-red-700">
                    <p><strong>Error:</strong> {result.error}</p>
                    <p><strong>Details:</strong> {JSON.stringify(result.details, null, 2)}</p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 mt-6">
          <h3 className="text-lg font-semibold text-yellow-800 mb-2">Troubleshooting Steps</h3>
          <ol className="list-decimal list-inside space-y-1 text-sm text-yellow-700">
            <li>Verify the Railway backend is running and accessible</li>
            <li>Check that CORS is configured to allow Vercel domain</li>
            <li>Ensure the /tiers/positions/QB endpoint exists on the backend</li>
            <li>Verify environment variables are set correctly in Vercel</li>
            <li>Check Railway logs for any errors</li>
          </ol>
        </div>
      </div>
    </div>
  )
}