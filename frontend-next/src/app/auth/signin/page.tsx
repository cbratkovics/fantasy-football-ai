'use client'

import { useState } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { useRouter } from 'next/navigation'
import { 
  TrophyIcon, 
  ChartBarIcon, 
  BoltIcon,
  CheckCircleIcon 
} from '@heroicons/react/24/outline'

const benefits = [
  {
    icon: BoltIcon,
    text: '93.1% accurate AI predictions'
  },
  {
    icon: ChartBarIcon,
    text: 'Smart player tiers & rankings'
  },
  {
    icon: TrophyIcon,
    text: 'Win your fantasy league'
  }
]

export default function SignInPage() {
  const router = useRouter()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      // TODO: Implement actual authentication
      console.log('Sign in:', { email, password })
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // Redirect to dashboard
      router.push('/dashboard')
    } catch (err) {
      setError('Invalid email or password')
    } finally {
      setLoading(false)
    }
  }

  const handleSocialLogin = async (provider: string) => {
    console.log(`${provider} login`)
    // TODO: Implement social login
  }

  return (
    <div className="flex min-h-full">
      {/* Left side - Sign in form */}
      <div className="flex flex-1 flex-col justify-center px-4 py-12 sm:px-6 lg:flex-none lg:px-20 xl:px-24">
        <div className="mx-auto w-full max-w-sm lg:w-96">
          <div>
            <Link href="/" className="text-2xl font-bold text-indigo-600">
              WinMyLeague.ai
            </Link>
            <h2 className="mt-8 text-2xl font-bold leading-9 tracking-tight text-gray-900">
              Welcome back
            </h2>
            <p className="mt-2 text-sm leading-6 text-gray-500">
              Not a member?{' '}
              <Link href="/auth/signup" className="font-semibold text-indigo-600 hover:text-indigo-500">
                Start your free trial
              </Link>
            </p>
          </div>

          <div className="mt-10">
            <form onSubmit={handleSubmit} className="space-y-6">
              {error && (
                <div className="rounded-md bg-red-50 p-4">
                  <p className="text-sm text-red-800">{error}</p>
                </div>
              )}

              <div>
                <label htmlFor="email" className="block text-sm font-medium leading-6 text-gray-900">
                  Email address
                </label>
                <div className="mt-2">
                  <input
                    id="email"
                    name="email"
                    type="email"
                    autoComplete="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="block w-full rounded-md border-0 py-1.5 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                  />
                </div>
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium leading-6 text-gray-900">
                  Password
                </label>
                <div className="mt-2">
                  <input
                    id="password"
                    name="password"
                    type="password"
                    autoComplete="current-password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="block w-full rounded-md border-0 py-1.5 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <input
                    id="remember-me"
                    name="remember-me"
                    type="checkbox"
                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-600"
                  />
                  <label htmlFor="remember-me" className="ml-3 block text-sm leading-6 text-gray-700">
                    Remember me
                  </label>
                </div>

                <div className="text-sm leading-6">
                  <a href="#" className="font-semibold text-indigo-600 hover:text-indigo-500">
                    Forgot password?
                  </a>
                </div>
              </div>

              <div>
                <button
                  type="submit"
                  disabled={loading}
                  className="flex w-full justify-center rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-semibold leading-6 text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Signing in...' : 'Sign in'}
                </button>
              </div>
            </form>

            <div className="mt-10">
              <div className="relative">
                <div className="absolute inset-0 flex items-center" aria-hidden="true">
                  <div className="w-full border-t border-gray-200" />
                </div>
                <div className="relative flex justify-center text-sm font-medium leading-6">
                  <span className="bg-white px-6 text-gray-900">Or continue with</span>
                </div>
              </div>

              <div className="mt-6 grid grid-cols-2 gap-4">
                <button
                  onClick={() => handleSocialLogin('google')}
                  className="flex w-full items-center justify-center gap-3 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                >
                  <svg className="h-5 w-5" viewBox="0 0 24 24">
                    <path
                      fill="#4285F4"
                      d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                    />
                    <path
                      fill="#34A853"
                      d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                    />
                    <path
                      fill="#FBBC05"
                      d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                    />
                    <path
                      fill="#EA4335"
                      d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                    />
                  </svg>
                  <span className="text-sm font-semibold leading-6">Google</span>
                </button>

                <button
                  onClick={() => handleSocialLogin('apple')}
                  className="flex w-full items-center justify-center gap-3 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
                >
                  <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C17.523 2 22 6.477 22 12s-4.477 10-10 10S2 17.523 2 12 6.477 2 12 2zm2.823 4.014c-.296-.094-.64-.063-.957.093a1.203 1.203 0 00-.508.65c-.083.297-.042.622.124.892.182.296.472.5.793.547.296.047.622-.047.879-.25.234-.187.39-.468.421-.78a1.203 1.203 0 00-.752-1.152zm-1.448 2.479c-.514 0-1.03.156-1.448.453-.402.28-.683.718-.793 1.218-.094.5-.047 1.03.172 1.483.234.5.64.906 1.155 1.14.109.047.234.094.359.125v4.014c0 .234.187.422.421.422s.422-.188.422-.422v-4.014c.125-.031.25-.078.359-.125.515-.234.921-.64 1.155-1.14.219-.453.266-.983.172-1.483a2.082 2.082 0 00-.793-1.218 2.082 2.082 0 00-1.181-.453z" />
                  </svg>
                  <span className="text-sm font-semibold leading-6">Apple</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right side - Benefits showcase */}
      <div className="relative hidden w-0 flex-1 lg:block">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-600 to-purple-700" />
        <div className="absolute inset-0 bg-black/20" />
        
        <div className="relative flex h-full flex-col justify-center px-12 xl:px-20">
          <div className="max-w-xl">
            <h3 className="text-4xl font-bold tracking-tight text-white">
              Your championship season starts here
            </h3>
            <p className="mt-6 text-lg leading-8 text-white/90">
              Join 2,500+ fantasy players who are already winning more games with AI-powered insights.
            </p>
            
            <div className="mt-10 space-y-6">
              {benefits.map((benefit, index) => (
                <div key={index} className="flex items-center gap-x-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-white/10 backdrop-blur">
                    <benefit.icon className="h-6 w-6 text-white" />
                  </div>
                  <p className="text-base font-medium text-white">
                    {benefit.text}
                  </p>
                </div>
              ))}
            </div>

            <div className="mt-12 rounded-lg bg-white/10 p-6 backdrop-blur">
              <blockquote className="text-white">
                <p className="text-lg font-medium">
                  "WinMyLeague.ai helped me win 3 out of 4 leagues last season. 
                  The AI predictions are scary accurate."
                </p>
                <footer className="mt-4">
                  <p className="text-sm text-white/70">
                    — Mike R., Pro subscriber
                  </p>
                </footer>
              </blockquote>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}