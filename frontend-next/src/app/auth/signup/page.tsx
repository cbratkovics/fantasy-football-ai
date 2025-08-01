'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { 
  CheckIcon,
  ArrowLongRightIcon
} from '@heroicons/react/24/outline'

interface Plan {
  name: string
  price?: string
  features: string[]
}

const plans: Record<string, Plan> = {
  free: {
    name: 'Free',
    features: [
      'Basic player projections',
      'Weekly top 50 rankings',
      'Simple start/sit recommendations',
      '1 team',
    ]
  },
  pro: {
    name: 'Pro',
    price: '$19/month',
    features: [
      'Advanced AI predictions (93.1% accuracy)',
      'Complete player tier system',
      'Weekly lineup optimizer',
      'Mock draft simulator',
      'Real-time updates',
      '5 teams',
    ]
  },
  league: {
    name: 'League',
    price: '$49/month',
    features: [
      'Everything in Pro',
      'Unlimited teams',
      'Custom scoring systems',
      'API access',
      'Priority support',
    ]
  }
}

export default function SignUpPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const planParam = searchParams.get('plan') || 'free'
  
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: '',
    acceptTerms: false,
  })
  const [selectedPlan, setSelectedPlan] = useState(planParam)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [step, setStep] = useState(1)

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    // Validation
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match')
      return
    }

    if (formData.password.length < 8) {
      setError('Password must be at least 8 characters')
      return
    }

    if (!formData.acceptTerms) {
      setError('Please accept the terms and conditions')
      return
    }

    setLoading(true)

    try {
      // TODO: Implement actual registration
      console.log('Sign up:', { ...formData, plan: selectedPlan })
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // Redirect to onboarding or dashboard
      router.push('/dashboard')
    } catch (err) {
      setError('Failed to create account. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleSocialSignup = async (provider: string) => {
    console.log(`${provider} signup`)
    // TODO: Implement social signup
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex min-h-full flex-col justify-center py-12 sm:px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-md">
          <Link href="/" className="flex justify-center text-2xl font-bold text-indigo-600">
            WinMyLeague.ai
          </Link>
          <h2 className="mt-6 text-center text-2xl font-bold leading-9 tracking-tight text-gray-900">
            Create your account
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            Already have an account?{' '}
            <Link href="/auth/signin" className="font-semibold text-indigo-600 hover:text-indigo-500">
              Sign in
            </Link>
          </p>
        </div>

        <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-[600px]">
          <div className="bg-white px-6 py-12 shadow sm:rounded-lg sm:px-12">
            {/* Progress indicator */}
            <div className="mb-8">
              <div className="flex items-center justify-between">
                <div className={`flex items-center ${step >= 1 ? 'text-indigo-600' : 'text-gray-400'}`}>
                  <span className={`flex h-8 w-8 items-center justify-center rounded-full border-2 ${
                    step >= 1 ? 'border-indigo-600 bg-indigo-600 text-white' : 'border-gray-300'
                  }`}>
                    {step > 1 ? <CheckIcon className="h-5 w-5" /> : '1'}
                  </span>
                  <span className="ml-2 text-sm font-medium">Account</span>
                </div>
                <div className={`flex-1 mx-4 h-0.5 ${step >= 2 ? 'bg-indigo-600' : 'bg-gray-300'}`} />
                <div className={`flex items-center ${step >= 2 ? 'text-indigo-600' : 'text-gray-400'}`}>
                  <span className={`flex h-8 w-8 items-center justify-center rounded-full border-2 ${
                    step >= 2 ? 'border-indigo-600 bg-indigo-600 text-white' : 'border-gray-300'
                  }`}>
                    2
                  </span>
                  <span className="ml-2 text-sm font-medium">Plan</span>
                </div>
              </div>
            </div>

            {error && (
              <div className="mb-6 rounded-md bg-red-50 p-4">
                <p className="text-sm text-red-800">{error}</p>
              </div>
            )}

            {step === 1 ? (
              <form onSubmit={(e) => { e.preventDefault(); setStep(2); }} className="space-y-6">
                <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                  <div>
                    <label htmlFor="firstName" className="block text-sm font-medium leading-6 text-gray-900">
                      First name
                    </label>
                    <input
                      id="firstName"
                      name="firstName"
                      type="text"
                      required
                      value={formData.firstName}
                      onChange={handleInputChange}
                      className="mt-2 block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                    />
                  </div>

                  <div>
                    <label htmlFor="lastName" className="block text-sm font-medium leading-6 text-gray-900">
                      Last name
                    </label>
                    <input
                      id="lastName"
                      name="lastName"
                      type="text"
                      required
                      value={formData.lastName}
                      onChange={handleInputChange}
                      className="mt-2 block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                    />
                  </div>
                </div>

                <div>
                  <label htmlFor="email" className="block text-sm font-medium leading-6 text-gray-900">
                    Email address
                  </label>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    autoComplete="email"
                    required
                    value={formData.email}
                    onChange={handleInputChange}
                    className="mt-2 block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                  />
                </div>

                <div>
                  <label htmlFor="password" className="block text-sm font-medium leading-6 text-gray-900">
                    Password
                  </label>
                  <input
                    id="password"
                    name="password"
                    type="password"
                    autoComplete="new-password"
                    required
                    value={formData.password}
                    onChange={handleInputChange}
                    className="mt-2 block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                  />
                  <p className="mt-1 text-sm text-gray-500">Must be at least 8 characters</p>
                </div>

                <div>
                  <label htmlFor="confirmPassword" className="block text-sm font-medium leading-6 text-gray-900">
                    Confirm password
                  </label>
                  <input
                    id="confirmPassword"
                    name="confirmPassword"
                    type="password"
                    autoComplete="new-password"
                    required
                    value={formData.confirmPassword}
                    onChange={handleInputChange}
                    className="mt-2 block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                  />
                </div>

                <div className="flex items-center">
                  <input
                    id="acceptTerms"
                    name="acceptTerms"
                    type="checkbox"
                    checked={formData.acceptTerms}
                    onChange={handleInputChange}
                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-600"
                  />
                  <label htmlFor="acceptTerms" className="ml-3 block text-sm leading-6 text-gray-700">
                    I agree to the{' '}
                    <Link href="/terms" className="font-semibold text-indigo-600 hover:text-indigo-500">
                      terms
                    </Link>{' '}
                    and{' '}
                    <Link href="/privacy" className="font-semibold text-indigo-600 hover:text-indigo-500">
                      privacy policy
                    </Link>
                  </label>
                </div>

                <button
                  type="submit"
                  className="flex w-full justify-center items-center gap-2 rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
                >
                  Continue to plan selection
                  <ArrowLongRightIcon className="h-4 w-4" />
                </button>
              </form>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium leading-6 text-gray-900">Select your plan</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    You can change or cancel your plan anytime
                  </p>
                </div>

                <div className="space-y-4">
                  {Object.entries(plans).map(([key, plan]) => (
                    <label
                      key={key}
                      className={`relative flex cursor-pointer rounded-lg border p-4 shadow-sm focus:outline-none ${
                        selectedPlan === key
                          ? 'border-indigo-600 ring-2 ring-indigo-600'
                          : 'border-gray-300'
                      }`}
                    >
                      <input
                        type="radio"
                        name="plan"
                        value={key}
                        checked={selectedPlan === key}
                        onChange={(e) => setSelectedPlan(e.target.value)}
                        className="sr-only"
                      />
                      <div className="flex flex-1">
                        <div className="flex flex-col">
                          <span className="block text-sm font-medium text-gray-900">
                            {plan.name}
                            {plan.price && <span className="ml-2 text-gray-500">{plan.price}</span>}
                          </span>
                          <ul className="mt-2 space-y-1">
                            {plan.features.map((feature, idx) => (
                              <li key={idx} className="flex items-start">
                                <CheckIcon className="h-4 w-4 text-indigo-600 mt-0.5 mr-2 flex-shrink-0" />
                                <span className="text-sm text-gray-600">{feature}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </label>
                  ))}
                </div>

                <div className="flex gap-4">
                  <button
                    type="button"
                    onClick={() => setStep(1)}
                    className="flex-1 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50"
                  >
                    Back
                  </button>
                  <button
                    type="submit"
                    disabled={loading}
                    className="flex-1 rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? 'Creating account...' : 'Create account'}
                  </button>
                </div>
              </form>
            )}

            {step === 1 && (
              <>
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
                      type="button"
                      onClick={() => handleSocialSignup('google')}
                      className="flex w-full items-center justify-center gap-3 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50"
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
                      type="button"
                      onClick={() => handleSocialSignup('apple')}
                      className="flex w-full items-center justify-center gap-3 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50"
                    >
                      <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 2C17.523 2 22 6.477 22 12s-4.477 10-10 10S2 17.523 2 12 6.477 2 12 2zm2.823 4.014c-.296-.094-.64-.063-.957.093a1.203 1.203 0 00-.508.65c-.083.297-.042.622.124.892.182.296.472.5.793.547.296.047.622-.047.879-.25.234-.187.39-.468.421-.78a1.203 1.203 0 00-.752-1.152zm-1.448 2.479c-.514 0-1.03.156-1.448.453-.402.28-.683.718-.793 1.218-.094.5-.047 1.03.172 1.483.234.5.64.906 1.155 1.14.109.047.234.094.359.125v4.014c0 .234.187.422.421.422s.422-.188.422-.422v-4.014c.125-.031.25-.078.359-.125.515-.234.921-.64 1.155-1.14.219-.453.266-.983.172-1.483a2.082 2.082 0 00-.793-1.218 2.082 2.082 0 00-1.181-.453z" />
                      </svg>
                      <span className="text-sm font-semibold leading-6">Apple</span>
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}