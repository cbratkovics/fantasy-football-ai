'use client'

import { useState } from 'react'
import { 
  EnvelopeIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

const contactMethods = [
  {
    title: 'Email',
    description: 'The best way to reach me. I read everything, and I reply as soon as I can.',
    icon: EnvelopeIcon,
    contact: 'support@winmyleague.ai',
    color: 'blue'
  }
]

export default function ContactPage() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    category: 'general',
    message: '',
    urgency: 'normal'
  })
  const [submitted, setSubmitted] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)

    try {
      // TODO: Implement actual form submission
      console.log('Contact form submission:', formData)
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      setSubmitted(true)
    } catch (error) {
      console.error('Error submitting form:', error)
    } finally {
      setSubmitting(false)
    }
  }

  if (submitted) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        
        <main className="pt-20">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-16">
            <div className="text-center">
              <CheckCircleIcon className="mx-auto h-16 w-16 text-green-500" />
              <h1 className="mt-4 text-3xl font-bold text-gray-900">
                Message Sent Successfully!
              </h1>
              <p className="mt-2 text-lg text-gray-600 max-w-2xl mx-auto">
                Thank you for contacting us. We've received your message and will get back to you within 24 hours.
              </p>
              <div className="mt-8">
                <button
                  onClick={() => setSubmitted(false)}
                  className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 transition-colors"
                >
                  Send Another Message
                </button>
              </div>
            </div>
          </div>
        </main>

        <Footer />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="pt-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="mb-12">
            <Breadcrumb items={[{ name: 'Contact', current: true }]} />
            
            <div className="mt-4 text-center">
              <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl">
                Get in Touch
              </h1>
              <p className="mt-6 text-xl text-gray-600 max-w-3xl mx-auto">
                Questions about WinMyLeague.ai, or something not working right? Send a message and it comes straight to me.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
            {/* Contact Methods */}
            <div className="lg:col-span-1">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">How to Reach Me</h2>
              
              <div className="space-y-6 mb-8">
                {contactMethods.map((method) => (
                  <div
                    key={method.title}
                    className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
                  >
                    <div className={`w-10 h-10 rounded-lg bg-${method.color}-100 flex items-center justify-center mb-4`}>
                      <method.icon className={`h-5 w-5 text-${method.color}-600`} />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      {method.title}
                    </h3>
                    <p className="text-gray-600 mb-3">
                      {method.description}
                    </p>
                    <p className="text-sm font-medium text-gray-900 mb-4">
                      {method.contact}
                    </p>
                  </div>
                ))}
              </div>

            </div>

            {/* Contact Form */}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">Send a message</h2>
                
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                    <div>
                      <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
                        Full Name *
                      </label>
                      <input
                        type="text"
                        id="name"
                        name="name"
                        required
                        value={formData.name}
                        onChange={handleInputChange}
                        className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                      />
                    </div>

                    <div>
                      <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
                        Email Address *
                      </label>
                      <input
                        type="email"
                        id="email"
                        name="email"
                        required
                        value={formData.email}
                        onChange={handleInputChange}
                        className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                    <div>
                      <label htmlFor="category" className="block text-sm font-medium text-gray-700 mb-2">
                        Category
                      </label>
                      <select
                        id="category"
                        name="category"
                        value={formData.category}
                        onChange={handleInputChange}
                        className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                      >
                        <option value="general">General Question</option>
                        <option value="technical">Technical Support</option>
                        <option value="billing">Billing & Account</option>
                        <option value="feature">Feature Request</option>
                        <option value="bug">Bug Report</option>
                        <option value="partnership">Partnership</option>
                      </select>
                    </div>

                    <div>
                      <label htmlFor="urgency" className="block text-sm font-medium text-gray-700 mb-2">
                        Priority
                      </label>
                      <select
                        id="urgency"
                        name="urgency"
                        value={formData.urgency}
                        onChange={handleInputChange}
                        className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                      >
                        <option value="low">Low</option>
                        <option value="normal">Normal</option>
                        <option value="high">High</option>
                        <option value="urgent">Urgent</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-2">
                      Subject *
                    </label>
                    <input
                      type="text"
                      id="subject"
                      name="subject"
                      required
                      value={formData.subject}
                      onChange={handleInputChange}
                      className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                      placeholder="Brief description of your inquiry"
                    />
                  </div>

                  <div>
                    <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-2">
                      Message *
                    </label>
                    <textarea
                      id="message"
                      name="message"
                      rows={6}
                      required
                      value={formData.message}
                      onChange={handleInputChange}
                      className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                      placeholder="Please provide details about your question or issue..."
                    />
                  </div>

                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <p className="text-sm text-blue-800">
                      <strong>Before submitting:</strong> Check our <a href="/help" className="underline hover:text-blue-900">Help Center</a> for quick answers to common questions.
                    </p>
                  </div>

                  <button
                    type="submit"
                    disabled={submitting}
                    className="w-full py-3 px-6 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {submitting ? 'Sending...' : 'Send Message'}
                  </button>
                </form>
              </div>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}