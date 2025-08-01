import { Metadata } from 'next'
import { 
  DocumentTextIcon,
  UserIcon,
  CreditCardIcon,
  ShieldExclamationIcon,
  ScaleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

export const metadata: Metadata = {
  title: 'Terms of Service | WinMyLeague.ai - Legal Terms',
  description: 'WinMyLeague.ai terms of service. Read our legal terms and conditions for using our fantasy football AI platform.',
}

const sections = [
  {
    title: 'Acceptance of Terms',
    icon: DocumentTextIcon,
    content: [
      {
        subtitle: 'Agreement to Terms',
        text: 'By accessing and using WinMyLeague.ai, you accept and agree to be bound by the terms and provision of this agreement. If you do not agree to abide by the above, please do not use this service.'
      },
      {
        subtitle: 'Changes to Terms',
        text: 'We reserve the right to change these terms at any time. Changes will be effective immediately upon posting on the website. Your continued use of the service after changes are posted constitutes your acceptance of the modified terms.'
      },
      {
        subtitle: 'Eligibility',
        text: 'You must be at least 18 years old to use our service. By using WinMyLeague.ai, you represent and warrant that you are at least 18 years of age and have the legal capacity to enter into this agreement.'
      }
    ]
  },
  {
    title: 'Account Terms',
    icon: UserIcon,
    content: [
      {
        subtitle: 'Account Creation',
        text: 'You must provide accurate and complete information when creating an account. You are responsible for maintaining the security of your account and password and for all activities that occur under your account.'
      },
      {
        subtitle: 'Account Responsibilities',
        text: 'You are responsible for all content posted and activity that occurs under your account. You must not use your account to violate any laws, regulations, or the rights of others.'
      },
      {
        subtitle: 'Account Termination',
        text: 'We may terminate or suspend your account at any time for any reason, including violation of these terms. You may also terminate your account at any time by contacting our support team.'
      }
    ]
  },
  {
    title: 'Service Description & Availability',
    icon: CreditCardIcon,
    content: [
      {
        subtitle: 'Service Overview',
        text: 'WinMyLeague.ai provides AI-powered fantasy football analysis, predictions, and tools. We strive to provide accurate and helpful information, but we cannot guarantee the accuracy of predictions or analysis.'
      },
      {
        subtitle: 'Service Availability',
        text: 'We aim to provide continuous service availability but cannot guarantee uninterrupted access. We may suspend service for maintenance, updates, or other operational reasons with or without notice.'
      },
      {
        subtitle: 'Feature Changes',
        text: 'We reserve the right to modify, suspend, or discontinue any feature or aspect of our service at any time. We will attempt to provide reasonable notice of significant changes when possible.'
      },
      {
        subtitle: 'Third-Party Integrations',
        text: 'Our service may integrate with third-party fantasy platforms and data sources. We are not responsible for the availability, accuracy, or policies of these third-party services.'
      }
    ]
  },
  {
    title: 'Payment & Subscription Terms',
    icon: CreditCardIcon,
    content: [
      {
        subtitle: 'Subscription Plans',
        text: 'We offer various subscription plans with different features and pricing. By subscribing, you agree to pay the applicable fees and any applicable taxes.'
      },
      {
        subtitle: 'Billing & Payment',
        text: 'Subscription fees are billed in advance on a recurring basis. Payment is due immediately upon subscription or renewal. We accept major credit cards and other payment methods as available.'
      },
      {
        subtitle: 'Refunds & Cancellation',
        text: 'We offer a 7-day money-back guarantee for new subscribers. You may cancel your subscription at any time, and cancellation will take effect at the end of your current billing period.'
      },
      {
        subtitle: 'Price Changes',
        text: 'We reserve the right to change our pricing at any time. Existing subscribers will be notified of price changes at least 30 days in advance, and changes will take effect at your next billing cycle.'
      }
    ]
  },
  {
    title: 'Acceptable Use Policy',
    icon: ShieldExclamationIcon,
    content: [
      {
        subtitle: 'Prohibited Activities',
        text: 'You may not use our service for any illegal, harmful, or abusive activities. This includes but is not limited to harassment, spam, fraud, or violation of intellectual property rights.'
      },
      {
        subtitle: 'System Integrity',
        text: 'You may not attempt to interfere with, compromise, or disrupt our service, servers, or networks. This includes attempts to gain unauthorized access or to introduce viruses or malicious code.'
      },
      {
        subtitle: 'Data Usage',
        text: 'You may not scrape, harvest, or otherwise collect data from our service using automated means. You may not use our service to build competing products or services.'
      },
      {
        subtitle: 'Commercial Use',
        text: 'Our service is intended for personal use. Commercial use, resale, or redistribution of our content or services is prohibited without express written permission.'
      }
    ]
  },
  {
    title: 'Intellectual Property',
    icon: ScaleIcon,
    content: [
      {
        subtitle: 'Our Rights',
        text: 'All content, features, and functionality of WinMyLeague.ai, including but not limited to text, graphics, logos, software, and AI models, are owned by us and protected by copyright, trademark, and other intellectual property laws.'
      },
      {
        subtitle: 'Your Rights',
        text: 'Subject to these terms, we grant you a limited, non-exclusive, non-transferable license to access and use our service for your personal, non-commercial use.'
      },
      {
        subtitle: 'User Content',
        text: 'You retain ownership of any content you provide to us. By using our service, you grant us a license to use your content as necessary to provide our services.'
      },
      {
        subtitle: 'DMCA Policy',
        text: 'We respect intellectual property rights and will respond to valid DMCA takedown notices. If you believe your copyright has been infringed, please contact us with the required information.'
      }
    ]
  },
  {
    title: 'Disclaimers & Limitations',
    icon: ExclamationTriangleIcon,
    content: [
      {
        subtitle: 'No Guarantees',
        text: 'Fantasy football involves inherent uncertainty. While our AI models are highly accurate, we cannot guarantee the accuracy of predictions or the success of following our recommendations.'
      },
      {
        subtitle: 'Service "As Is"',
        text: 'Our service is provided "as is" without warranties of any kind, either express or implied. We disclaim all warranties, including but not limited to warranties of merchantability, fitness for a particular purpose, and non-infringement.'
      },
      {
        subtitle: 'Limitation of Liability',
        text: 'In no event shall WinMyLeague.ai be liable for any indirect, incidental, special, consequential, or punitive damages, including but not limited to loss of profits, data, or other intangible losses.'
      },
      {
        subtitle: 'Maximum Liability',
        text: 'Our total liability to you for any claims arising out of or relating to these terms or our service shall not exceed the amount you paid us in the twelve months preceding the claim.'
      }
    ]
  }
]

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="pt-20">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="mb-12">
            <Breadcrumb items={[{ name: 'Terms of Service', current: true }]} />
            
            <div className="mt-4">
              <h1 className="text-4xl font-bold text-gray-900">
                Terms of Service
              </h1>
              <p className="mt-6 text-lg text-gray-600">
                These terms of service govern your use of WinMyLeague.ai and our fantasy football AI platform. 
                Please read them carefully as they contain important information about your rights and obligations.
              </p>
              <div className="mt-4 text-sm text-gray-500">
                <p>Last updated: January 15, 2024</p>
                <p>Effective date: January 15, 2024</p>
              </div>
            </div>
          </div>

          {/* Important Notice */}
          <div className="mb-12 bg-amber-50 border border-amber-200 rounded-lg p-6">
            <div className="flex items-start gap-3">
              <ExclamationTriangleIcon className="h-6 w-6 text-amber-600 mt-1 flex-shrink-0" />
              <div>
                <h2 className="text-lg font-semibold text-amber-900 mb-2">Important Notice</h2>
                <p className="text-amber-800">
                  By using WinMyLeague.ai, you agree to these terms. Fantasy football predictions are for 
                  entertainment purposes and involve inherent uncertainty. Past performance does not guarantee 
                  future results. Please play responsibly and within your means.
                </p>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="space-y-12">
            {sections.map((section, index) => (
              <section key={index}>
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-lg bg-indigo-100 flex items-center justify-center">
                    <section.icon className="h-5 w-5 text-indigo-600" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900">
                    {section.title}
                  </h2>
                </div>
                
                <div className="space-y-6">
                  {section.content.map((item, itemIndex) => (
                    <div key={itemIndex} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                      <h3 className="text-lg font-semibold text-gray-900 mb-3">
                        {item.subtitle}
                      </h3>
                      <p className="text-gray-600 leading-relaxed">
                        {item.text}
                      </p>
                    </div>
                  ))}
                </div>
              </section>
            ))}
          </div>

          {/* Governing Law */}
          <section className="mt-12">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 rounded-lg bg-indigo-100 flex items-center justify-center">
                  <ScaleIcon className="h-4 w-4 text-indigo-600" />
                </div>
                <h2 className="text-xl font-semibold text-gray-900">Governing Law & Disputes</h2>
              </div>
              <div className="space-y-4 text-gray-600">
                <p>
                  These terms shall be governed by and construed in accordance with the laws of the 
                  State of California, without regard to its conflict of law provisions.
                </p>
                <p>
                  Any disputes arising out of or relating to these terms or our service shall be 
                  resolved through binding arbitration in accordance with the rules of the American 
                  Arbitration Association, rather than in court.
                </p>
                <p>
                  The arbitration will be conducted in San Francisco, California, and the arbitrator's 
                  decision will be final and binding. You agree to waive your right to a jury trial 
                  or to participate in a class action lawsuit.
                </p>
              </div>
            </div>
          </section>

          {/* Severability */}
          <section className="mt-12">
            <div className="bg-gray-100 rounded-lg p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-3">Severability</h2>
              <p className="text-gray-600">
                If any provision of these terms is found to be unenforceable or invalid, that provision 
                will be limited or eliminated to the minimum extent necessary so that these terms will 
                otherwise remain in full force and effect and enforceable.
              </p>
            </div>
          </section>

          {/* Entire Agreement */}
          <section className="mt-12">
            <div className="bg-gray-100 rounded-lg p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-3">Entire Agreement</h2>
              <p className="text-gray-600">
                These terms, together with our Privacy Policy and any other legal notices published 
                by us on our service, constitute the entire agreement between you and WinMyLeague.ai 
                concerning our service.
              </p>
            </div>
          </section>

          {/* Contact Information */}
          <section className="mt-12 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-8 text-center">
            <h2 className="text-2xl font-bold text-white mb-4">
              Questions About These Terms?
            </h2>
            <p className="text-indigo-100 mb-6">
              If you have any questions about these terms of service, please don't hesitate to contact us.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="/contact"
                className="px-6 py-3 bg-white text-indigo-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                Contact Support
              </a>
              <a
                href="mailto:legal@winmyleague.ai"
                className="px-6 py-3 bg-indigo-700 text-white rounded-lg font-semibold hover:bg-indigo-800 transition-colors"
              >
                Email Legal Team
              </a>
            </div>
            <div className="mt-6 text-sm text-indigo-200">
              <p>WinMyLeague.ai Legal Department</p>
              <p>123 Fantasy Drive, Suite 456</p>
              <p>San Francisco, CA 94105</p>
              <p>legal@winmyleague.ai</p>
            </div>
          </section>
        </div>
      </main>

      <Footer />
    </div>
  )
}