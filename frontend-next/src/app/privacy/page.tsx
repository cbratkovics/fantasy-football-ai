import { Metadata } from 'next'
import { 
  ShieldCheckIcon,
  EyeIcon,
  LockClosedIcon,
  UserIcon,
  GlobeAltIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

export const metadata: Metadata = {
  title: 'Privacy Policy | WinMyLeague.ai - Your Data Privacy',
  description: 'WinMyLeague.ai privacy policy. Learn how we collect, use, and protect your personal information.',
}

const sections = [
  {
    title: 'Information We Collect',
    icon: DocumentTextIcon,
    content: [
      {
        subtitle: 'Account Information',
        text: 'When you create an account, we collect your name, email address, and password. This information is necessary to provide our services and communicate with you.'
      },
      {
        subtitle: 'Fantasy Team Data',
        text: 'We collect information about your fantasy teams, including team names, league settings, and roster configurations. This data helps us provide personalized recommendations.'
      },
      {
        subtitle: 'Usage Data',
        text: 'We automatically collect information about how you use our service, including pages viewed, features used, and time spent on the platform. This helps us improve our service.'
      },
      {
        subtitle: 'Device Information',
        text: 'We collect information about the device you use to access our service, including IP address, browser type, and operating system for security and optimization purposes.'
      }
    ]
  },
  {
    title: 'How We Use Your Information',
    icon: EyeIcon,
    content: [
      {
        subtitle: 'Service Provision',
        text: 'We use your information to provide our fantasy football AI tools, generate predictions, and deliver personalized recommendations tailored to your teams and leagues.'
      },
      {
        subtitle: 'Communication',
        text: 'We may use your email address to send you service updates, important announcements, and optional marketing communications (which you can opt out of at any time).'
      },
      {
        subtitle: 'Improvement & Analytics',
        text: 'We analyze usage patterns and feedback to improve our AI models, add new features, and enhance the overall user experience.'
      },
      {
        subtitle: 'Security & Fraud Prevention',
        text: 'We use your information to protect our service and users from fraud, abuse, and security threats.'
      }
    ]
  },
  {
    title: 'Information Sharing & Disclosure',
    icon: GlobeAltIcon,
    content: [
      {
        subtitle: 'No Sale of Personal Data',
        text: 'We do not sell, rent, or trade your personal information to third parties for marketing purposes. Your privacy is not for sale.'
      },
      {
        subtitle: 'Service Providers',
        text: 'We may share information with trusted service providers who help us operate our service, such as hosting providers, email services, and analytics tools. These providers are bound by strict confidentiality agreements.'
      },
      {
        subtitle: 'Legal Requirements',
        text: 'We may disclose information if required by law, court order, or governmental regulation, or to protect the rights, property, or safety of WinMyLeague.ai, our users, or others.'
      },
      {
        subtitle: 'Business Transfers',
        text: 'In the event of a merger, acquisition, or sale of assets, your information may be transferred as part of that transaction. We will notify you of any such change in ownership.'
      }
    ]
  },
  {
    title: 'Data Security',
    icon: LockClosedIcon,
    content: [
      {
        subtitle: 'Encryption',
        text: 'We use industry-standard encryption to protect your data both in transit and at rest. All sensitive information is encrypted using strong cryptographic methods.'
      },
      {
        subtitle: 'Access Controls',
        text: 'We implement strict access controls to ensure that only authorized personnel can access your information, and only when necessary for service provision or support.'
      },
      {
        subtitle: 'Regular Security Audits',
        text: 'We regularly conduct security audits and assessments to identify and address potential vulnerabilities in our systems and processes.'
      },
      {
        subtitle: 'Incident Response',
        text: 'We have procedures in place to quickly detect, investigate, and respond to any security incidents that may affect your data.'
      }
    ]
  },
  {
    title: 'Your Rights & Choices',
    icon: UserIcon,
    content: [
      {
        subtitle: 'Access & Portability',
        text: 'You have the right to access your personal information and receive a copy of your data in a portable format. Contact us to request your data.'
      },
      {
        subtitle: 'Correction & Updates',
        text: 'You can update your account information at any time through your account settings. If you need help correcting information, contact our support team.'
      },
      {
        subtitle: 'Deletion',
        text: 'You can request deletion of your account and personal information. Note that some information may be retained for legal or legitimate business purposes.'
      },
      {
        subtitle: 'Marketing Communications',
        text: 'You can opt out of marketing emails at any time by clicking the unsubscribe link in any email or updating your preferences in your account settings.'
      }
    ]
  },
  {
    title: 'Data Retention',
    icon: ShieldCheckIcon,
    content: [
      {
        subtitle: 'Account Data',
        text: 'We retain your account information as long as your account is active or as needed to provide services. After account deletion, we may retain some information for legal compliance.'
      },
      {
        subtitle: 'Usage Analytics',
        text: 'Aggregated and anonymized usage data may be retained indefinitely for research and service improvement purposes.'
      },
      {
        subtitle: 'Legal Requirements',
        text: 'Some data may be retained longer if required by law, regulation, or for legitimate business purposes such as fraud prevention.'
      }
    ]
  }
]

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="pt-20">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="mb-12">
            <Breadcrumb items={[{ name: 'Privacy Policy', current: true }]} />
            
            <div className="mt-4">
              <h1 className="text-4xl font-bold text-gray-900">
                Privacy Policy
              </h1>
              <p className="mt-6 text-lg text-gray-600">
                At WinMyLeague.ai, we take your privacy seriously. This policy explains how we collect, 
                use, and protect your personal information when you use our fantasy football AI platform.
              </p>
              <div className="mt-4 text-sm text-gray-500">
                <p>Last updated: January 15, 2024</p>
                <p>Effective date: January 15, 2024</p>
              </div>
            </div>
          </div>

          {/* Quick Summary */}
          <div className="mb-12 bg-blue-50 border border-blue-200 rounded-lg p-6">
            <h2 className="text-xl font-semibold text-blue-900 mb-4">Privacy at a Glance</h2>
            <ul className="space-y-2 text-blue-800">
              <li className="flex items-start gap-2">
                <ShieldCheckIcon className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <span>We never sell your personal information to third parties</span>
              </li>
              <li className="flex items-start gap-2">
                <LockClosedIcon className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <span>Your data is encrypted and securely stored</span>
              </li>
              <li className="flex items-start gap-2">
                <UserIcon className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <span>You have full control over your data and can delete it anytime</span>
              </li>
              <li className="flex items-start gap-2">
                <EyeIcon className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                <span>We only collect data necessary to provide our AI-powered services</span>
              </li>
            </ul>
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

          {/* Cookies Notice */}
          <section className="mt-12">
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
              <h2 className="text-xl font-semibold text-yellow-900 mb-4">Cookies & Tracking</h2>
              <div className="text-yellow-800 space-y-3">
                <p>
                  We use cookies and similar technologies to improve your experience on our platform. 
                  These include:
                </p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li><strong>Essential cookies:</strong> Required for the website to function properly</li>
                  <li><strong>Analytics cookies:</strong> Help us understand how you use our service</li>
                  <li><strong>Preference cookies:</strong> Remember your settings and preferences</li>
                  <li><strong>Marketing cookies:</strong> Used to show relevant advertisements (optional)</li>
                </ul>
                <p>
                  You can control cookie preferences through your browser settings or our cookie preference center.
                </p>
              </div>
            </div>
          </section>

          {/* International Users */}
          <section className="mt-12">
            <div className="bg-green-50 border border-green-200 rounded-lg p-6">
              <h2 className="text-xl font-semibold text-green-900 mb-4">International Users</h2>
              <div className="text-green-800 space-y-3">
                <p>
                  WinMyLeague.ai is based in the United States. If you are accessing our service from outside 
                  the US, please be aware that your information may be transferred to, stored, and processed 
                  in the United States.
                </p>
                <p>
                  We comply with applicable data protection laws, including GDPR for European users and 
                  CCPA for California residents. If you have specific rights under these laws, please 
                  contact us to exercise them.
                </p>
              </div>
            </div>
          </section>

          {/* Children's Privacy */}
          <section className="mt-12">
            <div className="bg-red-50 border border-red-200 rounded-lg p-6">
              <h2 className="text-xl font-semibold text-red-900 mb-4">Children's Privacy</h2>
              <p className="text-red-800">
                Our service is not intended for children under 13 years of age. We do not knowingly 
                collect personal information from children under 13. If you are a parent or guardian 
                and believe your child has provided us with personal information, please contact us 
                immediately so we can delete such information.
              </p>
            </div>
          </section>

          {/* Changes to Policy */}
          <section className="mt-12">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Changes to This Policy</h2>
              <p className="text-gray-600 mb-4">
                We may update this privacy policy from time to time to reflect changes in our practices 
                or for other operational, legal, or regulatory reasons. When we make changes, we will:
              </p>
              <ul className="list-disc list-inside space-y-1 text-gray-600 ml-4 mb-4">
                <li>Update the "Last updated" date at the top of this policy</li>
                <li>Notify you via email if the changes are significant</li>
                <li>Post a notice on our website highlighting the changes</li>
              </ul>
              <p className="text-gray-600">
                We encourage you to review this policy periodically to stay informed about how we 
                protect your information.
              </p>
            </div>
          </section>

          {/* Contact Information */}
          <section className="mt-12 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-8 text-center">
            <h2 className="text-2xl font-bold text-white mb-4">
              Questions About This Policy?
            </h2>
            <p className="text-indigo-100 mb-6">
              If you have any questions about this privacy policy or our data practices, 
              we're here to help.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="/contact"
                className="px-6 py-3 bg-white text-indigo-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                Contact Support
              </a>
              <a
                href="mailto:privacy@winmyleague.ai"
                className="px-6 py-3 bg-indigo-700 text-white rounded-lg font-semibold hover:bg-indigo-800 transition-colors"
              >
                Email Privacy Team
              </a>
            </div>
            <div className="mt-6 text-sm text-indigo-200">
              <p>WinMyLeague.ai Privacy Team</p>
              <p>123 Fantasy Drive, Suite 456</p>
              <p>San Francisco, CA 94105</p>
              <p>privacy@winmyleague.ai</p>
            </div>
          </section>
        </div>
      </main>

      <Footer />
    </div>
  )
}