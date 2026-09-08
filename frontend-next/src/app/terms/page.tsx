import { Metadata } from 'next'
import {
  DocumentTextIcon,
  UserIcon,
  ChartBarIcon,
  ShieldCheckIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

export const metadata: Metadata = {
  title: 'Terms of Service | WinMyLeague.ai',
  description:
    'The terms that apply to using WinMyLeague.ai, an independent fantasy football projection tool.',
}

const sections = [
  {
    title: 'Using This Service',
    icon: DocumentTextIcon,
    content: [
      {
        subtitle: 'Agreement',
        text: 'By using WinMyLeague.ai you agree to these terms. If you do not agree with them, please do not use the service.'
      },
      {
        subtitle: 'Who can use it',
        text: 'You must be at least 13 years old to create an account. If you are under the age of majority where you live, you should have a parent or guardian review these terms with you.'
      },
      {
        subtitle: 'Changes to these terms',
        text: 'These terms may change as the service develops. Material changes will be posted on this page, and if you have an account you will be emailed about significant ones.'
      }
    ]
  },
  {
    title: 'Your Account',
    icon: UserIcon,
    content: [
      {
        subtitle: 'Creating an account',
        text: 'Accounts are created through Clerk, a third-party authentication provider. You are responsible for providing accurate information and for keeping your login credentials secure.'
      },
      {
        subtitle: 'Your responsibility',
        text: 'You are responsible for activity that happens under your account. If you believe your account has been accessed by someone else, email support@winmyleague.ai.'
      },
      {
        subtitle: 'Closing your account',
        text: 'You can have your account deleted at any time by emailing support@winmyleague.ai. Accounts may also be closed if they are used to attack or abuse the service.'
      }
    ]
  },
  {
    title: 'What the Projections Are',
    icon: ChartBarIcon,
    content: [
      {
        subtitle: 'No guarantee',
        text: 'WinMyLeague.ai produces statistical projections from machine learning models trained on historical NFL data. They are estimates, not predictions of fact, and they carry no guarantee of accuracy. Football outcomes depend on injuries, coaching decisions, weather, and chance, none of which any model captures reliably.'
      },
      {
        subtitle: 'For your own decisions',
        text: 'Projections are information to weigh, not advice to follow. You are responsible for your own lineup, draft, trade, and any other decisions you make. Do not treat this service as financial advice or use it as the basis for wagering.'
      },
      {
        subtitle: 'Availability',
        text: 'This is an independent project in beta. Features may change or be removed, and the service may be unavailable at times. It is provided as-is, without warranties of any kind.'
      }
    ]
  },
  {
    title: 'Acceptable Use',
    icon: ShieldCheckIcon,
    content: [
      {
        subtitle: 'What not to do',
        text: 'Do not attempt to break, overload, or gain unauthorized access to the service or its API. Do not scrape it at a volume that degrades it for other people, and do not use it to build a competing product by bulk-extracting its output.'
      },
      {
        subtitle: 'Content you submit',
        text: 'Anything you send through the contact form or support email may be used to answer you and to fix the problem you reported.'
      },
      {
        subtitle: 'Limitation of liability',
        text: 'To the extent permitted by law, WinMyLeague.ai and its operator are not liable for losses arising from your use of the service, including decisions made in reliance on its projections.'
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
              <h1 className="text-4xl font-bold text-gray-900">Terms of Service</h1>
              <p className="mt-6 text-lg text-gray-600">
                WinMyLeague.ai is an independent fantasy football projection tool, currently free
                and in beta. These are the terms that apply to using it.
              </p>
            </div>
          </div>

          {/* Important Notice */}
          <div className="mb-12 bg-amber-50 border border-amber-200 rounded-lg p-6">
            <div className="flex items-start gap-3">
              <ExclamationTriangleIcon className="h-6 w-6 text-amber-600 mt-1 flex-shrink-0" />
              <div>
                <h2 className="text-lg font-semibold text-amber-900 mb-2">The important part</h2>
                <p className="text-amber-800">
                  The projections are model estimates, not guarantees. They are one input into your
                  decisions, not a substitute for them. The service is free, in beta, and provided
                  as-is.
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
                  <h2 className="text-2xl font-bold text-gray-900">{section.title}</h2>
                </div>

                <div className="space-y-6">
                  {section.content.map((item, itemIndex) => (
                    <div
                      key={itemIndex}
                      className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
                    >
                      <h3 className="text-lg font-semibold text-gray-900 mb-3">
                        {item.subtitle}
                      </h3>
                      <p className="text-gray-600 leading-relaxed">{item.text}</p>
                    </div>
                  ))}
                </div>
              </section>
            ))}
          </div>

          {/* Contact */}
          <section className="mt-12 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-8 text-center">
            <h2 className="text-2xl font-bold text-white mb-4">Questions About These Terms?</h2>
            <p className="text-indigo-100 mb-6">
              Send a message and it comes straight to me.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="/contact"
                className="px-6 py-3 bg-white text-indigo-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                Contact
              </a>
              <a
                href="mailto:support@winmyleague.ai"
                className="px-6 py-3 bg-indigo-700 text-white rounded-lg font-semibold hover:bg-indigo-800 transition-colors"
              >
                support@winmyleague.ai
              </a>
            </div>
          </section>
        </div>
      </main>

      <Footer />
    </div>
  )
}
