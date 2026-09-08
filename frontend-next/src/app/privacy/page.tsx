import { Metadata } from 'next'
import {
  ShieldCheckIcon,
  LockClosedIcon,
  UserIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

export const metadata: Metadata = {
  title: 'Privacy Policy | WinMyLeague.ai',
  description:
    'What data WinMyLeague.ai collects, who processes it, and how to have it deleted.',
}

const sections = [
  {
    title: 'What Is Collected',
    icon: DocumentTextIcon,
    content: [
      {
        subtitle: 'Account information',
        text: 'If you create an account, your name and email address are collected so the account can exist and so you can sign back in. Authentication is handled by Clerk, a third-party provider, which stores your credentials. Your password is never stored by WinMyLeague.ai directly.'
      },
      {
        subtitle: 'Information you choose to send',
        text: 'If you use the contact form or email support, the message and your email address are retained so the conversation can be answered.'
      },
      {
        subtitle: 'What is not collected',
        text: 'No payment details are collected, because the service is free and there is no billing system. No advertising or cross-site tracking identifiers are used. Fantasy league credentials from other platforms are not requested or stored.'
      }
    ]
  },
  {
    title: 'How It Is Used',
    icon: UserIcon,
    content: [
      {
        subtitle: 'Running the service',
        text: 'Account data is used to authenticate you and to show you your own saved settings. That is its only purpose.'
      },
      {
        subtitle: 'Answering you',
        text: 'If you contact support, your message and email address are used to reply.'
      },
      {
        subtitle: 'What is not done with it',
        text: 'Your personal information is not sold, rented, or shared with advertisers or data brokers. You will not receive marketing email you did not ask for.'
      }
    ]
  },
  {
    title: 'Who Else Processes It',
    icon: ShieldCheckIcon,
    content: [
      {
        subtitle: 'Clerk',
        text: 'Clerk provides authentication and stores account credentials and email addresses on behalf of WinMyLeague.ai. Clerk maintains its own privacy policy governing that processing.'
      },
      {
        subtitle: 'Hosting providers',
        text: 'The site and its API run on third-party hosting infrastructure. As with any website, those providers process technical request data such as IP addresses in the course of serving pages.'
      },
      {
        subtitle: 'Legal requests',
        text: 'Information may be disclosed if required by valid legal process.'
      }
    ]
  },
  {
    title: 'Your Choices',
    icon: LockClosedIcon,
    content: [
      {
        subtitle: 'Deletion',
        text: 'Email support@winmyleague.ai to have your account and its associated data deleted. No reason is required, and the request will be honored.'
      },
      {
        subtitle: 'Access and correction',
        text: 'Email support@winmyleague.ai to ask what data is held about you, or to have it corrected.'
      },
      {
        subtitle: 'Retention',
        text: 'Account data is kept while the account exists. Support correspondence is kept only as long as it is useful for answering follow-up questions.'
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
              <h1 className="text-4xl font-bold text-gray-900">Privacy Policy</h1>
              <p className="mt-6 text-lg text-gray-600">
                WinMyLeague.ai is an independent project run by one person. This policy describes
                what data the service handles and what is done with it, in plain terms.
              </p>
            </div>
          </div>

          {/* Summary */}
          <div className="mb-12 bg-indigo-50 border border-indigo-200 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-indigo-900 mb-2">The short version</h2>
            <p className="text-indigo-800">
              An email address and name if you sign up, and whatever you send in a support
              message. Authentication is handled by Clerk. Nothing is sold or shared with
              advertisers. No payment details are collected. Email support@winmyleague.ai and
              your data will be deleted.
            </p>
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

          {/* Changes */}
          <section className="mt-12 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Changes to This Policy</h2>
            <p className="text-gray-600 leading-relaxed">
              If this policy changes in a way that affects how your data is handled, the change
              will be posted here. If the change is significant and you have an account, you will
              be emailed about it.
            </p>
          </section>

          {/* Contact */}
          <section className="mt-12 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-8 text-center">
            <h2 className="text-2xl font-bold text-white mb-4">Questions About This Policy?</h2>
            <p className="text-indigo-100 mb-6">
              Privacy questions and deletion requests go to the same place.
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
