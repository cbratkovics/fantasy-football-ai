import { Metadata } from 'next'
import Link from 'next/link'
import { 
  QuestionMarkCircleIcon,
  ChatBubbleLeftRightIcon,
  BookOpenIcon,
  CogIcon,
  CreditCardIcon,
  UserGroupIcon,
  ChevronDownIcon,
  MagnifyingGlassIcon
} from '@heroicons/react/24/outline'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

export const metadata: Metadata = {
  title: 'Help Center | WinMyLeague.ai - Fantasy Football Support',
  description: 'Get help with WinMyLeague.ai fantasy football tools. FAQs, guides, and support for all your questions.',
}

const categories = [
  {
    title: 'Getting Started',
    description: 'New to WinMyLeague.ai? Learn the basics',
    icon: BookOpenIcon,
    color: 'blue',
    articles: [
      'Setting up your first team',
      'Understanding player tiers',
      'How to read predictions',
      'Using the lineup optimizer'
    ]
  },
  {
    title: 'Account & Billing',
    description: 'Manage your account and subscription',
    icon: CreditCardIcon,
    color: 'green',
    articles: [
      'Changing your plan',
      'Updating payment method',
      'Canceling subscription',
      'Billing and invoices'
    ]
  },
  {
    title: 'Features & Tools',
    description: 'Make the most of our AI tools',
    icon: CogIcon,
    color: 'purple',
    articles: [
      'Draft strategy with AI',
      'Weekly lineup decisions',
      'Trade analysis tool',
      'Injury impact assessment'
    ]
  },
  {
    title: 'League Management',
    description: 'Managing multiple teams and leagues',
    icon: UserGroupIcon,
    color: 'orange',
    articles: [
      'Adding multiple teams',
      'Custom scoring systems',
      'League-specific settings',
      'Commissioner tools'
    ]
  }
]

const faqs = [
  {
    category: 'General',
    question: 'How accurate are WinMyLeague.ai predictions?',
    answer: 'Our AI models achieve 93.1% accuracy in weekly player projections, measured using a 15% margin of error on actual fantasy points scored. We continuously improve our models with new data and feedback.'
  },
  {
    category: 'General',
    question: 'What makes WinMyLeague.ai different from other fantasy sites?',
    answer: 'We use advanced machine learning that analyzes 100+ data points per player, including matchup data, weather, injury reports, and historical trends. Our transparent approach shows you exactly why we make each recommendation.'
  },
  {
    category: 'Features',
    question: 'How often are predictions updated?',
    answer: 'Predictions are updated hourly during the season to incorporate the latest injury reports, weather forecasts, and news. Major updates happen Tuesday evenings and Sunday mornings.'
  },
  {
    category: 'Features',
    question: 'Can I use WinMyLeague.ai for dynasty leagues?',
    answer: 'Yes! Our Pro and League plans include dynasty-specific features like long-term projections, rookie analysis, and keeper value calculations.'
  },
  {
    category: 'Account',
    question: 'Can I cancel my subscription anytime?',
    answer: 'Absolutely. You can cancel your subscription at any time from your account settings. You\'ll continue to have access through the end of your current billing period.'
  },
  {
    category: 'Account',
    question: 'Do you offer refunds?',
    answer: 'We offer a 7-day money-back guarantee for new subscribers. If you\'re not satisfied within the first week, contact support for a full refund.'
  },
  {
    category: 'Technical',
    question: 'Is there a mobile app?',
    answer: 'Our website is fully optimized for mobile devices. We\'re working on dedicated mobile apps for iOS and Android, which will be available in early 2024.'
  },
  {
    category: 'Technical',
    question: 'Can I integrate with my fantasy platform?',
    answer: 'We support direct integration with major platforms like ESPN, Yahoo, and Sleeper. More platforms are being added regularly based on user demand.'
  }
]

export default function HelpPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="pt-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="mb-12">
            <Breadcrumb items={[{ name: 'Help', current: true }]} />
            
            <div className="mt-4 text-center">
              <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl">
                How can we help you?
              </h1>
              <p className="mt-6 text-xl text-gray-600 max-w-3xl mx-auto">
                Find answers to common questions, browse our guides, or get in touch with our support team.
              </p>
            </div>

            {/* Search Bar */}
            <div className="mt-8 max-w-2xl mx-auto">
              <div className="relative">
                <MagnifyingGlassIcon className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search for help articles..."
                  className="w-full rounded-lg border-gray-300 pl-12 pr-4 py-3 text-lg focus:border-indigo-500 focus:ring-indigo-500"
                />
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <section className="mb-16">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Link
                href="/contact"
                className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow group"
              >
                <div className="w-12 h-12 rounded-lg bg-indigo-100 flex items-center justify-center mb-4 group-hover:bg-indigo-200 transition-colors">
                  <ChatBubbleLeftRightIcon className="h-6 w-6 text-indigo-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Contact Support
                </h3>
                <p className="text-gray-600">
                  Get personalized help from our support team
                </p>
              </Link>

              <Link
                href="/learn"
                className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow group"
              >
                <div className="w-12 h-12 rounded-lg bg-green-100 flex items-center justify-center mb-4 group-hover:bg-green-200 transition-colors">
                  <BookOpenIcon className="h-6 w-6 text-green-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Learning Center
                </h3>
                <p className="text-gray-600">
                  Master fantasy football with our tutorials
                </p>
              </Link>

              <Link
                href="/status"
                className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow group"
              >
                <div className="w-12 h-12 rounded-lg bg-yellow-100 flex items-center justify-center mb-4 group-hover:bg-yellow-200 transition-colors">
                  <CogIcon className="h-6 w-6 text-yellow-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  System Status
                </h3>
                <p className="text-gray-600">
                  Check if all systems are running smoothly
                </p>
              </Link>
            </div>
          </section>

          {/* Help Categories */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-8">Browse by Category</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {categories.map((category) => (
                <div
                  key={category.title}
                  className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
                >
                  <div className={`w-12 h-12 rounded-lg bg-${category.color}-100 flex items-center justify-center mb-4`}>
                    <category.icon className={`h-6 w-6 text-${category.color}-600`} />
                  </div>
                  
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    {category.title}
                  </h3>
                  <p className="text-gray-600 mb-4">
                    {category.description}
                  </p>
                  
                  <ul className="space-y-2">
                    {category.articles.map((article, idx) => (
                      <li key={idx}>
                        <Link
                          href={`/help/articles/${article.toLowerCase().replace(/\s+/g, '-')}`}
                          className="text-indigo-600 hover:text-indigo-800 text-sm flex items-center gap-1"
                        >
                          {article}
                          <ChevronDownIcon className="h-3 w-3 rotate-[-90deg]" />
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>

          {/* FAQ Section */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-8">Frequently Asked Questions</h2>
            
            {/* FAQ Categories */}
            <div className="mb-8">
              <div className="flex flex-wrap gap-2">
                {['All', 'General', 'Features', 'Account', 'Technical'].map((cat) => (
                  <button
                    key={cat}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      cat === 'All'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-white text-gray-700 hover:bg-gray-100 border border-gray-200'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-gray-200">
              {faqs.map((faq, index) => (
                <details
                  key={index}
                  className="group border-b border-gray-200 last:border-b-0"
                >
                  <summary className="flex items-center justify-between p-6 cursor-pointer hover:bg-gray-50">
                    <div>
                      <span className={`inline-block px-2 py-1 text-xs font-medium rounded-full mb-2 ${
                        faq.category === 'General' ? 'bg-blue-100 text-blue-800' :
                        faq.category === 'Features' ? 'bg-purple-100 text-purple-800' :
                        faq.category === 'Account' ? 'bg-green-100 text-green-800' :
                        'bg-orange-100 text-orange-800'
                      }`}>
                        {faq.category}
                      </span>
                      <h3 className="text-lg font-semibold text-gray-900">
                        {faq.question}
                      </h3>
                    </div>
                    <ChevronDownIcon className="h-5 w-5 text-gray-500 group-open:rotate-180 transition-transform" />
                  </summary>
                  <div className="px-6 pb-6">
                    <p className="text-gray-600 leading-relaxed">
                      {faq.answer}
                    </p>
                  </div>
                </details>
              ))}
            </div>
          </section>

          {/* Contact CTA */}
          <section className="text-center bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-12">
            <QuestionMarkCircleIcon className="h-16 w-16 text-white mx-auto mb-6" />
            <h2 className="text-3xl font-bold text-white mb-4">
              Still need help?
            </h2>
            <p className="text-xl text-indigo-100 mb-8 max-w-2xl mx-auto">
              Our support team is here to help. Get personalized assistance with your account or questions about our features.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/contact"
                className="px-8 py-3 bg-white text-indigo-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                Contact Support
              </Link>
              <Link
                href="/learn"
                className="px-8 py-3 bg-indigo-700 text-white rounded-lg font-semibold hover:bg-indigo-800 transition-colors"
              >
                Browse Tutorials
              </Link>
            </div>
          </section>
        </div>
      </main>

      <Footer />
    </div>
  )
}