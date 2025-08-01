import { Metadata } from 'next'
import Link from 'next/link'
import { 
  AcademicCapIcon,
  ChartBarIcon,
  LightBulbIcon,
  TrophyIcon,
  ClockIcon,
  BookOpenIcon,
  PlayCircleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

export const metadata: Metadata = {
  title: 'Learn Fantasy Football AI | WinMyLeague.ai - Master Fantasy Football',
  description: 'Learn how to use AI-powered fantasy football tools to dominate your league. Complete guides, tutorials, and strategies.',
}

const learningPaths = [
  {
    title: 'Fantasy Football Basics',
    description: 'New to fantasy football? Start here to learn the fundamentals.',
    icon: BookOpenIcon,
    color: 'blue',
    lessons: [
      'How Fantasy Football Works',
      'Understanding Scoring Systems',
      'Draft Strategy Fundamentals',
      'Waivers and Free Agency',
      'Setting Your Lineup'
    ],
    duration: '45 min',
    difficulty: 'Beginner'
  },
  {
    title: 'AI-Powered Strategies',
    description: 'Learn how to leverage WinMyLeague.ai tools for competitive advantage.',
    icon: ChartBarIcon,
    color: 'indigo',
    lessons: [
      'Understanding Player Tiers',
      'Reading Confidence Scores',
      'Using Weekly Predictions',
      'Trade Analysis with AI',
      'Injury Impact Assessment'
    ],
    duration: '60 min',
    difficulty: 'Intermediate'
  },
  {
    title: 'Advanced Analytics',
    description: 'Master advanced concepts and statistical analysis for expert-level play.',
    icon: AcademicCapIcon,
    color: 'purple',
    lessons: [
      'Value-Based Drafting',
      'Positional Scarcity Analysis',
      'Correlation and Stacking',
      'Playoff Strategy Planning',
      'Dynasty League Management'
    ],
    duration: '90 min',
    difficulty: 'Advanced'
  }
]

const quickTips = [
  {
    title: 'Draft Strategy',
    tip: 'Focus on tiers, not rankings. Players in the same tier have similar value.',
    icon: TrophyIcon
  },
  {
    title: 'Weekly Management',
    tip: 'Check predictions Tuesday night for the most accurate projections.',
    icon: ClockIcon
  },
  {
    title: 'Trade Analysis',
    tip: 'Look for trades where both teams fill positional needs, not just overall value.',
    icon: ChartBarIcon
  },
  {
    title: 'Injury Management',
    tip: 'Handcuff your RB1 with their backup - it\'s insurance, not wasted roster space.',
    icon: LightBulbIcon
  }
]

const tutorials = [
  {
    title: 'Your First Draft with WinMyLeague.ai',
    description: 'Step-by-step walkthrough of using our AI tools during your draft.',
    duration: '15 min',
    video: true
  },
  {
    title: 'Setting Weekly Lineups',
    description: 'How to use predictions and matchup analysis for optimal lineup decisions.',
    duration: '10 min',
    video: true
  },
  {
    title: 'Understanding Player Tiers',
    description: 'Deep dive into our tier system and how to use it strategically.',
    duration: '12 min',
    video: false
  },
  {
    title: 'Trade Evaluation Masterclass',
    description: 'Learn to evaluate trades like a pro using our analysis tools.',
    duration: '20 min',
    video: true
  }
]

const faqs = [
  {
    question: 'How accurate are the AI predictions?',
    answer: 'Our AI models achieve 93.1% accuracy in weekly player projections, significantly outperforming traditional fantasy advice. We measure accuracy using a 15% margin of error on actual fantasy points scored.'
  },
  {
    question: 'What makes WinMyLeague.ai different from other sites?',
    answer: 'We use advanced machine learning models that analyze 100+ data points per player, including matchup data, weather, injury reports, and historical trends. Our transparent approach shows you exactly why we make each recommendation.'
  },
  {
    question: 'Should I always follow the AI recommendations?',
    answer: 'Our AI provides data-driven insights, but fantasy football still involves some unpredictability. Use our predictions as a strong foundation, but consider your league context, gut feelings, and any information we might not have.'
  },
  {
    question: 'How often are predictions updated?',
    answer: 'Predictions are updated hourly during the season to incorporate the latest injury reports, weather forecasts, and news. Major updates happen Tuesday evenings and Sunday mornings.'
  }
]

export default function LearnPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="pt-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="mb-12">
            <Breadcrumb items={[{ name: 'Learn', current: true }]} />
            
            <div className="mt-4 text-center">
              <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl">
                Master Fantasy Football with AI
              </h1>
              <p className="mt-6 text-xl text-gray-600 max-w-3xl mx-auto">
                Learn how to leverage artificial intelligence to dominate your fantasy league. 
                From basic strategies to advanced analytics, we'll teach you everything you need to know.
              </p>
            </div>
          </div>

          {/* Learning Paths */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-8">Learning Paths</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {learningPaths.map((path, index) => (
                <div
                  key={path.title}
                  className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow"
                >
                  <div className={`w-12 h-12 rounded-lg bg-${path.color}-100 flex items-center justify-center mb-4`}>
                    <path.icon className={`h-6 w-6 text-${path.color}-600`} />
                  </div>
                  
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    {path.title}
                  </h3>
                  <p className="text-gray-600 mb-4">
                    {path.description}
                  </p>
                  
                  <div className="flex items-center gap-4 text-sm text-gray-500 mb-4">
                    <span className="flex items-center gap-1">
                      <ClockIcon className="h-4 w-4" />
                      {path.duration}
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      path.difficulty === 'Beginner' ? 'bg-green-100 text-green-800' :
                      path.difficulty === 'Intermediate' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {path.difficulty}
                    </span>
                  </div>
                  
                  <ul className="space-y-2 mb-6">
                    {path.lessons.map((lesson, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm text-gray-600">
                        <CheckCircleIcon className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                        {lesson}
                      </li>
                    ))}
                  </ul>
                  
                  <button className={`w-full py-2 px-4 rounded-lg font-medium transition-colors bg-${path.color}-600 text-white hover:bg-${path.color}-700`}>
                    Start Learning
                  </button>
                </div>
              ))}
            </div>
          </section>

          {/* Quick Tips */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-8">Quick Tips</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {quickTips.map((tip, index) => (
                <div
                  key={tip.title}
                  className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-lg bg-indigo-100 flex items-center justify-center flex-shrink-0">
                      <tip.icon className="h-5 w-5 text-indigo-600" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-2">
                        {tip.title}
                      </h3>
                      <p className="text-gray-600">
                        {tip.tip}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Video Tutorials */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-8">Video Tutorials</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {tutorials.map((tutorial, index) => (
                <div
                  key={tutorial.title}
                  className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow"
                >
                  {/* Video Thumbnail */}
                  <div className="aspect-video bg-gray-100 flex items-center justify-center">
                    <div className="text-center">
                      <PlayCircleIcon className="h-16 w-16 text-indigo-600 mx-auto mb-2" />
                      <p className="text-sm text-gray-500">
                        {tutorial.video ? 'Video Tutorial' : 'Article'}
                      </p>
                    </div>
                  </div>
                  
                  <div className="p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      {tutorial.title}
                    </h3>
                    <p className="text-gray-600 mb-4">
                      {tutorial.description}
                    </p>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-500 flex items-center gap-1">
                        <ClockIcon className="h-4 w-4" />
                        {tutorial.duration}
                      </span>
                      <button className="text-indigo-600 hover:text-indigo-700 font-medium text-sm">
                        {tutorial.video ? 'Watch Now' : 'Read Now'} →
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* FAQ Section */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-8">Frequently Asked Questions</h2>
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 divide-y divide-gray-200">
              {faqs.map((faq, index) => (
                <div key={index} className="p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">
                    {faq.question}
                  </h3>
                  <p className="text-gray-600 leading-relaxed">
                    {faq.answer}
                  </p>
                </div>
              ))}
            </div>
          </section>

          {/* CTA Section */}
          <section className="text-center bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-12">
            <h2 className="text-3xl font-bold text-white mb-4">
              Ready to Start Winning?
            </h2>
            <p className="text-xl text-indigo-100 mb-8 max-w-2xl mx-auto">
              Put your new knowledge to work. Start using WinMyLeague.ai's AI-powered tools 
              to dominate your fantasy football league.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/auth/signup"
                className="px-8 py-3 bg-white text-indigo-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                Start Free Trial
              </Link>
              <Link
                href="/features"
                className="px-8 py-3 bg-indigo-700 text-white rounded-lg font-semibold hover:bg-indigo-800 transition-colors"
              >
                Explore Features
              </Link>
            </div>
          </section>
        </div>
      </main>

      <Footer />
    </div>
  )
}