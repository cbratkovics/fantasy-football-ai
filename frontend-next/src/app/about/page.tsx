import { Metadata } from 'next'
import Link from 'next/link'
import Image from 'next/image'
import { 
  ChartBarIcon,
  LightBulbIcon,
  UsersIcon,
  TrophyIcon,
  HeartIcon,
  RocketLaunchIcon,
  AcademicCapIcon,
  StarIcon
} from '@heroicons/react/24/outline'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

export const metadata: Metadata = {
  title: 'About WinMyLeague.ai | Fantasy Football AI Company',
  description: 'Learn about WinMyLeague.ai - the team behind the most accurate fantasy football AI predictions and tools.',
}

const values = [
  {
    title: 'Data-Driven Excellence',
    description: 'Every recommendation is backed by rigorous analysis of 100+ data points',
    icon: ChartBarIcon,
    color: 'blue'
  },
  {
    title: 'Transparent Innovation',
    description: 'We show you exactly why we make each recommendation',
    icon: LightBulbIcon,
    color: 'yellow'
  },
  {
    title: 'Community First',
    description: 'Built by fantasy players, for fantasy players',
    icon: UsersIcon,
    color: 'green'
  },
  {
    title: 'Continuous Improvement',
    description: 'Our models learn and adapt every week to stay ahead',
    icon: RocketLaunchIcon,
    color: 'purple'
  }
]

const team = [
  {
    name: 'Alex Chen',
    role: 'CEO & Co-Founder',
    bio: 'Former Google ML engineer and 15-year fantasy football veteran. Alex combines deep technical expertise with a passion for fantasy sports.',
    image: '/images/team/alex-chen.jpg',
    credentials: 'PhD Computer Science, Stanford'
  },
  {
    name: 'Sarah Martinez',
    role: 'CTO & Co-Founder',
    bio: 'Ex-Facebook data scientist with expertise in sports analytics. Sarah leads our AI development and model optimization.',
    image: '/images/team/sarah-martinez.jpg',
    credentials: 'MS Data Science, MIT'
  },
  {
    name: 'Mike Thompson',
    role: 'Head of Fantasy Research',
    bio: 'Former NFL scout and fantasy industry veteran. Mike ensures our models understand the nuances of football.',
    image: '/images/team/mike-thompson.jpg',
    credentials: '20+ years NFL experience'
  },
  {
    name: 'Jessica Park',
    role: 'Lead Product Designer',
    bio: 'Award-winning UX designer focused on making complex data accessible and actionable for fantasy players.',
    image: '/images/team/jessica-park.jpg',
    credentials: 'BFA Design, RISD'
  }
]

const stats = [
  { label: 'Prediction Accuracy', value: '93.1%' },
  { label: 'Fantasy Players Served', value: '50K+' },
  { label: 'Data Points Analyzed', value: '100+' },
  { label: 'Years of NFL Data', value: '20+' }
]

const timeline = [
  {
    year: '2020',
    title: 'The Idea',
    description: 'Founded by two frustrated fantasy players who knew AI could revolutionize fantasy sports'
  },
  {
    year: '2021',
    title: 'First Models',
    description: 'Developed initial ML models achieving 87% accuracy, outperforming industry standards'
  },
  {
    year: '2022',
    title: 'Public Launch',
    description: 'Launched beta version to 1,000 fantasy players with overwhelming positive feedback'
  },
  {
    year: '2023',
    title: 'Breakthrough Year',
    description: 'Achieved 93.1% accuracy and grew to serve over 50,000 fantasy players'
  },
  {
    year: '2024',
    title: 'What\'s Next',
    description: 'Expanding to other sports and building the future of AI-powered fantasy tools'
  }
]

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="pt-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="mb-16">
            <Breadcrumb items={[{ name: 'About', current: true }]} />
            
            <div className="mt-4 text-center">
              <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl">
                Revolutionizing Fantasy Football with AI
              </h1>
              <p className="mt-6 text-xl text-gray-600 max-w-4xl mx-auto">
                WinMyLeague.ai was born from a simple frustration: why do fantasy "experts" get it wrong so often? 
                We knew artificial intelligence could do better, and we were right.
              </p>
            </div>
          </div>

          {/* Stats */}
          <section className="mb-16">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              {stats.map((stat) => (
                <div key={stat.label} className="text-center">
                  <div className="text-4xl font-bold text-indigo-600 mb-2">
                    {stat.value}
                  </div>
                  <div className="text-sm text-gray-600">
                    {stat.label}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Mission */}
          <section className="mb-16">
            <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-12 text-center text-white">
              <h2 className="text-3xl font-bold mb-4">Our Mission</h2>
              <p className="text-xl text-indigo-100 max-w-4xl mx-auto leading-relaxed">
                To democratize access to elite fantasy football analysis through artificial intelligence, 
                giving every fantasy player the tools they need to compete at the highest level.
              </p>
            </div>
          </section>

          {/* Values */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">What Drives Us</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {values.map((value) => (
                <div
                  key={value.title}
                  className="bg-white rounded-xl shadow-sm border border-gray-200 p-8"
                >
                  <div className={`w-12 h-12 rounded-lg bg-${value.color}-100 flex items-center justify-center mb-6`}>
                    <value.icon className={`h-6 w-6 text-${value.color}-600`} />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">
                    {value.title}
                  </h3>
                  <p className="text-gray-600 leading-relaxed">
                    {value.description}
                  </p>
                </div>
              ))}
            </div>
          </section>

          {/* Timeline */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">Our Journey</h2>
            <div className="relative">
              {/* Timeline line */}
              <div className="absolute left-4 md:left-1/2 top-0 bottom-0 w-0.5 bg-indigo-200 transform md:-translate-x-px"></div>
              
              <div className="space-y-12">
                {timeline.map((item, index) => (
                  <div key={item.year} className={`relative flex items-center ${index % 2 === 0 ? 'md:flex-row' : 'md:flex-row-reverse'}`}>
                    {/* Timeline dot */}
                    <div className="absolute left-4 md:left-1/2 w-3 h-3 bg-indigo-600 rounded-full transform -translate-x-1.5 md:-translate-x-1.5"></div>
                    
                    {/* Content */}
                    <div className={`ml-12 md:ml-0 md:w-1/2 ${index % 2 === 0 ? 'md:pr-12' : 'md:pl-12'}`}>
                      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                        <div className="text-2xl font-bold text-indigo-600 mb-2">
                          {item.year}
                        </div>
                        <h3 className="text-xl font-semibold text-gray-900 mb-3">
                          {item.title}
                        </h3>
                        <p className="text-gray-600">
                          {item.description}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Team */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold text-gray-900 text-center mb-4">Meet Our Team</h2>
            <p className="text-lg text-gray-600 text-center mb-12 max-w-3xl mx-auto">
              We're a team of AI experts, fantasy veterans, and product builders united by our passion 
              for using technology to revolutionize fantasy sports.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {team.map((member) => (
                <div
                  key={member.name}
                  className="bg-white rounded-xl shadow-sm border border-gray-200 p-8"
                >
                  <div className="flex items-start space-x-4">
                    <div className="w-20 h-20 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0">
                      <span className="text-2xl font-bold text-indigo-600">
                        {member.name.split(' ').map(n => n[0]).join('')}
                      </span>
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-gray-900 mb-1">
                        {member.name}
                      </h3>
                      <p className="text-indigo-600 font-medium mb-2">
                        {member.role}
                      </p>
                      <p className="text-sm text-gray-500 mb-3">
                        {member.credentials}
                      </p>
                      <p className="text-gray-600 leading-relaxed">
                        {member.bio}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Recognition */}
          <section className="mb-16">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
              <h2 className="text-2xl font-bold text-gray-900 text-center mb-8">Recognition & Awards</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div className="text-center">
                  <TrophyIcon className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
                  <h3 className="font-semibold text-gray-900 mb-2">Best Fantasy Tool 2023</h3>
                  <p className="text-sm text-gray-600">Fantasy Football Analytics</p>
                </div>
                <div className="text-center">
                  <StarIcon className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
                  <h3 className="font-semibold text-gray-900 mb-2">Top Prediction Accuracy</h3>
                  <p className="text-sm text-gray-600">Fantasy Sports Research</p>
                </div>
                <div className="text-center">
                  <AcademicCapIcon className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
                  <h3 className="font-semibold text-gray-900 mb-2">AI Innovation Award</h3>
                  <p className="text-sm text-gray-600">Sports Tech Conference</p>
                </div>
              </div>
            </div>
          </section>

          {/* CTA */}
          <section className="text-center bg-gradient-to-r from-green-600 to-blue-600 rounded-2xl p-12">
            <HeartIcon className="h-16 w-16 text-white mx-auto mb-6" />
            <h2 className="text-3xl font-bold text-white mb-4">
              Join the Revolution
            </h2>
            <p className="text-xl text-green-100 mb-8 max-w-2xl mx-auto">
              Ready to experience the future of fantasy football? Join thousands of players 
              who are already winning with AI-powered insights.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/auth/signup"
                className="px-8 py-3 bg-white text-green-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                Start Free Trial
              </Link>
              <Link
                href="/features"
                className="px-8 py-3 bg-green-700 text-white rounded-lg font-semibold hover:bg-green-800 transition-colors"
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