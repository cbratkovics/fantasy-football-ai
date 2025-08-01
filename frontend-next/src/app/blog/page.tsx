'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  CalendarIcon,
  UserIcon,
  ClockIcon,
  TagIcon,
  MagnifyingGlassIcon,
  ArrowTrendingUpIcon,
  ChartBarIcon,
  LightBulbIcon,
  NewspaperIcon
} from '@heroicons/react/24/outline'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

interface BlogPost {
  id: string
  title: string
  excerpt: string
  content: string
  author: string
  publishedAt: string
  readTime: string
  category: string
  tags: string[]
  featured: boolean
  slug: string
}

const categories = [
  { name: 'All', count: 24 },
  { name: 'Strategy', count: 8 },
  { name: 'AI Insights', count: 6 },
  { name: 'Player Analysis', count: 5 },
  { name: 'Industry News', count: 3 },
  { name: 'Tutorial', count: 2 }
]

const blogPosts: BlogPost[] = [
  {
    id: '1',
    title: 'How AI Achieved 93.1% Accuracy in Fantasy Football Predictions',
    excerpt: 'A deep dive into the machine learning techniques and data sources that power our industry-leading prediction accuracy.',
    content: '',
    author: 'Sarah Martinez',
    publishedAt: '2024-01-15',
    readTime: '8 min read',
    category: 'AI Insights',
    tags: ['Machine Learning', 'Accuracy', 'Data Science'],
    featured: true,
    slug: 'ai-93-percent-accuracy-fantasy-predictions'
  },
  {
    id: '2',
    title: 'Week 18 Championship Strategy: Leveraging Player Tiers for Victory',
    excerpt: 'Championship week requires a different approach. Learn how to use our tier system to identify the perfect lineup combinations.',
    content: '',
    author: 'Mike Thompson',
    publishedAt: '2024-01-12',
    readTime: '6 min read',
    category: 'Strategy',
    tags: ['Championship', 'Tiers', 'Strategy'],
    featured: true,
    slug: 'week-18-championship-strategy-player-tiers'
  },
  {
    id: '3',
    title: 'The Impact of Weather on Fantasy Football Performance',
    excerpt: 'Our AI models factor in weather conditions to improve predictions. Here\'s how different weather patterns affect player performance.',
    content: '',
    author: 'Alex Chen',
    publishedAt: '2024-01-10',
    readTime: '5 min read',
    category: 'Player Analysis',
    tags: ['Weather', 'Analysis', 'Performance'],
    featured: false,
    slug: 'weather-impact-fantasy-football-performance'
  },
  {
    id: '4',
    title: 'Building Your First AI-Powered Draft Strategy',
    excerpt: 'New to WinMyLeague.ai? This step-by-step guide will help you leverage our tools for your best draft ever.',
    content: '',
    author: 'Jessica Park',
    publishedAt: '2024-01-08',
    readTime: '10 min read',
    category: 'Tutorial',
    tags: ['Draft', 'Beginners', 'Tutorial'],
    featured: false,
    slug: 'building-first-ai-powered-draft-strategy'
  },
  {
    id: '5',
    title: 'Why Traditional Fantasy Advice Falls Short',
    excerpt: 'Exploring the limitations of traditional fantasy football analysis and how AI-powered insights provide a competitive edge.',
    content: '',
    author: 'Mike Thompson',
    publishedAt: '2024-01-05',
    readTime: '7 min read',
    category: 'Industry News',
    tags: ['Industry', 'Analysis', 'AI'],
    featured: false,
    slug: 'why-traditional-fantasy-advice-falls-short'
  },
  {
    id: '6',
    title: 'Understanding Confidence Scores: When to Trust the AI',
    excerpt: 'Learn how to interpret and act on our prediction confidence scores to make better lineup decisions.',
    content: '',
    author: 'Sarah Martinez',
    publishedAt: '2024-01-03',
    readTime: '4 min read',
    category: 'Tutorial',
    tags: ['Confidence', 'Predictions', 'Tutorial'],
    featured: false,
    slug: 'understanding-confidence-scores-trust-ai'
  }
]

const getCategoryIcon = (category: string) => {
  switch (category) {
    case 'Strategy':
      return ArrowTrendingUpIcon
    case 'AI Insights':
      return ChartBarIcon
    case 'Player Analysis':
      return UserIcon
    case 'Tutorial':
      return LightBulbIcon
    case 'Industry News':
      return NewspaperIcon
    default:
      return NewspaperIcon
  }
}

const getCategoryColor = (category: string) => {
  switch (category) {
    case 'Strategy':
      return 'bg-blue-100 text-blue-800'
    case 'AI Insights':
      return 'bg-purple-100 text-purple-800'
    case 'Player Analysis':
      return 'bg-green-100 text-green-800'
    case 'Tutorial':
      return 'bg-yellow-100 text-yellow-800'
    case 'Industry News':
      return 'bg-red-100 text-red-800'
    default:
      return 'bg-gray-100 text-gray-800'
  }
}

export default function BlogPage() {
  const [selectedCategory, setSelectedCategory] = useState('All')
  const [searchQuery, setSearchQuery] = useState('')

  const filteredPosts = blogPosts.filter(post => {
    const matchesCategory = selectedCategory === 'All' || post.category === selectedCategory
    const matchesSearch = searchQuery === '' || 
      post.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      post.excerpt.toLowerCase().includes(searchQuery.toLowerCase()) ||
      post.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
    
    return matchesCategory && matchesSearch
  })

  const featuredPosts = filteredPosts.filter(post => post.featured)
  const regularPosts = filteredPosts.filter(post => !post.featured)

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'long',
      day: 'numeric',
      year: 'numeric'
    })
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <main className="pt-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="mb-12">
            <Breadcrumb items={[{ name: 'Blog', current: true }]} />
            
            <div className="mt-4">
              <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl">
                Fantasy Football Insights
              </h1>
              <p className="mt-6 text-xl text-gray-600 max-w-3xl">
                Expert analysis, AI insights, and winning strategies from the WinMyLeague.ai team. 
                Stay ahead of the competition with our latest research and findings.
              </p>
            </div>
          </div>

          {/* Search and Filters */}
          <div className="mb-12">
            {/* Search Bar */}
            <div className="mb-8">
              <div className="relative max-w-md">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search articles..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full rounded-lg border-gray-300 pl-10 pr-4 py-2 focus:border-indigo-500 focus:ring-indigo-500"
                />
              </div>
            </div>

            {/* Categories */}
            <div className="flex flex-wrap gap-2">
              {categories.map((category) => (
                <button
                  key={category.name}
                  onClick={() => setSelectedCategory(category.name)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    selectedCategory === category.name
                      ? 'bg-indigo-600 text-white'
                      : 'bg-white text-gray-700 hover:bg-gray-100 border border-gray-200'
                  }`}
                >
                  {category.name} ({category.count})
                </button>
              ))}
            </div>
          </div>

          {/* Featured Posts */}
          {featuredPosts.length > 0 && (
            <section className="mb-16">
              <h2 className="text-2xl font-bold text-gray-900 mb-8">Featured Articles</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {featuredPosts.map((post) => {
                  const CategoryIcon = getCategoryIcon(post.category)
                  return (
                    <Link
                      key={post.id}
                      href={`/blog/${post.slug}`}
                      className="group bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow"
                    >
                      <div className="aspect-video bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                        <CategoryIcon className="h-16 w-16 text-white" />
                      </div>
                      <div className="p-6">
                        <div className="flex items-center gap-2 mb-3">
                          <span className={`px-2 py-1 text-xs font-medium rounded-full ${getCategoryColor(post.category)}`}>
                            {post.category}
                          </span>
                          <span className="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs font-medium rounded-full">
                            Featured
                          </span>
                        </div>
                        
                        <h3 className="text-xl font-semibold text-gray-900 mb-3 group-hover:text-indigo-600 transition-colors">
                          {post.title}
                        </h3>
                        <p className="text-gray-600 mb-4 line-clamp-3">
                          {post.excerpt}
                        </p>
                        
                        <div className="flex items-center justify-between text-sm text-gray-500">
                          <div className="flex items-center gap-4">
                            <span className="flex items-center gap-1">
                              <UserIcon className="h-4 w-4" />
                              {post.author}
                            </span>
                            <span className="flex items-center gap-1">
                              <CalendarIcon className="h-4 w-4" />
                              {formatDate(post.publishedAt)}
                            </span>
                          </div>
                          <span className="flex items-center gap-1">
                            <ClockIcon className="h-4 w-4" />
                            {post.readTime}
                          </span>
                        </div>
                      </div>
                    </Link>
                  )
                })}
              </div>
            </section>
          )}

          {/* Regular Posts */}
          {regularPosts.length > 0 && (
            <section>
              <h2 className="text-2xl font-bold text-gray-900 mb-8">All Articles</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {regularPosts.map((post) => {
                  const CategoryIcon = getCategoryIcon(post.category)
                  return (
                    <Link
                      key={post.id}
                      href={`/blog/${post.slug}`}
                      className="group bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow"
                    >
                      <div className="aspect-video bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center">
                        <CategoryIcon className="h-12 w-12 text-gray-400" />
                      </div>
                      <div className="p-6">
                        <div className="flex items-center gap-2 mb-3">
                          <span className={`px-2 py-1 text-xs font-medium rounded-full ${getCategoryColor(post.category)}`}>
                            {post.category}
                          </span>
                        </div>
                        
                        <h3 className="text-lg font-semibold text-gray-900 mb-2 group-hover:text-indigo-600 transition-colors">
                          {post.title}
                        </h3>
                        <p className="text-gray-600 text-sm mb-4 line-clamp-2">
                          {post.excerpt}
                        </p>
                        
                        {/* Tags */}
                        <div className="flex flex-wrap gap-1 mb-4">
                          {post.tags.slice(0, 3).map((tag) => (
                            <span
                              key={tag}
                              className="inline-flex items-center gap-1 px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded"
                            >
                              <TagIcon className="h-3 w-3" />
                              {tag}
                            </span>
                          ))}
                        </div>
                        
                        <div className="flex items-center justify-between text-xs text-gray-500">
                          <div className="flex items-center gap-2">
                            <span>{post.author}</span>
                            <span>•</span>
                            <span>{formatDate(post.publishedAt)}</span>
                          </div>
                          <span>{post.readTime}</span>
                        </div>
                      </div>
                    </Link>
                  )
                })}
              </div>
            </section>
          )}

          {/* No Results */}
          {filteredPosts.length === 0 && (
            <div className="text-center py-16">
              <NewspaperIcon className="mx-auto h-16 w-16 text-gray-400 mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                No articles found
              </h3>
              <p className="text-gray-600 mb-6">
                Try adjusting your search or filter criteria.
              </p>
              <button
                onClick={() => {
                  setSearchQuery('')
                  setSelectedCategory('All')
                }}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
              >
                Clear Filters
              </button>
            </div>
          )}

          {/* Newsletter Signup */}
          <section className="mt-16 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-12 text-center">
            <h2 className="text-3xl font-bold text-white mb-4">
              Stay Updated
            </h2>
            <p className="text-xl text-indigo-100 mb-8 max-w-2xl mx-auto">
              Get the latest fantasy football insights, AI updates, and winning strategies 
              delivered straight to your inbox.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center max-w-md mx-auto">
              <input
                type="email"
                placeholder="Enter your email"
                className="flex-1 rounded-lg border-0 px-4 py-3 text-gray-900 placeholder:text-gray-500 focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-indigo-600"
              />
              <button className="px-6 py-3 bg-white text-indigo-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
                Subscribe
              </button>
            </div>
          </section>
        </div>
      </main>

      <Footer />
    </div>
  )
}