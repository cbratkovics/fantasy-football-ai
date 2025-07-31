'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'
import { ArrowRightIcon, ChartBarIcon, CpuChipIcon, FlagIcon } from '@heroicons/react/24/outline'
import { useState } from 'react'

export function Hero() {
  const [isHovered, setIsHovered] = useState(false)

  return (
    <section className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 pt-32 pb-20">
      {/* Background decoration */}
      <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0.6))] opacity-10" />
      
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-4xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            {/* Accuracy Badge */}
            <div className="mb-8 inline-flex items-center rounded-full bg-blue-500/20 backdrop-blur-sm border border-blue-400/30 px-4 py-2 text-sm font-semibold text-blue-300">
              <ChartBarIcon className="mr-2 h-4 w-4" />
              93% Prediction Accuracy • 2019-2024 NFL Data
            </div>

            <h1 className="text-5xl font-bold tracking-tight text-white sm:text-6xl lg:text-7xl">
              Draft Smarter with{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
                AI-Powered Tiers
              </span>
            </h1>
            
            <p className="mt-6 text-xl leading-8 text-gray-300 max-w-3xl mx-auto">
              See player tiers, not just rankings. Our unique dual-algorithm approach combines 
              GMM clustering for strategic drafting with neural networks for precise weekly predictions.
            </p>

            {/* Dual Path CTAs */}
            <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/learn"
                className="group relative inline-flex items-center justify-center px-8 py-4 text-lg font-semibold text-white bg-white/10 backdrop-blur-sm border border-white/20 rounded-lg hover:bg-white/20 transition-all duration-200"
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
              >
                <CpuChipIcon className="mr-2 h-5 w-5" />
                Learn How It Works
                <ArrowRightIcon className={`ml-2 h-5 w-5 transition-transform duration-200 ${isHovered ? 'translate-x-1' : ''}`} />
              </Link>
              
              <Link
                href="/tiers"
                className="inline-flex items-center justify-center px-8 py-4 text-lg font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-500 transition-all duration-200 shadow-lg shadow-blue-600/25"
              >
                <FlagIcon className="mr-2 h-5 w-5" />
                Start Drafting Now
              </Link>
            </div>

            {/* Value Props */}
            <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
              <div className="flex items-center justify-center gap-2 text-gray-300">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                16 Scientific Player Tiers
              </div>
              <div className="flex items-center justify-center gap-2 text-gray-300">
                <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                Weekly Point Predictions
              </div>
              <div className="flex items-center justify-center gap-2 text-gray-300">
                <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                Draft + Season-Long Value
              </div>
            </div>
          </motion.div>

          {/* Dual Algorithm Preview */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mt-16"
          >
            <div className="grid md:grid-cols-2 gap-6">
              {/* GMM Tiers Preview */}
              <div className="relative rounded-xl bg-white/10 backdrop-blur-sm border border-white/20 overflow-hidden">
                <div className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-blue-600/20 rounded-lg">
                      <ChartBarIcon className="w-5 h-5 text-blue-400" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">Draft Strategy</h3>
                      <p className="text-sm text-blue-400">GMM Clustering</p>
                    </div>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between items-center text-gray-300">
                      <span>Tier 1: Elite</span>
                      <span className="text-blue-400">3 players</span>
                    </div>
                    <div className="flex justify-between items-center text-gray-300">
                      <span>Tier 2: High-End QB1</span>
                      <span className="text-blue-400">2 players</span>
                    </div>
                    <div className="flex justify-between items-center text-gray-300">
                      <span>Tier 3: Mid QB1</span>
                      <span className="text-blue-400">3 players</span>
                    </div>
                  </div>
                  <Link href="/tiers" className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300 text-sm font-medium mt-4">
                    View Full Tiers <ArrowRightIcon className="w-3 h-3" />
                  </Link>
                </div>
              </div>

              {/* Neural Network Predictions Preview */}
              <div className="relative rounded-xl bg-white/10 backdrop-blur-sm border border-white/20 overflow-hidden">
                <div className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-emerald-600/20 rounded-lg">
                      <CpuChipIcon className="w-5 h-5 text-emerald-400" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">Weekly Predictions</h3>
                      <p className="text-sm text-emerald-400">Neural Networks</p>
                    </div>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between items-center text-gray-300">
                      <span>P. Mahomes</span>
                      <span className="text-emerald-400">24.8 pts</span>
                    </div>
                    <div className="flex justify-between items-center text-gray-300">
                      <span>C. McCaffrey</span>
                      <span className="text-emerald-400">22.5 pts</span>
                    </div>
                    <div className="flex justify-between items-center text-gray-300">
                      <span>J. Jefferson</span>
                      <span className="text-emerald-400">19.2 pts</span>
                    </div>
                  </div>
                  <Link href="/predictions" className="inline-flex items-center gap-1 text-emerald-400 hover:text-emerald-300 text-sm font-medium mt-4">
                    View Predictions <ArrowRightIcon className="w-3 h-3" />
                  </Link>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}