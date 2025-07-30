'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'
import { ArrowRightIcon, SparklesIcon } from '@heroicons/react/24/outline'
import { useState } from 'react'

export function Hero() {
  const [isHovered, setIsHovered] = useState(false)

  return (
    <section className="relative overflow-hidden bg-gradient-to-br from-primary-50 via-white to-primary-50 pt-32 pb-20">
      {/* Background decoration */}
      <div className="absolute inset-0 bg-grid-slate-100 [mask-image:linear-gradient(0deg,white,rgba(255,255,255,0.6))] -z-10" />
      
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-3xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            {/* Accuracy Badge */}
            <div className="mb-8 inline-flex items-center rounded-full bg-success-100 px-4 py-2 text-sm font-semibold text-success-800">
              <SparklesIcon className="mr-2 h-4 w-4" />
              89.2% Prediction Accuracy
            </div>

            <h1 className="text-5xl font-bold tracking-tight text-gray-900 sm:text-6xl lg:text-7xl">
              Win Your Fantasy League with{' '}
              <span className="text-primary-600">AI-Powered</span> Predictions
            </h1>
            
            <p className="mt-6 text-lg leading-8 text-gray-600 max-w-2xl mx-auto">
              Get transparent, data-driven player predictions that explain the "why" behind every recommendation. 
              Our AI analyzes millions of data points to give you the edge you need.
            </p>

            <div className="mt-10 flex items-center justify-center gap-4">
              <Link
                href="/signup"
                className="group relative inline-flex items-center justify-center px-8 py-3 text-lg font-semibold text-white bg-primary-600 rounded-lg hover:bg-primary-700 transition-all duration-200 transform hover:scale-105"
                onMouseEnter={() => setIsHovered(true)}
                onMouseLeave={() => setIsHovered(false)}
              >
                Start 7-Day Free Trial
                <ArrowRightIcon className={`ml-2 h-5 w-5 transition-transform duration-200 ${isHovered ? 'translate-x-1' : ''}`} />
              </Link>
              
              <Link
                href="/demo"
                className="inline-flex items-center justify-center px-8 py-3 text-lg font-semibold text-gray-700 bg-white border-2 border-gray-300 rounded-lg hover:bg-gray-50 transition-colors duration-200"
              >
                View Demo
              </Link>
            </div>

            <div className="mt-10 flex items-center justify-center gap-8 text-sm text-gray-600">
              <div className="flex items-center gap-2">
                <svg className="h-5 w-5 text-success-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                No credit card required
              </div>
              <div className="flex items-center gap-2">
                <svg className="h-5 w-5 text-success-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Cancel anytime
              </div>
            </div>
          </motion.div>

          {/* Preview Image */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mt-16"
          >
            <div className="relative rounded-xl shadow-2xl overflow-hidden">
              <div className="aspect-[16/9] bg-gradient-to-br from-primary-100 to-primary-50 flex items-center justify-center">
                <div className="text-center p-8">
                  <h3 className="text-2xl font-semibold text-gray-900 mb-4">
                    Interactive Prediction Dashboard
                  </h3>
                  <p className="text-gray-600">
                    See live predictions with confidence scores and explanations
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}