'use client'

import { motion } from 'framer-motion'
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline'
import { useState } from 'react'
import { cn } from '@/lib/utils'

interface PredictionData {
  player: {
    id: string
    name: string
    position: string
    team: string
    status: string
  }
  prediction: {
    week: number
    season: number
    scoring_formats: {
      ppr: {
        point_estimate: number
        lower_bound: number
        upper_bound: number
      }
    }
  }
  confidence: {
    score: number
    level: string
  }
  explanation: {
    summary: string
    key_factors: Array<{
      factor: string
      explanation: string
      impact: string
      impact_icon: string
    }>
    risk_assessment: {
      level: string
      factors: string[]
      bust_probability: string
    }
    recommendation: string
  }
}

interface PredictionCardProps {
  prediction: PredictionData
}

export function PredictionCard({ prediction }: PredictionCardProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const { player, prediction: pred, confidence, explanation } = prediction

  const getConfidenceColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'high':
        return 'text-success-600 bg-success-50'
      case 'medium':
        return 'text-warning-600 bg-warning-50'
      case 'low':
        return 'text-danger-600 bg-danger-50'
      default:
        return 'text-gray-600 bg-gray-50'
    }
  }

  const getRiskColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low':
        return 'text-success-600'
      case 'medium':
        return 'text-warning-600'
      case 'high':
        return 'text-danger-600'
      default:
        return 'text-gray-600'
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow"
    >
      <div className="p-6">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">{player.name}</h3>
            <p className="text-sm text-gray-600">
              {player.position} - {player.team}
            </p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-primary-600">
              {pred.scoring_formats.ppr.point_estimate.toFixed(1)}
            </div>
            <div className="text-sm text-gray-600">Projected Points</div>
          </div>
        </div>

        {/* Quick Info */}
        <div className="mt-4 flex items-center gap-4">
          <span className={cn('px-3 py-1 rounded-full text-xs font-semibold', getConfidenceColor(confidence.level))}>
            {confidence.level} Confidence ({(confidence.score * 100).toFixed(0)}%)
          </span>
          <span className={cn('text-sm font-medium', getRiskColor(explanation.risk_assessment.level))}>
            {explanation.risk_assessment.level} Risk
          </span>
        </div>

        {/* Summary */}
        <p className="mt-4 text-sm text-gray-700">{explanation.summary}</p>

        {/* Expand/Collapse Button */}
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="mt-4 flex items-center gap-1 text-sm font-medium text-primary-600 hover:text-primary-700"
        >
          {isExpanded ? 'Show Less' : 'View Details'}
          {isExpanded ? (
            <ChevronUpIcon className="h-4 w-4" />
          ) : (
            <ChevronDownIcon className="h-4 w-4" />
          )}
        </button>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="border-t border-gray-200"
        >
          <div className="p-6 space-y-6">
            {/* Confidence Range */}
            <div>
              <h4 className="text-sm font-semibold text-gray-900 mb-2">Projection Range</h4>
              <div className="flex items-center gap-4">
                <span className="text-sm text-gray-600">
                  Floor: {pred.scoring_formats.ppr.lower_bound.toFixed(1)}
                </span>
                <div className="flex-1">
                  <div className="h-2 bg-gray-200 rounded-full relative">
                    <div
                      className="absolute h-2 bg-primary-500 rounded-full"
                      style={{
                        left: '20%',
                        width: '60%',
                      }}
                    />
                    <div
                      className="absolute w-3 h-3 bg-primary-600 rounded-full -top-0.5"
                      style={{
                        left: `${((pred.scoring_formats.ppr.point_estimate - pred.scoring_formats.ppr.lower_bound) / 
                                (pred.scoring_formats.ppr.upper_bound - pred.scoring_formats.ppr.lower_bound)) * 100}%`,
                      }}
                    />
                  </div>
                </div>
                <span className="text-sm text-gray-600">
                  Ceiling: {pred.scoring_formats.ppr.upper_bound.toFixed(1)}
                </span>
              </div>
            </div>

            {/* Key Factors */}
            <div>
              <h4 className="text-sm font-semibold text-gray-900 mb-3">Key Factors</h4>
              <div className="space-y-2">
                {explanation.key_factors.map((factor, index) => (
                  <div key={index} className="flex items-start gap-2">
                    <span className={cn(
                      'text-lg',
                      factor.impact === 'positive' ? 'text-success-600' : 
                      factor.impact === 'negative' ? 'text-danger-600' : 
                      'text-gray-400'
                    )}>
                      {factor.impact_icon}
                    </span>
                    <div className="flex-1">
                      <span className="text-sm font-medium text-gray-900">{factor.factor}:</span>
                      <span className="text-sm text-gray-700 ml-1">{factor.explanation}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Risk Assessment */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-gray-900 mb-2">Risk Assessment</h4>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">Bust Probability</span>
                  <span className={cn('text-sm font-medium', getRiskColor(explanation.risk_assessment.level))}>
                    {explanation.risk_assessment.bust_probability}
                  </span>
                </div>
                {explanation.risk_assessment.factors.length > 0 && (
                  <div className="text-sm text-gray-600">
                    {explanation.risk_assessment.factors.join(', ')}
                  </div>
                )}
              </div>
            </div>

            {/* Recommendation */}
            <div className={cn(
              'rounded-lg p-4 border',
              confidence.level === 'High' ? 'bg-success-50 border-success-200' :
              confidence.level === 'Low' ? 'bg-warning-50 border-warning-200' :
              'bg-primary-50 border-primary-200'
            )}>
              <p className={cn(
                'text-sm font-medium',
                confidence.level === 'High' ? 'text-success-800' :
                confidence.level === 'Low' ? 'text-warning-800' :
                'text-primary-800'
              )}>
                💡 {explanation.recommendation}
              </p>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}