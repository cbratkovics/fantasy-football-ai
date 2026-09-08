// Centralized constants for the application.
//
// This is a portfolio demo, not a commercial service. No prediction-accuracy,
// user-count, or usage figures are published here, because none have been measured
// against a held-out evaluation set on live data. Describe capability, not results.

export const METRICS = {
  accuracy: {
    description: 'Weekly point projections with confidence intervals',
  },
  models: {
    architecture: 'Ensemble of XGBoost, LightGBM, and Neural Networks',
    types: ['XGBoost', 'LightGBM', 'Neural Networks'],
  },
  features: {
    description: 'Engineered features spanning usage, efficiency, matchup, and momentum signals',
  },
  techStack: {
    python: '3.11',
    tensorflow: '2.16',
    nextjs: '14',
    fastapi: '0.104',
  },
} as const

export const FEATURES = {
  core: [
    'AI-powered weekly point projections',
    'Transparent explanations for every prediction',
    'Real-time injury and news integration',
    'Position-specific ML models',
    'Weather and matchup analysis',
    'Confidence intervals included',
    'Engineered features across usage, efficiency, and matchup categories',
    'Ensemble models combining multiple algorithms',
  ],
  tiers: {
    free: {
      predictions: '5 per week',
      features: ['Basic predictions', 'Limited explanations'],
    },
    pro: {
      predictions: 'Unlimited',
      features: ['Full predictions', 'Detailed explanations', 'Draft assistant'],
      price: '$14.99/month',
    },
    premium: {
      predictions: 'Unlimited',
      features: ['Everything in Pro', 'API access', 'Custom scoring', 'Priority support'],
      price: '$29.99/month',
    },
  },
} as const
