// Centralized constants for metrics and statistics
export const METRICS = {
  accuracy: {
    percentage: '93.1%',
    description: 'Validated predictions within 3 fantasy points',
    oldAccuracy: '54-57%', // Previous accuracy for comparison
  },
  mae: {
    current: '1.25',
    previous: '5.0',
    unit: 'points',
  },
  features: {
    count: '100+',
    categories: 10,
    playerAttributes: '50+',
  },
  models: {
    architecture: 'Ensemble of XGBoost, LightGBM, and Neural Networks',
    types: ['XGBoost', 'LightGBM', 'Neural Networks'],
    tiers: 16,
  },
  users: {
    active: '2,500+',
    predictions: '10K+',
    playersAnalyzed: '500+',
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
    'AI-Powered Predictions with 93.1% accuracy',
    'Transparent explanations for every prediction',
    'Real-time injury and news integration',
    'Position-specific ML models',
    'Weather and matchup analysis',
    'Historical accuracy tracking',
    'Confidence intervals included',
    '100+ engineered features across 10 categories',
    'Ensemble models combining multiple algorithms',
    '50+ player attributes analyzed',
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