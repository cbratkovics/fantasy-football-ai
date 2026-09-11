// Site-wide constants. No performance figures live here: every metric shown in the UI is read at
// runtime from the API, which serves the committed evaluation artifact.

export const SITE = {
  name: 'Win My League',
  shortName: 'WML',
  tagline: 'Decision Lab',
  supportEmail: 'support@winmyleague.ai',
  repoUrl: 'https://github.com/cbratkovics/fantasy-football-ai',
  authorGithub: 'https://github.com/cbratkovics',
  authorLinkedin: 'https://linkedin.com/in/cbratkovics',
} as const

export const ROUTES = {
  home: '/',
  predictions: '/predictions',
  tiers: '/tiers',
  draft: '/draft',
  startSit: '/start-sit',
  performance: '/performance',
  howItWorks: '/how-it-works',
  learn: '/learn',
  help: '/help',
  about: '/about',
  privacy: '/privacy',
  terms: '/terms',
} as const
