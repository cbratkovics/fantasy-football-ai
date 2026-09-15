// Project vocabulary and deployment names, read from the generated project.config.json
// (python -m ffai.config --frontend; tests/test_project_config.py fails if it drifts from
// ffai/config.py). Components on /performance, the decisions panel, and the player history read
// their labels from here instead of string literals (ADR-0028).
import config from './project.config.json'
import type { WithinKey } from './api/types'

export interface ProjectConfig {
  slug: string
  displayName: string
  domainSummary: string
  repoUrl: string
  apiUrl: string
  pagesUrl: string
  entity: { name: string; key: string }
  period: { name: string; season: string }
  cohort: { name: string; values: string[] }
  target: { column: string; units: string; format: string; formats: string[] }
  withinK: number[]
  candidates: string[]
}

export const PROJECT: ProjectConfig = config as ProjectConfig

/** "ALL" plus every cohort, in config order — the cohort selector everywhere. */
export const COHORTS: string[] = ['ALL', ...PROJECT.cohort.values]

/** "player-week" — the grain of one scored row. */
export const ENTITY_PERIOD = `${PROJECT.entity.name}-${PROJECT.period.name}`

/** "Within ±3" for the first tolerance band, "Within ±5" for the second. */
export const withinLabel = (k: number) => `Within ±${k}`

/** The metric keys the artifact carries for a tolerance band: within_3_rate, within_5_rate. */
export const withinKey = (k: number) => `within_${k}_rate` as WithinKey

export const UNITS = PROJECT.target.units
