import type { PredictionPeriod, PredictionRecord, PredictionsResponse, ScoringFormat } from '@/lib/api/types'
import { loadInputsSnapshot, loadManifest, policySpecDigestMatches, type LoadOptions } from '@/lib/decision-lab/bundle'
import SPEC_DIGEST from '@/lib/decision-lab/spec_digest.json'
import { actualPresentation } from '@/components/ui/ActualPoints'
import { selectionHref } from './display'

export type ProjectionPosition = 'FLEX' | 'ALL' | 'QB' | 'RB' | 'WR' | 'TE'
export const PROJECTION_POSITIONS: ProjectionPosition[] = ['FLEX', 'ALL', 'QB', 'RB', 'WR', 'TE']
export const PROJECTIONS_PREFERENCE = 'ffai.projections.preferences/1'
export function periodKey(p: { season: number; week: number }) { return `${p.season}-${p.week}` }

export interface ProjectionSettings {
  period: string | null
  scoring: ScoringFormat
  position: ProjectionPosition
  notice: string | null
}

/** Explicit validated URL fields win; invalid requested periods remain unselected with a reason. */
export function projectionDefaults(periods: PredictionPeriod[], query: URLSearchParams, saved: unknown): ProjectionSettings {
  const preference = saved && typeof saved === 'object' ? saved as Record<string, unknown> : {}
  const explicitPeriod = query.has('season') || query.has('week')
  const seasonValue = query.get('season') ?? '', weekValue = query.get('week') ?? ''
  const requested = /^\d+$/.test(seasonValue) && /^\d+$/.test(weekValue)
    ? `${Number(seasonValue)}-${Number(weekValue)}` : ''
  const old = typeof preference.period === 'string' ? preference.period : null
  const valid = (key: string | null) => key !== null && periods.some((p) => periodKey(p) === key)
  let notice: string | null = null
  const period = explicitPeriod ? (valid(requested) ? requested : null) : valid(old) ? old : periods[0] ? periodKey(periods[0]) : null
  if (explicitPeriod && period === null) notice = 'That requested period has no published projection file. Choose an available period below.'
  else if (old && !valid(old)) notice = 'Your saved period is no longer available. Showing the latest published period.'
  const scoringValue = query.get('scoring') ?? preference.scoring
  const scoring = ['ppr', 'half', 'standard'].includes(String(scoringValue)) ? scoringValue as ScoringFormat : 'ppr'
  const positionValue = query.get('position') ?? preference.position
  const position = PROJECTION_POSITIONS.includes(positionValue as ProjectionPosition) ? positionValue as ProjectionPosition : 'FLEX'
  if ((query.has('scoring') && scoringValue !== scoring) || (query.has('position') && positionValue !== position)) notice = 'An unsupported filter was cleared. Only the available scoring and position options are shown.'
  return { period, scoring, position, notice }
}

export function projectionGroup(rows: PredictionRecord[], position: ProjectionPosition) {
  return rows.filter((r) => position === 'ALL' || (position === 'FLEX' ? ['RB', 'WR', 'TE'].includes(r.position) : r.position === position))
    .slice().sort((a, b) => b.prediction - a.prediction || a.player_id.localeCompare(b.player_id))
}

export function rangeScale(rows: PredictionRecord[]) {
  const values = rows.flatMap((r) => [r.floor, r.prediction, r.ceiling]).filter(Number.isFinite)
  const min = Math.min(0, ...values), max = Math.max(0, ...values)
  return { min, max: max === min ? min + 1 : max }
}

export function actualCoverage(rows: PredictionRecord[], scoring: ScoringFormat) {
  const values = rows.map((r) => actualPresentation(r.actual, r.actual_scoring_format, scoring, r.actual_null_reason))
  const observed = values.filter((r) => r.value !== null).length
  const reasons = Array.from(new Set(values.filter((r) => r.value === null).map((r) => r.reason ?? 'outcome_unavailable')))
  const formatMissing = reasons.some((r) => ['actual_format_unverified', 'conversion_evidence_missing', 'scoring_rules_mismatch'].includes(r))
  const absent = reasons.every((r) => r === 'outcome_unavailable')
  return { observed, reasons, showColumn: observed > 0,
    message: observed > 0 ? `${observed} of ${rows.length} published projections have a usable recorded result.`
      : formatMissing ? 'Actual scores are unavailable in this scoring format.'
      : absent ? 'Results have not been recorded in this snapshot.'
      : reasons.includes('actual_reason_unverified') ? 'Actual scores are unavailable; this API does not declare why.'
      : 'Recorded results are withheld because their evidence is incomplete or conflicting.' }
}

/** Resolve the exact same-origin evidence before transferring IDs; never transfer URL numbers. */
export async function comparisonDestination(data: PredictionsResponse, ids: string[], options: LoadOptions = {}): Promise<string> {
  if (data.scoring !== 'ppr') throw new Error('Load the PPR evidence before preparing a comparison.')
  if (data.scoring_rules_id && data.scoring_rules_id !== 'ffai-nflverse-v1') throw new Error('These scoring rules are not supported by the PPR comparison policy.')
  if (ids.length < 2 || ids.length > 8 || new Set(ids).size !== ids.length) throw new Error('Choose two to eight distinct players for a comparison.')
  const result = await loadManifest(options.baseUrl, options.fetchImpl)
  if (!result.ok) throw new Error('Comparison evidence is unavailable. Your player selections are still here.')
  const manifest = result.manifest
  if (!policySpecDigestMatches(manifest, SPEC_DIGEST.sha256).matches) throw new Error('The comparison bundle does not match this application’s policy.')
  const entries = manifest.snapshots.filter((s) => s.source_family === 'weekly' && s.season === data.season && s.week === data.week)
  if (entries.length !== 1) throw new Error('No unique comparison snapshot matches this published period. No other week was substituted.')
  const loaded = await loadInputsSnapshot(manifest, entries[0].snapshot_id, options)
  if (!loaded.ok) throw new Error('The matching comparison evidence could not be verified.')
  const s = loaded.snapshot
  if (s.scoring_format !== 'ppr' || s.model_version !== data.model_version || s.feature_version !== data.feature_version ||
      s.generated_at_utc !== data.generated_at_utc || s.data_cutoff?.season !== data.data_through.season || s.data_cutoff?.week !== data.data_through.week) {
    throw new Error('The API and comparison bundle describe different releases. Refresh both sources before comparing; selected players were not substituted.')
  }
  for (const id of ids) {
    const api = data.predictions.find((r) => r.player_id === id)
    const row = s.rows.find((r) => r.player_id === id)
    if (!api || !row || row.candidate !== api.candidate || row.model_version !== api.model_version ||
        row.projection === null || Math.abs(row.projection - api.prediction) > 0.011) {
      throw new Error('At least one selected projection does not match the verified evidence. No alternative was removed.')
    }
  }
  const isLatest = s.snapshot_id === manifest.latest_weekly_snapshot_id && s.mode === 'published_weekly'
  return selectionHref(isLatest ? '/start-sit' : '/decision-lab', s.snapshot_id, ids, 'FLEX')
}
