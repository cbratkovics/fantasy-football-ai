import { fmtNum } from '@/lib/format'
import type { ScoringFormat } from '@/lib/api/types'

const NULL_REASONS: Record<string, string> = {
  outcome_unavailable: 'no recorded outcome',
  actual_format_unverified: 'scoring format is unverified or does not match',
  actual_reason_unverified: 'this API does not declare a missing-result reason',
  conversion_evidence_missing: 'missing realized conversion evidence',
  nonfinite_actual: 'invalid nonfinite outcome',
  conflicting_outcome_sources: 'conflicting outcome sources',
  conflicting_mart_outcomes: 'conflicting warehouse outcomes',
  scoring_rules_mismatch: 'scoring rules do not match',
  outcome_revision_mismatch: 'outcome revisions do not match',
  invalid_stat_components: 'invalid realized stat components',
}

export function actualPresentation(actual: number | null, actualFormat?: ScoringFormat, expectedFormat?: ScoringFormat, reason?: string | null) {
  const verified = expectedFormat === undefined || actualFormat === expectedFormat
    || (expectedFormat === 'ppr' && actualFormat === undefined)
  if (actual !== null && !verified) return { value: null, reason: 'actual_format_unverified' }
  if (actual !== null && Number.isFinite(actual)) return { value: actual, reason: null }
  return { value: null, reason: actual !== null ? 'nonfinite_actual' : reason ?? 'actual_reason_unverified' }
}

/** Render only observations as numbers, including valid zero and negative scores. */
export function ActualPoints({ actual, reason, source, actualFormat, expectedFormat }: {
  actual: number | null
  reason?: string | null
  source?: string | null
  actualFormat?: ScoringFormat
  expectedFormat?: ScoringFormat
}) {
  // Legacy prediction actuals mean PPR. Other formats require an explicit matching declaration;
  // a newer format request reaching an older API must never relabel its stored PPR actual.
  const resolved = actualPresentation(actual, actualFormat, expectedFormat, reason)
  if (resolved.value !== null) return <span title={source ? `Actual source: ${source}` : undefined}>{fmtNum(resolved.value, 1)}</span>
  const code = resolved.reason ?? 'outcome_unavailable'
  return <span className="text-[#52605d]">Actual unavailable: {NULL_REASONS[code] ?? code.replaceAll('_', ' ')}</span>
}
