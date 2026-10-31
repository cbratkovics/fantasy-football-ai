import { fmtNum } from '@/lib/format'
import type { ScoringFormat } from '@/lib/api/types'

const NULL_REASONS: Record<string, string> = {
  outcome_unavailable: 'no recorded outcome',
  actual_format_unverified: 'scoring format is unverified or does not match',
  conversion_evidence_missing: 'missing realized conversion evidence',
  nonfinite_actual: 'invalid nonfinite outcome',
  conflicting_outcome_sources: 'conflicting outcome sources',
  conflicting_mart_outcomes: 'conflicting warehouse outcomes',
  scoring_rules_mismatch: 'scoring rules do not match',
  outcome_revision_mismatch: 'outcome revisions do not match',
  invalid_stat_components: 'invalid realized stat components',
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
  const formatVerified = expectedFormat === undefined || actualFormat === expectedFormat
    || (expectedFormat === 'ppr' && actualFormat === undefined)
  if (actual !== null && !formatVerified) {
    return <span className="text-[#8c9a96]">Actual unavailable: {NULL_REASONS.actual_format_unverified}</span>
  }
  if (actual !== null && Number.isFinite(actual)) return <span title={source ? `Actual source: ${source}` : undefined}>{fmtNum(actual, 1)}</span>
  const code = actual !== null ? 'nonfinite_actual' : reason ?? 'outcome_unavailable'
  return <span className="text-[#8c9a96]">Actual unavailable: {NULL_REASONS[code] ?? code.replaceAll('_', ' ')}</span>
}
