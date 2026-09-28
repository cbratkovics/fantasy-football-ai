/**
 * Small pure helpers that turn codes from `policy_spec.json` into UI text. Nothing here changes a
 * decision; these are labels for statuses, severities, limitations and availability assumptions.
 */
import { SPEC } from './spec';
import type { Availability, AvailabilityBasis, EvidenceStatus, Severity, Status } from './types';

/** Human label for a decision status. */
export function statusLabel(status: Status | string): string {
  switch (status) {
    case 'hold':
      return 'Hold';
    case 'review':
      return 'Review';
    case 'recommend':
      return 'Recommend';
    default:
      return String(status);
  }
}

/** One-sentence meaning of a decision status. */
export function statusDescription(status: Status | string): string {
  switch (status) {
    case 'hold':
      return 'The evidence is invalid or incompatible; no comparison is made until it is fixed.';
    case 'review':
      return 'A comparison was made but a gate or an assumption needs a human decision.';
    case 'recommend':
      return 'The leading option clears every gate, conditional on the recorded assumptions.';
    default:
      return '';
  }
}

/** Human label for a reason severity. */
export function reasonSeverityLabel(severity: Severity | string): string {
  switch (severity) {
    case 'hold':
      return 'Blocking';
    case 'review':
      return 'Needs review';
    case 'exclusion':
      return 'Excluded';
    case 'info':
      return 'Note';
    default:
      return String(severity);
  }
}

/** The full text of a limitation code from the spec (the code itself when unknown). */
export function limitationText(code: string): string {
  return Object.prototype.hasOwnProperty.call(SPEC.limitations, code) ? SPEC.limitations[code] : code;
}

/** The raw template of a reason code from the spec (the code itself when unknown). */
export function reasonTemplate(code: string): string {
  return Object.prototype.hasOwnProperty.call(SPEC.reasons, code) ? SPEC.reasons[code].template : code;
}

/** Human label for an availability value. */
export function availabilityLabel(availability: Availability | string | null): string {
  switch (availability) {
    case 'assumed_available':
      return 'Assumed available';
    case 'unavailable':
      return 'Unavailable';
    case 'unknown':
      return 'Unknown';
    case 'realized_stats_row':
      return 'Played (hindsight row)';
    default:
      return availability === null ? '—' : String(availability);
  }
}

/** Human label for an availability basis. */
export function availabilityBasisLabel(basis: AvailabilityBasis | string | null): string {
  switch (basis) {
    case 'user_assumed':
      return 'user assumption';
    case 'user_excluded':
      return 'user exclusion';
    case 'not_verified':
      return 'not verified';
    case 'hindsight_stats_row':
      return 'appears in realized stats';
    case 'synthetic':
      return 'synthetic';
    default:
      return basis === null ? '—' : String(basis);
  }
}

/** One phrase describing an alternative's availability and where that assumption came from. */
export function describeAvailability(
  availability: Availability | string | null,
  basis: AvailabilityBasis | string | null,
): string {
  const a = availabilityLabel(availability);
  const b = availabilityBasisLabel(basis);
  return basis === null ? a : `${a} (${b})`;
}

/** Human label for a snapshot's evidence status. */
export function evidenceStatusLabel(status: EvidenceStatus | string): string {
  switch (status) {
    case 'verified':
      return 'Verified against the manifest';
    case 'unverified':
      return 'Not verified';
    case 'corrupt':
      return 'Failed integrity check';
    default:
      return String(status);
  }
}
