/** Quick comparison orchestration. Evidence, normalization and ranking belong to Decision Lab. */
import { loadInputsSnapshot, loadManifest, policySpecDigestMatches, type BundleFailure, type LoadOptions } from '@/lib/decision-lab/bundle'
import { evaluate } from '@/lib/decision-lab/policy'
import { buildDecisionInputs, type AvailabilityOverride } from '@/lib/decision-lab/receipts'
import type { DecisionInputs, DecisionResult, InputsSnapshot, LabManifest } from '@/lib/decision-lab/types'
import { deepEqual } from '@/lib/decision-lab/canonical'
import { validateInputsSnapshotShape } from '@/lib/decision-lab/validate'
import SPEC_DIGEST from '@/lib/decision-lab/spec_digest.json'

export type WeeklyEvidence = { ok: true; manifest: LabManifest; snapshot: InputsSnapshot } | BundleFailure

function incompatible(detail: string): BundleFailure {
  return { ok: false, kind: 'compat', error: 'Weekly evidence unavailable', detail, evidence_status: 'corrupt' }
}

function unknownWeeklyAvailability(snapshot: InputsSnapshot): boolean {
  // Policy-v1 weekly evidence has no provider-verified availability basis. Context facts stay
  // in their companion; all availability assumptions in this quick path must be intentional.
  return snapshot.rows.every((row) => row.availability_default === 'unknown')
}

/** Never substitute an older week, an API forecast, or a synthetic case after a load failure. */
export async function loadLatestWeekly(options: LoadOptions = {}): Promise<WeeklyEvidence> {
  const manifest = await loadManifest(options.baseUrl, options.fetchImpl)
  if (!manifest.ok) return manifest
  if (policySpecDigestMatches(manifest.manifest, SPEC_DIGEST.sha256).matches !== true) {
    return incompatible('The bundle policy specification does not match the installed policy.')
  }
  const id = manifest.manifest.latest_weekly_snapshot_id
  const entry = manifest.manifest.snapshots.find((s) => s.snapshot_id === id)
  if (!id || !entry || entry.source_family !== 'weekly' || entry.mode !== 'published_weekly') {
    return incompatible('No published weekly snapshot is identified. Choose a historical replay explicitly in Decision Lab.')
  }
  const loaded = await loadInputsSnapshot(manifest.manifest, id, options)
  if (!loaded.ok) return loaded
  const snapshot = loaded.snapshot
  if (snapshot.source_family !== 'weekly' || snapshot.mode !== 'published_weekly' ||
      snapshot.publication.status !== 'published' || snapshot.scoring_format !== 'ppr' ||
      snapshot.season !== entry.season || snapshot.week !== entry.week ||
      snapshot.model_version !== entry.model_version ||
      !deepEqual(snapshot.candidate_by_position, entry.candidate_by_position)) {
    return incompatible('Snapshot period, publication, model, or scoring is incompatible with its manifest.')
  }
  if (!unknownWeeklyAvailability(snapshot)) {
    return incompatible('Weekly availability must start unknown; assumptions cannot be encoded as provider verification.')
  }
  return { ok: true, manifest: manifest.manifest, snapshot }
}

export interface QuickComparisonOptions {
  slot: string
  playerIds: string[]
  overrides?: Record<string, AvailabilityOverride>
  scoring?: string
}

/** Uses the same builder and pure v1 policy as the Lab; no baseline joins or UI ranking rule. */
export function compareWeekly(snapshot: InputsSnapshot, options: QuickComparisonOptions): { inputs: DecisionInputs; result: DecisionResult } {
  if ((options.scoring ?? 'ppr') !== 'ppr' || snapshot.scoring_format !== 'ppr') {
    throw new Error('Start/Sit supports only the implemented PPR rules. Half-PPR and standard browsing remain available in Predictions.')
  }
  if (snapshot.source_family !== 'weekly' || snapshot.mode !== 'published_weekly' || snapshot.publication.status !== 'published') {
    throw new Error('Select a published weekly snapshot for this dated comparison.')
  }
  // Also validate direct callers: TypeScript types and a caller-supplied digest do not verify
  // cardinality, finite measurements, model compatibility, or the absence of outcome fields.
  validateInputsSnapshotShape(snapshot)
  if (!unknownWeeklyAvailability(snapshot)) {
    throw new Error('Weekly availability must start unknown. Confirm assumptions explicitly for the dated comparison.')
  }
  const inputs = buildDecisionInputs(snapshot, options)
  return { inputs, result: evaluate(inputs) }
}

export function comparisonHeading(result: DecisionResult): string {
  if (result.status === 'hold') return 'Recommendation unavailable'
  if (result.status === 'recommend') return 'Suggested starter'
  if (result.reasons.some((r) => r.code === 'availability_unresolved')) return 'Check player status'
  if (result.reasons.some((r) => ['projection_tie', 'gap_below_minimum'].includes(r.code))) return 'Too close for a clear edge'
  return 'Comparison incomplete'
}
