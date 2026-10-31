import { syntheticBundle, fileText } from '@/lib/decision-lab/__tests__/fixtures'
import SPEC_DIGEST from '@/lib/decision-lab/spec_digest.json'
import type { InputsSnapshot } from '@/lib/decision-lab/types'

/** Artificial weekly-shaped evidence for orchestration tests; never provider verification. */
export function weeklyBundle(mutate?: (snapshot: InputsSnapshot) => void) {
  const b = syntheticBundle(({ inputs }) => {
    inputs.snapshot_id = 'weekly-fixture'
    inputs.mode = 'published_weekly'
    inputs.source_family = 'weekly'
    inputs.publication = { status: 'published', action: 'PUBLISH', run_id: 'fixture-run', note: 'Synthetic weekly-shaped test fixture' }
    inputs.generated_at_utc = '2099-09-01T00:00:00Z'
    inputs.population.conditioning = 'scored_eligible_players'
    inputs.rows.forEach((r) => { r.availability_default = 'unknown'; r.display_source = 'prediction_source' })
    mutate?.(inputs)
  })
  b.manifest.latest_weekly_snapshot_id = b.inputs.snapshot_id
  b.manifest.policy_spec.sha256 = SPEC_DIGEST.sha256
  b.files['manifest.json'] = fileText(b.manifest)
  return b
}
