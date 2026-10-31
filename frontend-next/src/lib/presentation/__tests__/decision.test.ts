import { expect, it } from 'vitest'
import { syntheticInputsSnapshot } from '@/lib/decision-lab/__tests__/fixtures'
import { buildDecisionInputs } from '@/lib/decision-lab/receipts'
import { evaluate } from '@/lib/decision-lab/policy'
import { contentId } from '@/lib/decision-lab/canonical'
import { decisionSummary } from '../decision'

it('describes the separate baseline preference on a model tie without treating display order as a winner', () => {
  const snapshot = syntheticInputsSnapshot()
  snapshot.rows[0].projection = 13.6
  const result = evaluate(buildDecisionInputs(snapshot, { slot: 'RB', playerIds: ['SYN-A', 'SYN-B'] }))
  const identity = contentId(result)
  const summary = decisionSummary(result, { 'SYN-A': 'Synthetic A', 'SYN-B': 'Synthetic B' })
  expect(result.status).toBe('review')
  expect(result.recommended_player_id).toBeNull()
  expect(summary.heading).toBe('No clear edge under this comparison rule')
  expect(summary.support).toContain('The historical baseline prefers Synthetic B; the model projections remain tied.')
  expect(summary.support.join(' ')).not.toContain('different player')
  expect(contentId(result)).toBe(identity)
})
