import type { DecisionResult } from '../decision-lab/types'

/** Explain structured v1 facts without rewriting the stored result or its identity. */
export function decisionSummary(result: DecisionResult, names: Record<string, string> = {}) {
  const has = (code: string) => result.reasons.some((r) => r.code === code)
  const player = result.recommended_player_id ? names[result.recommended_player_id] ?? result.recommended_player_id : null
  const gap = result.leading.gap
  const threshold = result.gates.min_projection_gap
  let heading = 'This comparison needs another check'
  if (result.status === 'hold') heading = 'This comparison is unavailable'
  else if (has('availability_unresolved')) heading = 'Check availability before choosing'
  else if (result.counts.comparable === 1) heading = 'Only one comparable player remains'
  else if (has('projection_tie') || has('gap_below_minimum')) heading = 'No clear edge under this comparison rule'
  else if (result.status === 'recommend' && player) heading = `${player} has the higher projection`
  const support: string[] = []
  if (gap !== null) support.push(has('projection_tie') ? 'The leading projections are tied.' : `The leading projections differ by ${gap.toFixed(2)} PPR points${result.status === 'review' && threshold !== null ? `; the comparison margin is ${threshold.toFixed(2)} points` : ''}.`)
  if (has('projection_tie') && result.baseline.preferred_player_id) support.push(`The historical baseline prefers ${names[result.baseline.preferred_player_id] ?? result.baseline.preferred_player_id}; the model projections remain tied.`)
  else if (result.baseline.agrees_with_model_leader !== null) support.push(result.baseline.agrees_with_model_leader ? 'The historical baseline prefers the same player.' : 'The historical baseline prefers a different player. Consider both views.')
  if (has('availability_unresolved')) support.push('Confirm which players can play, or explicitly compare under an availability assumption.')
  else if (has('projection_missing')) support.push('A selected player has no usable projection. Keep the choice set intact or explicitly change it.')
  else if (result.counts.excluded) support.push(`${result.counts.excluded} selected option${result.counts.excluded === 1 ? ' is' : 's are'} excluded by slot or your assumptions.`)
  else if (result.status === 'recommend') support.push('This is conditional on the selected players being available and eligible for your lineup.')
  return { heading, player, support: support.slice(0, 3) }
}

export function contextReason(code: string): string {
  if (code.startsWith('kickoff_passed')) return 'A selected player’s kickoff has passed. Check your league’s lock rules; this cannot support a current lineup change.'
  if (code.startsWith('known_bye')) return 'A selected player has a known bye in this period.'
  if (code.startsWith('known_unavailable')) return 'A selected player is reported unavailable.'
  if (code.startsWith('identity_')) return 'A selected player’s current identity or team could not be resolved.'
  if (code === 'context_invalid' || code.includes('reference_mismatch')) return 'The context evidence cannot be verified against this forecast.'
  if (code.includes('stale')) return 'This evidence is too old to support a current lineup change.'
  if (code.includes('not_verified') || code.includes('fixture')) return 'Current player and game coverage is unverified.'
  if (code.startsWith('game_')) return 'The selected player’s game or kickoff is unknown.'
  return 'Current lineup eligibility has not been established. Use a dated comparison only.'
}
