'use client'

import { useSearchParams } from 'next/navigation'
import { DecisionLabApp } from './DecisionLabApp'

/** Query values are identifiers only; all measurements resolve from verified same-origin files. */
export function DecisionLab() {
  const params = useSearchParams()
  const caseId = params?.get('case') ?? null
  const snapshotId = params?.get('snapshot') ?? null
  const playerIds = params?.get('players') === null ? [] : (params?.get('players') ?? '').split(',')
  const slot = params?.get('slot') ?? null
  return <DecisionLabApp key={params?.toString() ?? ''} initialCaseId={caseId} initialSnapshotId={snapshotId} initialPlayerIds={playerIds} initialSlot={slot} />
}
