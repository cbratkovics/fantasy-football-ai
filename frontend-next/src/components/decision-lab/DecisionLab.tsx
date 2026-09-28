'use client'

import { useSearchParams } from 'next/navigation'
import { DecisionLabApp } from './DecisionLabApp'

/** Route entry: reads the `?case=<case_id>` deep link and renders the lab. */
export function DecisionLab() {
  const params = useSearchParams()
  const caseId = params ? params.get('case') : null
  return <DecisionLabApp initialCaseId={caseId} />
}
