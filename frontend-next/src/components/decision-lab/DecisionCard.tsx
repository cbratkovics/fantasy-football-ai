'use client'

import { DECISION_LAB_COPY } from '@/content/decision-lab'
import { limitationText, reasonSeverityLabel, statusDescription } from '@/lib/decision-lab/explain'
import type { Receipt } from '@/lib/decision-lab/types'
import { Badge, Card, Hash, Notice } from './Bits'
import { num } from './draft'
import { StatusPill } from './StatusPill'

const COPY = DECISION_LAB_COPY.card

interface DecisionCardProps {
  receipt: Receipt | null
  stale: boolean
  saved: boolean
  emptyMessage: string
}

function nameOf(receipt: Receipt, playerId: string | null): string {
  if (!playerId) return '—'
  const alt = receipt.inputs.alternatives.find((a) => a.player_id === playerId)
  return alt && alt.display_name ? `${alt.display_name} (${playerId})` : playerId
}

/** The primary result card: status, recommended player, gates, baseline line, limitations and explanation. */
export function DecisionCard({ receipt, stale, saved, emptyMessage }: DecisionCardProps) {
  if (!receipt) {
    return (
      <Card title={COPY.title} id="decision-card" testId="decision-card">
        <Notice testId="decision-empty">{emptyMessage}</Notice>
      </Card>
    )
  }
  if (stale) {
    return (
      <Card title={COPY.title} id="decision-card" testId="decision-card" aside={<Badge tone="warn" testId="stale-badge">stale</Badge>}>
        <Notice testId="decision-stale">{DECISION_LAB_COPY.compute.stale}</Notice>
      </Card>
    )
  }
  const r = receipt.result
  const top = r.reasons.length ? r.explanation[0] : ''
  const gates = r.gates
  const baseline = r.baseline
  let baselineLine: string
  if (baseline.status === 'tie') {
    baselineLine = COPY.baselineTie(num(baseline.preferred_value))
  } else if (baseline.status === 'unavailable') {
    baselineLine = COPY.baselineUnavailable(baseline.missing_for.length ? baseline.missing_for.map((pid) => nameOf(receipt, pid)).join(', ') : 'the comparable set')
  } else if (baseline.agrees_with_model_leader) {
    baselineLine = COPY.baselineAgrees(nameOf(receipt, baseline.preferred_player_id), num(baseline.preferred_value))
  } else {
    baselineLine = COPY.baselinePrefers(nameOf(receipt, baseline.preferred_player_id), num(baseline.preferred_value))
  }
  let floorLine: string
  if (!gates.floor_checked) floorLine = COPY.floorOff
  else if (gates.leading_floor === null) floorLine = `${COPY.floorFail} — ${COPY.floorMissing} (guardrail ${num(gates.min_floor)})`
  else floorLine = `${gates.floor_ok ? COPY.floorOk : COPY.floorFail} — leading floor ${num(gates.leading_floor)} vs guardrail ${num(gates.min_floor)}`

  return (
    <Card
      title={COPY.title}
      id="decision-card"
      testId="decision-card"
      aside={saved ? <Badge tone="saved" testId="saved-badge">{COPY.recorded}</Badge> : <Badge testId="saved-badge">{COPY.unsaved}</Badge>}
    >
      <div role="status" aria-live="polite" data-testid="decision-result">
        <div className="flex flex-wrap items-center gap-3">
          <StatusPill status={r.status} />
          <span className="text-xs text-[#52605d]">{statusDescription(r.status)}</span>
        </div>
        <p className="mt-4 text-2xl font-semibold leading-tight tracking-[-0.03em]" data-testid="recommended-player">
          {r.recommended_player_id ? nameOf(receipt, r.recommended_player_id) : COPY.noRecommendation}
        </p>
        {top && <p className="mt-2 text-sm leading-6 text-[#3b4a46]">{top}</p>}
      </div>

      <dl className="mt-5 grid gap-px bg-[#d7d7d0] sm:grid-cols-3">
        <div className="bg-white p-3">
          <dt className="font-mono text-[9px] uppercase tracking-widest text-[#6e7875]">{COPY.gap}</dt>
          <dd className="mt-1 font-mono text-xs" data-testid="gap-line">
            {num(r.leading.gap)} vs {COPY.threshold} {num(gates.min_projection_gap)}
            {gates.gap_ok === null ? '' : gates.gap_ok ? ' — meets' : ' — fails'}
          </dd>
        </div>
        <div className="bg-white p-3">
          <dt className="font-mono text-[9px] uppercase tracking-widest text-[#6e7875]">{COPY.floorGate}</dt>
          <dd className="mt-1 font-mono text-xs" data-testid="floor-line">
            {floorLine}
          </dd>
        </div>
        <div className="bg-white p-3">
          <dt className="font-mono text-[9px] uppercase tracking-widest text-[#6e7875]">{COPY.baseline}</dt>
          <dd className="mt-1 text-xs" data-testid="baseline-line">
            {baselineLine}
          </dd>
        </div>
      </dl>

      <div className="mt-5 grid gap-5 md:grid-cols-2">
        <div>
          <h3 className="font-mono text-[9px] font-bold uppercase tracking-widest text-[#5c6966]">{COPY.explanation}</h3>
          <ol className="mt-2 space-y-2 text-xs leading-5 text-[#3b4a46]" data-testid="explanation">
            {r.reasons.map((reason, i) => (
              <li key={`${reason.code}-${i}`} className="border-l-2 border-[#c5c7c0] pl-3">
                <span className="mr-2 font-mono text-[9px] uppercase tracking-widest text-[#6e7875]">{reasonSeverityLabel(reason.severity)}</span>
                {r.explanation[i]}
              </li>
            ))}
          </ol>
        </div>
        <div>
          <h3 className="font-mono text-[9px] font-bold uppercase tracking-widest text-[#5c6966]">{COPY.limitations}</h3>
          <ul className="mt-2 space-y-2 text-xs leading-5 text-[#3b4a46]" data-testid="limitations">
            {r.limitations.map((code) => (
              <li key={code} data-limitation={code} className="border-l-2 border-ember pl-3">
                {limitationText(code)}
              </li>
            ))}
          </ul>
        </div>
      </div>

      <dl className="mt-5 flex flex-wrap gap-x-6 gap-y-1 font-mono text-[10px] text-[#6e7875]">
        <div>
          <dt className="inline">{COPY.decisionId}: </dt>
          <dd className="inline" data-testid="decision-id">
            <Hash value={receipt.decision_id} />
          </dd>
        </div>
        <div>
          <dt className="inline">{COPY.parent}: </dt>
          <dd className="inline" data-testid="parent-id">
            <Hash value={receipt.parent_decision_id} />
          </dd>
        </div>
        <div>
          <dt className="inline">{COPY.policy}: </dt>
          <dd className="inline">{r.policy_version}</dd>
        </div>
      </dl>
    </Card>
  )
}
