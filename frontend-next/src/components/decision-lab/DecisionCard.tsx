'use client'

import { DECISION_LAB_COPY } from '@/content/decision-lab'
import { limitationText, reasonSeverityLabel } from '@/lib/decision-lab/explain'
import { decisionSummary } from '@/lib/presentation/decision'
import type { Receipt } from '@/lib/decision-lab/types'
import { Badge, Card, Hash, Notice } from './Bits'
import { num } from './draft'
import { StatusPill } from './StatusPill'

const COPY = DECISION_LAB_COPY.card
interface DecisionCardProps { receipt: Receipt | null; stale: boolean; saved: boolean; emptyMessage: string }
function nameOf(receipt: Receipt, id: string | null): string {
  if (!id) return '—'
  const a = receipt.inputs.alternatives.find((row) => row.player_id === id)
  return a?.display_name ? `${a.display_name} (${id})` : id
}

/** Presentation only: every value and status comes from the unchanged canonical result. */
export function DecisionCard({ receipt, stale, saved, emptyMessage }: DecisionCardProps) {
  if (!receipt) return <Card title="Your comparison" id="decision-card" testId="decision-card"><Notice testId="decision-empty">{emptyMessage}</Notice></Card>
  if (stale) return <Card title="Your comparison" id="decision-card" testId="decision-card" aside={<Badge tone="warn" testId="stale-badge">Update needed</Badge>}><Notice testId="decision-stale">{DECISION_LAB_COPY.compute.stale}</Notice></Card>
  const r = receipt.result
  const names = Object.fromEntries(receipt.inputs.alternatives.map((a) => [a.player_id, a.display_name ?? a.player_id]))
  const summary = decisionSummary(r, names)
  const baseline = r.baseline
  const baselineLine = baseline.status === 'tie' ? COPY.baselineTie(num(baseline.preferred_value))
    : baseline.status === 'unavailable' ? COPY.baselineUnavailable(baseline.missing_for.length ? baseline.missing_for.map((id) => nameOf(receipt, id)).join(', ') : 'the comparable set')
    : baseline.agrees_with_model_leader ? COPY.baselineAgrees(nameOf(receipt, baseline.preferred_player_id), num(baseline.preferred_value))
    : COPY.baselinePrefers(nameOf(receipt, baseline.preferred_player_id), num(baseline.preferred_value))
  const gates = r.gates
  const floorLine = !gates.floor_checked ? COPY.floorOff : gates.leading_floor === null
    ? `${COPY.floorFail} — ${COPY.floorMissing} (guardrail ${num(gates.min_floor)})`
    : `${gates.floor_ok ? COPY.floorOk : COPY.floorFail} — model range lower value ${num(gates.leading_floor)} vs guardrail ${num(gates.min_floor)}`
  return <Card title="Your comparison" id="decision-card" testId="decision-card" aside={<Badge testId="saved-badge" tone={saved ? 'saved' : 'neutral'}>{saved ? COPY.recorded : COPY.unsaved}</Badge>}>
    <div data-testid="decision-result" role="status" aria-live="polite">
      <StatusPill status={r.status} />
      <h3 className="mt-4 text-2xl font-semibold leading-tight text-[#173e35]">{summary.heading}</h3>
      <p className="mt-2 text-sm text-[#52605d]" data-testid="recommended-player">{r.recommended_player_id ? nameOf(receipt, r.recommended_player_id) : COPY.noRecommendation}</p>
      {r.status === 'recommend' && <p className="mt-2 font-medium">Assuming the selected players are available and eligible for your lineup.</p>}
      <ul className="mt-4 space-y-2 text-base leading-6">{summary.support.map((line) => <li key={line}>{line}</li>)}</ul>
    </div>
    <details className="product-details mt-5"><summary>Why this result?</summary>
      <dl className="mt-4 grid gap-3 text-sm">
        {r.leading.gap !== null && <div><dt className="font-semibold">Point difference and comparison margin</dt><dd data-testid="gap-line">{num(r.leading.gap)} vs {COPY.threshold} {num(gates.min_projection_gap)}{gates.gap_ok === null ? '' : gates.gap_ok ? ' — meets' : ' — fails'}</dd></div>}
        {gates.floor_checked && <div><dt className="font-semibold">Optional model-range guardrail</dt><dd data-testid="floor-line">{floorLine}</dd></div>}
        {r.counts.comparable > 0 && <div><dt className="font-semibold">Historical baseline</dt><dd data-testid="baseline-line">{baselineLine}</dd></div>}
      </dl>
      <p className="mt-4 text-sm text-[#52605d]">The comparison margin is a configured rule, not statistical significance or a confidence estimate. Model ranges use validation-residual offsets; actual scores can fall outside them.</p>
      <ol className="mt-4 space-y-2 text-sm" data-testid="explanation">{r.reasons.map((reason, i) => <li key={`${reason.code}-${i}`}>{r.explanation[i]}</li>)}</ol>
      <ul className="mt-4 space-y-2 text-sm text-[#52605d]" data-testid="limitations">{r.limitations.map((code) => <li key={code} data-limitation={code}>{limitationText(code)}</li>)}</ul>
    </details>
    <details className="product-details mt-3"><summary>Decision identity and policy details</summary>
      <dl className="mt-3 space-y-2 break-all text-sm"><div><dt>Decision ID</dt><dd data-testid="decision-id"><Hash value={receipt.decision_id} /></dd></div><div><dt>Parent decision</dt><dd data-testid="parent-id"><Hash value={receipt.parent_decision_id} /></dd></div><div><dt>Policy version</dt><dd>{r.policy_version}</dd></div></dl>
      <ul className="mt-4 space-y-1 text-sm">{r.reasons.map((reason, i) => <li key={`${reason.code}-${i}`}><code>{reason.code}</code> · {reasonSeverityLabel(reason.severity)}</li>)}</ul>
    </details>
  </Card>
}
