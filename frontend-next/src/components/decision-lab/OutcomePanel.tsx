'use client'

import { DECISION_LAB_COPY } from '@/content/decision-lab'
import { limitationText } from '@/lib/decision-lab/explain'
import { fmtUtc } from '@/lib/format'
import type { LabCase, OutcomeMetrics, Receipt } from '@/lib/decision-lab/types'
import { Alert, Button, Expandable, Hash, Help, Notice } from './Bits'
import { num } from './draft'

const COPY = DECISION_LAB_COPY.outcome

export type OutcomeUiState = { status: 'idle' } | { status: 'loading' } | { status: 'none' } | { status: 'error'; message: string } | { status: 'attached' }

interface OutcomePanelProps {
  receipt: Receipt | null
  labCase: LabCase | null
  open: boolean
  onToggle: () => void
  state: OutcomeUiState
  onReveal: () => void
}

function reasonText(code: string | null): string {
  if (!code) return ''
  return COPY.reasons[code] ?? code
}

function Metric({ label, value, reason, testId }: { label: string; value: number | null; reason: string | null; testId: string }) {
  return (
    <div className="bg-white p-3">
      <dt className="font-mono text-[9px] uppercase tracking-widest text-[#6e7875]">{label}</dt>
      <dd className="mt-1 font-mono text-sm" data-testid={testId}>
        {value === null ? <span className="text-xs text-[#52605d]">null — {reasonText(reason)}</span> : num(value)}
      </dd>
    </div>
  )
}

/** Outcome metrics for a receipt; reveals only after an action so the sequence stays prediction → action → evidence. */
export function OutcomeMetricsView({ receipt, metrics, labCase }: { receipt: Receipt; metrics: OutcomeMetrics; labCase: LabCase | null }) {
  const names: Record<string, string> = {}
  receipt.inputs.alternatives.forEach((a) => {
    names[a.player_id] = a.display_name ?? a.player_id
  })
  const sealed = labCase && labCase.case_id === receipt.case_id ? labCase.selection?.sealed_pattern ?? null : null
  return (
    <div data-testid="outcome-metrics">
      <h3 className="font-mono text-[9px] font-bold uppercase tracking-widest text-[#5c6966]">{COPY.perAlternative}</h3>
      <ul className="mt-2 divide-y divide-[#e3e3dc] border border-[#d7d7d0] bg-white font-mono text-xs">
        {metrics.per_alternative.map((row) => (
          <li key={row.player_id} className="flex items-center justify-between gap-3 p-2" data-testid={`actual-${row.player_id}`}>
            <span>
              {names[row.player_id] ?? row.player_id} <span className="text-[10px] text-[#8c9a96]">{row.player_id}</span>
            </span>
            <span>{row.observed ? num(row.actual) : COPY.notObserved}</span>
          </li>
        ))}
      </ul>
      <dl className="mt-4 grid gap-px bg-[#d7d7d0] sm:grid-cols-2">
        <Metric label={COPY.chosen} value={metrics.chosen_actual_points} reason={metrics.chosen_actual_reason} testId="metric-chosen" />
        <Metric label={COPY.regret} value={metrics.choice_set_regret} reason={metrics.choice_set_regret_reason} testId="metric-regret" />
        <Metric label={COPY.vsBaseline} value={metrics.points_vs_baseline_choice} reason={metrics.points_vs_baseline_choice_reason} testId="metric-vs-baseline" />
        <Metric label={COPY.modelVsBaseline} value={metrics.model_policy_vs_baseline_choice} reason={metrics.model_policy_vs_baseline_choice_reason} testId="metric-model-vs-baseline" />
        <div className="bg-white p-3">
          <dt className="font-mono text-[9px] uppercase tracking-widest text-[#6e7875]">{COPY.coverage}</dt>
          <dd className="mt-1 font-mono text-xs" data-testid="metric-coverage">
            {metrics.coverage.observed} of {metrics.coverage.choice_set_size} comparable observed · {metrics.coverage.nominated} nominated
            {metrics.coverage.missing.length ? ` · missing: ${metrics.coverage.missing.map((pid) => names[pid] ?? pid).join(', ')}` : ''}
          </dd>
        </div>
        <div className="bg-white p-3">
          <dt className="font-mono text-[9px] uppercase tracking-widest text-[#6e7875]">{COPY.bestInSet}</dt>
          <dd className="mt-1 font-mono text-xs">{metrics.best_in_choice_set ? `${names[metrics.best_in_choice_set.player_id] ?? metrics.best_in_choice_set.player_id} · ${num(metrics.best_in_choice_set.actual)}` : '—'}</dd>
        </div>
      </dl>
      {sealed && (
        <p className="mt-4 border border-[#c5c7c0] bg-white p-3 text-xs" data-testid="sealed-pattern">
          <b>{COPY.sealed}:</b> <code className="font-mono">{sealed}</code>
        </p>
      )}
      <ul className="mt-4 space-y-1 text-[11px] leading-5 text-[#52605d]">
        {metrics.notes.map((n) => (
          <li key={n}>{n}</li>
        ))}
        <li>{limitationText('outcomes_inspectable')}</li>
      </ul>
    </div>
  )
}

export function OutcomePanel({ receipt, labCase, open, onToggle, state, onReveal }: OutcomePanelProps) {
  const canReveal = receipt !== null && receipt.action.state !== 'not_recorded' && receipt.outcome.state !== 'attached' && state.status !== 'loading'
  const attached = receipt !== null && receipt.outcome.state === 'attached' && receipt.outcome.metrics !== null
  return (
    <Expandable id="outcomes" title={COPY.title} open={open} onToggle={onToggle} testId="outcome-panel">
      <Help>{COPY.revealHelp}</Help>
      {!attached && (
        <div className="mt-3">
          <Button tone="primary" data-testid="reveal-outcomes" disabled={!canReveal} onClick={onReveal}>
            {COPY.reveal}
          </Button>
        </div>
      )}
      {state.status === 'loading' && (
        <p role="status" className="mt-3 font-mono text-[10px] uppercase tracking-widest text-[#61706c]">
          {COPY.loading}
        </p>
      )}
      {state.status === 'none' && (
        <div className="mt-3">
          <Notice testId="outcome-none">{COPY.none}</Notice>
        </div>
      )}
      {state.status === 'error' && (
        <div className="mt-3">
          <Alert title={COPY.failed} testId="outcome-error">
            {state.message}
          </Alert>
        </div>
      )}
      {attached && receipt && receipt.outcome.metrics && (
        <div className="mt-4">
          <p className="font-mono text-[10px] text-[#5c6966]" data-testid="outcome-ids">
            {COPY.idUnchanged}: <Hash value={receipt.decision_id} /> → <Hash value={receipt.decision_id} /> · {COPY.outcomeId} <Hash value={receipt.outcome.outcome_id} /> · {COPY.attachedAt} {fmtUtc(receipt.outcome.attached_at_utc)}
          </p>
          <div className="mt-3">
            <OutcomeMetricsView receipt={receipt} metrics={receipt.outcome.metrics} labCase={labCase} />
          </div>
        </div>
      )}
    </Expandable>
  )
}
