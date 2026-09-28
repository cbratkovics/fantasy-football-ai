'use client'

import { useState } from 'react'
import { DECISION_LAB_COPY } from '@/content/decision-lab'
import { fmtUtc } from '@/lib/format'
import type { ActionKind, Receipt } from '@/lib/decision-lab/types'
import { Alert, Button, Card, FOCUS, Hash, Help, MicroLabel } from './Bits'

const COPY = DECISION_LAB_COPY.action
const NOTE_MAX = 500

interface ActionPanelProps {
  receipt: Receipt
  onRecord: (chosenPlayerId: string, kind: ActionKind, note: string | null) => void
  onDecline: (note: string | null) => void
  error: string | null
}

/** Record the human choice for a computed decision. The recommendation is never preselected. */
export function ActionPanel({ receipt, onRecord, onDecline, error }: ActionPanelProps) {
  const [chosen, setChosen] = useState<string | null>(null)
  const [kind, setKind] = useState<ActionKind>('hypothetical_replay')
  const [note, setNote] = useState('')
  const action = receipt.action
  const r = receipt.result

  if (action.state !== 'not_recorded') {
    const chosenRow = r.ranking.find((row) => row.player_id === action.chosen_player_id)
    return (
      <Card title={COPY.title} id="action" testId="action-panel">
        <div role="status" aria-live="polite" data-testid="action-state" className="text-sm">
          <p className="font-semibold">{action.state === 'recorded' ? COPY.recorded : COPY.declined}</p>
          <dl className="mt-2 grid gap-1 font-mono text-[11px] text-[#52605d] sm:grid-cols-2">
            {action.state === 'recorded' && (
              <div>
                <dt className="inline">{COPY.choose}: </dt>
                <dd className="inline" data-testid="action-chosen">
                  {chosenRow?.display_name ?? action.chosen_player_id} ({action.chosen_player_id})
                </dd>
              </div>
            )}
            <div>
              <dt className="inline">{COPY.kind}: </dt>
              <dd className="inline">{action.kind === 'self_reported_real' ? COPY.real : action.kind === 'hypothetical_replay' ? COPY.hypothetical : '—'}</dd>
            </div>
            <div>
              <dt className="inline">action id: </dt>
              <dd className="inline">
                <Hash value={action.action_id} />
              </dd>
            </div>
            <div>
              <dt className="inline">time: </dt>
              <dd className="inline">{fmtUtc(action.at_utc)}</dd>
            </div>
          </dl>
          {action.note && <p className="mt-2 whitespace-pre-wrap break-words text-xs text-[#3b4a46]">{action.note}</p>}
        </div>
      </Card>
    )
  }

  const trimmedNote = note.trim() ? note.trim().slice(0, NOTE_MAX) : null
  return (
    <Card title={COPY.title} id="action" testId="action-panel">
      <Help>{COPY.help}</Help>
      <fieldset className="mt-4">
        <MicroLabel as="legend">{COPY.choose}</MicroLabel>
        <ul className="mt-2 divide-y divide-[#e3e3dc] border border-[#d7d7d0] bg-white">
          {r.ranking.map((row) => {
            const id = `choose-${row.player_id}`
            const tag = row.exclusion ? `${COPY.excludedLabel}: ${row.exclusion}` : r.recommended_player_id === row.player_id ? COPY.recommendedLabel : COPY.notRecommended
            return (
              <li key={row.player_id} className="flex items-center gap-3 p-2 text-xs">
                <input id={id} data-testid={id} type="radio" name="chosen" value={row.player_id} checked={chosen === row.player_id} onChange={() => setChosen(row.player_id)} className={`h-4 w-4 border-[#bfc3bb] text-ink ${FOCUS}`} />
                <label htmlFor={id}>
                  <b>{row.display_name ?? row.player_id}</b> <span className="font-mono text-[10px] text-[#6e7875]">{row.player_id} · {tag}</span>
                </label>
              </li>
            )
          })}
        </ul>
      </fieldset>
      <fieldset className="mt-4">
        <MicroLabel as="legend">{COPY.kind}</MicroLabel>
        <div className="mt-2 flex flex-wrap gap-4 text-xs">
          {(
            [
              ['hypothetical_replay', COPY.hypothetical],
              ['self_reported_real', COPY.real],
            ] as Array<[ActionKind, string]>
          ).map(([value, label]) => (
            <span key={value} className="flex items-center gap-2">
              <input id={`kind-${value}`} data-testid={`kind-${value}`} type="radio" name="kind" value={value} checked={kind === value} onChange={() => setKind(value)} className={`h-4 w-4 border-[#bfc3bb] text-ink ${FOCUS}`} />
              <label htmlFor={`kind-${value}`}>{label}</label>
            </span>
          ))}
        </div>
      </fieldset>
      <div className="mt-4 flex flex-col gap-1">
        <MicroLabel htmlFor="action-note">{COPY.note}</MicroLabel>
        <textarea id="action-note" value={note} maxLength={NOTE_MAX} rows={2} onChange={(e) => setNote(e.target.value)} className={`w-full border border-[#bfc3bb] bg-white p-2 text-xs ${FOCUS}`} />
        <Help>
          {note.length}/{NOTE_MAX}
        </Help>
      </div>
      {error && (
        <div className="mt-4">
          <Alert title={COPY.saveFailed} testId="action-error">
            {error}
          </Alert>
        </div>
      )}
      <div className="mt-4 flex flex-wrap gap-2">
        <Button tone="primary" data-testid="record-choice" disabled={chosen === null} onClick={() => chosen && onRecord(chosen, kind, trimmedNote)}>
          {COPY.record}
        </Button>
        <Button data-testid="decline-choice" onClick={() => onDecline(trimmedNote)}>
          {COPY.decline}
        </Button>
        {chosen === null && <Help>{COPY.needChoice}</Help>}
      </div>
    </Card>
  )
}
