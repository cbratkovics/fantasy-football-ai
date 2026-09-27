'use client'

import { useRef } from 'react'
import { DECISION_LAB_COPY } from '@/content/decision-lab'
import type { ImportReport, LoadedReceipts } from '@/lib/decision-lab/storage'
import { TRUST_NOTE } from '@/lib/decision-lab/receipts'
import type { Receipt } from '@/lib/decision-lab/types'
import { fmtUtc } from '@/lib/format'
import { Alert, Button, Expandable, FOCUS, Hash, Help, Notice } from './Bits'
import { StatusPill } from './StatusPill'

const COPY = DECISION_LAB_COPY.saved

export interface PanelMessage {
  kind: 'ok' | 'error'
  text: string
}

interface SavedDecisionsPanelProps {
  loaded: LoadedReceipts
  onOpen: (receipt: Receipt) => void
  onAttach: (receipt: Receipt) => void
  onDelete: (decisionId: string) => void
  exportText: () => string
  onImport: (text: string) => void
  onClear: () => void
  report: ImportReport | null
  message: PanelMessage | null
  open: boolean
  onToggle: () => void
}

function download(text: string, filename: string) {
  if (typeof URL === 'undefined' || typeof URL.createObjectURL !== 'function') return
  const blob = new Blob([text], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

/** Receipts in browser storage: open, attach outcomes, export, import, delete, clear. */
export function SavedDecisionsPanel({ loaded, onOpen, onAttach, onDelete, exportText, onImport, onClear, report, message, open, onToggle }: SavedDecisionsPanelProps) {
  const fileRef = useRef<HTMLInputElement | null>(null)
  const ids = Object.keys(loaded.receipts).sort((a, b) => (loaded.receipts[a].created_at_utc < loaded.receipts[b].created_at_utc ? 1 : -1))
  const count = ids.length

  const readFile = (file: File | undefined) => {
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => onImport(typeof reader.result === 'string' ? reader.result : '')
    reader.onerror = () => onImport('')
    reader.readAsText(file)
  }

  return (
    <Expandable id="saved" title={`${COPY.title} · ${count}`} open={open} onToggle={onToggle} testId="saved-panel">
      <Help>{COPY.help}</Help>
      {loaded.problems.length > 0 && (
        <div className="mt-3">
          <Alert title={COPY.problems} testId="storage-problems">
            <ul className="list-disc pl-5 text-xs">
              {loaded.problems.map((p, i) => (
                <li key={`${p.kind}-${i}`}>
                  {p.kind}
                  {p.decision_id ? ` (${p.decision_id.slice(0, 12)})` : ''}: {p.message}
                </li>
              ))}
            </ul>
          </Alert>
        </div>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        <Button data-testid="export-all" disabled={count === 0} onClick={() => download(exportText(), `decision-lab-receipts-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '')}.json`)}>
          {COPY.exportAll}
        </Button>
        <Button data-testid="import-button" onClick={() => fileRef.current?.click()}>
          {COPY.import}
        </Button>
        <input ref={fileRef} data-testid="import-file" type="file" accept="application/json,.json" aria-label={COPY.import} className="sr-only" onChange={(e) => readFile(e.target.files?.[0])} />
        <Button
          tone="danger"
          data-testid="clear-all"
          disabled={count === 0}
          onClick={() => {
            if (window.confirm(COPY.confirmClear(count))) onClear()
          }}
        >
          {COPY.clear} ({count})
        </Button>
      </div>
      {message && (
        <div className="mt-3">
          {message.kind === 'error' ? (
            <Alert title={message.text} testId="saved-message" />
          ) : (
            <Notice testId="saved-message">{message.text}</Notice>
          )}
        </div>
      )}
      {report && (
        <div className="mt-3 border border-[#d7d7d0] bg-white p-3 text-xs" data-testid="import-report" role="status" aria-live="polite">
          <p className="font-mono">
            {report.imported} {COPY.imported} · {report.duplicates} {COPY.duplicates} · {report.rejected.length} {COPY.rejected}
            {!report.persisted ? ` · not persisted${report.storage_error ? `: ${report.storage_error}` : ''}` : ''}
          </p>
          {report.rejected.length > 0 && (
            <ul className="mt-2 list-disc pl-5 text-ember" data-testid="import-rejections">
              {report.rejected.map((r, i) => (
                <li key={i}>
                  {r.decision_id ? `${r.decision_id.slice(0, 12)}: ` : ''}
                  {r.reason}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {count === 0 ? (
        <div className="mt-3">
          <Notice testId="saved-empty">{COPY.empty}</Notice>
        </div>
      ) : (
        <div className="mt-3 overflow-x-auto">
          <table className="table w-full min-w-[960px] text-left font-mono text-xs" data-testid="saved-table">
            <thead className="bg-[#f8f7f2]">
              <tr>
                {[COPY.columns.case, COPY.columns.snapshot, COPY.columns.slot, COPY.columns.status, COPY.columns.recommended, COPY.columns.action, COPY.columns.outcome, COPY.columns.created, COPY.columns.parent, ''].map((h, i) => (
                  <th key={`${h}-${i}`} scope="col" className="border-b border-[#c5c7c0] px-2 py-2 text-[9px] font-bold uppercase tracking-widest text-[#5c6966]">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ids.map((id) => {
                const r = loaded.receipts[id]
                return (
                  <tr key={id} data-testid={`saved-${id}`} data-decision-id={id} className="border-b border-[#e3e3dc] bg-white align-top">
                    <td className="px-2 py-2">
                      {r.case_id ?? 'custom'}
                      <br />
                      <Hash value={id} />
                    </td>
                    <td className="px-2 py-2 break-all">{r.inputs.snapshot.snapshot_id}</td>
                    <td className="px-2 py-2">{r.inputs.slot}</td>
                    <td className="px-2 py-2">
                      <StatusPill status={r.result.status} size="sm" />
                    </td>
                    <td className="px-2 py-2">{r.result.recommended_player_id ?? '—'}</td>
                    <td className="px-2 py-2">
                      {r.action.state}
                      {r.action.chosen_player_id ? ` · ${r.action.chosen_player_id}` : ''}
                    </td>
                    <td className="px-2 py-2">{r.outcome.state}</td>
                    <td className="px-2 py-2 whitespace-nowrap">{fmtUtc(r.created_at_utc)}</td>
                    <td className="px-2 py-2">
                      {r.parent_decision_id ? (
                        <button type="button" className={`underline ${FOCUS}`} onClick={() => loaded.receipts[r.parent_decision_id as string] && onOpen(loaded.receipts[r.parent_decision_id as string])}>
                          <Hash value={r.parent_decision_id} />
                        </button>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="px-2 py-2">
                      <div className="flex flex-wrap gap-1">
                        <Button data-testid={`open-${id}`} onClick={() => onOpen(r)}>
                          {COPY.open}
                        </Button>
                        <Button data-testid={`attach-${id}`} disabled={r.outcome.state === 'attached'} onClick={() => onAttach(r)}>
                          {COPY.attach}
                        </Button>
                        <Button
                          tone="danger"
                          data-testid={`delete-${id}`}
                          onClick={() => {
                            if (window.confirm(COPY.confirmDelete)) onDelete(id)
                          }}
                        >
                          {COPY.delete}
                        </Button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="mt-4 text-[11px] leading-5 text-[#52605d]">{TRUST_NOTE}</p>
    </Expandable>
  )
}
