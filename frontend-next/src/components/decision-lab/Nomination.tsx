'use client'

import { useState } from 'react'
import { DECISION_LAB_COPY } from '@/content/decision-lab'
import { SLOTS } from '@/lib/decision-lab/policy'
import type { InputsSnapshot, ManifestSnapshot } from '@/lib/decision-lab/types'
import { Alert, Button, Card, FOCUS, Help, MicroLabel, Notice } from './Bits'
import { MAX_ALTERNATIVES, SLOT_OPTIONS, num, searchRows, type Draft } from './draft'

const COPY = DECISION_LAB_COPY.nomination

interface NominationProps {
  draft: Draft
  snapshots: ManifestSnapshot[]
  snapshot: InputsSnapshot | null
  snapshotState: 'idle' | 'loading' | 'error'
  snapshotError: string | null
  onChange: (draft: Draft) => void
  disabled: boolean
}

export function snapshotLabel(s: ManifestSnapshot): string {
  return `${s.season} week ${String(s.week).padStart(2, '0')} · ${s.source_family} · ${s.model_version}`
}

/** Snapshot, slot and alternative selection for a custom nomination (or edits to a case). */
export function Nomination({ draft, snapshots, snapshot, snapshotState, snapshotError, onChange, disabled }: NominationProps) {
  const [query, setQuery] = useState('')
  const eligible = SLOTS[draft.slot] ?? []
  const matches = snapshot ? searchRows(snapshot.rows, eligible, query).filter((r) => draft.playerIds.indexOf(r.player_id) < 0) : []
  const rowsById: Record<string, InputsSnapshot['rows'][number]> = {}
  if (snapshot) snapshot.rows.forEach((r) => (rowsById[r.player_id] = r))
  const full = draft.playerIds.length >= MAX_ALTERNATIVES

  return (
    <Card title={COPY.title} id="nomination" testId="nomination">
      <div className="grid gap-4 md:grid-cols-2">
        <div className="flex flex-col gap-1">
          <MicroLabel htmlFor="snapshot-select">{COPY.snapshot}</MicroLabel>
          <select
            id="snapshot-select"
            value={draft.snapshotId ?? ''}
            disabled={disabled}
            aria-describedby="snapshot-help"
            onChange={(e) => onChange({ ...draft, caseId: null, snapshotId: e.target.value || null, playerIds: [], overrides: {} })}
            className={`h-10 w-full border border-[#bfc3bb] bg-white px-2 font-mono text-xs disabled:bg-sand ${FOCUS}`}
          >
            <option value="">—</option>
            {snapshots.map((s) => (
              <option key={s.snapshot_id} value={s.snapshot_id}>
                {snapshotLabel(s)}
              </option>
            ))}
          </select>
          <Help id="snapshot-help">{COPY.snapshotHelp}</Help>
        </div>
        <fieldset className="flex flex-col gap-1">
          <MicroLabel as="legend">{COPY.slot}</MicroLabel>
          <div role="group" aria-label={COPY.slot} aria-describedby="slot-help" className="flex flex-wrap border border-[#bfc3bb]">
            {SLOT_OPTIONS.map((slot) => (
              <button
                key={slot}
                type="button"
                data-testid={`slot-${slot}`}
                aria-pressed={draft.slot === slot}
                disabled={disabled}
                onClick={() => onChange({ ...draft, slot })}
                className={`min-w-[56px] flex-1 border-r border-[#bfc3bb] px-3 py-2 font-mono text-[10px] font-bold uppercase last:border-r-0 disabled:cursor-not-allowed ${draft.slot === slot ? 'bg-ink text-acid' : 'bg-white text-[#5c6966] hover:bg-sand'} ${FOCUS}`}
              >
                {slot}
              </button>
            ))}
          </div>
          <Help id="slot-help">{COPY.slotHelp}</Help>
        </fieldset>
      </div>

      {snapshotState === 'loading' && (
        <div role="status" className="mt-4 font-mono text-[10px] uppercase tracking-widest text-[#61706c]">
          {COPY.loadingSnapshot}
        </div>
      )}
      {snapshotState === 'error' && (
        <div className="mt-4">
          <Alert title={DECISION_LAB_COPY.load.failedTitle} testId="snapshot-error">
            <p>{snapshotError}</p>
            <p className="mt-2 text-xs text-[#6e7875]">{DECISION_LAB_COPY.load.failedBody}</p>
          </Alert>
        </div>
      )}

      {snapshot && (
        <div className="mt-5 grid gap-5 md:grid-cols-2">
          <div>
            <div className="flex flex-col gap-1">
              <MicroLabel htmlFor="player-search">{COPY.search}</MicroLabel>
              <input
                id="player-search"
                type="search"
                value={query}
                disabled={disabled || full}
                aria-describedby="player-search-help"
                onChange={(e) => setQuery(e.target.value)}
                className={`h-10 w-full border border-[#bfc3bb] bg-white px-3 font-mono text-xs disabled:bg-sand ${FOCUS}`}
              />
              <Help id="player-search-help">{full ? COPY.full : COPY.searchHelp}</Help>
            </div>
            {!full && (
              <ul className="mt-2 max-h-64 divide-y divide-[#e3e3dc] overflow-y-auto border border-[#d7d7d0] bg-white" aria-label={COPY.search}>
                {matches.length === 0 && <li className="p-3 text-xs text-[#6e7875]">{COPY.noMatches}</li>}
                {matches.map((r) => (
                  <li key={r.player_id} className="flex items-center justify-between gap-3 p-2 text-xs">
                    <span>
                      <b>{r.display_name ?? r.player_id}</b>
                      <span className="ml-2 font-mono text-[10px] text-[#6e7875]">
                        {r.position} · {r.team ?? '—'} · proj {num(r.projection)}
                      </span>
                    </span>
                    <Button
                      data-testid={`add-${r.player_id}`}
                      disabled={disabled}
                      aria-label={`${COPY.add} ${r.display_name ?? r.player_id}`}
                      onClick={() => onChange({ ...draft, playerIds: [...draft.playerIds, r.player_id] })}
                    >
                      {COPY.add}
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div>
            <MicroLabel as="span">
              Nominated · {draft.playerIds.length}/{MAX_ALTERNATIVES}
            </MicroLabel>
            {draft.playerIds.length === 0 ? (
              <div className="mt-2">
                <Notice>{COPY.none}</Notice>
              </div>
            ) : (
              <ul className="mt-2 divide-y divide-[#e3e3dc] border border-[#d7d7d0] bg-white" data-testid="nominated-list">
                {draft.playerIds.map((pid) => {
                  const r = rowsById[pid]
                  return (
                    <li key={pid} className="flex items-center justify-between gap-3 p-2 text-xs">
                      <span>
                        <b>{r?.display_name ?? pid}</b>
                        <span className="ml-2 font-mono text-[10px] text-[#6e7875]">
                          {r ? `${r.position} · ${r.team ?? '—'} · proj ${num(r.projection)}` : 'not in this snapshot'}
                        </span>
                      </span>
                      <Button
                        data-testid={`remove-${pid}`}
                        disabled={disabled}
                        aria-label={`${COPY.remove} ${r?.display_name ?? pid}`}
                        onClick={() => {
                          const overrides = { ...draft.overrides }
                          delete overrides[pid]
                          onChange({ ...draft, playerIds: draft.playerIds.filter((p) => p !== pid), overrides })
                        }}
                      >
                        {COPY.remove}
                      </Button>
                    </li>
                  )
                })}
              </ul>
            )}
          </div>
        </div>
      )}

      {draft.mode === 'synthetic' && (
        <div className="mt-5 flex items-start gap-3">
          <input
            id="simulate-corrupt"
            type="checkbox"
            checked={draft.simulateCorrupt}
            disabled={disabled}
            aria-describedby="simulate-corrupt-help"
            onChange={(e) => onChange({ ...draft, simulateCorrupt: e.target.checked })}
            className={`mt-0.5 h-4 w-4 border-[#bfc3bb] text-ink ${FOCUS}`}
          />
          <div>
            <MicroLabel htmlFor="simulate-corrupt">{COPY.simulateCorrupt}</MicroLabel>
            <Help id="simulate-corrupt-help">{COPY.simulateCorruptHelp}</Help>
          </div>
        </div>
      )}
    </Card>
  )
}
