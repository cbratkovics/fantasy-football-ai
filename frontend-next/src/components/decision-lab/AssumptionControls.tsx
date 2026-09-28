'use client'

import { DECISION_LAB_COPY } from '@/content/decision-lab'
import { describeAvailability } from '@/lib/decision-lab/explain'
import { PARAM_SPEC } from '@/lib/decision-lab/policy'
import type { Experiment, InputsSnapshot, LabCase } from '@/lib/decision-lab/types'
import { Button, Card, FOCUS, Hash, Help, MicroLabel } from './Bits'
import { availabilityChoice, withAvailability, type DiffRow, type Draft } from './draft'
import { NumberField } from './NumberField'

const COPY = DECISION_LAB_COPY.parameters
const NCOPY = DECISION_LAB_COPY.nomination

export interface DiffView {
  rows: DiffRow[]
  controls: string[]
  parentId: string
}

interface AssumptionControlsProps {
  draft: Draft
  snapshot: InputsSnapshot | null
  onChange: (draft: Draft) => void
  disabled: boolean
  frozen: boolean
  onUnlock: () => void
  labCase: LabCase | null
  onApplyExperiment: (experiment: Experiment) => void
  diff: DiffView | null
}

function Checkbox({ id, label, help, checked, disabled, onChange }: { id: string; label: string; help: string; checked: boolean; disabled: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-start gap-3">
      <input id={id} type="checkbox" checked={checked} disabled={disabled} aria-describedby={`${id}-help`} onChange={(e) => onChange(e.target.checked)} className={`mt-0.5 h-4 w-4 border-[#bfc3bb] text-ink ${FOCUS}`} />
      <div>
        <MicroLabel htmlFor={id}>{label}</MicroLabel>
        <Help id={`${id}-help`}>{help}</Help>
      </div>
    </div>
  )
}

/** Policy parameters, per-alternative availability, case experiments and the "what changed" panel. */
export function AssumptionControls({ draft, snapshot, onChange, disabled, frozen, onUnlock, labCase, onApplyExperiment, diff }: AssumptionControlsProps) {
  const p = draft.params
  const setParams = (patch: Partial<Draft['params']>) => onChange({ ...draft, params: { ...p, ...patch } })
  const rowsById: Record<string, InputsSnapshot['rows'][number]> = {}
  if (snapshot) snapshot.rows.forEach((r) => (rowsById[r.player_id] = r))
  const locked = disabled || frozen

  return (
    <Card
      title={COPY.title}
      id="assumptions"
      testId="assumption-controls"
      aside={
        frozen ? (
          <Button data-testid="unlock" onClick={onUnlock}>
            {COPY.unlock}
          </Button>
        ) : undefined
      }
    >
      {frozen && (
        <p role="status" className="mb-4 border border-dashed border-[#bfc3bb] bg-white p-3 text-xs text-[#52605d]" data-testid="frozen-note">
          {COPY.frozen}
        </p>
      )}
      <div className="grid gap-5 md:grid-cols-2">
        <NumberField id="min-gap" label={COPY.minGap} help={COPY.minGapHelp} value={p.minGap} onChange={(v) => setParams({ minGap: v })} min={PARAM_SPEC.min_projection_gap.min as number} max={PARAM_SPEC.min_projection_gap.max as number} step={0.1} disabled={locked} />
        <div className="flex flex-col gap-3">
          <Checkbox id="floor-enabled" label={COPY.floorEnabled} help={COPY.minFloorHelp} checked={p.floorEnabled} disabled={locked} onChange={(v) => setParams({ floorEnabled: v, minFloor: v && p.minFloor === '' ? '0' : p.minFloor })} />
          {p.floorEnabled && <NumberField id="min-floor" label={COPY.minFloor} help={COPY.minFloorHelp} value={p.minFloor} onChange={(v) => setParams({ minFloor: v })} min={PARAM_SPEC.min_floor.min as number} max={PARAM_SPEC.min_floor.max as number} step={0.5} disabled={locked} />}
        </div>
        <Checkbox id="require-baseline" label={COPY.requireBaseline} help={COPY.requireBaselineHelp} checked={p.requireBaseline} disabled={locked} onChange={(v) => setParams({ requireBaseline: v })} />
        <Checkbox id="model-only" label={COPY.modelOnly} help={COPY.modelOnlyHelp} checked={p.modelOnly} disabled={locked} onChange={(v) => setParams({ modelOnly: v })} />
      </div>

      {snapshot && draft.playerIds.length > 0 && (
        <fieldset className="mt-6">
          <MicroLabel as="legend">{NCOPY.availability}</MicroLabel>
          <Help id="availability-help">{NCOPY.availabilityHelp}</Help>
          <ul className="mt-2 divide-y divide-[#e3e3dc] border border-[#d7d7d0] bg-white" data-testid="availability-list">
            {draft.playerIds.map((pid) => {
              const row = rowsById[pid]
              if (!row) return null
              const choice = availabilityChoice(draft, pid)
              const defaultLabel = describeAvailability(row.availability_default, row.availability_default === 'unknown' ? 'not_verified' : row.availability_default === 'realized_stats_row' ? 'hindsight_stats_row' : row.availability_default === 'assumed_available' ? (snapshot.mode === 'synthetic' ? 'synthetic' : 'user_assumed') : 'user_excluded')
              const id = `avail-${pid}`
              return (
                <li key={pid} className="grid gap-2 p-2 text-xs sm:grid-cols-[1fr_auto] sm:items-center">
                  <label htmlFor={id} className="font-semibold">
                    {row.display_name ?? pid} <span className="ml-1 font-mono text-[10px] font-normal text-[#6e7875]">{row.position}</span>
                  </label>
                  <select
                    id={id}
                    data-testid={id}
                    value={choice}
                    disabled={locked}
                    aria-describedby="availability-help"
                    onChange={(e) => onChange(withAvailability(draft, pid, e.target.value as 'default' | 'assumed_available' | 'unavailable', row.availability_default))}
                    className={`h-9 border border-[#bfc3bb] bg-white px-2 font-mono text-[11px] disabled:bg-sand ${FOCUS}`}
                  >
                    <option value="default">
                      {NCOPY.availabilityDefault}: {defaultLabel}
                    </option>
                    {row.availability_default !== 'assumed_available' && <option value="assumed_available">{NCOPY.availabilityAssume}</option>}
                    <option value="unavailable">{NCOPY.availabilityExclude}</option>
                  </select>
                </li>
              )
            })}
          </ul>
        </fieldset>
      )}

      {labCase && labCase.experiments.length > 0 && (
        <div className="mt-6">
          <MicroLabel as="span">{COPY.experiments}</MicroLabel>
          <Help>{COPY.experimentsHelp}</Help>
          <ul className="mt-2 flex flex-wrap gap-2">
            {labCase.experiments.map((exp) => (
              <li key={exp.label}>
                <Button data-testid={`experiment-${exp.label}`} disabled={locked} onClick={() => onApplyExperiment(exp)} aria-label={`${COPY.apply}: ${exp.label}`}>
                  {COPY.apply} · {exp.label}
                </Button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {diff && (
        <div className="mt-6 border border-[#c5c7c0] bg-white p-4" data-testid="what-changed" role="status" aria-live="polite">
          <h3 className="font-mono text-[10px] font-bold uppercase tracking-widest text-moss">{DECISION_LAB_COPY.diff.title}</h3>
          <Help>{DECISION_LAB_COPY.diff.help}</Help>
          <p className="mt-2 font-mono text-[10px] text-[#5c6966]">
            {DECISION_LAB_COPY.diff.controls}: <span data-testid="changed-controls">{diff.controls.length ? diff.controls.join(', ') : '—'}</span> · parent <Hash value={diff.parentId} />
          </p>
          {diff.rows.length === 0 ? (
            <p className="mt-2 text-xs text-[#52605d]">{DECISION_LAB_COPY.diff.noneChanged}</p>
          ) : (
            <div className="mt-2 overflow-x-auto">
              <table className="w-full min-w-[420px] text-left font-mono text-xs">
                <thead className="bg-[#f8f7f2]">
                  <tr>
                    <th scope="col" className="px-2 py-1 text-[9px] uppercase tracking-widest text-[#5c6966]">Field</th>
                    <th scope="col" className="px-2 py-1 text-[9px] uppercase tracking-widest text-[#5c6966]">Before</th>
                    <th scope="col" className="px-2 py-1 text-[9px] uppercase tracking-widest text-[#5c6966]">After</th>
                  </tr>
                </thead>
                <tbody>
                  {diff.rows.map((row) => (
                    <tr key={row.key} data-testid={`diff-row-${row.key}`} className="border-t border-[#e3e3dc]">
                      <td className="px-2 py-1">{row.label}</td>
                      <td className="px-2 py-1">{row.before || '—'}</td>
                      <td className="px-2 py-1">{row.after || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}
