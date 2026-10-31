'use client'

import { DECISION_LAB_COPY } from '@/content/decision-lab'
import type { LabCase, Mode } from '@/lib/decision-lab/types'
import { FOCUS, Help } from './Bits'

const ORDER: Mode[] = ['historical_replay', 'published_weekly', 'synthetic']
const COPY = DECISION_LAB_COPY.library

interface CaseLibraryProps {
  cases: LabCase[]
  selectedId: string | null
  onSelect: (labCase: LabCase) => void
}

/**
 * The curated case list grouped by context. Titles and setup sentences come from cases.json;
 * the sealed selection pattern is never rendered here (it belongs to the outcome step).
 */
export function CaseLibrary({ cases, selectedId, onSelect }: CaseLibraryProps) {
  return (
    <nav aria-label={COPY.title} data-testid="case-library" className="border border-[#c5c7c0] bg-[#f8f7f2]">
      <div className="border-b border-[#d7d7d0] p-4">
        <h2 className="text-sm font-semibold text-moss">{COPY.title}</h2>
        <Help>{COPY.help}</Help>
      </div>
      {ORDER.map((mode) => {
        const group = cases.filter((c) => c.mode === mode).slice().sort((a, b) => a.case_id.localeCompare(b.case_id))
        if (!group.length) return null
        const ctx = DECISION_LAB_COPY.contexts[mode]
        return (
          <div key={mode} className="border-b border-[#d7d7d0] last:border-b-0">
            <h3 className="px-4 pb-1 pt-3 text-sm font-semibold text-[#5c6966]">
              {ctx.label} · {group.length}
            </h3>
            <ul className="divide-y divide-[#e3e3dc]">
              {group.map((c) => {
                const selected = c.case_id === selectedId
                return (
                  <li key={c.case_id}>
                    <button
                      type="button"
                      data-testid={`case-${c.case_id}`}
                      aria-current={selected ? 'true' : undefined}
                      onClick={() => onSelect(c)}
                      className={`block w-full px-4 py-3 text-left ${selected ? 'bg-ink text-white' : 'bg-transparent hover:bg-sand'} ${FOCUS}`}
                    >
                      <span className="block text-sm font-semibold leading-5">{c.title}</span>
                      <span className={`mt-1 block text-sm leading-5 ${selected ? 'text-[#c9d3cf]' : 'text-[#52605d]'}`}>{c.setup}</span>
                      <span className={`mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm ${selected ? 'text-[#c9d3cf]' : 'text-[#52605d]'}`}>
                        <span>{c.slot}</span>
                        <span>
                          {c.alternatives.length} {COPY.alternatives}
                        </span>



                      </span>
                    </button>
                  </li>
                )
              })}
            </ul>
          </div>
        )
      })}
    </nav>
  )
}
