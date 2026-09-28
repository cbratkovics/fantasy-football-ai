'use client'

import { statusLabel } from '@/lib/decision-lab/explain'
import { DECISION_LAB_STATUS_ICON } from '@/content/decision-lab'

const STYLES: Record<string, string> = {
  recommend: 'bg-ink text-acid border-ink',
  review: 'bg-[#fff1ea] text-ink border-ember',
  hold: 'bg-ember text-white border-ember',
}

/** Status as text plus an icon shape; colour is never the only encoding. */
export function StatusPill({ status, size = 'md' }: { status: string; size?: 'sm' | 'md' }) {
  const icon = DECISION_LAB_STATUS_ICON[status] ?? '·'
  const cls = STYLES[status] ?? 'bg-sand text-ink border-[#bfc3bb]'
  return (
    <span
      data-testid="status-pill"
      data-status={status}
      className={`inline-flex items-center gap-2 border font-mono font-bold uppercase tracking-widest ${size === 'sm' ? 'px-2 py-0.5 text-[9px]' : 'px-3 py-1.5 text-[11px]'} ${cls}`}
    >
      {statusLabel(status)} <span aria-hidden="true">{icon}</span>
    </span>
  )
}
