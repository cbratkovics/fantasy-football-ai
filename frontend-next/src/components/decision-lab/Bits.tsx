'use client'

import type { ReactNode } from 'react'

/** Shared focus ring: visible on keyboard focus for every interactive element in the lab. */
export const FOCUS = 'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ember'

export function MicroLabel({ htmlFor, children, as = 'label' }: { htmlFor?: string; children: ReactNode; as?: 'label' | 'span' | 'legend' }) {
  const cls = 'font-mono text-[9px] font-bold uppercase tracking-widest text-ink'
  if (as === 'legend') return <legend className={cls}>{children}</legend>
  if (as === 'span') return <span className={cls}>{children}</span>
  return (
    <label htmlFor={htmlFor} className={cls}>
      {children}
    </label>
  )
}

export function Help({ id, children }: { id?: string; children: ReactNode }) {
  return (
    <p id={id} className="text-[11px] leading-5 text-[#6e7875]">
      {children}
    </p>
  )
}

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & { tone?: 'primary' | 'secondary' | 'danger' }

export function Button({ tone = 'secondary', className = '', type = 'button', ...rest }: ButtonProps) {
  const tones: Record<string, string> = {
    primary: 'bg-ink text-acid border-ink hover:bg-[#1d2c29] disabled:bg-[#8c9a96] disabled:text-white disabled:border-[#8c9a96]',
    secondary: 'bg-white text-ink border-[#bfc3bb] hover:bg-sand disabled:text-[#8c9a96] disabled:hover:bg-white',
    danger: 'bg-white text-ember border-ember hover:bg-[#fff1ea] disabled:text-[#8c9a96] disabled:border-[#bfc3bb]',
  }
  return (
    <button
      type={type}
      className={`inline-flex items-center gap-2 border px-3 py-2 font-mono text-[10px] font-bold uppercase tracking-widest disabled:cursor-not-allowed ${tones[tone]} ${FOCUS} ${className}`}
      {...rest}
    />
  )
}

export function Card({ title, aside, children, id, testId }: { title: string; aside?: ReactNode; children: ReactNode; id?: string; testId?: string }) {
  return (
    <section id={id} data-testid={testId} aria-labelledby={id ? `${id}-title` : undefined} className="border border-[#c5c7c0] bg-[#f8f7f2] p-5 md:p-6">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <h2 id={id ? `${id}-title` : undefined} className="font-mono text-[10px] font-bold uppercase tracking-widest text-moss">
          {title}
        </h2>
        {aside}
      </div>
      {children}
    </section>
  )
}

/** A collapsible panel with an explicit aria-expanded toggle button. */
export function Expandable({ id, title, open, onToggle, aside, children, testId }: { id: string; title: string; open: boolean; onToggle: () => void; aside?: ReactNode; children: ReactNode; testId?: string }) {
  return (
    <section data-testid={testId} className="border border-[#c5c7c0] bg-[#f8f7f2]">
      <div className="flex flex-wrap items-center justify-between gap-3 p-4 md:px-6">
        <button type="button" aria-expanded={open} aria-controls={`${id}-body`} onClick={onToggle} className={`flex items-center gap-3 font-mono text-[10px] font-bold uppercase tracking-widest text-ink ${FOCUS}`}>
          <span aria-hidden="true" className="inline-block w-3">
            {open ? '−' : '+'}
          </span>
          {title}
        </button>
        {aside}
      </div>
      {open && (
        <div id={`${id}-body`} className="border-t border-[#d7d7d0] p-4 md:p-6">
          {children}
        </div>
      )}
    </section>
  )
}

export function Badge({ children, tone = 'neutral', testId }: { children: ReactNode; tone?: 'neutral' | 'saved' | 'warn'; testId?: string }) {
  const cls = tone === 'saved' ? 'border-moss text-moss' : tone === 'warn' ? 'border-ember text-ember' : 'border-[#bfc3bb] text-[#5c6966]'
  return (
    <span data-testid={testId} className={`inline-block border px-2 py-1 font-mono text-[9px] font-bold uppercase tracking-widest ${cls}`}>
      {children}
    </span>
  )
}

export function Dl({ items }: { items: Array<[string, ReactNode]> }) {
  return (
    <dl className="grid gap-px bg-[#d7d7d0] sm:grid-cols-2">
      {items.map(([k, v]) => (
        <div key={k} className="bg-white p-3">
          <dt className="font-mono text-[9px] uppercase tracking-widest text-[#6e7875]">{k}</dt>
          <dd className="mt-1 break-words text-xs text-ink">{v ?? '—'}</dd>
        </div>
      ))}
    </dl>
  )
}

export function Hash({ value, length = 12 }: { value: string | null | undefined; length?: number }) {
  if (!value) return <span>—</span>
  return (
    <code title={value} className="font-mono text-[11px] text-[#5c6966]">
      {value.slice(0, length)}
    </code>
  )
}

export function Alert({ title, children, testId }: { title: string; children?: ReactNode; testId?: string }) {
  return (
    <div role="alert" data-testid={testId} className="border border-ember bg-[#fff1ea] p-4">
      <p className="font-mono text-[10px] uppercase tracking-widest text-ember">{title}</p>
      {children && <div className="mt-2 text-sm text-ink">{children}</div>}
    </div>
  )
}

export function Notice({ children, testId }: { children: ReactNode; testId?: string }) {
  return (
    <div role="status" aria-live="polite" data-testid={testId} className="border border-dashed border-[#bfc3bb] bg-white p-4 text-sm text-[#52605d]">
      {children}
    </div>
  )
}
