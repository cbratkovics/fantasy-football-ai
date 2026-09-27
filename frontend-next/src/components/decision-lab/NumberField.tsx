'use client'

import { numberError } from './draft'
import { FOCUS, Help, MicroLabel } from './Bits'

interface NumberFieldProps {
  id: string
  label: string
  help: string
  value: string
  onChange: (raw: string) => void
  min: number
  max: number
  step: number
  disabled?: boolean
}

/**
 * A numeric input that never yields NaN into state: the raw string is kept as typed, the parent
 * parses it when it computes, and an inline error (also reported through aria-invalid) marks
 * anything outside the allowed range.
 */
export function NumberField({ id, label, help, value, onChange, min, max, step, disabled }: NumberFieldProps) {
  const error = numberError(value, min, max)
  return (
    <div className="flex flex-col gap-1">
      <MicroLabel htmlFor={id}>{label}</MicroLabel>
      <input
        id={id}
        type="number"
        inputMode="decimal"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        aria-describedby={`${id}-help${error ? ` ${id}-error` : ''}`}
        aria-invalid={error ? true : undefined}
        onChange={(e) => onChange(e.target.value)}
        className={`h-10 w-full border bg-white px-3 font-mono text-xs text-ink disabled:bg-sand disabled:text-[#6e7875] ${error ? 'border-ember' : 'border-[#bfc3bb]'} ${FOCUS}`}
      />
      <Help id={`${id}-help`}>{help}</Help>
      {error && (
        <p id={`${id}-error`} role="alert" className="font-mono text-[10px] text-ember">
          {error}
        </p>
      )}
    </div>
  )
}
