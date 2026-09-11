'use client'

interface SegmentedProps<T extends string> {
  label: string
  options: readonly T[]
  value: T
  onChange: (value: T) => void
  render?: (value: T) => string
}

export function Segmented<T extends string>({ label, options, value, onChange, render }: SegmentedProps<T>) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
      <small className="font-mono text-[9px] font-bold uppercase tracking-widest">{label}</small>
      <div role="group" aria-label={label} className="flex overflow-x-auto border border-[#bfc3bb]">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            className={`min-w-[56px] border-r border-[#bfc3bb] px-3 py-2 font-mono text-[10px] font-bold uppercase last:border-r-0 ${
              option === value ? 'bg-ink text-acid' : 'text-[#5c6966] hover:bg-sand'
            }`}
          >
            {render ? render(option) : option}
          </button>
        ))}
      </div>
    </div>
  )
}

export function TextInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={`h-10 w-full border border-[#bfc3bb] bg-white px-3 font-mono text-xs text-ink placeholder:text-[#8c9a96] focus:border-ink focus:outline-none ${props.className ?? ''}`}
    />
  )
}
