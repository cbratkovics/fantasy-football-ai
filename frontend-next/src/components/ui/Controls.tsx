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
      <span className="text-sm font-semibold">{label}</span>
      <div role="group" aria-label={label} className="flex flex-wrap gap-1 rounded-lg border border-[#c4ccc5] p-1">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            aria-pressed={option === value}
            className={`min-h-[44px] min-w-[48px] rounded-md px-3 py-2 text-sm font-semibold focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#176e5a] ${
              option === value ? 'bg-[#174b3d] text-white' : 'text-[#374b42] hover:bg-white'
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
      className={`product-field w-full ${props.className ?? ''}`}
    />
  )
}
