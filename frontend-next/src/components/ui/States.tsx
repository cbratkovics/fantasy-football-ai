import { describeError } from '@/lib/api/client'

export function LoadingState({ label = 'Loading from the API…' }: { label?: string }) {
  return (
    <div role="status" className="border border-[#c5c7c0] bg-[#f8f7f2] p-8 font-mono text-xs uppercase tracking-widest text-[#61706c]">
      <span className="mr-3 inline-block h-2 w-2 animate-pulse rounded-full bg-ember align-middle" />
      {label}
    </div>
  )
}

export function ErrorState({ error, context }: { error: unknown; context?: string }) {
  return (
    <div role="alert" className="border border-ember bg-[#fff1ea] p-8">
      <p className="font-mono text-[10px] uppercase tracking-widest text-ember">Request failed</p>
      <p className="mt-2 text-sm text-ink">
        {context ? `${context}: ` : ''}
        {describeError(error)}
      </p>
      <p className="mt-3 text-xs text-[#6e7875]">Nothing is shown in place of missing data. Check that the API is reachable and that the requested artifact exists.</p>
    </div>
  )
}

export function EmptyState({ message }: { message: string }) {
  return <div className="border border-dashed border-[#c5c7c0] p-8 text-center text-sm text-[#6e7875]">{message}</div>
}
