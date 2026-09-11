import { fmtUtc } from '@/lib/format'

interface ProvenanceProps {
  items: Array<[label: string, value: React.ReactNode]>
  note?: string
}

/** Compact key/value strip that names the artifact behind a view. */
export function Provenance({ items, note }: ProvenanceProps) {
  return (
    <div className="border border-[#c5c7c0] bg-ink text-[#dbe5e1]">
      <dl className="grid grid-cols-2 gap-px bg-[#2e3b38] md:grid-cols-4">
        {items.map(([label, value]) => (
          <div key={label} className="bg-ink p-4">
            <dt className="font-mono text-[9px] uppercase tracking-widest text-[#84918d]">{label}</dt>
            <dd className="mt-1 break-all font-mono text-xs text-white">{value ?? '—'}</dd>
          </div>
        ))}
      </dl>
      {note && <p className="border-t border-[#2e3b38] px-4 py-3 font-mono text-[10px] leading-relaxed text-[#9fb0aa]">{note}</p>}
    </div>
  )
}

export function generatedLine(modelVersion: string, featureVersion: string, season: number | undefined, week: number | undefined, generatedAt: string) {
  return `Model ${modelVersion} · features ${featureVersion} · data through season ${season ?? '—'} week ${week ?? '—'} · generated ${fmtUtc(generatedAt)}`
}
