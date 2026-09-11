/** Consistent building blocks for the text-only routes. */

export function Section({ number, title, children }: { number: string; title: string; children: React.ReactNode }) {
  return (
    <section className="grid gap-6 border-t border-[#c5c7c0] py-10 md:grid-cols-[180px_1fr]">
      <div className="flex items-start gap-3">
        <span className="section-number">{number}</span>
        <h2 className="font-mono text-[10px] font-bold uppercase leading-[34px] tracking-widest">{title}</h2>
      </div>
      <div className="space-y-4 text-[15px] leading-7 text-[#3b4a46]">{children}</div>
    </section>
  )
}

export function Cards({ items }: { items: Array<{ title: string; text: string }> }) {
  return (
    <div className="grid gap-px border border-[#bec2b8] bg-[#bec2b8] sm:grid-cols-2">
      {items.map((item) => (
        <article key={item.title} className="bg-[#f8f7f2] p-5">
          <b className="block text-sm">{item.title}</b>
          <p className="mt-2 text-xs leading-6 text-[#63706d]">{item.text}</p>
        </article>
      ))}
    </div>
  )
}

export function Faq({ items }: { items: Array<{ q: string; a: React.ReactNode }> }) {
  return (
    <dl className="divide-y divide-[#d7d7d0] border border-[#c5c7c0] bg-white">
      {items.map((item) => (
        <div key={item.q} className="p-5">
          <dt className="font-semibold text-ink">{item.q}</dt>
          <dd className="mt-2 text-sm leading-6 text-[#52605d]">{item.a}</dd>
        </div>
      ))}
    </dl>
  )
}
