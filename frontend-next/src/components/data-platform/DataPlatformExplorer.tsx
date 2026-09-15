'use client'

import { useEffect, useMemo, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import metadata from '@/data/dbt-showcase.json'
import { dbtDoc, repoFile } from '@/content/data-platform'

type Node = (typeof metadata.nodes)[number]

export function DataPlatformExplorer() {
  const params = useSearchParams(); const router = useRouter()
  const requested = params.get('model')
  const [query, setQuery] = useState('')
  const initial = metadata.nodes.find((n) => n.uniqueId === requested) ?? metadata.nodes.find((n) => n.name === 'fct_weekly_eval')!
  const [selectedId, setSelectedId] = useState(initial.uniqueId)
  useEffect(() => {
    if (requested && metadata.nodes.some((node) => node.uniqueId === requested)) setSelectedId(requested)
  }, [requested])
  const matches = useMemo(() => metadata.nodes.filter((n) => `${n.name} ${n.layer} ${n.description}`.toLowerCase().includes(query.toLowerCase())), [query])
  const selected: Node = metadata.nodes.find((n) => n.uniqueId === selectedId) ?? initial
  function select(id: string) { setSelectedId(id); router.replace(`/data-platform?model=${encodeURIComponent(id)}#model-inspector`, { scroll: false }) }
  return <section id="model-inspector" className="case-section scroll-mt-24">
    <div className="section-heading compact"><div><span className="section-number">04</span><p>MODEL INSPECTOR</p></div><h2>Inspect the contract, not just the model name.</h2><p>{metadata.inventory.executableModelNodes} executable nodes represent {metadata.inventory.logicalModelNames} logical models; gold has {metadata.inventory.layers.gold} nodes across nine logical families because both policy versions count separately. One snapshot is listed separately.</p></div>
    <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
      <aside className="border border-[#d1d3cd] bg-white p-4">
        <label className="block text-xs font-bold uppercase tracking-widest" htmlFor="model-search">Search project models</label>
        <input id="model-search" className="mt-2 w-full border-[#9da8a4]" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="weekly eval, gold…" />
        <div className="mt-3 max-h-[560px] overflow-y-auto" role="listbox" aria-label="dbt models">
          {matches.map((n) => <button key={n.uniqueId} role="option" aria-selected={n.uniqueId === selected.uniqueId} onClick={() => select(n.uniqueId)} className={`block w-full border-t px-2 py-3 text-left ${n.uniqueId === selected.uniqueId ? 'bg-[#e8efe9]' : 'bg-white'}`}><b className="font-mono text-xs">{n.name}{n.version ? ` v${n.version}` : ''}</b><small className="block uppercase text-[#61706c]">{n.layer} · {n.materialization}</small></button>)}
          {!matches.length && <p className="py-6 text-sm">No model matches this search.</p>}
        </div>
      </aside>
      <article className="min-w-0 border border-[#d1d3cd] bg-white p-5 md:p-7" aria-live="polite">
        <div className="flex flex-wrap items-center gap-2"><span className="rounded bg-[#152e27] px-2 py-1 text-xs text-white">{selected.layer}</span><span className="font-mono text-xs">{selected.uniqueId}</span></div>
        <h3 className="mt-4 text-2xl font-bold">{selected.name}{selected.version ? ` · version ${selected.version}` : ''}</h3>
        <p className="mt-2">{selected.description || 'Purpose is documented in the linked SQL and schema contract.'}</p>
        <dl className="mt-5 grid gap-3 sm:grid-cols-2"><div><dt>Full grain</dt><dd>{selected.grain}</dd></div><div><dt>Relation alias</dt><dd><code>{selected.alias}</code></dd></div><div><dt>Resource / materialization</dt><dd>{selected.resourceType} / {selected.materialization}</dd></div><div><dt>Export / consumer</dt><dd>{selected.exported ? 'Exported' : 'Not exported'} · {selected.consumer}</dd></div></dl>
        <h4 className="mt-6 font-bold">Direct transformation dependencies</h4><p className="font-mono text-sm">{selected.upstream.join(' · ') || 'source-backed / none'}</p>
        <h4 className="mt-6 font-bold">SQL excerpt</h4><pre className="mt-2 max-h-80 overflow-auto bg-[#10251f] p-4 text-xs leading-relaxed text-[#e6eee8]"><code>{selected.sqlExcerpt}</code></pre>
        <details className="mt-5"><summary className="cursor-pointer font-bold">Documented columns ({selected.columns.length})</summary><div className="mt-3 overflow-x-auto"><table className="min-w-[560px] text-left text-sm"><thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead><tbody>{selected.columns.map((c) => <tr className="border-t" key={c.name}><td className="py-2 pr-3 font-mono">{c.name}</td><td className="pr-3">{'data_type' in c ? c.data_type : '—'}</td><td>{c.description}</td></tr>)}</tbody></table></div></details>
        <div className="mt-6 flex flex-wrap gap-3"><a className="portfolio-button secondary" href={repoFile(selected.sqlPath)} target="_blank" rel="noreferrer">Open SQL</a><a className="portfolio-button secondary" href={repoFile(selected.yamlPath)} target="_blank" rel="noreferrer">Open contract</a><a className="portfolio-button secondary" href={dbtDoc(selected.uniqueId)} target="_blank" rel="noreferrer">Open dbt docs</a></div>
      </article>
    </div>
  </section>
}
