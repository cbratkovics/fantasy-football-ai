'use client'

import { DECISION_LAB_COPY } from '@/content/decision-lab'
import type { PolicySpecCheck } from '@/lib/decision-lab/bundle'
import { evidenceStatusLabel } from '@/lib/decision-lab/explain'
import { TRUST_NOTE } from '@/lib/decision-lab/receipts'
import type { InputsSnapshot, LabManifest, ManifestSnapshot, Receipt } from '@/lib/decision-lab/types'
import { fmtUtc } from '@/lib/format'
import { Dl, Expandable, Hash, Notice } from './Bits'

const COPY = DECISION_LAB_COPY.evidence

interface EvidencePanelProps {
  manifest: LabManifest
  entry: ManifestSnapshot | null
  snapshot: InputsSnapshot | null
  receipt: Receipt | null
  specCheck: PolicySpecCheck
  build: Record<string, unknown> | null
  open: boolean
  onToggle: () => void
}

function str(v: unknown): string {
  return v === null || v === undefined ? '—' : String(v)
}

/** Everything that identifies the evidence behind the current decision; hashes are secondary text. */
export function EvidencePanel({ manifest, entry, snapshot, receipt, specCheck, build, open, onToggle }: EvidencePanelProps) {
  return (
    <Expandable id="evidence" title={COPY.title} open={open} onToggle={onToggle} testId="evidence-panel">
      {!snapshot || !entry ? (
        <Notice>{COPY.noSnapshot}</Notice>
      ) : (
        <div className="space-y-4">
          <Dl
            items={[
              [COPY.snapshot, <code key="s" className="font-mono text-[11px]">{snapshot.snapshot_id}</code>],
              [COPY.mode, `${DECISION_LAB_COPY.contexts[snapshot.mode].label} (${snapshot.mode})`],
              [COPY.family, snapshot.source_family],
              [COPY.period, `${snapshot.season} / week ${snapshot.week}`],
              [COPY.model, snapshot.model_version],
              [COPY.candidates, Object.keys(snapshot.candidate_by_position).sort().map((k) => `${k}: ${snapshot.candidate_by_position[k]}`).join(' · ')],
              [COPY.features, str(snapshot.feature_version)],
              [COPY.cutoff, snapshot.data_cutoff ? `${snapshot.data_cutoff.season} / week ${snapshot.data_cutoff.week}` : '—'],
              [COPY.generated, fmtUtc(snapshot.generated_at_utc)],
              [COPY.publication, `${snapshot.publication.status}${snapshot.publication.run_id ? ` · run ${snapshot.publication.run_id}` : ''}${snapshot.publication.action ? ` · ${snapshot.publication.action}` : ''}`],
              [COPY.evidenceStatus, receipt ? `${evidenceStatusLabel(receipt.inputs.snapshot.evidence_status)}${receipt.inputs.snapshot.evidence_detail ? ` — ${receipt.inputs.snapshot.evidence_detail}` : ''}` : evidenceStatusLabel('verified')],
              [COPY.coverage, entry.outcomes ? `${entry.outcomes.n_observed} observed of ${entry.outcomes.n_rows} rows (paired outcome file)` : 'no outcome file published'],
            ]}
          />
          <div className="border border-[#d7d7d0] bg-white p-3 text-xs leading-5">
            <p className="font-mono text-[9px] uppercase tracking-widest text-[#6e7875]">{COPY.population}</p>
            <p className="mt-1">
              <b>{snapshot.population.conditioning}</b> · {snapshot.population.n_rows} rows. {snapshot.population.description}
            </p>
            {snapshot.population.exclusions.length > 0 && (
              <>
                <p className="mt-2 font-mono text-[9px] uppercase tracking-widest text-[#6e7875]">{COPY.exclusions}</p>
                <ul className="mt-1 list-disc pl-5">
                  {snapshot.population.exclusions.map((e) => (
                    <li key={e}>{e}</li>
                  ))}
                </ul>
              </>
            )}
          </div>
          <div className="border border-[#d7d7d0] bg-white p-3 text-xs leading-5">
            <p className="font-mono text-[9px] uppercase tracking-widest text-[#6e7875]">{COPY.baseline}</p>
            <p className="mt-1">
              <b>{snapshot.baseline.name}</b> · provenance basis {snapshot.baseline.provenance_basis} · reconciled {snapshot.baseline.reconciled === null ? 'not applicable' : snapshot.baseline.reconciled ? 'yes' : 'no'}
              {snapshot.baseline.reconciled_to ? ` to ${snapshot.baseline.reconciled_to}` : ''}
            </p>
            {snapshot.baseline.note && <p className="mt-1 text-[#52605d]">{snapshot.baseline.note}</p>}
            {entry.reconciliation && (
              <p className="mt-2">
                <span className="font-mono text-[9px] uppercase tracking-widest text-[#6e7875]">{COPY.reconciliation}</span> {entry.reconciliation.status}
                {entry.reconciliation.reference ? ` · reference ${entry.reconciliation.reference}` : ''}
                {entry.reconciliation.tolerance !== null ? ` · tolerance ${entry.reconciliation.tolerance}` : ''}
              </p>
            )}
          </div>
          <div className="border border-[#d7d7d0] bg-white p-3 text-xs leading-5">
            <p className="font-mono text-[9px] uppercase tracking-widest text-[#6e7875]">{COPY.sources}</p>
            {snapshot.source.files.length === 0 ? (
              <p className="mt-1 text-[#52605d]">none (synthetic)</p>
            ) : (
              <ul className="mt-1 space-y-1 font-mono text-[11px]">
                {snapshot.source.files.map((f) => (
                  <li key={`${f.role}-${f.path}`} className="break-all">
                    <span className="text-[#6e7875]">{f.role}</span> {f.path} <Hash value={f.sha256} />
                    {f.rows !== null && f.rows !== undefined ? ` · ${f.rows} rows` : ''}
                  </li>
                ))}
              </ul>
            )}
            {snapshot.source.mart_export && (
              <p className="mt-2 text-[#52605d]">
                mart export {fmtUtc(snapshot.source.mart_export.exported_at_utc)} · target {str(snapshot.source.mart_export.target)} · code commit <Hash value={snapshot.source.mart_export.code_commit} />
              </p>
            )}
          </div>
          <Dl
            items={[
              [COPY.inputsDigest, <Hash key="i" value={entry.inputs.content_sha256} length={16} />],
              [COPY.decision, receipt ? <Hash key="d" value={receipt.decision_id} length={16} /> : '—'],
              [COPY.policy, `${manifest.policy_version} · spec ${specCheck.checked ? (specCheck.matches ? 'digest matches this build' : 'digest differs from this build') : 'not checked'}`],
              [COPY.revision, <span key="r"><Hash value={manifest.code_revision.produced_at} /> — {manifest.code_revision.note}</span>],
              [COPY.build, build ? `built ${fmtUtc(str(build.built_at_utc))} · exporter ${str(build.exporter_version)} · python ${str(build.python_version)} · duckdb ${str(build.duckdb_version)}` : 'not available'],
            ]}
          />
          <p className="border border-dashed border-[#bfc3bb] bg-white p-3 text-[11px] leading-5 text-[#52605d]">
            <b>{COPY.trust}.</b> {TRUST_NOTE}
          </p>
        </div>
      )}
    </Expandable>
  )
}
