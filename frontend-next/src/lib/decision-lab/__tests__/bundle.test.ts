import { describe, expect, it } from 'vitest';

import {
  createMemoryFetch,
  loadCases,
  loadInputsSnapshot,
  loadManifest,
  loadOutcomeSnapshot,
  policySpecDigestMatches,
  safeJoin,
} from '../bundle';
import { contentId } from '../canonical';
import { sha256 } from '../sha256';
import type { LabManifest } from '../types';
import { fileText, syntheticBundle } from './fixtures';

async function manifestFrom(files: Record<string, string>): Promise<LabManifest> {
  const m = await loadManifest('/decision-lab', createMemoryFetch(files));
  if (!m.ok) throw new Error(m.detail);
  return m.manifest;
}

describe('bundle loader happy path', () => {
  it('loads and verifies manifest, inputs, outcomes and cases', async () => {
    const b = syntheticBundle();
    const fetchImpl = createMemoryFetch(b.files);
    const opts = { baseUrl: '/decision-lab', fetchImpl };
    const manifest = await manifestFrom(b.files);
    expect(manifest).toEqual(b.manifest);

    const inputs = await loadInputsSnapshot(manifest, 'syn-bundle', opts);
    expect(inputs.ok).toBe(true);
    if (!inputs.ok) return;
    expect(inputs.snapshot).toEqual(b.inputs);
    expect(inputs.content_sha256).toBe(contentId(b.inputs));
    expect(inputs.file_sha256).toBe(sha256(b.files['inputs/syn-bundle.json']));

    const outcomes = await loadOutcomeSnapshot(manifest, 'syn-bundle', inputs.snapshot, opts);
    expect(outcomes.ok).toBe(true);
    if (!outcomes.ok) return;
    expect(outcomes.snapshot).toEqual(b.outcomes);
    expect(outcomes.outcome_id).toBe(contentId(b.outcomes));

    const cases = await loadCases(manifest, opts);
    expect(cases.ok).toBe(true);
    if (!cases.ok) return;
    expect(cases.cases).toEqual(b.cases);
  });

  it('returns snapshot null when the manifest lists no outcomes', async () => {
    const b = syntheticBundle();
    const manifest = { ...b.manifest, snapshots: [{ ...b.manifest.snapshots[0], outcomes: null }] };
    const files = { ...b.files, 'manifest.json': fileText(manifest) };
    const m = await manifestFrom(files);
    const opts = { fetchImpl: createMemoryFetch(files) };
    const inputs = await loadInputsSnapshot(m, 'syn-bundle', opts);
    if (!inputs.ok) throw new Error(inputs.detail);
    const out = await loadOutcomeSnapshot(m, 'syn-bundle', inputs.snapshot, opts);
    expect(out).toEqual({ ok: true, snapshot: null, outcome_id: null, file_sha256: null });
  });

  it('reports unknown snapshot ids and missing files without throwing', async () => {
    const b = syntheticBundle();
    const opts = { fetchImpl: createMemoryFetch({ 'manifest.json': b.files['manifest.json'] }) };
    const m = await manifestFrom(b.files);
    const unknown = await loadInputsSnapshot(m, 'nope', opts);
    expect(unknown).toMatchObject({ ok: false, kind: 'unknown_snapshot', evidence_status: 'corrupt' });
    const missing = await loadInputsSnapshot(m, 'syn-bundle', opts);
    expect(missing).toMatchObject({ ok: false, kind: 'unavailable', evidence_status: 'corrupt' });
    expect((missing as { detail: string }).detail).toMatch(/HTTP 404/);
  });
});

describe('bundle loader fails closed', () => {
  it('byte tamper (whitespace only) → corrupt with byte digest mismatch', async () => {
    const b = syntheticBundle();
    const files = { ...b.files, 'inputs/syn-bundle.json': b.files['inputs/syn-bundle.json'].replace('\n', '\n\n') };
    const m = await manifestFrom(files);
    const r = await loadInputsSnapshot(m, 'syn-bundle', { fetchImpl: createMemoryFetch(files) });
    expect(r).toEqual({
      ok: false,
      kind: 'byte_digest',
      error: 'byte digest mismatch',
      detail: 'inputs/syn-bundle.json: byte digest mismatch (corrupt or edited)',
      evidence_status: 'corrupt',
    });
  });

  it('content tamper with a matching byte digest → content digest mismatch', async () => {
    const b = syntheticBundle();
    const edited = JSON.parse(b.files['inputs/syn-bundle.json']);
    edited.rows[0].projection = 40;
    const text = fileText(edited);
    const manifest = JSON.parse(b.files['manifest.json']) as LabManifest;
    manifest.snapshots[0].inputs.file_sha256 = sha256(text);
    const files = { ...b.files, 'inputs/syn-bundle.json': text, 'manifest.json': fileText(manifest) };
    const m = await manifestFrom(files);
    const r = await loadInputsSnapshot(m, 'syn-bundle', { fetchImpl: createMemoryFetch(files) });
    expect(r).toMatchObject({ ok: false, kind: 'content_digest', evidence_status: 'corrupt' });
  });

  it('row count mismatch → corrupt', async () => {
    const b = syntheticBundle();
    const manifest = JSON.parse(b.files['manifest.json']) as LabManifest;
    manifest.snapshots[0].inputs.n_rows = 2;
    const files = { ...b.files, 'manifest.json': fileText(manifest) };
    const m = await manifestFrom(files);
    const r = await loadInputsSnapshot(m, 'syn-bundle', { fetchImpl: createMemoryFetch(files) });
    expect(r).toMatchObject({ ok: false, kind: 'rows', detail: 'inputs/syn-bundle.json: row count 3 != manifest 2' });
  });

  it('contract violation inside a verified file → corrupt', async () => {
    const b = syntheticBundle((d) => {
      d.inputs.rows[0].floor = 99; // floor above projection: digests are computed after the edit
    });
    const m = await manifestFrom(b.files);
    const r = await loadInputsSnapshot(m, 'syn-bundle', { fetchImpl: createMemoryFetch(b.files) });
    expect(r).toMatchObject({ ok: false, kind: 'shape', evidence_status: 'corrupt' });
    expect((r as { detail: string }).detail).toMatch(/floor above projection/);
  });

  it('snapshot identity mismatch → corrupt', async () => {
    const b = syntheticBundle();
    const manifest = JSON.parse(b.files['manifest.json']) as LabManifest;
    manifest.snapshots[0].mode = 'historical_replay';
    manifest.snapshots[0].source_family = 'frozen_test';
    const files = { ...b.files, 'manifest.json': fileText(manifest) };
    const m = await manifestFrom(files);
    const r = await loadInputsSnapshot(m, 'syn-bundle', { fetchImpl: createMemoryFetch(files) });
    expect(r).toMatchObject({ ok: false, kind: 'identity' });
  });

  it('outcome digest mismatch against the inputs snapshot → corrupt', async () => {
    const b = syntheticBundle((d) => {
      d.outcomes.inputs_content_sha256 = '0'.repeat(64);
    });
    const opts = { fetchImpl: createMemoryFetch(b.files) };
    const m = await manifestFrom(b.files);
    const inputs = await loadInputsSnapshot(m, 'syn-bundle', opts);
    if (!inputs.ok) throw new Error(inputs.detail);
    const r = await loadOutcomeSnapshot(m, 'syn-bundle', inputs.snapshot, opts);
    expect(r).toMatchObject({ ok: false, kind: 'compat', evidence_status: 'corrupt' });
    expect((r as { detail: string }).detail).toMatch(/inputs_content_sha256 does not match/);
  });

  it('outcome observed count mismatch → corrupt', async () => {
    const b = syntheticBundle();
    const manifest = JSON.parse(b.files['manifest.json']) as LabManifest;
    (manifest.snapshots[0].outcomes as { n_observed: number }).n_observed = 3;
    const files = { ...b.files, 'manifest.json': fileText(manifest) };
    const opts = { fetchImpl: createMemoryFetch(files) };
    const m = await manifestFrom(files);
    const inputs = await loadInputsSnapshot(m, 'syn-bundle', opts);
    if (!inputs.ok) throw new Error(inputs.detail);
    const r = await loadOutcomeSnapshot(m, 'syn-bundle', inputs.snapshot, opts);
    expect(r).toMatchObject({ ok: false, kind: 'rows' });
  });

  it('cases referencing an unknown snapshot → corrupt', async () => {
    const b = syntheticBundle((d) => {
      d.cases.cases[0].snapshot_id = 'elsewhere';
    });
    const m = await manifestFrom(b.files);
    const r = await loadCases(m, { fetchImpl: createMemoryFetch(b.files) });
    expect(r).toMatchObject({ ok: false, kind: 'compat' });
  });

  it('invalid manifest JSON or shape → corrupt', async () => {
    const b = syntheticBundle();
    const bad = await loadManifest('/decision-lab', createMemoryFetch({ 'manifest.json': '{not json' }));
    expect(bad).toMatchObject({ ok: false, kind: 'json' });
    const manifest = JSON.parse(b.files['manifest.json']) as Record<string, unknown>;
    manifest.surprise = 1;
    const shape = await loadManifest('/decision-lab', createMemoryFetch({ 'manifest.json': fileText(manifest) }));
    expect(shape).toMatchObject({ ok: false, kind: 'shape' });
  });

  it('absolute and protocol-relative URLs are refused before any fetch', async () => {
    const b = syntheticBundle();
    let calls = 0;
    const spy = async (url: string) => {
      calls += 1;
      return createMemoryFetch(b.files)(url);
    };
    expect(await loadManifest('https://example.com/decision-lab', spy)).toMatchObject({ ok: false, kind: 'url' });
    expect(await loadManifest('//example.com/decision-lab', spy)).toMatchObject({ ok: false, kind: 'url' });
    expect(calls).toBe(0);

    const manifest = JSON.parse(b.files['manifest.json']) as LabManifest;
    manifest.snapshots[0].inputs.path = 'https://example.com/x.json';
    const files = { ...b.files, 'manifest.json': fileText(manifest) };
    const m = await manifestFrom(files);
    const r = await loadInputsSnapshot(m, 'syn-bundle', { fetchImpl: spy });
    expect(r).toMatchObject({ ok: false, kind: 'url' });
    expect(calls).toBe(0); // refused before any request

    expect(() => safeJoin('/decision-lab', '../secrets.json')).toThrow();
    expect(() => safeJoin('/decision-lab', '/etc/passwd')).toThrow();
    expect(() => safeJoin('/decision-lab', '//evil/x.json')).toThrow();
    expect(safeJoin('/decision-lab/', 'inputs/a.json')).toBe('/decision-lab/inputs/a.json');
    expect(safeJoin('decision-lab', 'cases.json')).toBe('decision-lab/cases.json');
  });
});

describe('policySpecDigestMatches', () => {
  it('skips with a warning when no digest is supplied, otherwise compares', () => {
    const b = syntheticBundle();
    expect(policySpecDigestMatches(b.manifest)).toMatchObject({ checked: false, matches: null });
    expect(policySpecDigestMatches(b.manifest, null).warning).toMatch(/not supplied/);
    expect(policySpecDigestMatches(b.manifest, 'f'.repeat(64))).toEqual({ checked: true, matches: true });
    expect(policySpecDigestMatches(b.manifest, 'e'.repeat(64))).toMatchObject({ checked: true, matches: false });
  });
});
