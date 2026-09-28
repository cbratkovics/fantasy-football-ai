/**
 * Fail-closed browser loader for an exported Decision Lab bundle — the mirror of
 * `ffai/decision_lab/bundle.py` over `fetch` instead of a directory.
 *
 * A bundle is `manifest.json` plus the files it lists (`inputs/*.json`, `outcomes/*.json`,
 * `cases.json`) served under one same-origin base URL (default `/decision-lab`). Every file is
 * verified on load: byte digest of the raw text, content digest (canonical JSON), row count,
 * schema, semantic checks, and cross-file compatibility (an outcome snapshot must name the content
 * digest of its inputs snapshot). Any mismatch resolves to `{ ok: false, evidence_status:
 * 'corrupt', ... }`; nothing is substituted, defaulted, or partially returned.
 *
 * Only same-origin relative URLs are fetched: a base URL or path containing `://` or starting with
 * `//` is rejected before any request, as are paths that start with `/` or contain `..` segments.
 *
 * Byte digests are computed over the UTF-8 encoding of the response text, which equals the file
 * bytes for the exporter's output (UTF-8 without a byte-order mark).
 */
import { contentId } from './canonical';
import { sha256 } from './sha256';
import type { CasesFile, InputsSnapshot, LabManifest, OutcomeSnapshot } from './types';
import {
  ContractError,
  validateCasesShape,
  validateInputsSnapshotShape,
  validateManifestShape,
  validateOutcomeSnapshotShape,
} from './validate';

/** Default base URL the Next.js app serves the bundle from (`public/decision-lab`). */
export const DEFAULT_BASE_URL = '/decision-lab';
/** File names inside a bundle. */
export const MANIFEST_NAME = 'manifest.json';

/** The subset of `fetch` the loader needs; lets tests inject {@link createMemoryFetch}. */
export type FetchLike = (url: string) => Promise<{
  ok: boolean;
  status: number;
  text(): Promise<string>;
}>;

/** Why a load failed; `evidence_status` is always `corrupt` so the UI can pass it straight into `snapshotRef`. */
export interface BundleFailure {
  ok: false;
  /** Machine-readable category. */
  kind: 'url' | 'unavailable' | 'json' | 'byte_digest' | 'content_digest' | 'rows' | 'shape' | 'identity' | 'compat' | 'unknown_snapshot';
  /** Short message (safe to show). */
  error: string;
  /** Longer detail naming the path and the mismatch; suitable as `evidence_detail`. */
  detail: string;
  evidence_status: 'corrupt';
}

export type ManifestLoad = { ok: true; manifest: LabManifest } | BundleFailure;
export type InputsLoad =
  | { ok: true; snapshot: InputsSnapshot; content_sha256: string; file_sha256: string }
  | BundleFailure;
export type OutcomeLoad =
  | { ok: true; snapshot: OutcomeSnapshot; outcome_id: string; file_sha256: string }
  | { ok: true; snapshot: null; outcome_id: null; file_sha256: null }
  | BundleFailure;
export type CasesLoad = { ok: true; cases: CasesFile; content_sha256: string } | BundleFailure;

/** Options shared by the loaders. */
export interface LoadOptions {
  baseUrl?: string;
  fetchImpl?: FetchLike;
}

function failure(kind: BundleFailure['kind'], error: string, detail: string): BundleFailure {
  return { ok: false, kind, error, detail, evidence_status: 'corrupt' };
}

function unsafe(part: string): boolean {
  return part.indexOf('://') >= 0 || part.indexOf('//') === 0 || part.indexOf('\\') >= 0;
}

/**
 * Join a base URL and a bundle-relative path into a same-origin URL. Throws on anything absolute,
 * protocol-relative, backslash-containing, root-relative (`/x`) or with `..` segments.
 */
export function safeJoin(baseUrl: string, path: string): string {
  if (unsafe(baseUrl)) {
    throw new Error(`refusing non-relative base URL ${JSON.stringify(baseUrl)}`);
  }
  if (unsafe(path) || path.indexOf('/') === 0 || path.split('/').indexOf('..') >= 0 || path === '') {
    throw new Error(`refusing unsafe bundle path ${JSON.stringify(path)}`);
  }
  const base = baseUrl.replace(/\/+$/, '');
  return `${base}/${path}`;
}

function defaultFetch(): FetchLike {
  if (typeof fetch !== 'function') {
    throw new Error('fetch is not available in this environment');
  }
  return (url: string) => fetch(url, { credentials: 'same-origin', cache: 'no-store' });
}

async function fetchText(
  path: string,
  options: LoadOptions,
): Promise<{ ok: true; text: string; url: string } | BundleFailure> {
  let url: string;
  try {
    url = safeJoin(options.baseUrl ?? DEFAULT_BASE_URL, path);
  } catch (e) {
    return failure('url', 'unsafe URL', e instanceof Error ? e.message : String(e));
  }
  try {
    const impl = options.fetchImpl ?? defaultFetch();
    const res = await impl(url);
    if (!res.ok) {
      return failure('unavailable', 'file unavailable', `${path}: HTTP ${res.status}`);
    }
    return { ok: true, text: await res.text(), url };
  } catch (e) {
    return failure('unavailable', 'fetch failed', `${path}: ${e instanceof Error ? e.message : String(e)}`);
  }
}

function parseJson(path: string, text: string): { ok: true; doc: unknown } | BundleFailure {
  try {
    return { ok: true, doc: JSON.parse(text) };
  } catch (e) {
    return failure('json', 'invalid JSON', `${path} is not valid JSON: ${e instanceof Error ? e.message : String(e)}`);
  }
}

interface FileRefLike {
  path: string;
  file_sha256: string;
  content_sha256: string;
  n_rows: number;
}

/** Fetch a listed file and verify byte digest, content digest and row count (Python `_verified`). */
async function verified(
  ref: FileRefLike,
  options: LoadOptions,
): Promise<{ ok: true; doc: Record<string, unknown>; file_sha256: string; content_sha256: string } | BundleFailure> {
  const fetched = await fetchText(ref.path, options);
  if (!fetched.ok) {
    return fetched;
  }
  const fileDigest = sha256(fetched.text);
  if (fileDigest !== ref.file_sha256) {
    return failure('byte_digest', 'byte digest mismatch', `${ref.path}: byte digest mismatch (corrupt or edited)`);
  }
  const parsed = parseJson(ref.path, fetched.text);
  if (!parsed.ok) {
    return parsed;
  }
  const doc = parsed.doc;
  if (doc === null || typeof doc !== 'object' || Array.isArray(doc)) {
    return failure('shape', 'not an object', `${ref.path}: top-level JSON value is not an object`);
  }
  const contentDigest = contentId(doc);
  if (contentDigest !== ref.content_sha256) {
    return failure('content_digest', 'content digest mismatch', `${ref.path}: content digest mismatch`);
  }
  const record = doc as Record<string, unknown>;
  const rows = record.rows !== undefined ? record.rows : record.cases;
  if (!Array.isArray(rows) || rows.length !== ref.n_rows) {
    const n = Array.isArray(rows) ? String(rows.length) : '?';
    return failure('rows', 'row count mismatch', `${ref.path}: row count ${n} != manifest ${ref.n_rows}`);
  }
  return { ok: true, doc: record, file_sha256: fileDigest, content_sha256: contentDigest };
}

function shapeFailure(path: string, e: unknown): BundleFailure {
  const msg = e instanceof Error ? e.message : String(e);
  return failure('shape', 'contract violation', `${path}: ${msg}`);
}

/** Fetch and validate `manifest.json`. */
export async function loadManifest(
  baseUrl: string = DEFAULT_BASE_URL,
  fetchImpl?: FetchLike,
): Promise<ManifestLoad> {
  const options: LoadOptions = { baseUrl, fetchImpl };
  const fetched = await fetchText(MANIFEST_NAME, options);
  if (!fetched.ok) {
    return fetched;
  }
  const parsed = parseJson(MANIFEST_NAME, fetched.text);
  if (!parsed.ok) {
    return parsed;
  }
  try {
    return { ok: true, manifest: validateManifestShape(parsed.doc) };
  } catch (e) {
    if (e instanceof ContractError) {
      return shapeFailure(MANIFEST_NAME, e);
    }
    throw e;
  }
}

/** Result of {@link policySpecDigestMatches}. */
export interface PolicySpecCheck {
  /** Whether a comparison happened at all. */
  checked: boolean;
  /** `true`/`false` when checked; `null` when skipped. */
  matches: boolean | null;
  /** Present when the check was skipped or failed. */
  warning?: string;
}

/**
 * Compare the manifest's `policy_spec.sha256` (digest of the exporter's `policy_spec.json` bytes)
 * with the digest of this build's `policy_spec.json`. The browser cannot hash the imported JSON
 * module byte-exactly, so the UI passes a digest constant produced at build time. When no digest
 * is supplied the check is skipped with a warning — the bundle may then have been exported with a
 * different spec than the one this code evaluates with.
 */
export function policySpecDigestMatches(
  manifest: Pick<LabManifest, 'policy_spec'>,
  expectedSha256?: string | null,
): PolicySpecCheck {
  if (!expectedSha256) {
    return {
      checked: false,
      matches: null,
      warning:
        'policy_spec.json digest not supplied; the bundle was not verified against this build\'s policy specification',
    };
  }
  const matches = manifest.policy_spec.sha256 === expectedSha256;
  return matches
    ? { checked: true, matches: true }
    : {
        checked: true,
        matches: false,
        warning: 'bundle was exported with a different policy_spec.json; rebuild it with this code',
      };
}

/** The manifest entry for a snapshot id, or null. */
export function snapshotEntry(manifest: LabManifest, snapshotId: string): LabManifest['snapshots'][number] | null {
  for (let i = 0; i < manifest.snapshots.length; i += 1) {
    if (manifest.snapshots[i].snapshot_id === snapshotId) {
      return manifest.snapshots[i];
    }
  }
  return null;
}

/**
 * Fetch, verify and validate one inputs snapshot. Verifies byte digest, content digest, row
 * count, schema + semantic checks, and that the file's identity matches the manifest entry.
 */
export async function loadInputsSnapshot(
  manifest: LabManifest,
  snapshotId: string,
  options: LoadOptions = {},
): Promise<InputsLoad> {
  const entry = snapshotEntry(manifest, snapshotId);
  if (!entry) {
    return failure('unknown_snapshot', 'unknown snapshot', `unknown snapshot ${snapshotId}`);
  }
  const v = await verified(entry.inputs, options);
  if (!v.ok) {
    return v;
  }
  let doc: InputsSnapshot;
  try {
    doc = validateInputsSnapshotShape(v.doc);
  } catch (e) {
    if (e instanceof ContractError) {
      return shapeFailure(entry.inputs.path, e);
    }
    throw e;
  }
  if (doc.snapshot_id !== snapshotId || doc.mode !== entry.mode) {
    return failure('identity', 'snapshot identity mismatch', `${entry.inputs.path}: snapshot identity differs from the manifest`);
  }
  return { ok: true, snapshot: doc, content_sha256: v.content_sha256, file_sha256: v.file_sha256 };
}

/**
 * Fetch, verify and validate the outcome snapshot paired with `snapshotId`, checking it against the
 * already-loaded inputs snapshot (same id, matching content digest, same period and model, no extra
 * players). Resolves to `{ ok: true, snapshot: null }` when the manifest lists no outcomes.
 */
export async function loadOutcomeSnapshot(
  manifest: LabManifest,
  snapshotId: string,
  inputs: InputsSnapshot,
  options: LoadOptions = {},
): Promise<OutcomeLoad> {
  const entry = snapshotEntry(manifest, snapshotId);
  if (!entry) {
    return failure('unknown_snapshot', 'unknown snapshot', `unknown snapshot ${snapshotId}`);
  }
  const ref = entry.outcomes;
  if (ref === null) {
    return { ok: true, snapshot: null, outcome_id: null, file_sha256: null };
  }
  const v = await verified(ref, options);
  if (!v.ok) {
    return v;
  }
  const coverage = v.doc.coverage as { observed?: unknown } | undefined;
  if (!coverage || coverage.observed !== ref.n_observed) {
    return failure('rows', 'observed count mismatch', `${ref.path}: observed count differs from the manifest`);
  }
  try {
    const doc = validateOutcomeSnapshotShape(v.doc, inputs);
    return { ok: true, snapshot: doc, outcome_id: v.content_sha256, file_sha256: v.file_sha256 };
  } catch (e) {
    if (e instanceof ContractError) {
      const f = shapeFailure(ref.path, e);
      return { ...f, kind: 'compat' };
    }
    throw e;
  }
}

/** Fetch, verify and validate `cases.json`; every case must reference a listed snapshot. */
export async function loadCases(manifest: LabManifest, options: LoadOptions = {}): Promise<CasesLoad> {
  const v = await verified(manifest.cases, options);
  if (!v.ok) {
    return v;
  }
  let doc: CasesFile;
  try {
    doc = validateCasesShape(v.doc);
  } catch (e) {
    if (e instanceof ContractError) {
      return shapeFailure(manifest.cases.path, e);
    }
    throw e;
  }
  for (let i = 0; i < doc.cases.length; i += 1) {
    const c = doc.cases[i];
    if (!snapshotEntry(manifest, c.snapshot_id)) {
      return failure('compat', 'case references unknown snapshot', `case ${c.case_id} references unknown snapshot ${c.snapshot_id}`);
    }
  }
  return { ok: true, cases: doc, content_sha256: v.content_sha256 };
}

/**
 * A `fetch` stand-in serving an in-memory map of bundle-relative path → file text (for tests).
 * Keys are relative to `baseUrl` (e.g. `manifest.json`, `inputs/syn.json`); unknown paths 404.
 */
export function createMemoryFetch(files: Record<string, string>, baseUrl: string = DEFAULT_BASE_URL): FetchLike {
  const base = baseUrl.replace(/\/+$/, '') + '/';
  return async (url: string) => {
    const rel = url.indexOf(base) === 0 ? url.slice(base.length) : url;
    if (Object.prototype.hasOwnProperty.call(files, rel)) {
      const text = files[rel];
      return { ok: true, status: 200, text: async () => text };
    }
    return { ok: false, status: 404, text: async () => '' };
  };
}
