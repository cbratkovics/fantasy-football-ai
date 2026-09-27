/**
 * localStorage persistence for decision receipts.
 *
 * Receipts live under one key, `ffai.decision-lab/<receipt schema version>/receipts`, as a JSON
 * object `{ [decision_id]: Receipt }`. Everything read back is treated as untrusted text: each entry
 * is shape-validated and malformed entries are dropped and reported (never thrown); a disabled or
 * full storage degrades to an empty set with a `storage_unavailable` problem or a `quota` result.
 *
 * Import replays the policy for every receipt (`validateReceipt`) and merges idempotently through
 * `merge`; a conflicting receipt with the same id is rejected, never overwritten. Nothing here
 * evaluates code, follows URLs found in imported data, or interprets notes as anything but text.
 */
import { validateReceipt, merge, ConflictError } from './receipts';
import { utf8ByteLength } from './sha256';
import { SCHEMA_VERSIONS } from './spec';
import type { Receipt } from './types';
import { validateReceiptShape } from './validate';

/** The localStorage key receipts are stored under. */
export const STORAGE_KEY = `ffai.decision-lab/${SCHEMA_VERSIONS.receipt}/receipts`;
/** Schema version of the export envelope written by {@link exportAll}. */
export const EXPORT_SCHEMA_VERSION = 'decision_receipt_export/1.0';
/** Largest import text accepted (UTF-8 bytes). */
export const IMPORT_MAX_BYTES = 2 * 1024 * 1024;

/** A non-fatal problem found while reading storage. */
export interface StorageProblem {
  kind: 'storage_unavailable' | 'malformed_store' | 'malformed_entry';
  decision_id?: string;
  message: string;
}

export interface LoadedReceipts {
  receipts: Record<string, Receipt>;
  problems: StorageProblem[];
}

export type SaveResult =
  | { ok: true; receipt: Receipt; merged: boolean }
  | { ok: false; reason: 'quota' | 'storage_unavailable' | 'invalid' | 'conflict'; message: string };

export interface ImportRejection {
  decision_id?: string;
  reason: string;
}

export interface ImportReport {
  /** New receipts stored, plus existing receipts extended with more events. */
  imported: number;
  /** Receipts identical to (or a subset of) what was already stored. */
  duplicates: number;
  rejected: ImportRejection[];
  /** False when the merged set could not be written back (quota or unavailable storage). */
  persisted: boolean;
  storage_error?: string;
}

/** The export envelope shape. */
export interface ReceiptExport {
  schema_version: typeof EXPORT_SCHEMA_VERSION;
  exported_at_utc: string;
  receipts: Receipt[];
}

/** A minimal storage interface (window.localStorage or a test double). */
export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/** `window.localStorage` when usable, else null (SSR, disabled storage, throwing getter). */
export function getStorage(): StorageLike | null {
  try {
    if (typeof window === 'undefined') {
      return null;
    }
    const s = window.localStorage;
    if (!s) {
      return null;
    }
    // Some browsers expose the object but throw on use in private mode.
    const probe = `${STORAGE_KEY}/__probe__`;
    s.setItem(probe, '1');
    s.removeItem(probe);
    return s;
  } catch {
    return null;
  }
}

function isQuotaError(e: unknown): boolean {
  if (!e || typeof e !== 'object') {
    return false;
  }
  const err = e as { name?: string; code?: number };
  return (
    err.name === 'QuotaExceededError' ||
    err.name === 'NS_ERROR_DOM_QUOTA_REACHED' ||
    err.code === 22 ||
    err.code === 1014
  );
}

function readStore(storage: StorageLike): LoadedReceipts {
  const problems: StorageProblem[] = [];
  const receipts: Record<string, Receipt> = {};
  let raw: string | null;
  try {
    raw = storage.getItem(STORAGE_KEY);
  } catch (e) {
    return {
      receipts,
      problems: [{ kind: 'storage_unavailable', message: e instanceof Error ? e.message : String(e) }],
    };
  }
  if (raw === null || raw === '') {
    return { receipts, problems };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (e) {
    problems.push({ kind: 'malformed_store', message: `stored receipts are not valid JSON: ${e instanceof Error ? e.message : String(e)}` });
    return { receipts, problems };
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    problems.push({ kind: 'malformed_store', message: 'stored receipts are not an object' });
    return { receipts, problems };
  }
  const entries = parsed as Record<string, unknown>;
  const ids = Object.keys(entries);
  for (let i = 0; i < ids.length; i += 1) {
    const id = ids[i];
    try {
      const receipt = validateReceiptShape(entries[id]);
      if (receipt.decision_id !== id) {
        problems.push({ kind: 'malformed_entry', decision_id: id, message: 'stored under a different decision_id' });
        continue;
      }
      receipts[id] = receipt;
    } catch (e) {
      problems.push({ kind: 'malformed_entry', decision_id: id, message: e instanceof Error ? e.message : String(e) });
    }
  }
  return { receipts, problems };
}

function writeStore(storage: StorageLike, receipts: Record<string, Receipt>): { ok: true } | { ok: false; reason: 'quota' | 'storage_unavailable'; message: string } {
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(receipts));
    return { ok: true };
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return { ok: false, reason: isQuotaError(e) ? 'quota' : 'storage_unavailable', message };
  }
}

/** Every valid stored receipt plus the problems found; never throws. */
export function loadReceipts(storage: StorageLike | null = getStorage()): LoadedReceipts {
  if (!storage) {
    return { receipts: {}, problems: [{ kind: 'storage_unavailable', message: 'localStorage is not available' }] };
  }
  return readStore(storage);
}

/**
 * Validate and persist one receipt. An existing receipt with the same id is merged through
 * `merge` (identical or superset events); a conflicting one is refused.
 */
export function saveReceipt(receipt: Receipt, storage: StorageLike | null = getStorage()): SaveResult {
  if (!storage) {
    return { ok: false, reason: 'storage_unavailable', message: 'localStorage is not available' };
  }
  let valid: Receipt;
  try {
    valid = validateReceiptShape(receipt);
  } catch (e) {
    return { ok: false, reason: 'invalid', message: e instanceof Error ? e.message : String(e) };
  }
  const { receipts } = readStore(storage);
  let merged = false;
  let toStore = valid;
  if (Object.prototype.hasOwnProperty.call(receipts, valid.decision_id)) {
    try {
      toStore = merge(receipts[valid.decision_id], valid);
      merged = true;
    } catch (e) {
      if (e instanceof ConflictError) {
        return { ok: false, reason: 'conflict', message: e.message };
      }
      throw e;
    }
  }
  receipts[valid.decision_id] = toStore;
  const written = writeStore(storage, receipts);
  if (!written.ok) {
    return { ok: false, reason: written.reason, message: written.message };
  }
  return { ok: true, receipt: toStore, merged };
}

/** Remove one receipt; returns whether it existed. */
export function deleteReceipt(decisionId: string, storage: StorageLike | null = getStorage()): boolean {
  if (!storage) {
    return false;
  }
  const { receipts } = readStore(storage);
  if (!Object.prototype.hasOwnProperty.call(receipts, decisionId)) {
    return false;
  }
  delete receipts[decisionId];
  return writeStore(storage, receipts).ok;
}

/** Remove every stored receipt. */
export function clearAll(storage: StorageLike | null = getStorage()): void {
  if (!storage) {
    return;
  }
  try {
    storage.removeItem(STORAGE_KEY);
  } catch {
    // nothing to clear when storage is unusable
  }
}

/**
 * JSON text of every valid stored receipt, sorted by `created_at_utc` then `decision_id`, in a
 * `decision_receipt_export/1.0` envelope.
 */
export function exportAll(
  exportedAtUtc: string = new Date().toISOString(),
  storage: StorageLike | null = getStorage(),
): string {
  const { receipts } = loadReceipts(storage);
  const list = Object.keys(receipts)
    .map((id) => receipts[id])
    .sort((a, b) =>
      a.created_at_utc < b.created_at_utc
        ? -1
        : a.created_at_utc > b.created_at_utc
          ? 1
          : a.decision_id < b.decision_id
            ? -1
            : a.decision_id > b.decision_id
              ? 1
              : 0,
    );
  const envelope: ReceiptExport = {
    schema_version: EXPORT_SCHEMA_VERSION,
    exported_at_utc: exportedAtUtc,
    receipts: list,
  };
  return JSON.stringify(envelope, null, 1);
}

/**
 * Import receipts from export text (or a bare JSON array of receipts). Size-bounded (2 MB), every
 * receipt shape-validated and policy-replayed, merged idempotently; conflicts are rejected, never
 * overwritten. Returns counts and per-receipt rejections; never throws on bad input.
 */
export function importReceipts(text: string, storage: StorageLike | null = getStorage()): ImportReport {
  const report: ImportReport = { imported: 0, duplicates: 0, rejected: [], persisted: false };
  if (typeof text !== 'string') {
    report.rejected.push({ reason: 'import must be JSON text' });
    return report;
  }
  const bytes = utf8ByteLength(text);
  if (bytes > IMPORT_MAX_BYTES) {
    report.rejected.push({ reason: `import is ${bytes} bytes; the limit is ${IMPORT_MAX_BYTES}` });
    return report;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    report.rejected.push({ reason: `not valid JSON: ${e instanceof Error ? e.message : String(e)}` });
    return report;
  }
  let items: unknown[];
  if (Array.isArray(parsed)) {
    items = parsed;
  } else if (parsed && typeof parsed === 'object') {
    const env = parsed as { schema_version?: unknown; receipts?: unknown };
    if (env.schema_version !== EXPORT_SCHEMA_VERSION) {
      report.rejected.push({ reason: `unsupported export schema_version ${JSON.stringify(env.schema_version)}` });
      return report;
    }
    if (!Array.isArray(env.receipts)) {
      report.rejected.push({ reason: 'export envelope has no receipts array' });
      return report;
    }
    items = env.receipts;
  } else {
    report.rejected.push({ reason: 'import must be an export envelope or an array of receipts' });
    return report;
  }

  if (!storage) {
    report.rejected.push({ reason: 'localStorage is not available' });
    report.storage_error = 'localStorage is not available';
    return report;
  }
  const { receipts } = readStore(storage);
  let changed = false;
  for (let i = 0; i < items.length; i += 1) {
    const item = items[i];
    const maybeId =
      item && typeof item === 'object' && typeof (item as { decision_id?: unknown }).decision_id === 'string'
        ? ((item as { decision_id: string }).decision_id)
        : undefined;
    let receipt: Receipt;
    try {
      receipt = validateReceiptShape(item);
      validateReceipt(receipt);
    } catch (e) {
      report.rejected.push({ decision_id: maybeId, reason: e instanceof Error ? e.message : String(e) });
      continue;
    }
    const existing = Object.prototype.hasOwnProperty.call(receipts, receipt.decision_id)
      ? receipts[receipt.decision_id]
      : undefined;
    if (!existing) {
      receipts[receipt.decision_id] = receipt;
      report.imported += 1;
      changed = true;
      continue;
    }
    try {
      const mergedReceipt = merge(existing, receipt);
      if (mergedReceipt.events.length > existing.events.length) {
        receipts[receipt.decision_id] = mergedReceipt;
        report.imported += 1;
        changed = true;
      } else {
        report.duplicates += 1;
      }
    } catch (e) {
      if (e instanceof ConflictError) {
        report.rejected.push({ decision_id: receipt.decision_id, reason: `conflict: ${e.message}` });
      } else {
        throw e;
      }
    }
  }
  if (!changed) {
    report.persisted = true;
    return report;
  }
  const written = writeStore(storage, receipts);
  if (written.ok) {
    report.persisted = true;
  } else {
    report.storage_error = `${written.reason}: ${written.message}`;
  }
  return report;
}
