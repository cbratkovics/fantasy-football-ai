import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  EXPORT_SCHEMA_VERSION,
  IMPORT_MAX_BYTES,
  STORAGE_KEY,
  clearAll,
  deleteReceipt,
  exportAll,
  getStorage,
  importReceipts,
  loadReceipts,
  saveReceipt,
} from '../storage';
import type { Receipt } from '../types';
import { clone, loadGolden } from './golden';

const G = loadGolden().receipt;

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('localStorage round trip', () => {
  it('saves, loads, deletes and clears receipts', () => {
    expect(loadReceipts()).toEqual({ receipts: {}, problems: [] });
    const saved = saveReceipt(clone(G.receipt_new));
    expect(saved).toMatchObject({ ok: true, merged: false });
    expect(STORAGE_KEY).toBe('ffai.decision-lab/decision_receipt/1.0/receipts');
    expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) as string)).toEqual({
      [G.receipt_new.decision_id]: G.receipt_new,
    });
    const loaded = loadReceipts();
    expect(loaded.problems).toEqual([]);
    expect(loaded.receipts[G.receipt_new.decision_id]).toEqual(G.receipt_new);

    // Saving the extended receipt merges (superset of events).
    const again = saveReceipt(clone(G.receipt_after_outcome));
    expect(again).toMatchObject({ ok: true, merged: true });
    expect(loadReceipts().receipts[G.receipt_new.decision_id]).toEqual(G.receipt_after_outcome);
    // Saving the older version back is a no-op merge, never a downgrade.
    expect(saveReceipt(clone(G.receipt_new))).toMatchObject({ ok: true, merged: true });
    expect(loadReceipts().receipts[G.receipt_new.decision_id]).toEqual(G.receipt_after_outcome);
    // A conflicting receipt with the same id is refused.
    expect(saveReceipt(clone(G.receipt_declined))).toMatchObject({ ok: false, reason: 'conflict' });

    expect(deleteReceipt(G.receipt_new.decision_id)).toBe(true);
    expect(deleteReceipt(G.receipt_new.decision_id)).toBe(false);
    expect(loadReceipts().receipts).toEqual({});
    saveReceipt(clone(G.receipt_new));
    clearAll();
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it('rejects an invalid receipt before touching storage', () => {
    const bad = clone(G.receipt_new) as unknown as Record<string, unknown>;
    bad.extra = true;
    expect(saveReceipt(bad as unknown as Receipt)).toMatchObject({ ok: false, reason: 'invalid' });
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it('drops malformed entries with a problem instead of throwing', () => {
    const tampered = clone(G.receipt_after_action) as unknown as Record<string, unknown>;
    tampered.result_sha256 = 'zzz';
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        [G.receipt_new.decision_id]: G.receipt_new,
        'not-a-receipt': { hello: 'world' },
        wrong_key: G.receipt_declined,
        [G.receipt_after_action.decision_id + 'x']: tampered,
      }),
    );
    const loaded = loadReceipts();
    expect(Object.keys(loaded.receipts)).toEqual([G.receipt_new.decision_id]);
    expect(loaded.problems.map((p) => p.kind)).toEqual(['malformed_entry', 'malformed_entry', 'malformed_entry']);
    expect(loaded.problems[1]).toMatchObject({ kind: 'malformed_entry', decision_id: 'wrong_key' });

    window.localStorage.setItem(STORAGE_KEY, '{oops');
    expect(loadReceipts().problems).toEqual([
      { kind: 'malformed_store', message: expect.stringMatching(/not valid JSON/) },
    ]);
    window.localStorage.setItem(STORAGE_KEY, '[1,2]');
    expect(loadReceipts().problems[0].kind).toBe('malformed_store');
  });

  it('degrades gracefully when localStorage is disabled', () => {
    vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
      throw new Error('SecurityError: storage disabled');
    });
    expect(getStorage()).toBeNull();
    expect(loadReceipts()).toEqual({
      receipts: {},
      problems: [{ kind: 'storage_unavailable', message: 'localStorage is not available' }],
    });
    expect(saveReceipt(clone(G.receipt_new))).toMatchObject({ ok: false, reason: 'storage_unavailable' });
    expect(deleteReceipt(G.receipt_new.decision_id)).toBe(false);
    expect(() => clearAll()).not.toThrow();
    expect(JSON.parse(exportAll('2099-01-01T00:00:00Z')).receipts).toEqual([]);
    const report = importReceipts(JSON.stringify([G.receipt_new]));
    expect(report.persisted).toBe(false);
    expect(report.rejected[0].reason).toMatch(/not available/);
  });

  it('reports quota errors', () => {
    const original = Storage.prototype.setItem;
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(function (this: Storage, key: string, value: string) {
      if (key === STORAGE_KEY) {
        const e = new Error('QuotaExceededError: storage full');
        e.name = 'QuotaExceededError';
        throw e;
      }
      return original.call(this, key, value);
    });
    expect(saveReceipt(clone(G.receipt_new))).toMatchObject({ ok: false, reason: 'quota' });
    const report = importReceipts(JSON.stringify([G.receipt_new]));
    expect(report.imported).toBe(1);
    expect(report.persisted).toBe(false);
    expect(report.storage_error).toMatch(/^quota:/);
  });
});

describe('export / import', () => {
  it('round-trips receipts exactly and imports idempotently', () => {
    saveReceipt(clone(G.receipt_after_outcome));
    const text = exportAll('2099-01-01T00:00:00Z');
    const envelope = JSON.parse(text);
    expect(envelope.schema_version).toBe(EXPORT_SCHEMA_VERSION);
    expect(envelope.exported_at_utc).toBe('2099-01-01T00:00:00Z');
    expect(envelope.receipts).toEqual([G.receipt_after_outcome]);

    clearAll();
    const first = importReceipts(text);
    expect(first).toEqual({ imported: 1, duplicates: 0, rejected: [], persisted: true });
    expect(loadReceipts().receipts[G.receipt_new.decision_id]).toEqual(G.receipt_after_outcome);

    const second = importReceipts(text);
    expect(second).toEqual({ imported: 0, duplicates: 1, rejected: [], persisted: true });
    expect(JSON.parse(exportAll('2099-01-01T00:00:00Z'))).toEqual(envelope);
  });

  it('accepts a bare array, extends with a superset and rejects conflicts and tampering', () => {
    expect(importReceipts(JSON.stringify([G.receipt_new]))).toMatchObject({ imported: 1, duplicates: 0 });
    // A superset of events extends the stored receipt.
    expect(importReceipts(JSON.stringify([G.receipt_after_action]))).toMatchObject({ imported: 1, duplicates: 0 });
    expect(loadReceipts().receipts[G.receipt_new.decision_id]).toEqual(G.receipt_after_action);
    // A subset is a duplicate.
    expect(importReceipts(JSON.stringify([G.receipt_new]))).toMatchObject({ imported: 0, duplicates: 1 });
    // A conflicting receipt (same id, different first event) is rejected, not overwritten.
    const conflict = importReceipts(JSON.stringify({ schema_version: EXPORT_SCHEMA_VERSION, exported_at_utc: 'x', receipts: [G.receipt_declined] }));
    expect(conflict.imported).toBe(0);
    expect(conflict.rejected).toEqual([{ decision_id: G.receipt_new.decision_id, reason: expect.stringMatching(/^conflict: event 1 differs/) }]);
    expect(loadReceipts().receipts[G.receipt_new.decision_id]).toEqual(G.receipt_after_action);
    // Tampered receipts fail the policy replay.
    const tampered = clone(G.receipt_after_outcome);
    tampered.result.recommended_player_id = 'SYN-B';
    const rep = importReceipts(JSON.stringify([tampered, { nonsense: true }]));
    expect(rep.imported).toBe(0);
    expect(rep.rejected.length).toBe(2);
    expect(rep.rejected[0]).toEqual({ decision_id: tampered.decision_id, reason: expect.stringMatching(/^result_replay:/) });
    expect(rep.rejected[1].reason).toMatch(/receipt: <root>/);
  });

  it('rejects oversized, malformed and unknown-schema imports without throwing', () => {
    expect(importReceipts('x'.repeat(IMPORT_MAX_BYTES + 1)).rejected[0].reason).toMatch(/limit is 2097152/);
    expect(importReceipts('{nope').rejected[0].reason).toMatch(/not valid JSON/);
    expect(importReceipts('42').rejected[0].reason).toMatch(/export envelope or an array/);
    expect(importReceipts(JSON.stringify({ schema_version: 'other/1.0', receipts: [] })).rejected[0].reason).toMatch(/unsupported export schema_version/);
    expect(importReceipts(JSON.stringify({ schema_version: EXPORT_SCHEMA_VERSION })).rejected[0].reason).toMatch(/no receipts array/);
    expect(importReceipts(null as unknown as string).rejected[0].reason).toMatch(/JSON text/);
    expect(loadReceipts().receipts).toEqual({});
  });
});
