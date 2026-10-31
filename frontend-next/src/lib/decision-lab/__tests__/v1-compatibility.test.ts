import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { evaluate } from '../policy';
import { validateReceipt } from '../receipts';
import { importReceipts, loadReceipts } from '../storage';
import type { InputsSnapshot, OutcomeSnapshot, Receipt } from '../types';

const root = path.resolve(process.cwd(), '../tests/fixtures/start_sit_compat');
const read = <T,>(file: string): T => JSON.parse(readFileSync(path.join(root, file), 'utf8'));
const index = read<{
  examples: Array<{
    label: string;
    snapshot_id: string;
    inputs: { path: string };
    outcomes: { path: string };
    receipts: Array<{ path: string; decision_id: string; result_sha256: string; event_ids: string[] }>;
  }>;
}>('index.json');

describe('original v1 receipt imports retain their paired evidence and event meaning', () => {
  for (const example of index.examples) {
    for (const entry of example.receipts) {
      it(entry.path, () => {
        const receipt = read<Receipt>(entry.path);
        const inputs = read<InputsSnapshot>(example.inputs.path);
        const outcomes = read<OutcomeSnapshot>(example.outcomes.path);
        const checks = validateReceipt(receipt, {
          inputsLookup: (id) => id === example.snapshot_id ? inputs : null,
          outcomeLookup: (id) => id === example.snapshot_id ? outcomes : null,
        });
        expect(checks.every((check) => check.ok)).toBe(true);
        expect(checks.some((check) => check.check === 'snapshot_digest')).toBe(true);
        expect(evaluate(receipt.inputs)).toEqual(receipt.result);
        expect(receipt.decision_id).toBe(entry.decision_id);
        expect(receipt.result_sha256).toBe(entry.result_sha256);
        expect(receipt.events.map((event) => event.event_id)).toEqual(entry.event_ids);

        const store = new Map<string, string>();
        const storage = {
          getItem: (key: string) => store.get(key) ?? null,
          setItem: (key: string, value: string) => { store.set(key, value); },
          removeItem: (key: string) => { store.delete(key); },
        };
        const report = importReceipts(JSON.stringify([receipt]), storage);
        expect(report.rejected).toEqual([]);
        expect(report.imported).toBe(1);
        const imported = loadReceipts(storage).receipts;
        expect(imported).toEqual({ [receipt.decision_id]: receipt });
        if (!receipt.events.length) {
          expect(imported[receipt.decision_id].action.state).toBe('not_recorded');
          expect(imported[receipt.decision_id].action.chosen_player_id).toBeNull();
        }
      });
    }
  }
});
