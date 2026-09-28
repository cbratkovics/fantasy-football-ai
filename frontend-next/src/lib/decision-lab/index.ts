/**
 * Decision Lab core — the TypeScript mirror of `ffai/decision_lab`.
 *
 * - `canonical` / `sha256`: canonical JSON, deterministic numbers, content identities.
 * - `policy`: the reference decision policy (`evaluate`), parity-tested against the golden file.
 * - `metrics`: outcome metrics joined after a decision exists.
 * - `receipts`: receipts, events, validation/replay, merge.
 * - `cases`: decision inputs for a case-library case or experiment.
 * - `validate`: strict document contracts (schemas + semantic checks).
 * - `bundle`: fail-closed loader for the exported bundle under `/decision-lab`.
 * - `storage`: localStorage persistence, export and import.
 * - `explain`: UI labels for codes.
 */
export * from './types';
export * from './spec';
export * from './sha256';
export * from './canonical';
export * from './policy';
export * from './metrics';
export * from './validate';
export * from './receipts';
export * from './cases';
export * from './bundle';
export * from './storage';
export * from './explain';
