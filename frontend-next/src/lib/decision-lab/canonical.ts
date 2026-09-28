/**
 * Canonical serialization and content identities — the mirror of `ffai/decision_lab/canonical.py`.
 *
 * Every identity in the lab is `sha256(canonicalJson(value))`. Canonical JSON is deterministic
 * across Python and JavaScript by construction:
 *
 * - object keys sorted by Unicode code point, no insignificant whitespace;
 * - strings escaped exactly like Python's `_escape` (`"`, `\`, `\n`, `\r`, `\t`, `\b`, `\f`,
 *   other control characters below 0x20 as `\u00xx` lowercase hex, everything else raw UTF-8);
 * - numbers rendered by one rule in both languages: `floor(x * 1e6 + 0.5) / 1e6` (halves toward
 *   +infinity), printed with six decimals, trailing zeros stripped, `-0` → `0`, non-finite and
 *   `|x| > 1e12` rejected. Integer-valued numbers therefore print as integers (`14`, not `14.0`).
 *
 * Key-order note: Python sorts `str` keys by code point. JavaScript's default `sort()` compares
 * UTF-16 code units, which differs only for astral characters (U+10000 and above, encoded as
 * surrogate pairs). {@link compareCodePoint} compares by code point so both agree everywhere.
 *
 * `undefined` handling (no Python equivalent): object properties whose value is `undefined` are
 * skipped, and `undefined` inside arrays renders as `null` — the same choices `JSON.stringify`
 * makes, so hashing an in-memory object equals hashing its JSON round-trip. A top-level
 * `undefined` is rejected.
 */
import { sha256 } from './sha256';

/** Decimal places kept by canonical numbers and by {@link q}. */
export const DECIMALS = 6;
const SCALE = 1e6;
/** Largest magnitude a canonical number may have. */
export const MAX_ABS = 1e12;

/** A value that cannot be serialized canonically (non-finite number, unsupported type). */
export class CanonicalError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CanonicalError';
  }
}

/**
 * `Math.round`-style rounding to `decimals` places with halves toward +infinity, computed as
 * `floor(x * scale + 0.5) / scale` — the exact expression Python uses, so both languages agree
 * bit for bit (including the rare inputs where `Math.round` and `floor(x + 0.5)` differ).
 * A zero result is normalized to `+0`.
 */
export function roundHalfUp(x: number, decimals: number = DECIMALS): number {
  const scale = Math.pow(10, decimals);
  const r = Math.floor(x * scale + 0.5) / scale;
  return r === 0 ? 0 : r;
}

/**
 * Canonical text for a number: rounded to six decimals, trailing zeros stripped, `-0` → `"0"`.
 * Rejects booleans, non-numbers, NaN, ±Infinity and magnitudes above 1e12.
 */
export function formatNumber(x: unknown): string {
  if (typeof x === 'boolean') {
    throw new CanonicalError('booleans are not numbers');
  }
  if (typeof x !== 'number') {
    throw new CanonicalError(`unsupported type ${typeof x}`);
  }
  if (!Number.isFinite(x)) {
    throw new CanonicalError(`non-finite number: ${String(x)}`);
  }
  if (Math.abs(x) > MAX_ABS) {
    throw new CanonicalError(`number out of range: ${String(x)}`);
  }
  const r = roundHalfUp(x);
  let s = r.toFixed(DECIMALS);
  if (s.indexOf('.') >= 0) {
    s = s.replace(/0+$/, '').replace(/\.$/, '');
  }
  if (s === '-0' || s === '') {
    s = '0';
  }
  return s;
}

/** Escape a string exactly like Python's `canonical._escape`. */
export function escapeString(s: string): string {
  let out = '"';
  for (let i = 0; i < s.length; i += 1) {
    const ch = s.charAt(i);
    const o = s.charCodeAt(i);
    if (ch === '"') {
      out += '\\"';
    } else if (ch === '\\') {
      out += '\\\\';
    } else if (ch === '\n') {
      out += '\\n';
    } else if (ch === '\r') {
      out += '\\r';
    } else if (ch === '\t') {
      out += '\\t';
    } else if (ch === '\b') {
      out += '\\b';
    } else if (ch === '\f') {
      out += '\\f';
    } else if (o < 0x20) {
      out += '\\u' + ('0000' + o.toString(16)).slice(-4);
    } else {
      out += ch;
    }
  }
  return out + '"';
}

/**
 * Compare two strings by Unicode code point (Python's `str` ordering). Returns -1, 0 or 1.
 * Differs from the default `<` comparison only when astral characters are involved.
 */
export function compareCodePoint(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i += 1) {
    let ca = a.charCodeAt(i);
    let cb = b.charCodeAt(i);
    if (ca === cb) {
      continue;
    }
    // Map surrogates above BMP code units so pairs sort after U+FFFF, as code points do.
    if (ca >= 0xd800 && ca <= 0xdfff) ca += 0x10000;
    if (cb >= 0xd800 && cb <= 0xdfff) cb += 0x10000;
    return ca < cb ? -1 : 1;
  }
  return a.length === b.length ? 0 : a.length < b.length ? -1 : 1;
}

/** Sort a copy of `keys` by code point. */
export function sortedKeys(keys: string[]): string[] {
  return keys.slice().sort(compareCodePoint);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/** Deterministic JSON text for `value` (plain objects, arrays, strings, numbers, booleans, null). */
export function canonicalJson(value: unknown): string {
  if (value === null) {
    return 'null';
  }
  if (value === true) {
    return 'true';
  }
  if (value === false) {
    return 'false';
  }
  if (typeof value === 'string') {
    return escapeString(value);
  }
  if (typeof value === 'number') {
    return formatNumber(value);
  }
  if (Array.isArray(value)) {
    const parts: string[] = [];
    for (let i = 0; i < value.length; i += 1) {
      const v = value[i];
      parts.push(v === undefined ? 'null' : canonicalJson(v));
    }
    return '[' + parts.join(',') + ']';
  }
  if (isPlainObject(value)) {
    const keys = sortedKeys(Object.keys(value));
    const parts: string[] = [];
    for (let i = 0; i < keys.length; i += 1) {
      const k = keys[i];
      const v = value[k];
      if (v === undefined) {
        continue;
      }
      parts.push(escapeString(k) + ':' + canonicalJson(v));
    }
    return '{' + parts.join(',') + '}';
  }
  if (value === undefined) {
    throw new CanonicalError('unsupported type undefined');
  }
  throw new CanonicalError(`unsupported type ${typeof value}`);
}

/** `sha256(canonicalJson(value))` — the identity of a JSON value, format-independent. */
export function contentId(value: unknown): string {
  return sha256(canonicalJson(value));
}

/** SHA-256 hex of a UTF-8 string (re-exported for convenience; equals Python `sha256_hex`). */
export function sha256Hex(text: string): string {
  return sha256(text);
}

/**
 * Quantize a measurement to the canonical precision (`null` passes through). Throws
 * {@link CanonicalError} for booleans, non-numbers, NaN and ±Infinity.
 */
export function q(x: number | null): number | null;
export function q(x: unknown): number | null;
export function q(x: unknown): number | null {
  if (x === null) {
    return null;
  }
  if (typeof x !== 'number' || !Number.isFinite(x)) {
    throw new CanonicalError(`not a finite number: ${String(x)}`);
  }
  return roundHalfUp(x);
}

/**
 * Structural equality with Python `==` semantics for JSON-like values: dict key order is
 * irrelevant, `-0 == 0`, and (as in {@link canonicalJson}) object properties set to `undefined`
 * count as absent.
 */
export function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) {
    return true;
  }
  if (typeof a === 'number' && typeof b === 'number') {
    return a === b; // NaN !== NaN, matching Python
  }
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      return false;
    }
    for (let i = 0; i < a.length; i += 1) {
      if (!deepEqual(a[i], b[i])) {
        return false;
      }
    }
    return true;
  }
  if (isPlainObject(a) && isPlainObject(b)) {
    const ka = Object.keys(a).filter((k) => a[k] !== undefined);
    const kb = Object.keys(b).filter((k) => b[k] !== undefined);
    if (ka.length !== kb.length) {
      return false;
    }
    for (let i = 0; i < ka.length; i += 1) {
      const k = ka[i];
      if (!Object.prototype.hasOwnProperty.call(b, k) || !deepEqual(a[k], b[k])) {
        return false;
      }
    }
    return true;
  }
  return false;
}
