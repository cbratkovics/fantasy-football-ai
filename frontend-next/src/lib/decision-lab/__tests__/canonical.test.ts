import { createHash } from 'node:crypto';

import { describe, expect, it } from 'vitest';

import {
  CanonicalError,
  canonicalJson,
  compareCodePoint,
  contentId,
  deepEqual,
  formatNumber,
  q,
  roundHalfUp,
} from '../canonical';
import { sha256, utf8Encode } from '../sha256';
import { loadGolden } from './golden';

const golden = loadGolden();

describe('sha256', () => {
  it('matches the standard test vectors', () => {
    expect(sha256('')).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
    expect(sha256('abc')).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
    expect(sha256('a'.repeat(1000))).toBe(
      '41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3',
    );
    // Multi-block message crossing the 55/56/64-byte padding boundaries.
    expect(sha256('The quick brown fox jumps over the lazy dog')).toBe(
      'd7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592',
    );
    expect(sha256('x'.repeat(55))).toBe(sha256Node('x'.repeat(55)));
    expect(sha256('x'.repeat(56))).toBe(sha256Node('x'.repeat(56)));
    expect(sha256('x'.repeat(64))).toBe(sha256Node('x'.repeat(64)));
    expect(sha256('Ja\'Marr Chase — été 🏈')).toBe(sha256Node('Ja\'Marr Chase — été 🏈'));
  });

  it('matches every golden text/sha256 vector', () => {
    for (const v of golden.canonical.filter((c) => typeof c.text === 'string')) {
      expect(sha256(v.text as string)).toBe(v.sha256);
    }
  });

  it('utf8Encode agrees with TextEncoder including astral characters', () => {
    const s = 'aé—🏈\ud800'; // includes a lone surrogate
    expect(Array.from(utf8Encode(s))).toEqual(Array.from(new TextEncoder().encode(s)));
  });
});

function sha256Node(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

describe('formatNumber', () => {
  it('implements the Python rule exactly', () => {
    expect(formatNumber(14)).toBe('14');
    expect(formatNumber(14.0)).toBe('14');
    expect(formatNumber(13.6)).toBe('13.6');
    expect(formatNumber(-0)).toBe('0');
    expect(formatNumber(0)).toBe('0');
    expect(formatNumber(1 / 128)).toBe('0.007813');
    expect(formatNumber(0.1 + 0.2)).toBe('0.3');
    expect(formatNumber(1e-7)).toBe('0');
    expect(formatNumber(-1e-7)).toBe('0');
    expect(formatNumber(0.000001)).toBe('0.000001');
    expect(formatNumber(-13.6)).toBe('-13.6');
    expect(formatNumber(123456789.123456789)).toBe('123456789.123457');
    expect(formatNumber(1e12)).toBe('1000000000000');
    expect(formatNumber(2.5e-6)).toBe('0.000003');
    expect(formatNumber(-2.5e-6)).toBe('-0.000002');
  });

  it('rejects non-finite values, booleans and out-of-range magnitudes', () => {
    expect(() => formatNumber(NaN)).toThrow(CanonicalError);
    expect(() => formatNumber(Infinity)).toThrow(CanonicalError);
    expect(() => formatNumber(-Infinity)).toThrow(CanonicalError);
    expect(() => formatNumber(true as unknown as number)).toThrow(CanonicalError);
    expect(() => formatNumber('1' as unknown as number)).toThrow(CanonicalError);
    expect(() => formatNumber(1e12 + 1)).toThrow(CanonicalError);
  });
});

describe('roundHalfUp / q', () => {
  it('rounds halves toward +infinity at six decimals and normalizes -0', () => {
    expect(roundHalfUp(0.0000005)).toBe(0.000001);
    expect(roundHalfUp(-0.0000005)).toBe(0);
    expect(Object.is(roundHalfUp(-0.0000001), 0)).toBe(true);
    expect(roundHalfUp(1.2345675)).toBe(1.234568);
    expect(roundHalfUp(2.5, 0)).toBe(3);
    expect(roundHalfUp(-2.5, 0)).toBe(-2);
  });

  it('q passes null through and throws on non-finite', () => {
    expect(q(null)).toBeNull();
    expect(q(14 - 13.6)).toBe(0.4);
    expect(() => q(NaN)).toThrow(CanonicalError);
    expect(() => q(Infinity)).toThrow(CanonicalError);
    expect(() => q('x')).toThrow(CanonicalError);
    expect(() => q(true)).toThrow(CanonicalError);
  });
});

describe('canonicalJson', () => {
  it('reproduces every golden canonical vector and its content id', () => {
    const vectors = golden.canonical.filter((c) => typeof c.canonical === 'string');
    expect(vectors.length).toBe(9);
    for (const v of vectors) {
      expect(canonicalJson(v.value)).toBe(v.canonical);
      expect(contentId(v.value)).toBe(v.sha256);
    }
  });

  it('escapes control characters like Python and leaves non-ASCII raw', () => {
    expect(canonicalJson('a\u0001b\u001f')).toBe('"a\\u0001b\\u001f"');
    expect(canonicalJson('\b\f\n\r\t"\\')).toBe('"\\b\\f\\n\\r\\t\\"\\\\"');
    expect(canonicalJson('é—🏈')).toBe('"é—🏈"');
    expect(canonicalJson('\u007f')).toBe('"\u007f"');
  });

  it('sorts keys by code point, not UTF-16 code unit', () => {
    // U+FF5E (BMP, code unit 0xFF5E) sorts after U+1F3C8 by code units but before by code point.
    const obj = { '～': 1, '🏈': 2, a: 3 };
    expect(canonicalJson(obj)).toBe('{"a":3,"～":1,"🏈":2}');
    expect(compareCodePoint('～', '🏈')).toBe(-1);
    expect(compareCodePoint('a', 'b')).toBe(-1);
    expect(compareCodePoint('b', 'a')).toBe(1);
    expect(compareCodePoint('ab', 'a')).toBe(1);
    expect(compareCodePoint('a', 'a')).toBe(0);
  });

  it('treats undefined like JSON.stringify and rejects unsupported types', () => {
    expect(canonicalJson({ a: undefined, b: 1 })).toBe('{"b":1}');
    expect(canonicalJson([undefined, 1])).toBe('[null,1]');
    expect(() => canonicalJson(undefined)).toThrow(CanonicalError);
    expect(() => canonicalJson(() => 1)).toThrow(CanonicalError);
    expect(() => canonicalJson(new Date(0))).toThrow(CanonicalError);
    expect(() => canonicalJson({ a: NaN })).toThrow(CanonicalError);
  });

  it('is invariant to key insertion order and float formatting noise', () => {
    expect(contentId({ b: 1, a: 2 })).toBe(contentId({ a: 2, b: 1 }));
    expect(contentId({ x: 0.1 + 0.2 })).toBe(contentId({ x: 0.3 }));
    expect(contentId({ x: 14.0 })).toBe(contentId({ x: 14 }));
  });
});

describe('deepEqual', () => {
  it('follows Python equality for JSON-like values', () => {
    expect(deepEqual({ a: [1, { b: null }] }, { a: [1, { b: null }] })).toBe(true);
    expect(deepEqual({ a: 1, b: 2 }, { b: 2, a: 1 })).toBe(true);
    expect(deepEqual(-0, 0)).toBe(true);
    expect(deepEqual(NaN, NaN)).toBe(false);
    expect(deepEqual({ a: undefined }, {})).toBe(true);
    expect(deepEqual([1, 2], [2, 1])).toBe(false);
    expect(deepEqual({ a: 1 }, { a: 1, b: 2 })).toBe(false);
    expect(deepEqual(null, {})).toBe(false);
  });
});
