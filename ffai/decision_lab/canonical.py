"""Canonical serialization and content identities (mirrored by ``canonical.ts``).

Every identity in the lab is ``sha256(canonical_json(value))``. Canonical JSON is deterministic
across Python and JavaScript by construction:

* object keys sorted by code point, no insignificant whitespace;
* strings escaped like ``JSON.stringify`` (``"``, ``\\``, control characters);
* numbers rendered by one rule in both languages: round to six decimals with
  ``floor(x * 1e6 + 0.5) / 1e6`` (half toward +infinity, the ``Math.round`` rule), print with six
  decimals, strip trailing zeros, ``-0`` becomes ``0``. Non-finite numbers are rejected. Integer
  valued numbers therefore print as integers (``14`` not ``14.0``), which is what
  ``JSON.stringify`` does.

Volatile fields (build timestamps, tool versions) are kept in separate files so they never enter
an identity; see ``digest_coverage`` in the exported manifest for what each digest covers.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

DECIMALS = 6
_SCALE = 10**DECIMALS
MAX_ABS = 1e12


class CanonicalError(ValueError):
    """A value that cannot be serialized canonically (non-finite number, unsupported type)."""


def round_half_up(x: float, decimals: int = DECIMALS) -> float:
    """``Math.round``-style rounding: halves go toward +infinity. Identical in JavaScript as
    ``Math.round(x * 10**decimals) / 10**decimals``."""
    scale = 10**decimals
    r = math.floor(x * scale + 0.5) / scale
    return 0.0 if r == 0 else r


def format_number(x: float | int) -> str:
    if isinstance(x, bool):
        raise CanonicalError("booleans are not numbers")
    if isinstance(x, int):
        if abs(x) > MAX_ABS:
            raise CanonicalError(f"number out of range: {x}")
        return str(x)
    if not math.isfinite(x):
        raise CanonicalError(f"non-finite number: {x!r}")
    if abs(x) > MAX_ABS:
        raise CanonicalError(f"number out of range: {x}")
    r = round_half_up(float(x))
    s = f"{r:.{DECIMALS}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def _escape(s: str) -> str:
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif o < 0x20:
            out.append(f"\\u{o:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def canonical_json(value: Any) -> str:
    """Deterministic JSON text for ``value`` (dicts, lists, str, numbers, bool, None)."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _escape(value)
    if isinstance(value, int | float):
        return format_number(value)
    if isinstance(value, dict):
        keys = sorted(str(k) for k in value)
        if len(set(keys)) != len(value):
            raise CanonicalError("object keys must be unique strings")
        parts = [f"{_escape(k)}:{canonical_json(value[k])}" for k in keys]
        return "{" + ",".join(parts) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(canonical_json(v) for v in value) + "]"
    raise CanonicalError(f"unsupported type {type(value).__name__}")


def sha256_hex(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def content_id(value: Any) -> str:
    """``sha256(canonical_json(value))`` — the identity of a JSON value, format-independent."""
    return sha256_hex(canonical_json(value))


def file_sha256(path) -> str:  # noqa: ANN001 - Path-like
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def q(x: float | None) -> float | None:
    """Quantize a measurement to the canonical precision (``None`` passes through)."""
    if x is None:
        return None
    if isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x):
        raise CanonicalError(f"not a finite number: {x!r}")
    return round_half_up(float(x))
