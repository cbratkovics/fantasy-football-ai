"""Canonical JSON and content identities for the Decision Lab (``ffai.decision_lab.canonical``).

The TypeScript mirror must reproduce every vector here byte for byte, so the assertions are on
exact strings and digests, not on round-trips.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from ffai.decision_lab import canonical, golden
from ffai.decision_lab.canonical import (
    CanonicalError,
    canonical_json,
    content_id,
    format_number,
    q,
    sha256_hex,
)

GOLDEN = json.loads(golden.GOLDEN_PATH.read_text(encoding="utf-8"))


# --- format_number -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (14.0, "14"),
        (13.6, "13.6"),
        (-0.0, "0"),
        (0.0, "0"),
        (1 / 128, "0.007813"),
        (0.1 + 0.2, "0.3"),
        (1e-7, "0"),
        (-1e-7, "0"),
        (0.000001, "0.000001"),
        (123456789.123456789, "123456789.123457"),
        (-13.6, "-13.6"),
        (2.5, "2.5"),
        (1e12, "1000000000000"),
    ],
)
def test_format_number_floats(value: float, text: str) -> None:
    assert format_number(value) == text


@pytest.mark.parametrize(
    ("value", "text"), [(0, "0"), (14, "14"), (-3, "-3"), (10**12, "1000000000000")]
)
def test_format_number_ints(value: int, text: str) -> None:
    assert format_number(value) == text


def test_format_number_half_up_at_six_decimals() -> None:
    # 1/128 = 0.0078125 exactly: the seventh decimal is a 5 and rounds toward +infinity.
    assert format_number(0.0078125) == "0.007813"
    assert format_number(-0.0078125) == "-0.007812"
    assert canonical.round_half_up(0.0078125) == 0.007813


def test_format_number_int_and_float_agree() -> None:
    assert format_number(14) == format_number(14.0) == "14"


@pytest.mark.parametrize("value", [True, False])
def test_format_number_rejects_bool(value: bool) -> None:
    with pytest.raises(CanonicalError):
        format_number(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_format_number_rejects_non_finite(value: float) -> None:
    with pytest.raises(CanonicalError):
        format_number(value)


@pytest.mark.parametrize("value", [1e12 + 1, -1e12 - 1, 10**13, 1e300])
def test_format_number_rejects_out_of_range(value: float | int) -> None:
    with pytest.raises(CanonicalError):
        format_number(value)


# --- canonical_json ----------------------------------------------------------------------------


def test_canonical_json_sorts_keys_by_code_point() -> None:
    assert canonical_json({"b": 1, "a": 2, "B": 3, "_": 4}) == '{"B":3,"_":4,"a":2,"b":1}'


def test_canonical_json_has_no_whitespace_and_renders_scalars() -> None:
    assert (
        canonical_json({"x": [1, 2.5, None, True, False, "s"]})
        == '{"x":[1,2.5,null,true,false,"s"]}'
    )
    assert canonical_json(None) == "null"
    assert canonical_json([]) == "[]"
    assert canonical_json({}) == "{}"
    assert canonical_json((1, 2)) == "[1,2]"


def test_canonical_json_escapes_like_json_stringify() -> None:
    assert canonical_json('x"y\\z\n') == '"x\\"y\\\\z\\n"'
    assert canonical_json("\r\t\b\f") == '"\\r\\t\\b\\f"'
    assert canonical_json("\x00\x1f") == '"\\u0000\\u001f"'
    # DEL (0x7f) and everything above 0x1f are kept raw.
    assert canonical_json("\x7f") == '"\x7f"'


def test_canonical_json_keeps_unicode_raw() -> None:
    text = "Ja'Marr Chase — été 日本"
    assert canonical_json(text) == f'"{text}"'
    assert "\\u" not in canonical_json(text)


def test_canonical_json_nested_structures() -> None:
    value = {"nested": {"k": [{"y": 1}, {"x": 2}], "a": {"z": [], "y": {}}}}
    assert canonical_json(value) == '{"nested":{"a":{"y":{},"z":[]},"k":[{"y":1},{"x":2}]}}'


def test_canonical_json_rejects_unsupported_types_and_non_finite() -> None:
    with pytest.raises(CanonicalError):
        canonical_json({"s": {1, 2}})
    with pytest.raises(CanonicalError):
        canonical_json({"n": float("nan")})
    with pytest.raises(CanonicalError):
        canonical_json([float("inf")])
    with pytest.raises(CanonicalError):
        canonical_json(object())


def test_canonical_json_rejects_non_unique_stringified_keys() -> None:
    with pytest.raises(CanonicalError):
        canonical_json({1: "a", "1": "b"})


# --- content_id / sha256 -----------------------------------------------------------------------


def test_sha256_hex_known_digests() -> None:
    assert sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256_hex(b"abc") == sha256_hex("abc")
    assert sha256_hex("été") == hashlib.sha256("été".encode()).hexdigest()


def test_content_id_is_stable_and_independent_of_key_order() -> None:
    a = {"projection": 14.0, "player_id": "SYN-A", "floor": 6.0}
    b = {"floor": 6.0, "player_id": "SYN-A", "projection": 14.0}
    assert content_id(a) == content_id(b) == content_id(dict(a))
    assert content_id(a) == sha256_hex(canonical_json(a))
    assert len(content_id(a)) == 64


def test_content_id_is_independent_of_float_formatting() -> None:
    assert content_id({"v": 14}) == content_id({"v": 14.0})
    assert content_id({"v": 0.1 + 0.2}) == content_id({"v": 0.3})
    assert content_id({"v": -0.0}) == content_id({"v": 0})
    assert content_id({"v": 1e-7}) == content_id({"v": 0})


def test_content_id_differs_on_semantic_change() -> None:
    assert content_id({"v": 14.0}) != content_id({"v": 14.000001})
    assert content_id({"v": 1}) != content_id({"v": "1"})
    assert content_id([1, 2]) != content_id([2, 1])


# --- q (quantize) ------------------------------------------------------------------------------


def test_q_quantizes_to_canonical_precision() -> None:
    assert q(None) is None
    assert q(0.1 + 0.2) == 0.3
    assert q(14) == 14.0
    assert q(-0.0) == 0.0 and not math.copysign(1, q(-0.0)) < 0
    assert q(0.0078125) == 0.007813
    for bad in (True, "1", float("nan"), float("inf")):
        with pytest.raises(CanonicalError):
            q(bad)  # type: ignore[arg-type]


# --- golden vectors ----------------------------------------------------------------------------


def test_golden_canonical_vectors_reproduce() -> None:
    vectors = GOLDEN["canonical"]
    assert len(vectors) >= 11
    for vec in vectors:
        if "text" in vec:
            assert sha256_hex(vec["text"]) == vec["sha256"]
        else:
            assert canonical_json(vec["value"]) == vec["canonical"]
            assert content_id(vec["value"]) == vec["sha256"]
            assert sha256_hex(vec["canonical"]) == vec["sha256"]
            # The canonical text is itself valid JSON that parses back to the value (modulo the
            # canonical number rule, which json.dumps does not apply).
            assert canonical_json(json.loads(vec["canonical"])) == vec["canonical"]


def test_golden_canonical_vectors_cover_the_number_rules() -> None:
    texts = [v["canonical"] for v in GOLDEN["canonical"] if "canonical" in v]
    assert '{"big":123456789.123457,"m":13.6,"n":14,"s":0.3,"t":0.007813,"z":0}' in texts
    assert "0.000001" in texts
    assert "-13.6" in texts
    assert "0" in texts  # 1e-7 collapses to 0
