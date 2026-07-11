"""Submission payload encode/decode — miner side fails loud, validator side fails quiet."""

from __future__ import annotations

import json

import pytest

from optima.chain.payload import (
    MAX_PAYLOAD_BYTES,
    PAYLOAD_VERSION,
    PayloadError,
    decode_payload,
    decode_payload_for_testing,
    encode_payload,
    encode_payload_for_testing,
)

HASH = "a" * 64


def test_roundtrip():
    data = encode_payload(HASH, "https://example.com/b.tar.gz")
    ref = decode_payload("hk1", 42, data)
    assert ref is not None
    assert ref.content_hash == HASH and ref.url == "https://example.com/b.tar.gz"
    assert ref.hotkey == "hk1" and ref.block == 42


def test_encode_rejects_bad_hash_and_scheme():
    with pytest.raises(PayloadError):
        encode_payload("nothex", "https://x")
    with pytest.raises(PayloadError):
        encode_payload(HASH.upper(), "https://x")  # uppercase hex is not canonical
    with pytest.raises(PayloadError):
        encode_payload(HASH, "ftp://example.com/b.tar.gz")
    with pytest.raises(PayloadError):
        encode_payload(HASH, "not-a-url")
    for url in (
        "https://user:pass@example.com/b.tar.gz",
        "https://example.com/b.tar.gz#fragment",
        "https://example.com/a b.tar.gz",
        "https://example.com/é.tar.gz",
    ):
        with pytest.raises(PayloadError):
            encode_payload(HASH, url)


def test_encode_rejects_oversize():
    with pytest.raises(PayloadError):
        encode_payload(HASH, "https://example.com/" + "x" * MAX_PAYLOAD_BYTES)


def test_decode_never_raises_on_garbage():
    for garbage in ("", "not json", "[]", '{"v":99,"h":"x","u":"y"}',
                    json.dumps({"v": PAYLOAD_VERSION}),  # missing fields
                    json.dumps({"v": PAYLOAD_VERSION, "h": "short", "u": "https://x"}),
                    json.dumps({"v": PAYLOAD_VERSION, "h": HASH, "u": "javascript:x"}),
                    json.dumps({"v": PAYLOAD_VERSION, "h": HASH, "u": 7}),
                    json.dumps({"v": True, "h": HASH, "u": "https://x"}),
                    json.dumps({"v": PAYLOAD_VERSION, "h": HASH, "u": "https://x", "x": 1}),
                    '{"v":1,"h":"' + HASH + '","h":"' + HASH + '","u":"https://x"}',
                    "x" * (MAX_PAYLOAD_BYTES + 1)):
        assert decode_payload("hk", 1, garbage) is None
    assert decode_payload("hk", 1, None) is None  # type: ignore[arg-type]
    assert decode_payload("", 1, encode_payload(HASH, "https://x")) is None


def test_production_rejects_file_and_plain_http_urls():
    for url in ("file:///tmp/b.tar.gz", "http://example.com/b.tar.gz"):
        with pytest.raises(PayloadError):
            encode_payload(HASH, url)
        raw = json.dumps({"v": PAYLOAD_VERSION, "h": HASH, "u": url})
        assert decode_payload("hk", 7, raw) is None


def test_explicit_test_decoder_accepts_file_url():
    data = encode_payload_for_testing(HASH, "file:///tmp/b.tar.gz")
    assert decode_payload("hk", 7, data) is None
    ref = decode_payload_for_testing("hk", 7, data)
    assert ref is not None and ref.url.startswith("file://")
