"""Offline contract tests for the miner submission entry point.

``submit_bundle(dry_run=True)`` returns before any chain object is touched,
so the full miner-side pipeline — bundle identity hash, payload encoding,
wire-policy enforcement — is testable with no subtensor and no wallet.
Passing ``None`` for both proves it: any chain interaction would raise
``AttributeError``, not the typed refusals asserted here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheon.bundle_hash import content_hash
from cacheon.chain.payload import MAX_PAYLOAD_BYTES, PayloadError, decode_payload
from cacheon.chain.submit import submit_bundle

_BUNDLE = Path(__file__).resolve().parent.parent / "examples" / "miner_silu_torch"
_URL = "https://example.com/bundles/miner-silu-torch.tar"


def test_dry_run_builds_the_exact_commitment_without_a_chain() -> None:
    result = submit_bundle(None, None, 1, _BUNDLE, _URL, dry_run=True)
    assert result["dry_run"] is True
    assert result["submitted"] is False
    assert result["content_hash"] == content_hash(_BUNDLE)
    assert result["payload"] == result["data"]
    assert len(result["payload"].encode("utf-8")) <= MAX_PAYLOAD_BYTES

    ref = decode_payload("hotkey-under-test", 7, result["payload"])
    assert ref is not None
    assert ref.content_hash == result["content_hash"]
    assert ref.url == _URL
    assert ref.hotkey == "hotkey-under-test"
    assert ref.block == 7


def test_dry_run_refuses_plaintext_http_before_the_chain() -> None:
    with pytest.raises(PayloadError):
        submit_bundle(
            None, None, 1, _BUNDLE, "http://example.com/bundle.tar", dry_run=True
        )


def test_dry_run_refuses_a_payload_over_the_chain_cap() -> None:
    oversized = "https://example.com/" + "a" * MAX_PAYLOAD_BYTES
    with pytest.raises(PayloadError):
        submit_bundle(None, None, 1, _BUNDLE, oversized, dry_run=True)
