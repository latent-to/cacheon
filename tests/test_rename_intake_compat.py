"""Durable intake identities derive from the Cacheon hash domains."""

from __future__ import annotations

from cacheon.chain.intake import FinalizedArrival
from cacheon.stack_identity import canonical_digest


def test_finalized_arrival_identity_derives_from_cacheon_domains() -> None:
    arrival = FinalizedArrival(
        hotkey="miner",
        content_hash="2" * 64,
        url="https://example.com/candidate.tar.gz",
        block=10,
        block_hash="0x" + "3" * 64,
        event_index=4,
    )
    payload_digest = canonical_digest(
        "cacheon.chain.finalized-payload",
        {"content_hash": arrival.content_hash, "url": arrival.url},
    )
    assert arrival.payload_digest == payload_digest
    assert arrival.reservation_id == canonical_digest(
        "cacheon.chain.finalized-arrival",
        {
            "block": arrival.block,
            "block_hash": arrival.block_hash,
            "content_hash": arrival.content_hash,
            "event_index": arrival.event_index,
            "event_subindex": arrival.event_subindex,
            "hotkey": arrival.hotkey,
            "payload_digest": payload_digest,
            "url": arrival.url,
        },
    )
