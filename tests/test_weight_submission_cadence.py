"""A changed offer must not create phantom pending state inside the chain rate limit."""

import pytest

from cacheon.chain import ChainWeightStateError
from cacheon.chain.weights import WeightPublicationRecord, reconcile_weight_publication
from tests.test_weight_publication import Chain, Journal, _projection, _wallet


@pytest.mark.parametrize("netuid,rate_limit", [(1, 10), (14, 100)])
def test_changed_offer_waits_before_intent_then_submits_once(netuid, rate_limit):
    chain = Chain(apply=True)
    chain.rate_limit = rate_limit
    chain.install({"alice": 1.0}, update=95)
    previous = _projection(block=95, netuid=netuid)
    confirmed = WeightPublicationRecord(
        previous.digest, "confirmed", confirmed_block=95,
        confirmed_last_update=95, reason="authoritative_readback")
    journal = Journal(confirmed, retained=(previous,))
    proposed = _projection(weights=(("bob", 1_000_000),), netuid=netuid)

    for block in (100, 95 + rate_limit):
        chain.block = block
        proposed = _projection(weights=(("bob", 1_000_000),), block=block, netuid=netuid)
        with pytest.raises(ChainWeightStateError, match=f"rate limit until block {96 + rate_limit}") as error:
            reconcile_weight_publication(chain, _wallet(), proposed, journal, refresh_blocks=600)
        assert error.value.retryable
    assert journal.row is confirmed and journal.history == []
    assert chain.submit_calls == 0

    chain.block = 96 + rate_limit
    proposed = _projection(weights=(("bob", 1_000_000),), block=chain.block, netuid=netuid)
    result = reconcile_weight_publication(chain, _wallet(), proposed, journal, refresh_blocks=600)
    assert result.status == "confirmed" and result.chain_matches
    assert chain.submit_calls == 1
    assert [row.status for row in journal.history] == ["intent", "pending", "confirmed"]
