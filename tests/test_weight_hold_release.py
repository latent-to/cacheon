"""A retained publication hold releases itself only when finalized state proves its attempt never landed."""

import pytest

from cacheon.chain.weights import WeightPublicationRecord, reconcile_weight_publication
from tests.test_weight_publication import Chain, Journal, _projection, _wallet


@pytest.mark.parametrize("record,last_update,commits,released", [
    ({}, 99, [("alice", 101, "cipher", 7)], True),  # only another validator's commit is outstanding
    ({}, 99, [("validator", 101, "cipher", 7)], False),  # our own commit may still reveal
    ({}, 100, [], False),  # the row is not older than the attempt
    ({"retry_after_block": 140}, 99, [], False),  # the attempt's deadline has not passed
    ({"confirmed_block": 110, "confirmed_last_update": 105}, 99, [], False),  # a changed confirmed vector: operator's call
])
def test_hold_over_an_attempt_absent_from_chain_releases_itself(record, last_update, commits, released):
    """2026-09-19: the signer died after journaling intent; nothing reached chain, and it held for 41 hours."""
    original = _projection(block=100)
    held = WeightPublicationRecord(original.digest, "held", **{
        "submit_block": 100, "retry_after_block": 120, "reason": "publication_readback_deadline_expired", **record})
    journal, chain = Journal(held), Chain(apply=True, block=121)
    chain.install(original.weights, update=last_update)
    chain.get_timelocked_weight_commits = lambda netuid, block=None: commits

    result = reconcile_weight_publication(chain, _wallet(), _projection(block=121), journal, refresh_blocks=20)

    if released:
        assert result.status == "confirmed" and chain.submit_calls == 1
        assert [row.status for row in journal.history] == ["released", "intent", "pending", "confirmed"]
        assert journal.history[0].reason == "attempt_absent_from_finalized_chain"
    else:
        assert result.status == "held" and chain.submit_calls == 0
        assert [row.reason for row in journal.history] == ["projection_changed_while_unresolved"]
