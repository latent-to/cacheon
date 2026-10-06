"""A successful commit ends follower publication; old holds and failed commits do not gate it."""

import pytest

from cacheon.chain.weights import WeightPublicationRecord
from cacheon.chain.weight_commit import commit_weight_publication
from tests.test_weight_publication import Chain, Journal, _projection, _wallet


@pytest.mark.parametrize("status", ["intent", "pending", "held", "released"])
@pytest.mark.parametrize("netuid,recipient", [(14, "alice"), (307, "bob")])
def test_old_unresolved_attempt_does_not_block_latest_commit(status, netuid, recipient):
    old = _projection(block=80, netuid=netuid)
    record = WeightPublicationRecord(old.digest, status, submit_block=80,
                                    retry_after_block=90, reason="historical attempt")
    journal = Journal(record, retained=(old,))
    live = Chain()
    latest = _projection(netuid=netuid, weights=((recipient, 1_000_000),))
    result = commit_weight_publication(live, _wallet(), latest, journal, refresh_blocks=20)
    assert result.submitted and result.status == "confirmed" and not result.chain_matches
    assert result.record.reason == "block_inclusion"
    assert live.submit_calls == 1 and live.weight_reads == 0
    assert result.projection_digest == latest.digest


@pytest.mark.parametrize("failure", [False, TimeoutError("lost RPC response")])
def test_failed_commit_retries_latest_offer_next_pass(failure, capsys):
    live, journal = Chain(), Journal()
    original = _projection()
    journal.retained[original.digest] = original
    submit = live.set_weights

    def fail(**kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure

    live.set_weights = fail
    result = commit_weight_publication(live, _wallet(), original, journal, refresh_blocks=20)
    assert result.status == "released" and not result.submitted
    assert "retry next pass" in capsys.readouterr().out
    live.set_weights = submit
    live.block += 1
    latest = _projection(block=live.block, weights=(("bob", 1_000_000),))
    result = commit_weight_publication(live, _wallet(), latest, journal, refresh_blocks=20)
    assert result.submitted and result.projection_digest == latest.digest
    assert live.weight_reads == 0


def test_unchanged_commit_refreshes_on_cadence_without_readback():
    live, journal = Chain(), Journal()
    original = _projection()
    journal.retained[original.digest] = original
    commit_weight_publication(live, _wallet(), original, journal, refresh_blocks=20)
    live.block = 119
    result = commit_weight_publication(live, _wallet(), _projection(block=119), journal, refresh_blocks=20)
    assert not result.submitted
    live.block = 120
    result = commit_weight_publication(live, _wallet(), _projection(block=120), journal, refresh_blocks=20)
    assert result.submitted and live.submit_calls == 2 and live.weight_reads == 0
