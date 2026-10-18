"""Publication-journal guards not yet proven through the signer CLI, and the follower tests' doubles.

``tests/test_validator_weights.py`` drives the rest of the journal through ``cacheon set-weights``.
"""

from __future__ import annotations

import types

import pytest

from cacheon import chain as chain_module
from cacheon.chain.weights import (
    StaleWeightProjectionError,
    WeightProjection,
    WeightPublicationError,
    WeightPublicationRecord,
    reconcile_weight_publication,
    resume_weight_projection,
)
from cacheon.stack_identity import canonical_digest


def _d(char: str) -> str:
    return char * 64


class Journal:
    def __init__(self, row=None, retained=()):
        self.row = row
        self.history = []
        self.retained = {projection.digest: projection for projection in retained}

    def load(self):
        return self.row

    def compare_and_swap(self, expected_record_digest, replacement):
        assert expected_record_digest == (self.row.digest if self.row else None)
        self.row = replacement
        self.history.append(replacement)

    def retained_projection(self, projection_digest):
        return self.retained[projection_digest]


class Chain:
    def __init__(self, *, apply=False, block=100):
        self.block = block
        self.hotkeys = ["validator", "alice", "bob"]
        self.last_update = [0, 0, 0]
        self.row = []
        self.apply = apply
        self.submit_calls = 0
        self.weight_reads = 0
        self.metagraph_reads: list[int | None] = []
        self.weight_read_blocks: list[int | None] = []
        self.submit_options: list[tuple[bool, bool]] = []

    def metagraph(self, netuid=None, block=None):
        self.metagraph_reads.append(block)
        return types.SimpleNamespace(
            uids=[0, 1, 2], hotkeys=self.hotkeys,
            last_update=self.last_update, validator_permit=[True, True, True],
            block=self.block if block is None else block,
        )

    def weights(self, netuid=None, block=None):
        self.weight_reads += 1
        self.weight_read_blocks.append(block)
        return [(0, list(self.row))] if self.row else []

    def get_current_block(self):
        return self.block

    def get_finalized_block_number(self):
        return self.block

    def get_block_hash(self, block):
        return "0x" + f"{block:064x}"

    def get_hyperparameter(self, param_name, netuid=None, block=None):
        assert param_name in {"WeightsVersionKey", "WeightsSetRateLimit"}
        return {"WeightsVersionKey": 29, "WeightsSetRateLimit": getattr(self, "rate_limit", 0)}[param_name]

    def set_weights(self, **kwargs):
        self.submit_calls += 1
        self.submit_options.append(
            (kwargs["wait_for_inclusion"], kwargs["wait_for_finalization"])
        )
        if self.apply:
            self.row = [
                (uid, round(weight * 65_535))
                for uid, weight in zip(kwargs["uids"], kwargs["weights"], strict=True)
            ]
            self.last_update[0] = self.block
        return True

    def install(self, weights, *, update):
        index = {hotkey: uid for uid, hotkey in enumerate(self.hotkeys)}
        self.row = [
            (index[hotkey], round(weight * 65_535))
            for hotkey, weight in weights.items()
        ]
        self.last_update[0] = update


class ReassigningChain(Chain):
    """Historical UID 1 is alice; finalized block 101 reassigns it."""

    def __init__(self, *, finalized_heads, apply=False):
        super().__init__(apply=apply, block=100)
        self._finalized_heads = iter(finalized_heads)
        self._last_finalized_head = 100

    def get_finalized_block_number(self):
        try:
            self._last_finalized_head = next(self._finalized_heads)
        except StopIteration:
            pass
        return self._last_finalized_head

    def metagraph(self, netuid=None, block=None):
        self.metagraph_reads.append(block)
        requested = self._last_finalized_head if block is None else block
        hotkeys = (
            ["validator", "alice", "bob"]
            if requested <= 100
            else ["validator", "mallory", "alice"]
        )
        return types.SimpleNamespace(
            uids=[0, 1, 2],
            hotkeys=hotkeys,
            last_update=self.last_update,
            validator_permit=[True, True, True],
            block=requested,
        )


class SameHeightHashChangingChain(Chain):
    """Returns a different canonical hash on the pre-sign same-height fetch."""

    def __init__(self):
        super().__init__(block=100)
        self._hash_reads = 0

    def get_block_hash(self, block):
        self._hash_reads += 1
        suffix = block if self._hash_reads <= 6 else block + 1
        return "0x" + f"{suffix:064x}"


def _projection(
    *, crowns=1, weights=(("alice", 1_000_000),), marker="a", block=100, netuid=1
):
    metagraph_digest = canonical_digest(
        "cacheon.economics.metagraph-membership",
        {
            "block": block,
            "block_hash": "0x" + f"{block:064x}",
            "chain_scope_digest": _d("1"),
            "members": [
                {"hotkey": hotkey, "uid": uid}
                for uid, hotkey in enumerate(("validator", "alice", "bob"))
            ],
        },
    )
    return WeightProjection(
        _d("1"), netuid, "validator", _d("2"), _d(marker), _d("4"),
        metagraph_digest, (_d("6"),), 3, block, crowns,
        ((_d("5"),) if crowns else ()), tuple(weights),
    )


def _wallet(hotkey="validator"):
    return types.SimpleNamespace(hotkey=types.SimpleNamespace(ss58_address=hotkey))


def test_stale_preexisting_vector_without_a_journal_is_refreshed_once():
    chain, journal = Chain(apply=True, block=121), Journal()
    projection = _projection(block=121)
    chain.install(projection.weights, update=100)

    result = reconcile_weight_publication(
        chain, _wallet(), projection, journal, refresh_blocks=20
    )

    assert result.status == "confirmed"
    assert result.submitted is True
    assert chain.submit_calls == 1
    assert [row.status for row in journal.history] == [
        "intent", "pending", "confirmed"
    ]
    assert journal.row.confirmed_last_update == 121


def test_pending_resume_rejects_a_different_chain_authority():
    original = _projection()
    pending = WeightPublicationRecord(
        original.digest,
        "pending",
        submit_block=100,
        retry_after_block=120,
        reason="sdk_result_unconfirmed",
    )
    journal = Journal(pending, retained=(original,))
    proposed = WeightProjection.from_dict(
        {**_projection(block=101).to_dict(), "validator_hotkey": "other"}
    )

    with pytest.raises(WeightPublicationError, match="current chain authority"):
        resume_weight_projection(proposed, journal)


def test_real_submission_requires_crown_and_exact_signer_before_intent():
    for projection, wallet, message in (
        (_projection(crowns=0), _wallet(), "current crown"),
        (_projection(), _wallet("other"), "signer wallet"),
    ):
        chain, journal = Chain(), Journal()
        with pytest.raises(WeightPublicationError, match=message):
            reconcile_weight_publication(
                chain, wallet, projection, journal, refresh_blocks=20
            )
        assert journal.history == [] and chain.submit_calls == 0


def test_stale_projection_is_rejected_before_journal_or_signing():
    chain, journal = Chain(block=101), Journal()
    with pytest.raises(WeightPublicationError, match="stale"):
        reconcile_weight_publication(
            chain, _wallet(), _projection(block=100), journal, refresh_blocks=20
        )
    assert journal.history == [] and chain.submit_calls == 0


def test_explicit_stale_initial_catch_up_requires_stable_recipient_uids():
    projection = _projection(block=100)
    stable = Chain(apply=True, block=101)
    journal = Journal()

    result = reconcile_weight_publication(
        stable,
        _wallet(),
        projection,
        journal,
        refresh_blocks=20,
        allow_stale_initial=True,
        max_stale_initial_blocks=20,
    )

    assert result.status == "confirmed"
    assert result.submitted is True
    assert stable.submit_calls == 1

    reassigned = ReassigningChain(finalized_heads=[101, 101])
    reassigned.block = 101
    journal = Journal()
    with pytest.raises(WeightPublicationError, match="UID mapping changed"):
        reconcile_weight_publication(
            reassigned,
            _wallet(),
            projection,
            journal,
            refresh_blocks=20,
            allow_stale_initial=True,
            max_stale_initial_blocks=20,
        )
    assert journal.history == [] and reassigned.submit_calls == 0


def test_stale_initial_catch_up_is_bounded_and_retryable():
    projection = _projection(block=100)
    stale = Chain(block=121)
    journal = Journal()

    with pytest.raises(
        StaleWeightProjectionError, match="freshness window"
    ) as raised:
        reconcile_weight_publication(
            stale,
            _wallet(),
            projection,
            journal,
            refresh_blocks=20,
            allow_stale_initial=True,
            max_stale_initial_blocks=20,
        )

    assert raised.value.retryable is True
    assert journal.history == []
    assert stale.submit_calls == 0


def test_same_height_authority_hash_change_aborts_without_intent():
    chain = SameHeightHashChangingChain()
    journal = Journal()

    with pytest.raises(
        chain_module.ChainWeightStateError,
        match="changed at the projection block",
    ):
        reconcile_weight_publication(
            chain, _wallet(), _projection(block=100), journal, refresh_blocks=20
        )

    assert journal.history == []
    assert chain.submit_calls == 0


def test_same_height_post_submit_authority_change_holds_not_confirms(
    monkeypatch,
):
    projection = _projection(block=100)
    subtensor = Chain(apply=True, block=100)
    journal = Journal()
    original_fetch = chain_module.fetch_metagraph
    calls = 0

    def changing_fetch(target, netuid, *, block=None):
        nonlocal calls
        calls += 1
        view = original_fetch(target, netuid, block=block)
        if calls == 5:
            return chain_module.MetagraphView(
                netuid=view.netuid,
                block=view.block,
                block_hash="0x" + f"{view.block + 1:064x}",
                uids=list(view.uids),
                hotkeys=list(view.hotkeys),
                validator_permit=list(view.validator_permit),
                last_update=list(view.last_update),
            )
        return view

    monkeypatch.setattr(chain_module, "fetch_metagraph", changing_fetch)
    result = reconcile_weight_publication(
        subtensor, _wallet(), projection, journal, refresh_blocks=20
    )

    assert result.status == "held"
    assert result.record is not None
    assert result.record.reason == "post_submit_authority_unavailable"
    assert result.submitted is True
    assert [row.status for row in journal.history] == ["intent", "pending", "held"]


def test_best_head_only_sparse_row_never_confirms():
    class BestHeadOnlyChain(Chain):
        def set_weights(self, **kwargs):
            self.submit_calls += 1
            self.submit_options.append(
                (kwargs["wait_for_inclusion"], kwargs["wait_for_finalization"])
            )
            # Simulate a misleading best-head row which the exact finalized
            # weights(block=100) reader below intentionally never exposes.
            self.best_head_row = [(1, 65_535)]
            return True

    chain = BestHeadOnlyChain()
    journal = Journal()
    result = reconcile_weight_publication(
        chain, _wallet(), _projection(block=100), journal, refresh_blocks=20
    )

    assert result.status == "pending"
    assert result.chain_matches is False
    assert result.submitted is True
    assert set(chain.weight_read_blocks) == {100}
    assert chain.submit_options == [(True, True)]


def test_released_post_submit_but_refresh_due_row_is_refreshed_once():
    original = _projection(block=100)
    released = WeightPublicationRecord(
        original.digest,
        "released",
        submit_block=100,
        retry_after_block=120,
        reason="late reveal reviewed",
    )
    journal = Journal(released)
    chain = Chain(apply=True, block=121)
    chain.install(original.weights, update=100)
    current = _projection(block=121)

    result = reconcile_weight_publication(
        chain, _wallet(), current, journal, refresh_blocks=20
    )

    assert result.status == "confirmed"
    assert result.submitted is True
    assert chain.submit_calls == 1
    assert [row.status for row in journal.history] == [
        "intent", "pending", "confirmed"
    ]
    assert journal.row.confirmed_last_update == 121


def test_reconcile_only_advances_fresh_confirmed_projection_from_recent_readback():
    original = _projection(block=100)
    confirmed = WeightPublicationRecord(
        original.digest,
        "confirmed",
        confirmed_block=100,
        confirmed_last_update=100,
        reason="authoritative_readback",
    )
    journal = Journal(confirmed)
    chain = Chain(block=101)
    chain.install(original.weights, update=100)
    current = _projection(block=101)

    result = reconcile_weight_publication(
        chain,
        None,
        current,
        journal,
        refresh_blocks=20,
        reconcile_only=True,
    )

    assert result.status == "confirmed"
    assert result.projection_digest == current.digest
    assert result.submitted is False
    assert chain.submit_calls == 0
    assert [row.status for row in journal.history] == ["confirmed"]
    assert journal.row.projection_digest == current.digest
    assert journal.row.prior_record_digest == confirmed.digest


def test_reconcile_only_released_refresh_due_row_fails_without_mutation():
    projection = _projection(block=100)
    released = WeightPublicationRecord(
        projection.digest,
        "released",
        submit_block=100,
        retry_after_block=120,
        reason="late reveal reviewed",
    )
    journal = Journal(released)
    chain = Chain(block=121)
    chain.install(projection.weights, update=100)

    with pytest.raises(WeightPublicationError, match="refresh-due"):
        reconcile_weight_publication(
            chain,
            None,
            projection,
            journal,
            refresh_blocks=20,
            reconcile_only=True,
        )

    assert journal.row == released
    assert journal.history == []
    assert chain.submit_calls == 0


def test_reconcile_only_rejects_a_signer_before_chain_or_journal_access():
    chain, journal = Chain(), Journal()

    with pytest.raises(WeightPublicationError, match="forbids a signer"):
        reconcile_weight_publication(
            chain,
            _wallet(),
            _projection(),
            journal,
            refresh_blocks=20,
            reconcile_only=True,
        )

    assert chain.metagraph_reads == []
    assert chain.weight_reads == 0
    assert journal.history == []
