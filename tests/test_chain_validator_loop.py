from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import cacheon.chain.validator_loop as loop
from cacheon.arena_service import AdmissionDecision, ArenaService, ArenaServiceRegistry
from cacheon.bundle_hash import content_hash
from cacheon.chain import FinalizedRevealSnapshot, RevealedCommitment
from cacheon.chain.eval_cost import (
    EvalCostFetchError,
    EvalCostPaymentProof,
    EvalCostPolicy,
    EvalCostRequest,
    encode_payment_remark,
    quote_eval_cost,
    verify_eval_cost_payment,
)
from cacheon.chain.intake import FinalizedIntakeStore, IntakePolicy, IntakeScope
from cacheon.chain.payload import encode_payload


BLOCK = 90
BLOCK_HASH = "0x" + "9" * 64
SCOPE = IntakeScope("0x" + "0" * 64, 307)
# Not the identity body: a copy of a public example is demoted before publication.
_NODE_BODY = (
    "def forward(module, hidden_states, *args, **kwargs):\n"
    "    return module.forward(hidden_states.contiguous(), *args, **kwargs)\n"
)


def _bundle(
    root: Path,
    body: str,
    *,
    slot: str = "model.layers.*.mlp",
    entry: str = "forward",
) -> Path:
    (root / "kernels").mkdir(parents=True)
    (root / "manifest.toml").write_text(
        'bundle_id = "test"\n'
        'abi_version = "cacheon-op-abi-v0"\n\n'
        '[[ops]]\n'
        f'slot = "{slot}"\n'
        'source = "kernels/k.py"\n'
        f'entry = "{entry}"\n'
    )
    (root / "kernels/k.py").write_text(body)
    for directory in (root, root / "kernels"):
        directory.chmod(0o700)
    for file in (root / "manifest.toml", root / "kernels/k.py"):
        file.chmod(0o600)
    return root


def _snapshot(rows: list[tuple[str, str]]) -> FinalizedRevealSnapshot:
    reveals = tuple(
        RevealedCommitment(hotkey, payload, BLOCK, BLOCK_HASH, index)
        for index, (hotkey, payload) in enumerate(rows)
    )
    return FinalizedRevealSnapshot(BLOCK, BLOCK_HASH, reveals)


class _NoWeightsSubtensor:
    def get_block_hash(self, block):
        assert block == 0
        return SCOPE.genesis_hash


def _run(
    tmp_path,
    monkeypatch,
    snapshot,
    sources,
    **changes,
):
    monkeypatch.setattr(
        loop.chain,
        "read_finalized_reveal_history",
        lambda *_, **__: snapshot,
    )
    provider = lambda: (snapshot.finalized_block, snapshot.finalized_block_hash)  # noqa: E731
    monkeypatch.setattr(
        loop.chain,
        "read_finalized_head",
        lambda *_: provider(),
    )
    calls = []

    def fetcher(_url, expected, _root):
        calls.append(expected)
        return sources[expected]

    monkeypatch.setattr(loop, "fetch_bundle", fetcher)

    options = dict(
        intake_db=tmp_path / "state" / "intake.sqlite3",
        private_root=tmp_path / "private-cache",
        publication_root=tmp_path / "worker",
        intake_only=True,
    )
    options.update(changes)
    return loop.run_pass(_NoWeightsSubtensor(), 307, **options), calls, options


def test_finalized_reveal_publishes_once_and_restart_reopens(tmp_path, monkeypatch):
    source = _bundle(
        tmp_path / "source",
        _NODE_BODY,
    )
    digest = content_hash(source)
    snapshot = _snapshot([("miner", encode_payload(digest, "https://example.com/a"))])
    result, calls, options = _run(tmp_path, monkeypatch, snapshot, {digest: source})

    assert result.seen == 1 and len(result.reserved) == 1
    assert len(result.published) == 1 and result.decisions == {}
    assert calls == [digest]
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        row = store.all()[0]
        assert row.status == "published"
        assert row.publication_digest == next(iter(result.published.values()))
        assert row.arrival.content_hash == digest

    second, second_calls, _ = _run(
        tmp_path, monkeypatch, snapshot, {digest: source}
    )
    assert second.reserved == [] and second.published == {}
    assert second_calls == []


def test_disabled_eval_cost_ignores_v2_pointer_without_consuming_it(
    tmp_path, monkeypatch
):
    source = _bundle(
        tmp_path / "source",
        _NODE_BODY,
    )
    digest = content_hash(source)
    snapshot = _snapshot(
        [
            (
                "miner",
                encode_payload(
                    digest,
                    "https://example.com/a",
                    payment_block=80,
                    payment_extrinsic_index=4,
                ),
            )
        ]
    )

    def unexpected_lookup(*_args, **_kwargs):
        raise AssertionError("disabled eval-cost must not read a payment")

    monkeypatch.setattr(loop, "read_eval_cost_payment", unexpected_lookup)
    result, calls, options = _run(
        tmp_path, monkeypatch, snapshot, {digest: source}
    )
    assert calls == [digest] and len(result.rejected) == 0
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        assert store.all()[0].status == "published"
        assert store._db.execute(
            "SELECT COUNT(*) FROM eval_cost_payments"
        ).fetchone()[0] == 0


def test_unpaid_v1_is_failed_when_eval_cost_is_required(tmp_path, monkeypatch):
    source = _bundle(
        tmp_path / "source",
        _NODE_BODY,
    )
    digest = content_hash(source)
    snapshot = _snapshot([("miner", encode_payload(digest, "https://example.com/a"))])
    result, calls, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {digest: source},
        policy=IntakePolicy(expiry_blocks=100),
        eval_cost_policy=EvalCostPolicy(amount_rao=10),
    )
    assert calls == [] and len(result.rejected) == 1
    with FinalizedIntakeStore(
        options["intake_db"],
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        row = store.all()[0]
        assert row.status == "failed"
        assert row.reason == "missing_eval_cost_payment"


def _paid_proof(digest: str) -> EvalCostPaymentProof:
    request = EvalCostRequest(netuid=307, hotkey="miner", content_hash=digest)
    quote = quote_eval_cost(
        request,
        policy=EvalCostPolicy(amount_rao=10, destination="treasury"),
        at_block=70,
    )
    return EvalCostPaymentProof(
        block=80,
        extrinsic_index=4,
        signer="coldkey",
        payer="coldkey",
        destination="treasury",
        amount_rao=10,
        remark=encode_payment_remark(request, quote),
    )


def _stub_owner(monkeypatch, dest: str = "treasury") -> None:
    monkeypatch.setattr(
        loop, "read_subnet_owner_coldkey", lambda *_args, **_kwargs: dest
    )


@pytest.mark.parametrize("required_amount", [5, 10])
def test_paid_v2_is_admitted_when_eval_cost_is_required(
    tmp_path, monkeypatch, required_amount,
):
    source = _bundle(
        tmp_path / "source",
        _NODE_BODY,
    )
    digest = content_hash(source)
    snapshot = _snapshot(
        [
            (
                "miner",
                encode_payload(
                    digest,
                    "https://example.com/a",
                    payment_block=80,
                    payment_extrinsic_index=4,
                ),
            )
        ]
    )
    monkeypatch.setattr(
        loop,
        "read_eval_cost_payment",
        lambda *_args, **_kwargs: _paid_proof(digest),
    )
    _stub_owner(monkeypatch)
    result, calls, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {digest: source},
        policy=IntakePolicy(expiry_blocks=100),
        eval_cost_policy=EvalCostPolicy(amount_rao=required_amount),
    )
    assert calls == [digest] and len(result.rejected) == 0
    with FinalizedIntakeStore(
        options["intake_db"],
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        row = store.all()[0]
        assert row.status == "published"
        assert row.arrival.payment_block == 80


@pytest.mark.parametrize("quoted,transferred", [(4, 10), (10, 5)])
def test_payment_must_cover_both_the_fee_and_its_declared_quote(quoted, transferred):
    """An overpayment cannot conceal an underquote or an underfunded remark."""
    request = EvalCostRequest(netuid=307, hotkey="miner", content_hash="a" * 64)
    quote = quote_eval_cost(
        request,
        policy=EvalCostPolicy(amount_rao=quoted, destination="treasury"),
        at_block=70,
    )
    proof = replace(
        _paid_proof(request.content_hash),
        amount_rao=transferred,
        remark=encode_payment_remark(request, quote),
    )
    assert verify_eval_cost_payment(
        request=request,
        policy=EvalCostPolicy(amount_rao=5, destination="treasury"),
        proof=proof,
        reveal_block=90,
    ) == "eval_cost_payment_invalid"


def test_payment_to_a_stale_owner_is_invalid(tmp_path, monkeypatch):
    source = _bundle(
        tmp_path / "source",
        _NODE_BODY,
    )
    digest = content_hash(source)
    snapshot = _snapshot(
        [
            (
                "miner",
                encode_payload(
                    digest,
                    "https://example.com/a",
                    payment_block=80,
                    payment_extrinsic_index=4,
                ),
            )
        ]
    )
    monkeypatch.setattr(
        loop,
        "read_eval_cost_payment",
        lambda *_args, **_kwargs: _paid_proof(digest),
    )
    _stub_owner(monkeypatch, "owner-b")
    result, calls, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {digest: source},
        policy=IntakePolicy(expiry_blocks=100),
        eval_cost_policy=EvalCostPolicy(amount_rao=10),
    )
    assert calls == [] and len(result.rejected) == 1
    with FinalizedIntakeStore(
        options["intake_db"],
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        row = store.all()[0]
        assert row.status == "failed"
        assert row.reason == "eval_cost_payment_invalid"


def test_unrecognizable_payment_pointer_is_invalid(tmp_path, monkeypatch):
    source = _bundle(
        tmp_path / "source",
        _NODE_BODY,
    )
    digest = content_hash(source)
    snapshot = _snapshot(
        [
            (
                "miner",
                encode_payload(
                    digest,
                    "https://example.com/a",
                    payment_block=80,
                    payment_extrinsic_index=4,
                ),
            )
        ]
    )
    monkeypatch.setattr(loop, "read_eval_cost_payment", lambda *_args, **_kwargs: None)
    result, calls, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {digest: source},
        policy=IntakePolicy(expiry_blocks=100),
        eval_cost_policy=EvalCostPolicy(amount_rao=10),
    )
    assert calls == [] and len(result.rejected) == 1
    with FinalizedIntakeStore(
        options["intake_db"],
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        row = store.all()[0]
        assert row.status == "failed"
        assert row.reason == "eval_cost_payment_invalid"


def test_eval_cost_fetch_error_does_not_advance_the_cursor(tmp_path, monkeypatch):
    source = _bundle(
        tmp_path / "source",
        _NODE_BODY,
    )
    digest = content_hash(source)
    snapshot = _snapshot(
        [
            (
                "miner",
                encode_payload(
                    digest,
                    "https://example.com/a",
                    payment_block=80,
                    payment_extrinsic_index=4,
                ),
            )
        ]
    )

    def boom(*_args, **_kwargs):
        raise EvalCostFetchError("rpc blip")

    monkeypatch.setattr(loop, "read_eval_cost_payment", boom)
    with pytest.raises(EvalCostFetchError, match="rpc blip"):
        _run(
            tmp_path,
            monkeypatch,
            snapshot,
            {digest: source},
            policy=IntakePolicy(expiry_blocks=100),
            eval_cost_policy=EvalCostPolicy(amount_rao=10),
        )
    with FinalizedIntakeStore(
        tmp_path / "state" / "intake.sqlite3",
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        assert store.finalized_cursor() is None
        assert store.all() == ()


def test_malformed_finalized_payload_is_reserved_and_never_fetched(tmp_path, monkeypatch):
    snapshot = _snapshot([("miner", "not-json")])
    result, calls, options = _run(tmp_path, monkeypatch, snapshot, {})
    assert calls == [] and len(result.reserved) == 1
    assert len(result.rejected) == 1
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        row = store.all()[0]
        assert row.status == "failed" and row.reason == "invalid_payload"


def test_deterministically_unpublishable_submission_is_not_retried(
    tmp_path, monkeypatch
):
    source = _bundle(
        tmp_path / "source",
        _NODE_BODY,
    )
    reserved = source / ".cacheon-native-artifact.json"
    reserved.write_text("{}\n")
    reserved.chmod(0o600)
    digest = content_hash(source)
    snapshot = _snapshot(
        [("miner", encode_payload(digest, "https://example.com/a"))]
    )
    result, _calls, options = _run(
        tmp_path, monkeypatch, snapshot, {digest: source}
    )
    assert len(result.rejected) == 1
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        row = store.all()[0]
        assert row.status == "failed"
        assert row.reason.startswith("publication_source:")


def test_reformatted_later_delta_is_copy_without_any_weight_edge(tmp_path, monkeypatch):
    first = _bundle(
        tmp_path / "first",
        "import torch\n\ndef forward(module, x):\n"
        "    d = x.shape[-1] // 2\n"
        "    return module.forward(torch.nn.functional.silu(x[..., :d]) * x[..., d:])\n",
    )
    second = _bundle(
        tmp_path / "second",
        "import torch\n\n# formatting only\ndef forward(module, x):\n"
        "    d = (x.shape[-1] // 2)\n"
        "    return module.forward((torch.nn.functional.silu(x[..., :d]) * x[..., d:]))\n",
    )
    first_hash, second_hash = content_hash(first), content_hash(second)
    assert first_hash != second_hash
    snapshot = _snapshot([
        ("author", encode_payload(first_hash, "https://example.com/a")),
        ("copycat", encode_payload(second_hash, "https://example.com/b")),
    ])
    result, _calls, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {first_hash: first, second_hash: second},
    )
    assert len(result.published) == 1 and len(result.copies) == 1
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        rows = store.all()
        assert [row.status for row in rows] == ["published", "failed"]
        assert rows[1].reason.startswith("copy_of:")


def _colliding_store(monkeypatch, collisions):
    """Install a store factory that reports a peer's lock hold ``collisions`` times."""

    from cacheon.chain.intake import IntakeError, _LOCK_COLLISION_MESSAGE

    attempts = []

    class Store:
        def __init__(self, *_args, **_kwargs):
            attempts.append(1)
            if len(attempts) <= collisions:
                raise IntakeError(_LOCK_COLLISION_MESSAGE)

    monkeypatch.setattr(loop, "FinalizedIntakeStore", Store)
    return Store, attempts


def test_intake_pass_waits_out_a_peer_controllers_lock_hold(monkeypatch, tmp_path):
    store_type, attempts = _colliding_store(monkeypatch, collisions=2)
    naps = []

    store = loop._open_store(tmp_path / "i.sqlite3", None, None, sleep=naps.append)

    assert isinstance(store, store_type)
    assert len(attempts) == 3
    assert naps == [loop._LOCK_RETRY_PAUSE_S] * 2


def test_intake_pass_gives_up_on_a_lock_that_outlasts_the_wait(monkeypatch, tmp_path):
    from cacheon.chain.intake import IntakeError

    _store_type, attempts = _colliding_store(monkeypatch, collisions=10_000)
    naps = []

    with pytest.raises(IntakeError, match="another intake controller"):
        loop._open_store(tmp_path / "i.sqlite3", None, None, sleep=naps.append)

    assert len(attempts) == loop._LOCK_RETRY_ATTEMPTS
    assert len(naps) == loop._LOCK_RETRY_ATTEMPTS - 1


def test_intake_pass_does_not_retry_other_store_errors(monkeypatch, tmp_path):
    from cacheon.chain.intake import IntakeError

    attempts = []

    class Store:
        def __init__(self, *_args, **_kwargs):
            attempts.append(1)
            raise IntakeError("intake store schema is corrupt")

    monkeypatch.setattr(loop, "FinalizedIntakeStore", Store)
    naps = []

    with pytest.raises(IntakeError, match="corrupt"):
        loop._open_store(tmp_path / "i.sqlite3", None, None, sleep=naps.append)

    assert attempts == [1] and naps == []


def test_intake_only_pass_never_moves_the_incumbent(tmp_path, monkeypatch):
    calls = []

    def recorder(store, *, current_block, finalized_block_provider):
        calls.append(current_block)
        return {"lease-digest": "plan-digest"}

    monkeypatch.setattr(loop, "_settle_pending", recorder)
    snapshot = _snapshot([])
    result, _fetches, _options = _run(
        tmp_path, monkeypatch, snapshot, {}, intake_only=True
    )
    assert calls == []
    assert result.settlements == {}


def test_settlement_refreshes_stale_pass_height_before_leasing():
    class Store:
        lease_blocks = []

        def has_pending_settlement(self):
            return True

        def lease_settlement_cohort(self, *, current_block):
            self.lease_blocks.append(current_block)
            return None

    store = Store()
    assert loop._settle_pending(
        store,
        current_block=BLOCK,
        finalized_block_provider=lambda: BLOCK + 100,
    ) == {}
    assert store.lease_blocks == [BLOCK + 100]


def test_settlement_head_refresh_failure_cannot_create_a_lease():
    class Store:
        lease_calls = 0

        def has_pending_settlement(self):
            return True

        def lease_settlement_cohort(self, *, current_block):
            self.lease_calls += 1
            return None

    def unavailable_head():
        raise RuntimeError("finalized head unavailable")

    store = Store()
    with pytest.raises(RuntimeError, match="finalized head unavailable"):
        loop._settle_pending(
            store,
            current_block=BLOCK,
            finalized_block_provider=unavailable_head,
        )
    assert store.lease_calls == 0


def test_closed_target_parks_by_name_only_and_fused_closed_slot_math_passes(
    tmp_path, monkeypatch
):
    """Closing a target closes its standalone lane only.

    The closed-family check keys on the SUBMITTED target name. A bundle for an
    open target whose kernel body computes -- and even names -- a closed slot's
    math is never parked for it: what an implementation absorbs inside its own
    boundary is not a rejection surface. Normative statement:
    docs/architecture/slot-contract.md.
    """
    closed = _bundle(
        tmp_path / "closed-src",
        _NODE_BODY,
    )
    fused = _bundle(
        tmp_path / "fused-src",
        '"""Folds the forward_pass model.layers.*.mlp node into the prefix cache."""\n'
        "def forward(cache):\n"
        "    forward_pass = type(cache)\n"
        "    return forward_pass\n",
        slot="tree_cache",
    )
    closed_digest = content_hash(closed)
    fused_digest = content_hash(fused)
    snapshot = _snapshot([
        ("miner-closed", encode_payload(closed_digest, "https://example.com/a")),
        ("miner-fused", encode_payload(fused_digest, "https://example.com/b")),
    ])

    service = object.__new__(ArenaService)
    service.manifest = type(
        "Manifest",
        (),
        {
            "digest": "e" * 64,
            "qualification_policy_digest": "f" * 64,
            "capacity": type("Capacity", (), {"max_cohort_size": 1})(),
            "closed_targets": ("forward_pass", "attention.sdpa"),
        },
    )()
    registry = object.__new__(ArenaServiceRegistry)
    monkeypatch.setattr(ArenaServiceRegistry, "require", lambda *_: service)
    monkeypatch.setattr(
        ArenaService, "admit_qualification", lambda *_args, **_kwargs: AdmissionDecision.QUEUE
    )

    result, _calls, options = _run(
        tmp_path,
        monkeypatch,
        snapshot,
        {closed_digest: closed, fused_digest: fused},
        intake_only=False,
        arena_registry=registry,
        arena_id="test-arena",
    )

    assert list(result.rejected.values()) == [
        "target_unavailable:forward_pass"
    ]
    assert len(result.published) == 1
    with FinalizedIntakeStore(options["intake_db"], scope=SCOPE) as store:
        by_hotkey = {row.arrival.hotkey: row for row in store.all()}
        parked = by_hotkey["miner-closed"]
        assert parked.status == "expired"
        assert parked.decision == "NO_DECISION"
        passed = by_hotkey["miner-fused"]
        assert passed.status == "published"
        assert passed.reason == ""
