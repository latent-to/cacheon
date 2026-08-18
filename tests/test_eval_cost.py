from __future__ import annotations

from pathlib import Path

import pytest

from cacheon.bundle_hash import content_hash
from cacheon.chain import mainnet_screen_dispatcher as dispatcher_module
from cacheon.chain.eval_cost import (
    DEFAULT_EVAL_COST_QUOTE_TTL_BLOCKS,
    PUBLISHED_EVAL_COST_TAO_RAO,
    REASON_INVALID,
    REASON_MISSING,
    REASON_QUOTE_EXPIRED,
    REASON_WINDOW,
    EvalCostCommitError,
    EvalCostError,
    EvalCostFetchError,
    EvalCostPaymentProof,
    EvalCostPolicy,
    EvalCostRequest,
    encode_payment_remark,
    proof_from_decoded_extrinsic,
    quote_eval_cost,
    unused_eval_cost_retry_flags,
    verify_eval_cost_payment,
)
from cacheon.chain.eval_cost_payment import bind_eval_cost_payment, read_subnet_owner_coldkey
from cacheon.chain.intake import (
    FinalizedArrival,
    FinalizedIntakeStore,
    IntakePolicy,
    IntakeScope,
)
from cacheon.chain.payload import (
    decode_payload,
    encode_payload,
)
from cacheon.chain.submit import submit_bundle


HASH = "a" * 64
SCOPE = IntakeScope("0x" + "0" * 64, 307)
TREASURY = "treasury"
_BUNDLE = Path(__file__).resolve().parent.parent / "examples" / "miner_silu_torch"
_PAID_AMOUNT = 10


def test_shared_intake_policy_keeps_the_sealed_dispatch_shape() -> None:
    fields = (
        "epoch_blocks",
        "cutoff_blocks",
        "max_pending",
        "max_per_hotkey_epoch",
        "max_per_target_epoch",
        "max_transport_retries",
        "max_qualification_retries",
        "max_cohort",
        "expiry_blocks",
    )
    assert tuple(IntakePolicy.__dataclass_fields__) == fields
    assert dispatcher_module._POLICY_FIELDS == frozenset(fields)


def _request(**changes) -> EvalCostRequest:
    values = dict(netuid=307, hotkey="miner", content_hash=HASH)
    values.update(changes)
    return EvalCostRequest(**values)


def _quote(**changes) -> tuple[EvalCostRequest, object]:
    request = _request()
    values = dict(amount_rao=10, destination=TREASURY, at_block=70)
    values.update(changes)
    at_block = values.pop("at_block")
    policy_kwargs = {
        key: values[key]
        for key in (
            "amount_rao",
            "destination",
            "quote_ttl_blocks",
            "payment_window_blocks",
        )
        if key in values
    }
    if "amount_rao" not in policy_kwargs:
        policy_kwargs["amount_rao"] = 10
    if "destination" not in policy_kwargs:
        policy_kwargs["destination"] = TREASURY
    quote = quote_eval_cost(
        request,
        policy=EvalCostPolicy(**policy_kwargs),
        at_block=at_block,
    )
    return request, quote


def _proof(**changes) -> EvalCostPaymentProof:
    request, quote = _quote()
    values = dict(
        block=80,
        extrinsic_index=4,
        signer="coldkey",
        payer="coldkey",
        destination=TREASURY,
        amount_rao=10,
        remark=encode_payment_remark(request, quote),
    )
    values.update(changes)
    return EvalCostPaymentProof(**values)


def test_v1_quote_ignores_submission_extras() -> None:
    base = quote_eval_cost(_request(), at_block=70)
    varied = quote_eval_cost(
        _request(target_id="activation.silu_and_mul", submission={"gpu_s": 1800}),
        at_block=70,
    )
    assert base == varied
    assert base.amount_rao == PUBLISHED_EVAL_COST_TAO_RAO
    assert base.instrument == "transfer_keep_alive"
    assert base.asset == "tao"
    assert base.issued_block == 70
    assert base.expires_block == 70 + DEFAULT_EVAL_COST_QUOTE_TTL_BLOCKS


def test_quote_stays_valid_through_payment_ttl() -> None:
    request, quote = _quote(quote_ttl_blocks=5, at_block=70)
    assert quote.expires_block == 75
    policy = EvalCostPolicy(
        amount_rao=10, destination=TREASURY, quote_ttl_blocks=5
    )
    assert (
        verify_eval_cost_payment(
            request=request,
            policy=policy,
            proof=_proof(remark=encode_payment_remark(request, quote), block=75),
            reveal_block=90,
        )
        == ""
    )
    assert (
        verify_eval_cost_payment(
            request=request,
            policy=policy,
            proof=_proof(remark=encode_payment_remark(request, quote), block=76),
            reveal_block=90,
        )
        == REASON_QUOTE_EXPIRED
    )


def test_payment_remark_round_trip_binds_content_hash_and_destination() -> None:
    request, quote = _quote()
    remark = encode_payment_remark(request, quote)
    other_hash = encode_payment_remark(_request(content_hash="b" * 64), quote)
    other_dest = encode_payment_remark(
        request, quote_eval_cost(request, policy=EvalCostPolicy(amount_rao=10, destination="other"), at_block=70)
    )
    assert remark != other_hash
    assert remark != other_dest
    assert "cacheon.chain.eval-cost-payment.v1" in remark
    assert "issued_block" in remark
    assert TREASURY in remark


def test_verify_accepts_matching_transfer() -> None:
    request, quote = _quote()
    assert (
        verify_eval_cost_payment(
            request=request,
            policy=EvalCostPolicy(amount_rao=10, destination=TREASURY),
            proof=_proof(),
            reveal_block=90,
        )
        == ""
    )
    assert quote.amount_rao == 10
    assert quote.destination == TREASURY


def test_verify_rejects_missing_wrong_stale_and_expired_payments() -> None:
    request = _request()
    policy = EvalCostPolicy(
        amount_rao=10, destination=TREASURY, payment_window_blocks=10
    )
    cases = [
        (None, REASON_MISSING),
        (_proof(signer="other"), REASON_INVALID),
        (_proof(destination="other"), REASON_INVALID),
        (_proof(amount_rao=9), REASON_INVALID),
        (_proof(block=91), REASON_WINDOW),
        (_proof(block=69), REASON_QUOTE_EXPIRED),
    ]
    for proof, reason in cases:
        assert (
            verify_eval_cost_payment(
                request=request,
                policy=policy,
                proof=proof,
                reveal_block=90,
            )
            == reason
        )


def test_proof_from_batch_all_and_transfer_event() -> None:
    request, quote = _quote()
    remark = encode_payment_remark(request, quote)
    proof = proof_from_decoded_extrinsic(
        block=80,
        extrinsic_index=4,
        signer="coldkey",
        call_module="Utility",
        call_function="batch_all",
        call_args={
            "calls": [
                {
                    "call_module": "Balances",
                    "call_function": "transfer_keep_alive",
                    "call_args": {"dest": TREASURY, "value": 10},
                },
                {
                    "call_module": "System",
                    "call_function": "remark_with_event",
                    "call_args": {"remark": remark},
                },
            ]
        },
        events=[
            {
                "extrinsic_idx": 4,
                "event": {
                    "module_id": "Balances",
                    "event_id": "Transfer",
                    "attributes": ("coldkey", TREASURY, 10),
                },
            }
        ],
    )
    assert proof is not None
    assert proof.amount_rao == 10
    assert proof.destination == TREASURY
    assert proof.remark == remark


def test_payment_proof_refuses_an_unattributed_transfer_event() -> None:
    request, quote = _quote()
    proof = proof_from_decoded_extrinsic(
        block=80,
        extrinsic_index=4,
        signer="coldkey",
        call_module="Utility",
        call_function="batch_all",
        call_args={
            "calls": [
                {
                    "call_module": "Balances",
                    "call_function": "transfer_keep_alive",
                    "call_args": {"dest": TREASURY, "value": 10},
                },
                {
                    "call_module": "System",
                    "call_function": "remark_with_event",
                    "call_args": {"remark": encode_payment_remark(request, quote)},
                },
            ]
        },
        events=[
            {
                "event": {
                    "module_id": "Balances",
                    "event_id": "Transfer",
                    "attributes": ("coldkey", TREASURY, 10),
                },
            }
        ],
    )
    assert proof is None


def test_proof_from_named_call_arg_lists() -> None:
    request, quote = _quote()
    remark = encode_payment_remark(request, quote)
    proof = proof_from_decoded_extrinsic(
        block=80,
        extrinsic_index=4,
        signer="coldkey",
        call_module="Utility",
        call_function="batch_all",
        call_args=[
            {
                "name": "calls",
                "value": [
                    {
                        "call_module": "Balances",
                        "call_function": "transfer_keep_alive",
                        "call_args": [
                            {"name": "dest", "value": {"Id": TREASURY}},
                            {"name": "value", "value": 10},
                        ],
                    },
                    {
                        "call_module": "System",
                        "call_function": "remark_with_event",
                        "call_args": [{"name": "remark", "value": remark}],
                    },
                ],
            }
        ],
        events=[
            {
                "extrinsic_idx": 4,
                "event": {
                    "module_id": "Balances",
                    "event_id": "Transfer",
                    "attributes": {
                        "from": "coldkey",
                        "to": TREASURY,
                        "amount": 10,
                    },
                },
            }
        ],
    )
    assert proof is not None
    assert proof.payer == "coldkey"
    assert proof.destination == TREASURY
    assert proof.remark == remark


def test_v2_payload_round_trip() -> None:
    data = encode_payload(HASH, "https://example.com/b.tar.gz", payment_block=80, payment_extrinsic_index=4)
    ref = decode_payload("hk1", 42, data)
    assert ref is not None
    assert ref.payment_block == 80 and ref.payment_extrinsic_index == 4
    assert '"v":2' in data
    assert decode_payload("hk1", 42, data.replace('"v":2', '"v":3')) is None


def test_dry_run_pay_quotes_without_a_chain() -> None:
    result = submit_bundle(
        None, None, 307, _BUNDLE, "https://example.com/bundles/miner-silu-torch.tar",
        dry_run=True,
        pay=True,
        eval_cost_policy=EvalCostPolicy(amount_rao=10, destination=TREASURY),
    )
    assert result["dry_run"] is True
    assert result["paid"] is True
    assert result["eval_cost_tao_rao"] == 10
    assert result["eval_cost_destination"] == TREASURY
    assert result["eval_cost_issued_block"] == 0
    assert result["eval_cost_expires_block"] == DEFAULT_EVAL_COST_QUOTE_TTL_BLOCKS
    assert result["eval_cost_payment"]["dry_run"] is True
    assert result["eval_cost_payment"]["expires_block"] == DEFAULT_EVAL_COST_QUOTE_TTL_BLOCKS
    ref = decode_payload("dry-run", 7, result["payload"])
    assert ref is not None
    assert ref.payment_block == 0
    assert ref.content_hash == content_hash(_BUNDLE)


def test_verify_accepts_distinct_subnet_owners() -> None:
    for dest in ("owner-a", "owner-b"):
        request, quote = _quote(destination=dest)
        proof = EvalCostPaymentProof(
            block=80,
            extrinsic_index=4,
            signer="coldkey",
            payer="coldkey",
            destination=dest,
            amount_rao=10,
            remark=encode_payment_remark(request, quote),
        )
        assert (
            verify_eval_cost_payment(
                request=request,
                policy=EvalCostPolicy(amount_rao=10, destination=dest),
                proof=proof,
                reveal_block=90,
            )
            == ""
        )


def test_read_subnet_owner_coldkey_two_profiles() -> None:
    class _Metagraph:
        def __init__(self, owner: str, block: int) -> None:
            self.owner_coldkey = owner
            self.block = block

    class _Subtensor:
        def __init__(self, owner: str) -> None:
            self.owner = owner

        def metagraph(self, netuid, block=None):
            assert netuid == 307
            return _Metagraph(self.owner, block)

    assert read_subnet_owner_coldkey(_Subtensor("owner-a"), 307, block=80) == "owner-a"
    assert read_subnet_owner_coldkey(_Subtensor("owner-b"), 307, block=81) == "owner-b"


def test_read_subnet_owner_coldkey_refuses_missing_owner() -> None:
    class _Metagraph:
        owner_coldkey = ""
        block = 80

    class _Subtensor:
        def metagraph(self, netuid, block=None):
            return _Metagraph()

    with pytest.raises(EvalCostFetchError, match="owner"):
        read_subnet_owner_coldkey(_Subtensor(), 307, block=80)


def test_dry_run_pay_refuses_without_a_destination() -> None:
    with pytest.raises(EvalCostError, match="destination"):
        submit_bundle(
            None, None, 307, _BUNDLE, "https://example.com/bundles/miner-silu-torch.tar",
            dry_run=True,
            pay=True,
            eval_cost_policy=EvalCostPolicy(amount_rao=10),
        )


def test_unpaid_v1_fails_when_eval_cost_is_required(tmp_path) -> None:
    arrival = FinalizedArrival(
        hotkey="miner",
        content_hash=HASH,
        url="https://example.com/a.tar.gz",
        block=10,
        block_hash="0x" + f"{10:064x}",
        event_index=0,
        invalid_reason="missing_eval_cost_payment",
    )
    with FinalizedIntakeStore(
        tmp_path / "private" / "intake.sqlite3",
        IntakePolicy(),
        scope=SCOPE,
    ) as store:
        row = store.reserve_finalized(
            (arrival,),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
            eval_cost_amount_tao_rao=_PAID_AMOUNT,
        )[0]
        assert row.status == "failed"
        assert row.reason == "missing_eval_cost_payment"


def test_valid_payment_is_consumed_once(tmp_path) -> None:
    first = FinalizedArrival(
        hotkey="miner",
        content_hash=HASH,
        url="https://example.com/a.tar.gz",
        block=10,
        block_hash="0x" + f"{10:064x}",
        event_index=0,
        payment_block=8,
        payment_extrinsic_index=4,
    )
    second = FinalizedArrival(
        hotkey="miner",
        content_hash="b" * 64,
        url="https://example.com/b.tar.gz",
        block=11,
        block_hash="0x" + f"{11:064x}",
        event_index=1,
        payment_block=8,
        payment_extrinsic_index=4,
    )
    with FinalizedIntakeStore(
        tmp_path / "private" / "intake.sqlite3",
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        rows = store.reserve_finalized(
            (first, second),
            finalized_block=11,
            finalized_block_hash="0x" + f"{11:064x}",
            eval_cost_amount_tao_rao=_PAID_AMOUNT,
        )
        assert rows[0].status == "reserved"
        assert rows[1].status == "failed"
        assert rows[1].reason == "eval_cost_payment_used"
        reopened = store.get(rows[0].reservation_id)
        assert reopened.arrival.payment_block == 8
        assert reopened.arrival.payment_extrinsic_index == 4


def test_disabled_gate_does_not_consume_or_poison_payment_pointer(tmp_path) -> None:
    def arrival(
        hotkey: str, digest: str, block: int, event_index: int
    ) -> FinalizedArrival:
        return FinalizedArrival(
            hotkey=hotkey,
            content_hash=digest,
            url=f"https://example.com/{hotkey}.tar.gz",
            block=block,
            block_hash="0x" + f"{block:064x}",
            event_index=event_index,
            payment_block=8,
            payment_extrinsic_index=4,
        )

    with FinalizedIntakeStore(
        tmp_path / "private" / "intake.sqlite3",
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        free_rows = store.reserve_finalized(
            (
                arrival("miner-a", "a" * 64, 10, 0),
                arrival("miner-b", "b" * 64, 11, 1),
            ),
            finalized_block=11,
            finalized_block_hash="0x" + f"{11:064x}",
        )
        assert [row.status for row in free_rows] == ["reserved", "reserved"]
        assert (
            store._db.execute(
                "SELECT COUNT(*) FROM eval_cost_payments"
            ).fetchone()[0]
            == 0
        )

        paid = store.reserve_finalized(
            (arrival("miner-c", "c" * 64, 12, 0),),
            finalized_block=12,
            finalized_block_hash="0x" + f"{12:064x}",
            eval_cost_amount_tao_rao=_PAID_AMOUNT,
        )[0]
        assert paid.status == "reserved"
        payment = store._db.execute(
            "SELECT amount_tao_rao FROM eval_cost_payments WHERE "
            "payment_block=8 AND payment_extrinsic_index=4"
        ).fetchone()
        assert payment is not None and payment[0] == _PAID_AMOUNT


def test_payment_may_precede_reveal_within_the_window() -> None:
    request, quote = _quote(payment_window_blocks=7_200, at_block=70)
    assert (
        verify_eval_cost_payment(
            request=request,
            policy=EvalCostPolicy(
                amount_rao=10,
                destination=TREASURY,
                payment_window_blocks=7_200,
            ),
            proof=_proof(remark=encode_payment_remark(request, quote), block=80),
            reveal_block=80 + 1_000,
        )
        == ""
    )


def test_failed_intake_leaves_eval_cost_payment_unused(tmp_path) -> None:
    failed = FinalizedArrival(
        hotkey="miner",
        content_hash=HASH,
        url="https://example.com/a.tar.gz",
        block=10,
        block_hash="0x" + f"{10:064x}",
        event_index=0,
        payment_block=8,
        payment_extrinsic_index=4,
        invalid_reason="malformed_payload",
    )
    retry = FinalizedArrival(
        hotkey="miner",
        content_hash=HASH,
        url="https://example.com/a.tar.gz",
        block=12,
        block_hash="0x" + f"{12:064x}",
        event_index=0,
        payment_block=8,
        payment_extrinsic_index=4,
    )
    with FinalizedIntakeStore(
        tmp_path / "private" / "intake.sqlite3",
        IntakePolicy(expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        first = store.reserve_finalized(
            (failed,),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
            eval_cost_amount_tao_rao=_PAID_AMOUNT,
        )[0]
        assert first.status == "failed"
        second = store.reserve_finalized(
            (retry,),
            finalized_block=12,
            finalized_block_hash="0x" + f"{12:064x}",
            eval_cost_amount_tao_rao=_PAID_AMOUNT,
        )[0]
        assert second.status == "reserved"
        assert second.arrival.payment_block == 8


class _Hotkey:
    ss58_address = "miner"


class _Wallet:
    hotkey = _Hotkey()


class _OwnerMetagraph:
    def __init__(self, owner: str, block: int) -> None:
        self.owner_coldkey = owner
        self.block = block


class _PaymentChain:
    def __init__(self, remark: str, *, current_block: int = 90) -> None:
        self._remark = remark
        self._current = current_block
        self.substrate = self

    def get_current_block(self) -> int:
        return self._current

    def get_block_hash(self, block: int) -> str:
        return f"0x{block:064x}"

    def metagraph(self, netuid, block=None):
        return _OwnerMetagraph(TREASURY, block)

    def get_block(self, block_hash=None):
        call = {
            "address": "coldkey",
            "call": {
                "call_module": "Utility",
                "call_function": "batch_all",
                "call_args": {
                    "calls": [
                        {
                            "call_module": "Balances",
                            "call_function": "transfer_keep_alive",
                            "call_args": {
                                "dest": TREASURY,
                                "value": 10,
                            },
                        },
                        {
                            "call_module": "System",
                            "call_function": "remark_with_event",
                            "call_args": {"remark": self._remark},
                        },
                    ]
                },
            },
        }

        class _Extrinsic:
            value = call

        return {"block": {"extrinsics": [{}, {}, {}, {}, _Extrinsic()]}}

    def get_events(self, block_hash=None):
        return [
            {
                "extrinsic_idx": 4,
                "event": {
                    "module_id": "Balances",
                    "event_id": "Transfer",
                    "attributes": ("coldkey", TREASURY, 10),
                },
            }
        ]


def _bundle_request_and_remark():
    request = EvalCostRequest(
        netuid=307,
        hotkey="miner",
        content_hash=content_hash(_BUNDLE),
    )
    quote = quote_eval_cost(
        request,
        policy=EvalCostPolicy(amount_rao=10, destination=TREASURY),
        at_block=70,
    )
    return request, encode_payment_remark(request, quote)


def test_bind_eval_cost_payment_rejects_a_different_proposal() -> None:
    _request, remark = _bundle_request_and_remark()
    chain = _PaymentChain(remark)
    with pytest.raises(EvalCostError, match="different proposal"):
        bind_eval_cost_payment(
            chain,
            EvalCostRequest(netuid=307, hotkey="miner", content_hash="b" * 64),
            payment_block=80,
            payment_extrinsic_index=4,
        )


def test_reuse_binds_an_unused_payment_and_encodes_v2() -> None:
    request, remark = _bundle_request_and_remark()
    result = submit_bundle(
        _PaymentChain(remark),
        _Wallet(),
        307,
        _BUNDLE,
        "https://example.com/bundles/miner-silu-torch.tar",
        dry_run=True,
        payment_block=80,
        payment_extrinsic_index=4,
        eval_cost_policy=EvalCostPolicy(amount_rao=10),
    )
    assert result["eval_cost_payment"]["reused"] is True
    assert result["eval_cost_payment_block"] == 80
    ref = decode_payload(request.hotkey, 100, result["payload"])
    assert ref is not None
    assert ref.payment_block == 80
    assert ref.payment_extrinsic_index == 4
    assert ref.content_hash == request.content_hash


def test_commit_failure_after_pay_preserves_the_unused_pointer(monkeypatch) -> None:
    monkeypatch.setattr(
        "cacheon.chain.submit.pay_eval_cost_tao",
        lambda *_args, **_kwargs: {
            "submitted": True,
            "payment_block": 80,
            "payment_extrinsic_index": 4,
            "amount_rao": 10,
            "destination": TREASURY,
            "issued_block": 70,
            "expires_block": 370,
            "remark": "paid",
        },
    )

    def boom(*_args, **_kwargs):
        raise RuntimeError("commitment timeout")

    monkeypatch.setattr("cacheon.chain.submit.post_reveal_commitment", boom)
    with pytest.raises(
        EvalCostCommitError, match="eval-cost-payment-block 80"
    ) as caught:
        submit_bundle(
            _PaymentChain("unused"),
            _Wallet(),
            307,
            _BUNDLE,
            "https://example.com/bundles/miner-silu-torch.tar",
            pay=True,
            eval_cost_policy=EvalCostPolicy(amount_rao=10, destination=TREASURY),
        )
    assert unused_eval_cost_retry_flags(80, 4) in str(caught.value)
    assert caught.value.payment_block == 80
    assert caught.value.payment_extrinsic_index == 4


def test_chain_rejection_after_pay_preserves_the_unused_pointer(monkeypatch) -> None:
    monkeypatch.setattr(
        "cacheon.chain.submit.pay_eval_cost_tao",
        lambda *_args, **_kwargs: {
            "submitted": True,
            "payment_block": 80,
            "payment_extrinsic_index": 4,
            "amount_rao": 10,
            "destination": TREASURY,
            "issued_block": 70,
            "expires_block": 370,
            "remark": "paid",
        },
    )
    monkeypatch.setattr(
        "cacheon.chain.submit.post_reveal_commitment",
        lambda *_args, **_kwargs: {
            "submitted": False,
            "message": "commit rejected on-chain",
        },
    )

    with pytest.raises(EvalCostCommitError, match="eval-cost-payment-block 80") as caught:
        submit_bundle(
            _PaymentChain("unused"),
            _Wallet(),
            307,
            _BUNDLE,
            "https://example.com/bundles/miner-silu-torch.tar",
            pay=True,
            eval_cost_policy=EvalCostPolicy(amount_rao=10, destination=TREASURY),
        )
    assert "commit rejected on-chain" in str(caught.value)
    assert caught.value.payment_extrinsic_index == 4
