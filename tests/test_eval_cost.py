from __future__ import annotations

from pathlib import Path

from cacheon.bundle_hash import content_hash
from cacheon.chain.eval_cost import (
    DEFAULT_EVAL_COST_QUOTE_TTL_BLOCKS,
    PUBLISHED_EVAL_COST_ALPHA_RAO,
    REASON_INVALID,
    REASON_MISSING,
    REASON_QUOTE_EXPIRED,
    REASON_WINDOW,
    EvalCostPaymentProof,
    EvalCostPolicy,
    EvalCostRequest,
    encode_payment_remark,
    proof_from_decoded_extrinsic,
    quote_eval_cost,
    verify_eval_cost_payment,
)
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
_BUNDLE = Path(__file__).resolve().parent.parent / "examples" / "miner_silu_torch"


def _request(**changes) -> EvalCostRequest:
    values = dict(netuid=307, hotkey="miner", content_hash=HASH)
    values.update(changes)
    return EvalCostRequest(**values)


def _quote(**changes) -> tuple[EvalCostRequest, object]:
    request = _request()
    values = dict(amount_alpha_rao=10, at_block=70)
    values.update(changes)
    at_block = values.pop("at_block")
    policy_kwargs = {
        key: values[key]
        for key in ("amount_alpha_rao", "quote_ttl_blocks", "payment_window_blocks")
        if key in values
    }
    if "amount_alpha_rao" not in policy_kwargs:
        policy_kwargs["amount_alpha_rao"] = 10
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
        burn_coldkey="coldkey",
        burn_hotkey="miner",
        burn_netuid=307,
        alpha_decrease=10,
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
    assert base.amount_alpha_rao == PUBLISHED_EVAL_COST_ALPHA_RAO
    assert base.instrument == "burn_alpha"
    assert base.issued_block == 70
    assert base.expires_block == 70 + DEFAULT_EVAL_COST_QUOTE_TTL_BLOCKS


def test_quote_stays_valid_through_payment_ttl() -> None:
    request, quote = _quote(quote_ttl_blocks=5, at_block=70)
    assert quote.expires_block == 75
    policy = EvalCostPolicy(amount_alpha_rao=10, quote_ttl_blocks=5)
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


def test_payment_remark_round_trip_binds_content_hash() -> None:
    request, quote = _quote()
    remark = encode_payment_remark(request, quote)
    other = encode_payment_remark(_request(content_hash="b" * 64), quote)
    assert remark != other
    assert "cacheon.chain.eval-cost-payment.v1" in remark
    assert "issued_block" in remark


def test_verify_accepts_matching_burn() -> None:
    request, quote = _quote()
    assert (
        verify_eval_cost_payment(
            request=request,
            policy=EvalCostPolicy(amount_alpha_rao=10),
            proof=_proof(),
            reveal_block=90,
        )
        == ""
    )
    assert quote.amount_alpha_rao == 10


def test_verify_rejects_missing_wrong_stale_and_expired_payments() -> None:
    request = _request()
    policy = EvalCostPolicy(amount_alpha_rao=10, payment_window_blocks=10)
    cases = [
        (None, REASON_MISSING),
        (_proof(signer="other"), REASON_INVALID),
        (_proof(burn_hotkey="other"), REASON_INVALID),
        (_proof(alpha_decrease=9), REASON_INVALID),
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


def test_proof_from_batch_all_and_alpha_burned_event() -> None:
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
                    "call_module": "SubtensorModule",
                    "call_function": "burn_alpha",
                    "call_args": {"hotkey": "miner", "amount": 10, "netuid": 307},
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
                    "module_id": "SubtensorModule",
                    "event_id": "AlphaBurned",
                    "attributes": ("coldkey", "miner", 10, 307),
                },
            }
        ],
    )
    assert proof is not None
    assert proof.alpha_decrease == 10
    assert proof.remark == remark


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
                        "call_module": "SubtensorModule",
                        "call_function": "burn_alpha",
                        "call_args": [
                            {"name": "hotkey", "value": "miner"},
                            {"name": "amount", "value": 10},
                            {"name": "netuid", "value": 307},
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
                    "module_id": "SubtensorModule",
                    "event_id": "AlphaBurned",
                    "attributes": {
                        "Coldkey": "coldkey",
                        "Hotkey": "miner",
                        "Actual Alpha Decrease": 10,
                        "Netuid": 307,
                    },
                },
            }
        ],
    )
    assert proof is not None
    assert proof.burn_hotkey == "miner"
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
        dry_run=True, pay=True, eval_cost_policy=EvalCostPolicy(amount_alpha_rao=10),
    )
    assert result["dry_run"] is True
    assert result["paid"] is True
    assert result["eval_cost_alpha_rao"] == 10
    assert result["eval_cost_issued_block"] == 0
    assert result["eval_cost_expires_block"] == DEFAULT_EVAL_COST_QUOTE_TTL_BLOCKS
    assert result["eval_cost_payment"]["dry_run"] is True
    assert result["eval_cost_payment"]["expires_block"] == DEFAULT_EVAL_COST_QUOTE_TTL_BLOCKS
    ref = decode_payload("dry-run", 7, result["payload"])
    assert ref is not None
    assert ref.payment_block == 0
    assert ref.content_hash == content_hash(_BUNDLE)


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
        IntakePolicy(eval_cost_alpha_rao=10),
        scope=SCOPE,
    ) as store:
        row = store.reserve_finalized(
            (arrival,),
            finalized_block=10,
            finalized_block_hash="0x" + f"{10:064x}",
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
        IntakePolicy(eval_cost_alpha_rao=10, expiry_blocks=100),
        scope=SCOPE,
    ) as store:
        rows = store.reserve_finalized(
            (first, second),
            finalized_block=11,
            finalized_block_hash="0x" + f"{11:064x}",
        )
        assert rows[0].status == "reserved"
        assert rows[1].status == "failed"
        assert rows[1].reason == "eval_cost_payment_used"
        reopened = store.get(rows[0].reservation_id)
        assert reopened.arrival.payment_block == 8
        assert reopened.arrival.payment_extrinsic_index == 4
