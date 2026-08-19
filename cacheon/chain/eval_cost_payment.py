"""Chain I/O for eval-cost TAO transfers. Pure quote/verify lives in eval_cost.py."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from cacheon.chain import _extrinsic_outcome
from cacheon.chain.eval_cost import (
    EvalCostError,
    EvalCostFetchError,
    EvalCostPaymentProof,
    EvalCostPolicy,
    EvalCostQuote,
    EvalCostRequest,
    decode_payment_remark,
    encode_payment_remark,
    proof_from_decoded_extrinsic,
    verify_eval_cost_payment,
)


def read_subnet_owner_coldkey(subtensor, netuid: int, *, block: int) -> str:
    """Return the subnet owner coldkey at ``block``.

    This is the eval-cost destination. It is the metagraph ``owner_coldkey``,
    not a miner-supplied wallet and not the owner-burn hotkey used for
    all-uncrowned weight projection.
    """

    if type(netuid) is not int or netuid < 0:
        raise EvalCostError("eval-cost netuid is malformed")
    if type(block) is not int or block < 0:
        raise EvalCostError("eval-cost owner block is malformed")
    reader = getattr(subtensor, "metagraph", None)
    if not callable(reader):
        raise EvalCostFetchError("subtensor exposes no metagraph API")
    try:
        mg = reader(netuid=netuid, block=block)
    except Exception as exc:
        raise EvalCostFetchError(
            f"cannot read subnet owner at block {block}: {exc}"
        ) from None
    reported = getattr(mg, "block", None)
    if type(reported) is int and reported != block:
        raise EvalCostFetchError(
            "metagraph response does not match the requested block"
        )
    owner = getattr(mg, "owner_coldkey", None)
    if (
        not isinstance(owner, str)
        or not owner
        or owner.strip() != owner
        or len(owner) > 256
        or any(char in owner for char in "\x00\r\n")
    ):
        raise EvalCostFetchError("subnet owner coldkey is missing or malformed")
    return owner


def pay_eval_cost_tao(
    subtensor,
    wallet,
    request: EvalCostRequest,
    quote: EvalCostQuote,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Coldkey-sign ``transfer_keep_alive`` + ``remark_with_event`` for this quote."""

    remark = encode_payment_remark(request, quote)
    if dry_run:
        return {
            "submitted": False,
            "dry_run": True,
            "amount_rao": quote.amount_rao,
            "destination": quote.destination,
            "issued_block": quote.issued_block,
            "expires_block": quote.expires_block,
            "remark": remark,
        }
    if subtensor is None or wallet is None:
        raise EvalCostError("eval-cost payment requires a chain client and wallet")
    current = current_eval_cost_block(subtensor)
    if current < quote.issued_block or current > quote.expires_block:
        raise EvalCostError("eval-cost quote has expired")
    owner = read_subnet_owner_coldkey(subtensor, request.netuid, block=current)
    if owner != quote.destination:
        raise EvalCostError("subnet owner changed; requote before paying")
    substrate = getattr(subtensor, "substrate", None)
    compose = getattr(substrate, "compose_call", None)
    send = getattr(subtensor, "sign_and_send_extrinsic", None)
    if not callable(compose) or not callable(send):
        raise EvalCostError("subtensor exposes no compose/sign extrinsic API")
    transfer = compose(
        call_module="Balances",
        call_function="transfer_keep_alive",
        call_params={
            "dest": quote.destination,
            "value": quote.amount_rao,
        },
    )
    noted = compose(
        call_module="System",
        call_function="remark_with_event",
        call_params={"remark": remark.encode("utf-8")},
    )
    batch = compose(
        call_module="Utility",
        call_function="batch_all",
        call_params={"calls": [transfer, noted]},
    )
    result = send(
        call=batch,
        wallet=wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    success, message = _extrinsic_outcome(result)
    if not success:
        raise EvalCostError(f"eval-cost transfer failed: {message or result}")
    block, index = _inclusion_pointer(result, subtensor)
    if block < quote.issued_block or block > quote.expires_block:
        raise EvalCostError("eval-cost transfer landed outside the quote window")
    return {
        "submitted": True,
        "result": result,
        "message": message,
        "payment_block": block,
        "payment_extrinsic_index": index,
        "remark": remark,
        "amount_rao": quote.amount_rao,
        "destination": quote.destination,
        "issued_block": quote.issued_block,
        "expires_block": quote.expires_block,
    }


def current_eval_cost_block(subtensor) -> int:
    """Read the live chain height used to issue or check a frozen quote."""

    reader = getattr(subtensor, "get_current_block", None)
    if not callable(reader):
        raise EvalCostError("subtensor exposes no current block API")
    block = reader()
    if type(block) is not int or block < 0:
        raise EvalCostError("current block is malformed")
    return block


def read_eval_cost_payment(
    subtensor, block: int, extrinsic_index: int
) -> EvalCostPaymentProof | None:
    """Fetch one extrinsic and decode it as an eval-cost payment, or ``None``."""

    if type(block) is not int or block <= 0:
        raise EvalCostError("payment block is malformed")
    if type(extrinsic_index) is not int or extrinsic_index < 0:
        raise EvalCostError("payment extrinsic index is malformed")
    substrate = getattr(subtensor, "substrate", None)
    get_block_hash = getattr(subtensor, "get_block_hash", None)
    get_block = getattr(substrate, "get_block", None)
    get_events = getattr(substrate, "get_events", None)
    if not callable(get_block_hash) or not callable(get_block) or not callable(get_events):
        raise EvalCostFetchError("subtensor exposes no finalized block/event API")
    try:
        block_hash = get_block_hash(block)
        packed = get_block(block_hash=block_hash)
        records = list(get_events(block_hash=block_hash))
    except Exception as exc:
        raise EvalCostFetchError(
            f"cannot read eval-cost payment at block {block}: {exc}"
        ) from None
    extrinsics = _block_extrinsics(packed)
    if extrinsic_index >= len(extrinsics):
        return None
    signer, module, function, args = _decode_extrinsic(extrinsics[extrinsic_index])
    if signer is None:
        return None
    events = [_normalize_event(record) for record in records]
    return proof_from_decoded_extrinsic(
        block=block,
        extrinsic_index=extrinsic_index,
        signer=signer,
        call_module=module,
        call_function=function,
        call_args=args,
        events=events,
    )


def bind_eval_cost_payment(
    subtensor,
    request: EvalCostRequest,
    *,
    payment_block: int,
    payment_extrinsic_index: int,
) -> EvalCostPaymentProof:
    """Fetch one included payment and refuse it when it names a different proposal."""

    proof = read_eval_cost_payment(subtensor, payment_block, payment_extrinsic_index)
    if proof is None:
        raise EvalCostError("eval-cost payment pointer is not a matching transfer")
    parsed = decode_payment_remark(proof.remark)
    if (
        parsed is None
        or parsed.get("content_hash") != request.content_hash
        or parsed.get("hotkey") != request.hotkey
        or parsed.get("netuid") != request.netuid
    ):
        raise EvalCostError("eval-cost payment is bound to a different proposal")
    return proof


def reopen_unused_eval_cost_payment(
    subtensor,
    request: EvalCostRequest,
    *,
    payment_block: int,
    payment_extrinsic_index: int,
    policy: EvalCostPolicy,
    reveal_block: int,
) -> EvalCostPaymentProof:
    """Reopen one included payment and refuse it if this reveal could not spend it."""

    if type(policy) is not EvalCostPolicy:
        raise EvalCostError("eval-cost policy is not typed")
    proof = bind_eval_cost_payment(
        subtensor,
        request,
        payment_block=payment_block,
        payment_extrinsic_index=payment_extrinsic_index,
    )
    owner = read_subnet_owner_coldkey(subtensor, request.netuid, block=proof.block)
    reason = verify_eval_cost_payment(
        request=request,
        policy=replace(policy, destination=owner),
        proof=proof,
        reveal_block=reveal_block,
    )
    if reason:
        raise EvalCostError(f"eval-cost payment cannot be reused ({reason})")
    return proof


def _inclusion_pointer(result: object, subtensor) -> tuple[int, int]:
    receipt = getattr(result, "extrinsic_receipt", None)
    block = _as_uint(
        getattr(receipt, "block_number", None)
        or getattr(receipt, "block_num", None)
        or getattr(receipt, "blockNum", None)
    )
    index = _as_uint(
        getattr(receipt, "extrinsic_idx", None)
        if getattr(receipt, "extrinsic_idx", None) is not None
        else getattr(receipt, "extrinsic_index", None)
    )
    block_hash = getattr(receipt, "block_hash", None)
    if block is None and block_hash is not None:
        reader = getattr(subtensor, "get_block_number", None) or getattr(
            getattr(subtensor, "substrate", None), "get_block_number", None
        )
        if callable(reader):
            block = _as_uint(reader(block_hash))
    if block is None or index is None:
        raise EvalCostError("eval-cost transfer inclusion pointer is unavailable")
    return block, index


def _block_extrinsics(packed: object) -> list[object]:
    raw = packed
    if isinstance(raw, Mapping) and "block" in raw:
        raw = raw["block"]
    if isinstance(raw, Mapping) and "extrinsics" in raw:
        rows = raw["extrinsics"]
        if isinstance(rows, list):
            return rows
    raise EvalCostFetchError("finalized block extrinsics are malformed")


def _decode_extrinsic(extrinsic: object) -> tuple[str | None, str, str, Mapping[str, object]]:
    value = extrinsic
    for attr in ("value_serialized", "value"):
        candidate = getattr(value, attr, None)
        if candidate is not None:
            value = candidate
            break
    if not isinstance(value, Mapping):
        return None, "", "", {}
    signer = _account_id(value.get("address", value.get("account_id", "")))
    call = value.get("call", value)
    if not isinstance(call, Mapping):
        return None, "", "", {}
    module = call.get("call_module", call.get("module_id", ""))
    function = call.get("call_function", call.get("function", ""))
    args = call.get("call_args", call.get("args", call.get("call_params", {})))
    if not isinstance(module, str) or not isinstance(function, str):
        return None, "", "", {}
    if isinstance(args, (str, bytes, bytearray)):
        return None, "", "", {}
    if not isinstance(args, (Mapping, Sequence)):
        return None, "", "", {}
    return signer or None, module, function, args


def _normalize_event(record: object) -> dict[str, object]:
    if isinstance(record, Mapping):
        return dict(record)
    value = getattr(record, "value", None) or getattr(record, "serialize", None)
    if callable(value):
        value = value()
    if isinstance(value, Mapping):
        return dict(value)
    return {"event": record}


def _account_id(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        inner = value.get("Id", value.get("id", value.get("Address32")))
        if isinstance(inner, str):
            return inner
        if isinstance(inner, Mapping):
            nested = inner.get("Id", inner.get("id"))
            if isinstance(nested, str):
                return nested
    return ""


def _as_uint(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
