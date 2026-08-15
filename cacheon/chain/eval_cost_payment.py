"""Chain I/O for eval-cost alpha burns. Pure quote/verify lives in eval_cost.py."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cacheon.chain import _extrinsic_outcome
from cacheon.chain.eval_cost import (
    EvalCostError,
    EvalCostFetchError,
    EvalCostPaymentProof,
    EvalCostQuote,
    EvalCostRequest,
    encode_payment_remark,
    proof_from_decoded_extrinsic,
)


def burn_eval_cost_alpha(
    subtensor,
    wallet,
    request: EvalCostRequest,
    quote: EvalCostQuote,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Coldkey-sign ``burn_alpha`` + ``remark_with_event`` for this quote."""

    remark = encode_payment_remark(request, quote)
    if dry_run:
        return {
            "submitted": False,
            "dry_run": True,
            "amount_alpha_rao": quote.amount_alpha_rao,
            "issued_block": quote.issued_block,
            "expires_block": quote.expires_block,
            "remark": remark,
        }
    if subtensor is None or wallet is None:
        raise EvalCostError("eval-cost burn requires a chain client and wallet")
    current = current_eval_cost_block(subtensor)
    if current < quote.issued_block or current > quote.expires_block:
        raise EvalCostError("eval-cost quote has expired")
    substrate = getattr(subtensor, "substrate", None)
    compose = getattr(substrate, "compose_call", None)
    send = getattr(subtensor, "sign_and_send_extrinsic", None)
    if not callable(compose) or not callable(send):
        raise EvalCostError("subtensor exposes no compose/sign extrinsic API")
    burn = compose(
        call_module="SubtensorModule",
        call_function="burn_alpha",
        call_params={
            "hotkey": request.hotkey,
            "amount": quote.amount_alpha_rao,
            "netuid": request.netuid,
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
        call_params={"calls": [burn, noted]},
    )
    result = send(
        call=batch,
        wallet=wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    success, message = _extrinsic_outcome(result)
    if not success:
        raise EvalCostError(f"eval-cost burn failed: {message or result}")
    block, index = _inclusion_pointer(result, subtensor)
    if block < quote.issued_block or block > quote.expires_block:
        raise EvalCostError("eval-cost burn landed outside the quote window")
    return {
        "submitted": True,
        "result": result,
        "message": message,
        "payment_block": block,
        "payment_extrinsic_index": index,
        "remark": remark,
        "amount_alpha_rao": quote.amount_alpha_rao,
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
        raise EvalCostError("eval-cost burn inclusion pointer is unavailable")
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
    return ""


def _as_uint(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
