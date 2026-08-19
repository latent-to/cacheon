"""Miner-side submission: bind a bundle to a URL and commit it on chain.

The miner's on-chain footprint is one timelock commitment. Version 1 carries
``{"v":1,"h":content_hash,"u":url}``. Version 2 adds a pointer to the coldkey
``transfer_keep_alive`` that paid for this proposal. Payment may precede the
reveal: a later commit retry attaches an unused pointer instead of paying again.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cacheon.bundle_hash import content_hash
from cacheon.chain import post_reveal_commitment
from cacheon.chain.eval_cost import (
    EvalCostCommitError,
    EvalCostError,
    EvalCostPolicy,
    EvalCostRequest,
    quote_eval_cost,
)
from cacheon.chain.eval_cost_payment import (
    current_eval_cost_block,
    pay_eval_cost_tao,
    read_subnet_owner_coldkey,
    reopen_unused_eval_cost_payment,
)
from cacheon.chain.payload import encode_payload


def submit_bundle(
    subtensor,
    wallet,
    netuid: int,
    bundle_dir: str | Path,
    url: str,
    *,
    blocks_until_reveal: int = 10,
    dry_run: bool = False,
    pay: bool = False,
    eval_cost_policy: EvalCostPolicy | None = None,
    payment_block: int = 0,
    payment_extrinsic_index: int = 0,
) -> dict:
    """Hash the bundle, optionally pay or reuse eval-cost, and commit the payload.

    ``pay`` and ``payment_block`` are exclusive. Intake consumes a pointer only
    on reserved or deferred admission.
    """
    ch = content_hash(bundle_dir)
    hotkey = ""
    if wallet is not None:
        hotkey = str(getattr(getattr(wallet, "hotkey", None), "ss58_address", "") or "")
    if (
        type(payment_block) is not int
        or payment_block < 0
        or type(payment_extrinsic_index) is not int
        or payment_extrinsic_index < 0
    ):
        raise EvalCostError("eval-cost payment pointer is malformed")
    reuse = payment_block > 0
    if pay and reuse:
        raise EvalCostError("cannot pay and reuse an eval-cost payment together")
    if payment_extrinsic_index and not reuse:
        raise EvalCostError("eval-cost payment pointer requires a block")
    if (pay or reuse) and not hotkey:
        if dry_run:
            hotkey = "dry-run"
        else:
            raise EvalCostError("eval-cost payment requires a miner hotkey")
    quote = None
    payment = None
    resolved_block = 0
    resolved_index = 0
    if pay:
        request = EvalCostRequest(netuid=netuid, hotkey=hotkey, content_hash=ch)
        policy = _eval_cost_policy(eval_cost_policy)
        if subtensor is not None:
            at_block = current_eval_cost_block(subtensor)
            dest = read_subnet_owner_coldkey(subtensor, netuid, block=at_block)
            policy = replace(policy, destination=dest)
        else:
            at_block = 0
        quote = quote_eval_cost(request, policy=policy, at_block=at_block)
        payment = pay_eval_cost_tao(
            subtensor, wallet, request, quote, dry_run=dry_run
        )
        if not dry_run:
            resolved_block = int(payment["payment_block"])
            resolved_index = int(payment["payment_extrinsic_index"])
    elif reuse:
        request = EvalCostRequest(netuid=netuid, hotkey=hotkey, content_hash=ch)
        if subtensor is not None:
            proof = reopen_unused_eval_cost_payment(
                subtensor,
                request,
                payment_block=payment_block,
                payment_extrinsic_index=payment_extrinsic_index,
                policy=_eval_cost_policy(eval_cost_policy),
                reveal_block=current_eval_cost_block(subtensor),
            )
            payment = {
                "reused": True,
                "payment_block": proof.block,
                "payment_extrinsic_index": proof.extrinsic_index,
                "destination": proof.destination,
                "amount_rao": proof.amount_rao,
            }
        else:
            payment = {
                "reused": True,
                "dry_run": True,
                "payment_block": payment_block,
                "payment_extrinsic_index": payment_extrinsic_index,
            }
        resolved_block = payment_block
        resolved_index = payment_extrinsic_index
    try:
        data = encode_payload(
            ch,
            url,
            payment_block=resolved_block,
            payment_extrinsic_index=resolved_index,
        )
        result = post_reveal_commitment(
            subtensor,
            wallet,
            netuid,
            data,
            blocks_until_reveal=blocks_until_reveal,
            dry_run=dry_run,
        )
        if resolved_block > 0 and not dry_run and not bool(result.get("submitted")):
            message = str(
                result.get("message") or "reveal commitment was not submitted"
            )
            raise EvalCostError(message)
    except Exception as exc:
        if resolved_block > 0 and not dry_run:
            raise EvalCostCommitError(resolved_block, resolved_index, exc) from exc
        raise
    payload = {
        **result,
        "content_hash": ch,
        "payload": data,
        "paid": bool(pay or reuse),
    }
    if quote is not None:
        payload["eval_cost_tao_rao"] = quote.amount_rao
        payload["eval_cost_destination"] = quote.destination
        payload["eval_cost_instrument"] = quote.instrument
        payload["eval_cost_issued_block"] = quote.issued_block
        payload["eval_cost_expires_block"] = quote.expires_block
    if payment is not None:
        payload["eval_cost_payment"] = payment
        payload["eval_cost_payment_block"] = resolved_block
        payload["eval_cost_payment_extrinsic_index"] = resolved_index
    return payload


def _eval_cost_policy(eval_cost_policy: EvalCostPolicy | None) -> EvalCostPolicy:
    base = eval_cost_policy if eval_cost_policy is not None else EvalCostPolicy()
    if type(base) is not EvalCostPolicy:
        raise EvalCostError("eval-cost policy is not typed")
    return base
