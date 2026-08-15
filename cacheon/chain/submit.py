"""Miner-side submission: bind a bundle to a URL and commit it on chain.

The miner's on-chain footprint is one timelock commitment. Version 1 carries
``{"v":1,"h":content_hash,"u":url}``. When an eval-cost payment is attached,
version 2 adds a pointer to the coldkey ``transfer_keep_alive`` that paid for
this proposal. The artifact at ``url`` must extract to a directory whose
``content_hash`` equals ``h`` (use ``package_bundle`` to produce it); the
validator rejects anything else. The reveal itself is hotkey-signed; paying
the eval cost requires the coldkey.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cacheon.bundle_hash import content_hash
from cacheon.chain import post_reveal_commitment
from cacheon.chain.eval_cost import (
    EvalCostError,
    EvalCostPolicy,
    EvalCostRequest,
    quote_eval_cost,
)
from cacheon.chain.eval_cost_payment import (
    current_eval_cost_block,
    pay_eval_cost_tao,
    read_subnet_owner_coldkey,
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
) -> dict:
    """Compute the bundle's identity hash, build the payload, and commit it.

    Raises PayloadError or EvalCostError before touching the chain if the
    payload or payment would be rejected.
    """
    ch = content_hash(bundle_dir)
    hotkey = ""
    if wallet is not None:
        hotkey = str(getattr(getattr(wallet, "hotkey", None), "ss58_address", "") or "")
    if pay and not hotkey:
        if dry_run:
            hotkey = "dry-run"
        else:
            raise EvalCostError("eval-cost payment requires a miner hotkey")
    quote = None
    payment = None
    payment_block = 0
    payment_index = 0
    if pay:
        request = EvalCostRequest(netuid=netuid, hotkey=hotkey, content_hash=ch)
        base = eval_cost_policy if eval_cost_policy is not None else EvalCostPolicy()
        if type(base) is not EvalCostPolicy:
            raise EvalCostError("eval-cost policy is not typed")
        if subtensor is not None:
            at_block = current_eval_cost_block(subtensor)
            dest = read_subnet_owner_coldkey(subtensor, netuid, block=at_block)
            policy = replace(base, destination=dest)
        else:
            at_block = 0
            policy = base
        quote = quote_eval_cost(request, policy=policy, at_block=at_block)
        payment = pay_eval_cost_tao(
            subtensor, wallet, request, quote, dry_run=dry_run
        )
        if not dry_run:
            payment_block = int(payment["payment_block"])
            payment_index = int(payment["payment_extrinsic_index"])
    data = encode_payload(
        ch,
        url,
        payment_block=payment_block,
        payment_extrinsic_index=payment_index,
    )
    result = post_reveal_commitment(
        subtensor,
        wallet,
        netuid,
        data,
        blocks_until_reveal=blocks_until_reveal,
        dry_run=dry_run,
    )
    payload = {
        **result,
        "content_hash": ch,
        "payload": data,
        "paid": bool(pay),
    }
    if quote is not None:
        payload["eval_cost_tao_rao"] = quote.amount_rao
        payload["eval_cost_destination"] = quote.destination
        payload["eval_cost_instrument"] = quote.instrument
        payload["eval_cost_issued_block"] = quote.issued_block
        payload["eval_cost_expires_block"] = quote.expires_block
    if payment is not None:
        payload["eval_cost_payment"] = payment
    return payload
