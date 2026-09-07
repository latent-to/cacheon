"""Miner-side check of the validator's currently commissioned baseline."""

import json
from urllib.request import urlopen

from cacheon.chain.eval_cost import EvalCostError
from cacheon.stack_identity import require_sha256_hex


def validate_baseline(validator_url: str, baseline_ref: str) -> None:
    """Check before quoting or paying; intake checks again after reveal."""

    try:
        require_sha256_hex(baseline_ref, field="baseline_ref")
    except ValueError as exc:
        raise EvalCostError(str(exc)) from exc
    if not validator_url.startswith(("https://", "http://")):
        raise EvalCostError("validator URL is required")
    try:
        with urlopen(validator_url.rstrip("/") + "/api/baseline", timeout=15) as response:
            head = json.load(response)
    except (OSError, ValueError) as exc:
        raise EvalCostError(f"cannot read finalized baseline: {exc}") from exc
    if not isinstance(head, dict) or head.get("baseline_ref") != baseline_ref:
        raise EvalCostError("stale baseline reference; request a quote against the current baseline")


def quote_command(args) -> int:
    """Quote the declared baseline through the miner CLI."""
    from cacheon.chain.eval_cost import (
        EvalCostPolicy,
        EvalCostRequest,
        quote_eval_cost,
    )
    from cacheon.chain.eval_cost_payment import (
        current_eval_cost_block,
        read_subnet_owner_coldkey,
    )

    try:
        validate_baseline(args.validator_url, args.baseline_ref)
    except EvalCostError as exc:
        print(f"REFUSED: {exc}")
        return 2
    request = EvalCostRequest(
        baseline_ref=args.baseline_ref,
        netuid=args.netuid,
        hotkey=args.hotkey or "query",
        content_hash=args.content_hash or "",
        target_id=args.target_id or "",
    )
    destination = ""
    at_block = 0
    network = str(getattr(args, "network", "") or "")
    if network:
        from cacheon import chain
        from cacheon.chain.eval_cost import EvalCostFetchError

        try:
            subtensor = chain.connect(network)
            at_block = current_eval_cost_block(subtensor)
            destination = read_subnet_owner_coldkey(
                subtensor, args.netuid, block=at_block
            )
        except EvalCostFetchError as exc:
            print(f"REFUSED: {exc}")
            return 2
    quote = quote_eval_cost(
        request,
        policy=EvalCostPolicy(
            amount_rao=int(
                getattr(
                    args,
                    "eval_cost_tao_rao",
                    1_000_000_000,
                )
            ),
            destination=destination,
        ),
        at_block=at_block,
    )
    print(f"baseline_ref: {quote.baseline_ref}")
    print(f"version:      {quote.version}")
    print(f"netuid:       {quote.netuid}")
    print(f"asset:        {quote.asset}")
    print(f"instrument:   {quote.instrument}")
    print(f"amount_rao:   {quote.amount_rao}")
    print(
        f"destination:  {quote.destination or 'current subnet owner coldkey (resolved at payment)'}"
    )
    print(f"quote_ttl:    {quote.expires_block - quote.issued_block} blocks")
    print(
        "This quote binds the baseline; admission checks the finalized HEAD again. "
        "Payment must be included within the TTL. chain-submit --pay "
        "quotes at the current block and transfers that frozen TAO amount to the "
        "current subnet owner coldkey."
    )
    return 0
