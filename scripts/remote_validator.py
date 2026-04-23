#!/usr/bin/env python3
"""CPU-side validator entrypoint.

Chain scan loop + local state + precheck (fetch + AST sandbox).
The actual GPU eval hook is not wired until PR3 — until then you
must pass ``--dry-run`` or the CLI exits with an error.

Usage:
    python scripts/remote_validator.py \\
        --network finney \\
        --netuid 14 \\
        --wallet-name my-validator \\
        --wallet-hotkey default \\
        --dry-run

Env vars (all optional; CLI wins):
    CACHEON_NETUID, CACHEON_NETWORK,
    CACHEON_WALLET_NAME, CACHEON_WALLET_HOTKEY,
    CACHEON_POLL_INTERVAL_S, CACHEON_STATE_DIR, CACHEON_DRY_RUN=1,
    CACHEON_POLICY_CACHE_DIR, CACHEON_POLICY_MAX_BYTES,
    CACHEON_HF_ETAG_TIMEOUT_S, CACHEON_HF_TOKEN
"""

from __future__ import annotations

import argparse
import functools
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validator import config as validator_config  # noqa: E402
from validator.chain import NotRegisteredError, preflight_check  # noqa: E402
from validator.loop import not_implemented_eval, run_forever  # noqa: E402
from validator.policy_fetch import fetch_policy_source  # noqa: E402
from validator.precheck import make_fetch_precheck  # noqa: E402
from validator.state import ValidatorState  # noqa: E402


def _configure_logging(verbose: bool) -> None:
    """Set up logging.

    On import, `bittensor` (a) installs its own root `basicConfig` and
    (b) walks `logging.Logger.manager.loggerDict` and sets every
    pre-existing `Logger` to `CRITICAL`. Because our `validator.*`
    modules are imported before `bittensor`, their loggers get silenced.

    We fix both in one shot: `basicConfig(force=True)` to reclaim the
    root handler, then reset `validator.*` levels to `NOTSET` so they
    inherit from root again.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )

    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(lg, logging.Logger) and name.startswith("validator"):
            lg.setLevel(logging.NOTSET)

    # Third-party chatter — loud at DEBUG, not useful for validator ops.
    for noisy in (
        "bittensor",
        "websockets",
        "websockets.client",
        "btdecode",
        "substrateinterface",
        "urllib3",
        "async_substrate_interface",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cacheon CPU-side validator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--network", default=validator_config.SUBTENSOR_NETWORK,
                   help="Bittensor network: finney | test | ws://...")
    p.add_argument("--netuid", type=int, default=validator_config.NETUID)
    p.add_argument("--wallet-name", default=validator_config.WALLET_NAME)
    p.add_argument("--wallet-hotkey", default=validator_config.WALLET_HOTKEY)
    p.add_argument("--poll-interval", type=int,
                   default=validator_config.POLL_INTERVAL_S,
                   help="Seconds to sleep between chain scans when idle.")
    p.add_argument("--state-dir", default=str(validator_config.STATE_DIR))
    p.add_argument("--dry-run", action="store_true",
                   default=validator_config.DRY_RUN,
                   help="Do not call subtensor.set_weights() and do not "
                        "run GPU eval — log only.")
    p.add_argument("--policy-cache-dir",
                   default=str(validator_config.POLICY_CACHE_DIR),
                   help="Directory for cached policy.py downloads.")
    p.add_argument("--policy-max-bytes", type=int,
                   default=validator_config.POLICY_MAX_BYTES,
                   help="Max size (bytes) for a single policy.py download.")
    p.add_argument("--hf-etag-timeout-s", type=float,
                   default=validator_config.HF_ETAG_TIMEOUT_S,
                   help="Timeout (seconds) for the HEAD/etag revalidation "
                        "inside hf_hub_download. Does NOT cap blob downloads.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Import bittensor BEFORE configuring logging so our basicConfig(force=True)
    # wins against whatever handlers bittensor installs on import.
    try:
        import bittensor as bt
    except ImportError:
        logging.basicConfig(level=logging.ERROR)
        logging.error(
            "bittensor is not installed. Install it before running "
            "remote_validator (e.g. `pip install 'bittensor>=10'`)."
        )
        return 2

    _configure_logging(args.verbose)
    logger = logging.getLogger("validator.cli")

    logger.info("Connecting to network=%s netuid=%d wallet=%s/%s",
                args.network, args.netuid, args.wallet_name, args.wallet_hotkey)

    subtensor = bt.Subtensor(network=args.network)
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    try:
        preflight_check(subtensor, wallet, netuid=args.netuid)
    except NotRegisteredError as exc:
        logger.error("%s", exc)
        return 4

    state = ValidatorState.load(args.state_dir)
    king_desc = (
        f"king=UID {state.king.uid} (score={state.king.score:.4f})"
        if state.king is not None
        else "no king yet"
    )
    logger.info(
        "Loaded state: %s | %d evaluation(s), %d precheck failure(s) | "
        "last_scan_block=%d, last_weights_set_block=%d",
        king_desc,
        len(state.evaluations),
        len(state.precheck_failures),
        state.last_scan_block,
        state.last_weights_set_block,
    )

    # Pre-create and writability-check the policy cache directory so a
    # permissions problem surfaces at startup, not as silent DEFERRED loops.
    policy_cache_dir = Path(args.policy_cache_dir).resolve()
    try:
        policy_cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error(
            "Cannot create policy cache directory %s: %s", policy_cache_dir, exc
        )
        return 7

    # Build the real precheck (fetch + AST sandbox)
    fetch_fn = functools.partial(
        fetch_policy_source,
        cache_dir=policy_cache_dir,
        max_bytes=args.policy_max_bytes,
        etag_timeout_s=args.hf_etag_timeout_s,
        hf_token=validator_config.HF_TOKEN,
    )
    precheck = make_fetch_precheck(fetch_fn)

    # Fail-fast until PR3 lands a real EvalFn
    if not args.dry_run:
        logger.error(
            "remote_validator currently has no EvalFn wired. "
            "Run with --dry-run until PR3 (Lium pod transport) lands."
        )
        return 6

    eval_fn = not_implemented_eval  # harmless under dry-run; loop never calls it

    try:
        run_forever(
            subtensor=subtensor,
            wallet=wallet,
            state=state,
            netuid=args.netuid,
            eval_fn=eval_fn,
            precheck=precheck,
            state_dir=args.state_dir,
            poll_interval_s=args.poll_interval,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down cleanly.")
        return 0
    except NotImplementedError as exc:
        logger.error("%s", exc)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
