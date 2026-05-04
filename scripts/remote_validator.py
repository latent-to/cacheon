#!/usr/bin/env python3
"""Cacheon validator entrypoint (single GPU machine).

Runs the chain-scan loop, evaluates Docker-based miner submissions locally,
and sets weights on-chain.

Usage:
    # Production:
    python scripts/remote_validator.py \
        --network finney --netuid 14 \
        --wallet-name my-validator --wallet-hotkey default

    # Dry-run (no eval, no weights):
    python scripts/remote_validator.py --dry-run

Env vars (all optional; CLI wins):
    CACHEON_NETUID, CACHEON_NETWORK,
    CACHEON_WALLET_NAME, CACHEON_WALLET_HOTKEY,
    CACHEON_POLL_INTERVAL_S, CACHEON_STATE_DIR, CACHEON_DRY_RUN=1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validator import config as validator_config  # noqa: E402
from validator.chain import NotRegisteredError, preflight_check  # noqa: E402
from validator.loop import not_implemented_eval, run_forever  # noqa: E402
from validator.state import ValidatorState  # noqa: E402


def _configure_logging(verbose: bool, log_dir: str) -> None:
    """Set up logging to both console and a rotating log file.

    On import, `bittensor` (a) installs its own root `basicConfig` and
    (b) walks `logging.Logger.manager.loggerDict` and sets every
    pre-existing `Logger` to `CRITICAL`. Because our `validator.*`
    modules are imported before `bittensor`, their loggers get silenced.

    We fix both in one shot: `basicConfig(force=True)` to reclaim the
    root handler, then reset `validator.*` levels to `NOTSET` so they
    inherit from root again.
    """
    from logging.handlers import RotatingFileHandler

    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )

    log_path = Path(log_dir) / "validator.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)

    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(lg, logging.Logger) and name.startswith("validator"):
            lg.setLevel(logging.NOTSET)

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

    logging.getLogger("validator.cli").info("Logging to %s", log_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cacheon validator (single GPU machine).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--network",
        default=validator_config.SUBTENSOR_NETWORK,
        help="Bittensor network: finney | test | ws://...",
    )
    p.add_argument("--netuid", type=int, default=validator_config.NETUID)
    p.add_argument("--wallet-name", default=validator_config.WALLET_NAME)
    p.add_argument("--wallet-hotkey", default=validator_config.WALLET_HOTKEY)
    p.add_argument(
        "--poll-interval",
        type=int,
        default=validator_config.POLL_INTERVAL_S,
        help="Seconds to sleep between chain scans when idle.",
    )
    p.add_argument("--state-dir", default=str(validator_config.STATE_DIR))
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=validator_config.DRY_RUN,
        help="Do not call subtensor.set_weights() and do not "
             "run Docker eval.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        import bittensor as bt
    except ImportError:
        logging.basicConfig(level=logging.ERROR)
        logging.error(
            "bittensor is not installed. Install it before running "
            "the validator (e.g. `pip install 'bittensor>=10'`)."
        )
        return 2

    _configure_logging(args.verbose, log_dir=args.state_dir)
    logger = logging.getLogger("validator.cli")

    logger.info(
        "Connecting to network=%s netuid=%d wallet=%s/%s",
        args.network, args.netuid, args.wallet_name, args.wallet_hotkey,
    )

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
        "Loaded state: %s | %d evaluation(s) | "
        "last_scan_block=%d, last_weights_set_block=%d",
        king_desc,
        len(state.evaluations),
        state.last_scan_block,
        state.last_weights_set_block,
    )

    # TODO(sprint-3): wire up Docker eval_fn here
    if args.dry_run:
        eval_fn = not_implemented_eval
    else:
        logger.error(
            "Live Docker evaluation not yet implemented. "
            "Run with --dry-run for now."
        )
        return 6

    try:
        run_forever(
            subtensor=subtensor,
            wallet=wallet,
            state=state,
            netuid=args.netuid,
            eval_fn=eval_fn,
            state_dir=args.state_dir,
            poll_interval_s=args.poll_interval,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down cleanly.")
        return 0
    except NotImplementedError as exc:
        logger.error("%s", exc)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
