#!/usr/bin/env python3
"""Phase 5 Part A — CPU-side validator entrypoint.

Chain scan loop + local state + (stubbed) GPU eval hook. The actual SSH
to the GPU pod and result ingestion is Phase 5 Part B — until that's
wired, pass `--dry-run` and leave `--eval-stub` on to exercise the loop
end-to-end against a live or testnet chain without trying to run any
real eval.

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
from validator.loop import not_implemented_eval, run_forever  # noqa: E402
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
        description="Cacheon CPU-side validator (Phase 5 Part A).",
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
                   help="Do not call subtensor.set_weights() — log only.")
    p.add_argument("--eval-stub", action="store_true",
                   help="Force the not-implemented eval stub (Phase 5 "
                        "Part B not yet wired).")
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
            "remote_validator (e.g. `pip install 'bittensor>=8'`)."
        )
        return 2

    _configure_logging(args.verbose)
    logger = logging.getLogger("validator.cli")

    logger.info("Connecting to network=%s netuid=%d wallet=%s/%s",
                args.network, args.netuid, args.wallet_name, args.wallet_hotkey)

    subtensor = bt.Subtensor(network=args.network)
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    state = ValidatorState.load(args.state_dir)
    if state.king is not None:
        logger.info(
            "Loaded state: king=UID %d (score=%.4f), %d evaluation(s) on record",
            state.king.uid, state.king.score, len(state.evaluations),
        )
    else:
        logger.info(
            "Loaded state: no king yet, %d evaluation(s) on record",
            len(state.evaluations),
        )

    eval_fn = not_implemented_eval
    if args.eval_stub:
        logger.warning(
            "Running with NotImplementedError eval stub — any new "
            "challenger will crash the tick. Use --dry-run on a network "
            "with no new commits, or wait for Phase 5 Part B."
        )

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
        logger.info("Interrupted — shutting down cleanly.")
        return 0
    except NotImplementedError as exc:
        logger.error("%s", exc)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
