"""CPU-side always-on validator: chain scan, challenger selection, weight setting.

Runs continuously on a lightweight VPS. Does NOT evaluate miners; that
happens on an ephemeral GPU pod reading ``eval_job.json`` from S3.

Loop (every CACHEON_POLL_INTERVAL_S, default 300s):
    1. Download latest state from Hippius S3
    2. Chain scan: fetch metagraph + commitments
    3. If king's hotkey deregistered, clear throne
    4. If new GPU eval results or weights stale: set_weights
    5. Select new challengers not yet evaluated
    6. If challengers found: write eval_job.json, upload to S3
    7. Sleep

Usage:
    python -m validator.cpu_validator
    python -m validator.cpu_validator --network test --netuid 470 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from . import config as validator_config
from .chain import (
    ChainError,
    NotRegisteredError,
    build_commitments,
    fetch_metagraph,
    fetch_revealed_commitments,
    preflight_check,
    set_weights,
)
from .challengers import select_challengers
from .eval_schema import ChallengerInfo, EvalJob
from .state import ValidatorState

logger = logging.getLogger(__name__)

WEIGHTS_REFRESH_BLOCKS: int = int(
    os.environ.get("CACHEON_WEIGHTS_REFRESH_BLOCKS", "360")
)
"""Re-affirm weights at least once per tempo (~72 min at 12s/block) so the
validator stays active in consensus even when no new GPU eval results arrive."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _configure_logging(verbose: bool, state_dir: str) -> None:
    from datetime import datetime

    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )

    logs_dir = Path(state_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"cpu_validator_{ts}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)

    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(lg, logging.Logger) and name.startswith("validator"):
            lg.setLevel(logging.NOTSET)
    logging.getLogger("validator").setLevel(logging.NOTSET)
    logger.setLevel(level)

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

    logger.info("Logging to %s", log_path)


def _needs_weight_set(state: ValidatorState, current_block: int) -> str | None:
    """Return a reason string if weights should be (re-)set, else None."""
    if state.king is None:
        return None
    if state.last_weights_set_block == 0:
        return "first weight set"
    if state.evaluations:
        max_eval_block = max(e.evaluation_block for e in state.evaluations.values())
        if max_eval_block > state.last_weights_set_block:
            return f"new evals (latest eval block {max_eval_block})"
    stale = current_block - state.last_weights_set_block
    if stale > WEIGHTS_REFRESH_BLOCKS:
        return f"stale ({stale} blocks since last set)"
    return None


def _reload_state(state: ValidatorState, state_dir: str) -> None:
    """Reload state from disk into the existing object (GPU may have updated it)."""
    fresh = ValidatorState.load(state_dir)
    state.king = fresh.king
    state.evaluations = fresh.evaluations
    state.precheck_failures = fresh.precheck_failures
    state.last_scan_block = fresh.last_scan_block
    state.last_weights_set_block = fresh.last_weights_set_block


_CPU_UPLOAD_ONLY = ["eval_job.json", "logs/"]
"""CPU never uploads state.json -- the GPU is the sole writer of eval
results and king. Prevents the CPU from overwriting GPU results on S3."""


def _try_upload(state_dir: str) -> None:
    try:
        from .sync import upload

        upload(state_dir, only=_CPU_UPLOAD_ONLY)
    except Exception as exc:
        logger.error("S3 upload failed: %s", exc)


# --------------------------------------------------------------------------- #
# Tick
# --------------------------------------------------------------------------- #


def run_tick(
    *,
    subtensor,
    wallet,
    state: ValidatorState,
    netuid: int,
    state_dir: str,
    dry_run: bool = False,
    version_key: int = validator_config.VERSION_KEY,
) -> dict:
    """One iteration of the CPU validator loop. Returns a summary dict."""

    # S3 download
    try:
        from .sync import download

        download(state_dir)
    except Exception as exc:
        logger.error("S3 download failed: %s -- using local state", exc)

    _reload_state(state, state_dir)

    king_desc = (
        f"UID {state.king.uid} score={state.king.score:.4f}" if state.king else "none"
    )
    logger.info(
        "📋 State: king=%s | %d eval(s) | last_weights_block=%d",
        king_desc,
        len(state.evaluations),
        state.last_weights_set_block,
    )

    # Chain scan
    metagraph, current_block, block_hash = fetch_metagraph(subtensor, netuid)
    revealed = fetch_revealed_commitments(subtensor, netuid)
    commitments = build_commitments(metagraph, revealed)
    state.last_scan_block = current_block

    logger.info(
        "🔍 Scan block %d: %d hotkey(s), %d commitment(s)",
        current_block,
        len(metagraph.hotkeys),
        len(commitments),
    )

    # UID recycling guard
    if state.king is not None:
        live_hotkey = (
            metagraph.hotkeys[state.king.uid]
            if state.king.uid < len(metagraph.hotkeys)
            else None
        )
        if live_hotkey != state.king.hotkey:
            logger.warning(
                "👑  King UID %d hotkey changed on chain (%s -> %s). Clearing throne.",
                state.king.uid,
                state.king.hotkey[:16],
                (live_hotkey or "<gone>")[:16],
            )
            state.king = None

    # Weight setting
    weights_set = False
    reason = _needs_weight_set(state, current_block)
    if reason:
        logger.info(
            "⚖️  Setting weights: king=UID %d (score=%.4f), reason=%s",
            state.king.uid,
            state.king.score,
            reason,
        )
        if dry_run:
            logger.info("[dry-run] would set_weights(winner_uid=%d)", state.king.uid)
            state.last_weights_set_block = current_block
            weights_set = True
        else:
            try:
                set_weights(
                    subtensor,
                    wallet,
                    netuid,
                    n_uids=len(metagraph.hotkeys),
                    winner_uid=state.king.uid,
                    version_key=version_key,
                )
                state.last_weights_set_block = current_block
                weights_set = True
            except ChainError as exc:
                logger.error("set_weights failed: %s", exc)
        if weights_set:
            state.save(state_dir)
            _try_upload(state_dir)

    # Challenger selection
    challenger_set = select_challengers(state, commitments.values())
    for com, rej_reason in challenger_set.newly_rejected:
        state.record_precheck_failure(com.hotkey, com.commit_block, rej_reason)

    logger.info(
        "⚔️  Challengers: %d new, %d rejected, %d deferred, %d known",
        len(challenger_set.challengers),
        len(challenger_set.newly_rejected),
        len(challenger_set.deferred),
        len(challenger_set.already_known),
    )

    n_challengers = len(challenger_set.challengers)
    if challenger_set.challengers:
        if block_hash is None:
            logger.warning(
                "block_hash is None; cannot write eval_job (prompts need "
                "deterministic seeding). Will retry next tick."
            )
        else:
            for com in challenger_set.challengers:
                logger.info(
                    "  New: UID %d  %s  %s", com.uid, com.hotkey[:16], com.image
                )
            eval_job = EvalJob(
                block=current_block,
                block_hash=block_hash,
                challengers=[
                    ChallengerInfo(
                        uid=c.uid,
                        hotkey=c.hotkey,
                        commit_block=c.commit_block,
                        image=c.image,
                        digest=c.digest,
                    )
                    for c in challenger_set.challengers
                ],
                created_at=time.time(),
            )
            eval_job.save(state_dir)
            state.save(state_dir)
            _try_upload(state_dir)
            logger.info(
                "📤 %d challenger(s) ready for GPU eval (eval_job.json uploaded)",
                n_challengers,
            )
    else:
        state.save(state_dir)

    return {
        "block": current_block,
        "commitments": len(commitments),
        "challengers": n_challengers,
        "weights_set": weights_set,
        "king_uid": state.king.uid if state.king else None,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cacheon CPU validator (always-on, no GPU).",
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
        help="Seconds between chain scans.",
    )
    p.add_argument("--state-dir", default=str(validator_config.STATE_DIR))
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=validator_config.DRY_RUN,
        help="Skip set_weights on chain.",
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
            "bittensor is not installed. "
            "pip install 'bittensor>=10' before running the CPU validator."
        )
        return 2

    _configure_logging(args.verbose, args.state_dir)

    logger.info(
        "🕗 CPU validator starting: network=%s netuid=%d wallet=%s/%s poll=%ds",
        args.network,
        args.netuid,
        args.wallet_name,
        args.wallet_hotkey,
        args.poll_interval,
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
        if state.king
        else "no king yet"
    )
    logger.info(
        "Loaded state: %s | %d eval(s) | last_weights_block=%d",
        king_desc,
        len(state.evaluations),
        state.last_weights_set_block,
    )

    try:
        while True:
            tick_start = time.time()
            try:
                summary = run_tick(
                    subtensor=subtensor,
                    wallet=wallet,
                    state=state,
                    netuid=args.netuid,
                    state_dir=args.state_dir,
                    dry_run=args.dry_run,
                )
                logger.info(
                    "☑️ Tick completed in %.1fs: block=%d commits=%d challengers=%d "
                    "weights=%s king=%s",
                    time.time() - tick_start,
                    summary["block"],
                    summary["commitments"],
                    summary["challengers"],
                    summary["weights_set"],
                    summary["king_uid"],
                )
            except Exception as exc:
                logger.exception(
                    "❌ Tick failed after %.1fs: %s", time.time() - tick_start, exc
                )

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
