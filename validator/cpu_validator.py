"""CPU-side always-on validator: chain scan, challenger selection, weight setting.

Runs continuously on a lightweight VPS. Does NOT evaluate miners; that
happens on an ephemeral GPU pod reading ``eval_job.json`` from S3.

Loop (every CACHEON_POLL_INTERVAL_S, default 600s):
    1. Download latest state from Hippius S3
    2. Chain scan: fetch metagraph + commitments
    3. If winner's hotkey deregistered, promote runner-up or clear
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
import signal
import time
from pathlib import Path

from . import config as validator_config
from .chain import (
    ChainError,
    NotRegisteredError,
    build_commitments,
    build_competition_weights,
    fetch_metagraph,
    fetch_revealed_commitments,
    preflight_check,
    set_weights,
)
from .challengers import select_challengers
from .eval_schema import ChallengerInfo, EvalJob
from .state import ValidatorState, WinnerRecord

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
        "paramiko",
        "paramiko.transport",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("Logging to %s", log_path)


def _needs_weight_set(state: ValidatorState, current_block: int) -> str | None:
    """Return a reason string if weights should be (re-)set, else None."""
    if state.winner is None:
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
    state.winner = fresh.winner
    state.runner_up_record = fresh.runner_up_record
    state.evaluations = fresh.evaluations
    state.precheck_failures = fresh.precheck_failures
    state.last_scan_block = fresh.last_scan_block
    state.last_weights_set_block = fresh.last_weights_set_block


_CPU_UPLOAD_ONLY = ["eval_job.json", "eval_progress.json", "logs/"]
"""CPU never uploads state.json -- the GPU is the sole writer of eval
results and winner. Prevents the CPU from overwriting GPU results on S3."""


def _try_upload(state_dir: str) -> None:
    try:
        from .sync import upload

        upload(state_dir, only=_CPU_UPLOAD_ONLY)
    except Exception as exc:
        logger.error("S3 upload failed: %s", exc)


def _clean_stale_eval_job(state: ValidatorState, state_dir: str) -> bool:
    """Remove ``eval_job.json`` if every challenger in it is already known.

    Returns True if the file was deleted."""
    from .eval_schema import EVAL_JOB_FILE, EvalJob

    path = Path(state_dir) / EVAL_JOB_FILE
    if not path.exists():
        return False
    job = EvalJob.load(state_dir)
    if job is None:
        return False
    if all(state.is_known(c.hotkey, c.commit_block) for c in job.challengers):
        try:
            path.unlink()
            logger.info(
                "Removed stale eval_job.json (%d challenger(s) all known)",
                len(job.challengers),
            )
        except OSError:
            return False
        try:
            from .sync import delete_remote_keys

            delete_remote_keys([EVAL_JOB_FILE])
        except Exception:
            logger.debug("Failed to delete eval_job.json from S3", exc_info=True)
        return True
    return False


def _hotkey_is_registered(metagraph, uid: int, hotkey: str) -> bool:
    """True if `uid` is in range and the on-chain hotkey matches."""
    if uid < 0 or uid >= len(metagraph.hotkeys):
        return False
    return metagraph.hotkeys[uid] == hotkey


def _resolve_runner_up_uid(state: ValidatorState, metagraph) -> int | None:
    """Return the runner-up's UID if one exists and is still registered."""
    ru = state.runner_up
    if ru is None:
        return None
    if _hotkey_is_registered(metagraph, ru.uid, ru.hotkey):
        return ru.uid
    return None


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
    _clean_stale_eval_job(state, state_dir)
    from .eval_progress import purge_old_logs

    purge_old_logs(state_dir)

    winner_desc = (
        f"UID {state.winner.uid} score={state.winner.score:.4f}"
        if state.winner
        else "none"
    )
    logger.info(
        "📋 State: winner=%s | %d eval(s) | last_weights_block=%d",
        winner_desc,
        len(state.evaluations),
        state.last_weights_set_block,
    )

    # Chain scan
    metagraph, current_block, block_hash = fetch_metagraph(subtensor, netuid)
    revealed = fetch_revealed_commitments(subtensor, netuid)
    commitments = build_commitments(metagraph, revealed)
    state.last_scan_block = current_block

    logger.info(
        "Scan block %d: %d hotkey(s), %d commitment(s)",
        current_block,
        len(metagraph.hotkeys),
        len(commitments),
    )

    # Winner UID recycling / deregistration guard
    if state.winner is not None:
        if not _hotkey_is_registered(metagraph, state.winner.uid, state.winner.hotkey):
            ru = state.runner_up
            if ru is not None and _hotkey_is_registered(metagraph, ru.uid, ru.hotkey):
                logger.warning(
                    "Winner UID %d deregistered (%s). Promoting runner-up UID %d.",
                    state.winner.uid,
                    state.winner.hotkey[:16],
                    ru.uid,
                )
                state.winner = WinnerRecord.from_evaluation(
                    ru, won_at_block=current_block
                )
                state.runner_up_record = None  # promoted; no runner-up until next eval
                state.last_weights_set_block = 0  # force immediate weight update
            else:
                reason = "runner-up also gone" if ru is not None else "no runner-up"
                logger.warning(
                    "Winner UID %d deregistered (%s). Clearing winner (%s).",
                    state.winner.uid,
                    state.winner.hotkey[:16],
                    reason,
                )
                state.winner = None
                state.runner_up_record = None

    # Weight setting
    weights_set = False
    dirty = False
    reason = _needs_weight_set(state, current_block)
    if reason:
        runner_up_uid = _resolve_runner_up_uid(state, metagraph)
        logger.info(
            "⚖️  Setting weights: winner=UID %d (score=%.4f), runner_up=%s, reason=%s",
            state.winner.uid,
            state.winner.score,
            runner_up_uid,
            reason,
        )

        w = build_competition_weights(
            n_uids=len(metagraph.hotkeys),
            winner_uid=state.winner.uid,
            winner_score=state.winner.score,
            runner_up_uid=runner_up_uid,
        )
        uid_list = [u for u, wt in enumerate(w) if wt > 0]
        w = [wt for wt in w if wt > 0]

        if dry_run:
            burn_uid = validator_config.BURN_UID
            logger.info(
                "🧪 [DRY-RUN] would set_weights: winner=%d (%.4f), runner_up=%s (%.4f),"
                " burn_uid=%d (%.4f), n_uids=%d",
                state.winner.uid,
                w[state.winner.uid] if state.winner.uid < len(w) else 0.0,
                runner_up_uid,
                w[runner_up_uid]
                if runner_up_uid is not None and runner_up_uid < len(w)
                else 0.0,
                burn_uid,
                w[burn_uid] if burn_uid < len(w) else 0.0,
                len(w),
            )
            state.last_weights_set_block = current_block
            weights_set = True
        else:
            burn_uid = validator_config.BURN_UID
            logger.info(
                "⚖️  weight vector: winner=%d (%.4f), runner_up=%s (%.4f),"
                " burn_uid=%d (%.4f), n_uids=%d",
                state.winner.uid,
                w[state.winner.uid] if state.winner.uid < len(w) else 0.0,
                runner_up_uid,
                w[runner_up_uid]
                if runner_up_uid is not None and runner_up_uid < len(w)
                else 0.0,
                burn_uid,
                w[burn_uid] if burn_uid < len(w) else 0.0,
                len(w),
            )
            try:
                set_weights(
                    subtensor,
                    wallet,
                    netuid,
                    uids=uid_list,
                    weights=w,
                    version_key=version_key,
                )
                state.last_weights_set_block = current_block
                weights_set = True
            except ChainError as exc:
                logger.error("set_weights failed: %s", exc)
        if weights_set:
            state.save(state_dir)
            dirty = True

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
            leader_info = None
            if state.winner is not None:
                leader_info = ChallengerInfo(
                    uid=state.winner.uid,
                    hotkey=state.winner.hotkey,
                    commit_block=state.winner.commit_block,
                    image=state.winner.image,
                    digest=state.winner.digest,
                )
                logger.info(
                    "  Leader (re-eval): UID %d  %s  %s",
                    state.winner.uid,
                    state.winner.hotkey[:16],
                    state.winner.image,
                )

            ru = state.runner_up
            ru_info = None
            if ru is not None:
                ru_info = ChallengerInfo(
                    uid=ru.uid,
                    hotkey=ru.hotkey,
                    commit_block=ru.commit_block,
                    image=ru.image,
                    digest=ru.digest,
                )
                logger.info(
                    "  Runner-up (re-eval): UID %d  %s  %s",
                    ru.uid,
                    ru.hotkey[:16],
                    ru.image,
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
                leader=leader_info,
                runner_up=ru_info,
            )
            from .eval_progress import update_progress

            update_progress(
                state_dir,
                phase="challengers_found",
                round_block=current_block,
                challengers=[
                    {"uid": c.uid, "hotkey": c.hotkey, "image": c.image}
                    for c in challenger_set.challengers
                ],
                leader=(
                    {
                        "uid": state.winner.uid,
                        "hotkey": state.winner.hotkey,
                        "image": state.winner.image,
                    }
                    if state.winner is not None
                    else None
                ),
                runner_up=(
                    {
                        "uid": ru.uid,
                        "hotkey": ru.hotkey,
                        "image": ru.image,
                    }
                    if ru is not None
                    else None
                ),
            )
            eval_job.save(state_dir)
            state.save(state_dir)
            dirty = True

    if not challenger_set.challengers:
        state.save(state_dir)

    # Single S3 upload per tick (covers weight-set + eval_job if both changed)
    if dirty:
        _try_upload(state_dir)

    if challenger_set.challengers and block_hash is not None:
        logger.info(
            "📤 %d challenger(s) ready for GPU eval (eval_job.json uploaded)",
            n_challengers,
        )

        if validator_config.AUTO_RENT:
            from .gpu_orchestrator import run_gpu_eval

            success = run_gpu_eval(state_dir, eval_job)
            if success:
                try:
                    from .sync import download

                    download(state_dir)
                except Exception as exc:
                    logger.error("Post-eval S3 download failed: %s", exc)
                _reload_state(state, state_dir)

            from .eval_progress import clear_progress

            clear_progress(state_dir)

    return {
        "block": current_block,
        "commitments": len(commitments),
        "challengers": n_challengers,
        "weights_set": weights_set,
        "winner_uid": state.winner.uid if state.winner else None,
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

    def _handle_sigterm(*_: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

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
    winner_desc = (
        f"winner=UID {state.winner.uid} (score={state.winner.score:.4f})"
        if state.winner
        else "no winner yet"
    )
    logger.info(
        "Loaded state: %s | %d eval(s) | last_weights_block=%d",
        winner_desc,
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
                    "weights=%s winner=%s",
                    time.time() - tick_start,
                    summary["block"],
                    summary["commitments"],
                    summary["challengers"],
                    summary["weights_set"],
                    summary["winner_uid"],
                )
            except Exception as exc:
                logger.exception(
                    "❌ Tick failed after %.1fs: %s", time.time() - tick_start, exc
                )

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down.")
        from .eval_progress import clear_progress

        clear_progress(args.state_dir)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
