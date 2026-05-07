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
    CACHEON_MODEL_VOLUME, CACHEON_BASELINE_IMAGE, CACHEON_BASELINE_DIGEST
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
    """Set up logging to both console and a timestamped log file.

    On import, `bittensor` (a) installs its own root `basicConfig` and
    (b) walks `logging.Logger.manager.loggerDict` and sets every
    pre-existing `Logger` to `CRITICAL`. Because our `validator.*`
    modules are imported before `bittensor`, their loggers get silenced.

    We fix both in one shot: `basicConfig(force=True)` to reclaim the
    root handler, then reset `validator.*` levels to `NOTSET` so they
    inherit from root again.
    """
    from datetime import datetime

    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )

    logs_dir = Path(log_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"validator_{ts}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
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
        help="Do not call subtensor.set_weights() and do not run Docker eval.",
    )
    p.add_argument("-v", "--verbose", action="store_true")

    p.add_argument(
        "--model-volume",
        default=validator_config.MODEL_VOLUME,
        help="Host path to the read-only model weights directory.",
    )
    p.add_argument(
        "--baseline-image",
        default=validator_config.BASELINE_IMAGE,
        help="Docker image for the vLLM baseline server.",
    )
    p.add_argument(
        "--baseline-digest",
        default=validator_config.BASELINE_DIGEST,
        help="Digest (sha256:...) of the baseline image. Auto-detected from local image if omitted.",
    )
    return p


def _detect_baseline_digest(image: str, logger: logging.Logger) -> str:
    """Try to read the digest of a locally-pulled baseline image."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                image,
                "--format",
                "{{index .RepoDigests 0}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return ""
        # output looks like "vllm/vllm-openai@sha256:abcd..."
        raw = result.stdout.strip()
        if "@" in raw:
            digest = raw.split("@", 1)[1]
            logger.info("Auto-detected baseline digest: %s", digest)
            return digest
    except Exception as exc:
        logger.debug("Could not auto-detect baseline digest: %s", exc)
    return ""


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
        args.network,
        args.netuid,
        args.wallet_name,
        args.wallet_hotkey,
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

    if args.dry_run:
        eval_fn = not_implemented_eval
    else:
        baseline_digest = args.baseline_digest
        if not baseline_digest:
            baseline_digest = _detect_baseline_digest(args.baseline_image, logger)
        if not baseline_digest:
            logger.error(
                "Could not determine baseline digest. Either set "
                "CACHEON_BASELINE_DIGEST, pass --baseline-digest, or "
                "pull the baseline image first (`docker pull %s`).",
                args.baseline_image,
            )
            return 6

        from validator.docker_eval import make_eval_fn  # noqa: E402

        eval_fn = make_eval_fn(
            model_volume=args.model_volume,
            baseline_cache_dir=str(Path(args.state_dir) / "baseline_cache"),
            baseline_image=args.baseline_image,
            baseline_digest=baseline_digest,
            gpu_count=validator_config.GPU_COUNT,
            state_dir=args.state_dir,
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
        logger.info("Interrupted, shutting down cleanly.")
        return 0
    except NotImplementedError as exc:
        logger.error("%s", exc)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
