#!/usr/bin/env python3
"""CPU-side validator entrypoint.

Chain scan loop + local state + precheck (fetch + AST sandbox).
When GPU pod SSH details are provided (``--gpu-pod-ssh-host`` +
``--gpu-pod-ssh-user``), evaluation runs remotely over SSH/SFTP.
Without those, you must pass ``--dry-run``.

Usage:
    # Production (remote GPU eval):
    python scripts/remote_validator.py \\
        --network finney --netuid 14 \\
        --wallet-name my-validator --wallet-hotkey default \\
        --gpu-pod-ssh-host ssh.deployments.targon.com \\
        --gpu-pod-ssh-user wrk-b6ptrqbmfkoj

    # Dry-run (no eval, no weights):
    python scripts/remote_validator.py --dry-run

Env vars (all optional; CLI wins):
    CACHEON_NETUID, CACHEON_NETWORK,
    CACHEON_WALLET_NAME, CACHEON_WALLET_HOTKEY,
    CACHEON_POLL_INTERVAL_S, CACHEON_STATE_DIR, CACHEON_DRY_RUN=1,
    CACHEON_POLICY_CACHE_DIR, CACHEON_POLICY_MAX_BYTES,
    CACHEON_HF_ETAG_TIMEOUT_S, CACHEON_HF_TOKEN,
    CACHEON_GPU_POD_SSH_HOST, CACHEON_GPU_POD_SSH_USER,
    CACHEON_GPU_POD_SSH_PORT, CACHEON_GPU_POD_WORK_DIR,
    CACHEON_GPU_POD_EVAL_TIMEOUT_S
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
from validator.eval_pod import (  # noqa: E402
    make_cache_policy_source_fn,
    make_remote_eval_fn,
)
from validator.loop import not_implemented_eval, run_forever  # noqa: E402
from validator.pod_transport import PodTransport  # noqa: E402
from validator.policy_fetch import fetch_policy_source  # noqa: E402
from validator.precheck import make_fetch_precheck  # noqa: E402
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
    fh = RotatingFileHandler(log_path, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)

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

    logging.getLogger("validator.cli").info("Logging to %s", log_path)


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
    # GPU pod SSH transport
    p.add_argument("--gpu-pod-ssh-host",
                   default=validator_config.GPU_POD_SSH_HOST,
                   help="SSH hostname of the GPU pod. Required for live eval.")
    p.add_argument("--gpu-pod-ssh-user",
                   default=validator_config.GPU_POD_SSH_USER,
                   help="SSH username on the GPU pod.")
    p.add_argument("--gpu-pod-ssh-port", type=int,
                   default=validator_config.GPU_POD_SSH_PORT)
    p.add_argument("--gpu-pod-work-dir",
                   default=validator_config.GPU_POD_WORK_DIR,
                   help="Repo checkout path on the GPU pod.")
    p.add_argument("--gpu-pod-eval-timeout", type=int,
                   default=validator_config.GPU_POD_EVAL_TIMEOUT_S,
                   help="Timeout (seconds) for one pod_eval.py run over SSH.")

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

    _configure_logging(args.verbose, log_dir=args.state_dir)
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

    # Build EvalFn — remote SSH/SFTP when GPU pod details are given,
    # otherwise stub (requires --dry-run).
    transport: PodTransport | None = None
    has_pod = bool(args.gpu_pod_ssh_host and args.gpu_pod_ssh_user)

    if has_pod:
        transport = PodTransport(
            host=args.gpu_pod_ssh_host,
            user=args.gpu_pod_ssh_user,
            port=args.gpu_pod_ssh_port,
        )
        transport.connect()
        logger.info(
            "SSH transport connected to %s@%s:%d",
            args.gpu_pod_ssh_user, args.gpu_pod_ssh_host, args.gpu_pod_ssh_port,
        )
        eval_fn = make_remote_eval_fn(
            policy_source_fn=make_cache_policy_source_fn(policy_cache_dir),
            transport=transport,
            pod_work_dir=args.gpu_pod_work_dir,
            timeout_s=float(args.gpu_pod_eval_timeout),
            state_dir=args.state_dir,
        )
    elif args.dry_run:
        eval_fn = not_implemented_eval
    else:
        logger.error(
            "No GPU pod SSH details provided and --dry-run not set. "
            "Either pass --gpu-pod-ssh-host + --gpu-pod-ssh-user for "
            "live evaluation, or run with --dry-run."
        )
        return 6

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
    finally:
        if transport is not None:
            transport.close()
            logger.info("SSH transport closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
