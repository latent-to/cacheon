"""Shared engine-launch context manager used by the eval modules.

Centralizes the spawn-safe, tamper-resistant launch: mark this process as the
driver (so it never imports miner code), set the seam env, build the sglang
Engine, and clean it up. Both the KL eval and the benchmark eval use this.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("optima.eval")


@contextmanager
def env(**overrides: str):
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def isolate_network() -> bool:
    """Put THIS process (and every child it spawns) into a fresh network namespace with
    NO egress, so untrusted miner code can't reach an external API to fake the output.
    Loopback is brought up so sglang's localhost IPC still works; the model is forced
    offline (it must already be cached). Self-checks that egress is actually gone.

    This is the boundary that makes the framework-mode token-match gate cheat-PROOF: the
    candidate must compute the right tokens — it can't see the trusted reference
    (separate process) and now can't fetch it either. Requires CAP_SYS_ADMIN (run the
    GPU box privileged; chain/cloud secrets live on a separate CPU control box). Returns
    True iff the candidate is confirmed no-egress; logs loudly and returns False if not.
    """
    import subprocess

    clone_newnet = getattr(os, "CLONE_NEWNET", None)
    if clone_newnet is None or not hasattr(os, "unshare"):
        logger.warning("optima: os.unshare/CLONE_NEWNET unavailable (need py>=3.12); candidate NOT isolated")
        return False
    try:
        os.unshare(clone_newnet)  # fresh netns: only `lo`, which starts DOWN
    except OSError as exc:
        logger.warning("optima: network isolation failed (%s); candidate NOT no-egress", exc)
        return False
    # Bring up loopback (the sglang scheduler<->detokenizer IPC uses localhost); external
    # stays unreachable because the netns has no route off-box.
    try:
        subprocess.run(["ip", "link", "set", "lo", "up"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("optima: could not bring up netns loopback (%s); sglang IPC may fail", exc)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Self-check: prove egress is actually gone (a fail-closed signal in the log).
    import socket

    try:
        socket.create_connection(("1.1.1.1", 443), timeout=2).close()
        logger.error("optima: ISOLATION FAILED — candidate still has network egress!")
        return False
    except OSError:
        logger.warning("optima: candidate network-isolated (no egress; loopback only)")
        return True


def engine_kwargs(cfg) -> dict[str, Any]:
    """Translate an ``EvalConfig`` into ``sglang.Engine`` kwargs.

    Shared by both eval paths so multi-GPU knobs (``tp_size`` / ``moe_runner_backend``
    / ``disable_custom_all_reduce``) and deterministic mode apply identically. New
    fields are read with ``getattr`` so an older/duck-typed cfg still works.
    """
    kwargs: dict[str, Any] = dict(
        model_path=cfg.model_path,
        dtype=cfg.dtype,
        mem_fraction_static=cfg.mem_fraction_static,
        random_seed=cfg.seed,
        log_level=cfg.log_level,
    )
    # Only pass these when explicitly set so sglang keeps its strong production
    # defaults otherwise (auto attention backend + CUDA graphs ON). A weak baseline
    # lets miners win against a crippled reference.
    if getattr(cfg, "attention_backend", None):
        kwargs["attention_backend"] = cfg.attention_backend
    if getattr(cfg, "disable_cuda_graph", False):
        kwargs["disable_cuda_graph"] = True
    if getattr(cfg, "deterministic", False):
        kwargs["enable_deterministic_inference"] = True
    if getattr(cfg, "tp_size", None):
        kwargs["tp_size"] = int(cfg.tp_size)
    if getattr(cfg, "moe_runner_backend", None):
        kwargs["moe_runner_backend"] = cfg.moe_runner_backend
    if getattr(cfg, "disable_custom_all_reduce", False):
        kwargs["disable_custom_all_reduce"] = True
    kwargs.update(getattr(cfg, "extra_engine_kwargs", {}) or {})
    return kwargs


@contextmanager
def launched_engine(cfg, *, bundle_path: str, active: bool):
    """Launch a sglang Engine with the Optima seam configured.

    ``cfg`` is an ``EvalConfig`` (see optima.eval.throughput_kl). The miner
    kernel runs only in the spawned scheduler child; THIS process is marked as
    the driver so it never imports miner code (timing stays tamper-resistant).
    """
    from optima import seam

    seam.mark_driver()
    with env(
        OPTIMA_BUNDLE_PATH=bundle_path or "",
        OPTIMA_ACTIVE="1" if active else "0",
        SGLANG_PLUGINS="optima",
    ):
        import sglang as sgl

        engine = sgl.Engine(**engine_kwargs(cfg))
        try:
            yield engine
        finally:
            try:
                engine.shutdown()
            except Exception:  # noqa: BLE001
                pass


def _subprocess_entry(out_path, fn, args, kwargs):
    """Run ``fn(*args, **kwargs)`` and pickle the result (or traceback) to a file."""
    import pickle
    import traceback

    try:
        payload = {"value": fn(*args, **kwargs), "error": None}
    except BaseException:  # noqa: BLE001 - report ANY failure back to the parent
        payload = {"value": None, "error": traceback.format_exc()}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)


def call_in_subprocess(fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` in a FRESH spawned process; return its result.

    Each model launch must run in its own process. sglang + deterministic mode set
    process-global state (torch deterministic algorithms, the cuBLAS workspace, the
    sampling backend) and hold a CUDA context; a second launch in the same driver
    process inherits that state and — observed on gpt-oss-120b in deterministic mode —
    the candidate launch then produces NaN/garbage. A fresh process makes the baseline
    and candidate launches independent and frees all GPU/host memory between them.

    ``fn`` must be a module-level (picklable) callable; the result travels back through
    a temp pickle file (avoids mp.Queue size limits / pipe deadlocks on large logprob
    payloads). Raises ``RuntimeError`` if the child crashes or ``fn`` raises.
    """
    import multiprocessing as mp
    import os
    import pickle
    import tempfile

    ctx = mp.get_context("spawn")
    fd, path = tempfile.mkstemp(prefix="optima_launch_", suffix=".pkl")
    os.close(fd)
    try:
        proc = ctx.Process(target=_subprocess_entry, args=(path, fn, args, kwargs))
        proc.start()
        proc.join()
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except (EOFError, FileNotFoundError, pickle.UnpicklingError) as exc:
            raise RuntimeError(
                f"launch subprocess crashed (exitcode={proc.exitcode}) with no result: {exc}"
            ) from None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if payload.get("error"):
        raise RuntimeError("launch subprocess failed:\n" + payload["error"])
    return payload["value"]
