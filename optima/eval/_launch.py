"""Shared engine-launch context manager used by the eval modules.

Centralizes the spawn-safe, tamper-resistant launch: mark this process as the
driver (so it never imports miner code), set the seam env, build the sglang
Engine, and clean it up. Both the KL eval and the benchmark eval use this.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any


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


def engine_kwargs(cfg) -> dict[str, Any]:
    """Translate an ``EvalConfig`` into ``sglang.Engine`` kwargs.

    Shared by both eval paths so multi-GPU knobs (``tp_size`` / ``moe_runner_backend``
    / ``disable_custom_all_reduce``) and deterministic mode apply identically. New
    fields are read with ``getattr`` so an older/duck-typed cfg still works.
    """
    kwargs: dict[str, Any] = dict(
        model_path=cfg.model_path,
        dtype=cfg.dtype,
        attention_backend=cfg.attention_backend,
        disable_cuda_graph=cfg.disable_cuda_graph,
        mem_fraction_static=cfg.mem_fraction_static,
        random_seed=cfg.seed,
        log_level=cfg.log_level,
    )
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
