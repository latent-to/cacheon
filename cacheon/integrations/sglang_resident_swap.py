"""Swap a resident engine's kernel bundle and rebuild its decode graphs.

Each rank acknowledges only after recapture succeeds or records the exact failure.
Candidate-prepared state and the prior graph are released before every rebuild.
"""

from __future__ import annotations

import functools
import gc
import json
import logging
import os
import sys
import time

from cacheon.registry import REGISTRY, KernelRegistry

logger = logging.getLogger(__name__)

_MODULE = "sglang.srt.model_executor.model_runner"
_CLASS = "ModelRunner"
_METHOD = "init_decode_cuda_graph"
_HOOK_FLAG = "_cacheon_resident_swap"

# Trigger surface: sglang's /flush_cache request is broadcast to every TP rank's
# scheduler and is idle-gated, so a post-flush hook is a weight-free, all-rank
# recapture trigger (measured 2026-07-20: the update-weights trigger crashes the
# quantized M3 arena — minimax_m3_vl load_weights is not re-entrant, so the swap
# trigger must never touch weights).
_SCHED_MODULE = "sglang.srt.managers.scheduler"
_SCHED_CLASS = "Scheduler"
_SCHED_METHOD = "flush_cache"
_SCHED_HOOK_FLAG = "_cacheon_resident_swap_flush"

# Last generation this rank applied (process-global; one scheduler rank per process).
_applied_generation = -1
_MOE_PREPARED_ATTR = "_cacheon_moe_prepared_by_impl"


def _read_command(control_dir: str) -> tuple[int, str | None] | None:
    path = os.path.join(control_dir, "command.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - malformed command must not kill the engine
        logger.exception("cacheon: resident swap command unreadable at %s", path)
        return None
    generation = raw.get("generation") if isinstance(raw, dict) else None
    bundle = raw.get("bundle") if isinstance(raw, dict) else None
    if not isinstance(generation, int) or isinstance(generation, bool):
        return None
    if bundle is not None and (not isinstance(bundle, str) or not bundle.strip()):
        return None
    return generation, bundle


def _write_ack(control_dir: str, rank: object, payload: dict[str, object]) -> None:
    path = os.path.join(control_dir, f"ack.rank{rank}.json")
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - ack loss is diagnosable, not fatal
        logger.exception("cacheon: resident swap ack write failed at %s", path)


def _release_cuda_state(model_runner: object) -> int:
    """Drop candidate layouts and the old graph before recapture."""

    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    evicted = 0
    modules = getattr(getattr(model_runner, "model", None), "modules", None)
    if callable(modules):
        for layer in modules():
            cache = getattr(layer, _MOE_PREPARED_ATTR, None)
            if isinstance(cache, dict):
                evicted += len(cache)
                cache.clear()
            if hasattr(layer, _MOE_PREPARED_ATTR):
                delattr(layer, _MOE_PREPARED_ATTR)
    setattr(model_runner, "decode_cuda_graph_runner", None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return evicted


def _apply_pending_swap(
    model_runner: object, control_dir: str
) -> tuple[int, object, float, dict[str, object]] | None:
    command = _read_command(control_dir)
    if command is None:
        return None
    generation, bundle = command
    if generation <= _applied_generation:
        return None
    prior_generation = _applied_generation
    rank = getattr(model_runner, "tp_rank", "unknown")
    started = time.perf_counter()
    ack: dict[str, object] = {
        "generation": generation,
        "bundle": bundle or "",
        "pid": os.getpid(),
    }
    ack["evicted_prepared_entries"] = _release_cuda_state(model_runner)
    try:
        from cacheon import receipts, seam

        # Closing-generation receipts are final before the new scope is armed.
        try:
            receipts.set_root(os.path.join(control_dir, "receipts"))
            ack["prior_generation"] = prior_generation
            ack["prior_receipts"] = (
                receipts.counts_for_scope(prior_generation, pid=os.getpid())
                if prior_generation >= 0
                else {}
            )
            ack["receipt_scope"] = receipts.set_scope(generation)
        except Exception:  # noqa: BLE001 - diagnostics never break an engine
            logger.exception("cacheon: receipt scope failed at generation %s", generation)

        result = seam.swap_resident_bundle(bundle)
        ack.update(result)
        ack["ok"] = True
    except Exception as exc:  # noqa: BLE001 - a bad bundle must not wedge the engine
        logger.exception("cacheon: resident swap failed for %s", bundle)
        REGISTRY.clear()
        ack["ok"] = False
        ack["error"] = str(exc)[:2048]
    return generation, rank, started, ack


def _finish_swap(
    control_dir: str,
    pending: tuple[int, object, float, dict[str, object]] | None,
    error: Exception | None = None,
) -> None:
    global _applied_generation
    if pending is None:
        return
    generation, rank, started, ack = pending
    if error is not None:
        ack["ok"] = False
        ack["error"] = f"recapture failed: {error}"[:2048]
    _applied_generation = generation
    ack["swap_seconds"] = time.perf_counter() - started
    _write_ack(control_dir, rank, ack)


def _swap_pending(control_dir: str) -> bool:
    command = _read_command(control_dir)
    return command is not None and command[0] > _applied_generation


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Install the pre-recapture swap and post-flush trigger hooks."""

    del registry
    control_dir = os.environ.get("CACHEON_RESIDENT_SWAP", "").strip()
    if not control_dir:
        return

    mod = sys.modules.get(_MODULE)
    cls = getattr(mod, _CLASS, None) if mod is not None else None
    fn = getattr(cls, _METHOD, None) if cls is not None else None
    if fn is not None and not getattr(fn, _HOOK_FLAG, False):

        @functools.wraps(fn)
        def init_decode_cuda_graph(self, *args, **kwargs):
            pending = None
            if not getattr(self, "is_draft_worker", False):
                pending = _apply_pending_swap(self, control_dir)
            try:
                result = fn(self, *args, **kwargs)
            except Exception as exc:
                _finish_swap(control_dir, pending, exc)
                raise
            _finish_swap(control_dir, pending)
            return result

        setattr(init_decode_cuda_graph, _HOOK_FLAG, True)
        init_decode_cuda_graph._cacheon_orig = fn  # type: ignore[attr-defined]
        setattr(cls, _METHOD, init_decode_cuda_graph)

    sched_mod = sys.modules.get(_SCHED_MODULE)
    sched_cls = getattr(sched_mod, _SCHED_CLASS, None) if sched_mod is not None else None
    sched_fn = getattr(sched_cls, _SCHED_METHOD, None) if sched_cls is not None else None
    if sched_fn is not None and not getattr(sched_fn, _SCHED_HOOK_FLAG, False):

        @functools.wraps(sched_fn)
        def flush_cache(self, *args, **kwargs):
            result = sched_fn(self, *args, **kwargs)
            if result and _swap_pending(control_dir):
                runner = getattr(
                    getattr(self, "tp_worker", None), "model_runner", None
                )
                if runner is not None:
                    runner.init_decode_cuda_graph()
            return result

        setattr(flush_cache, _SCHED_HOOK_FLAG, True)
        flush_cache._cacheon_orig = sched_fn  # type: ignore[attr-defined]
        setattr(sched_cls, _SCHED_METHOD, flush_cache)


def uninstall() -> None:
    mod = sys.modules.get(_MODULE)
    cls = getattr(mod, _CLASS, None) if mod is not None else None
    fn = getattr(cls, _METHOD, None) if cls is not None else None
    if fn is None or not getattr(fn, _HOOK_FLAG, False):
        return
    setattr(cls, _METHOD, fn._cacheon_orig)


def is_installed() -> bool:
    mod = sys.modules.get(_MODULE)
    cls = getattr(mod, _CLASS, None) if mod is not None else None
    fn = getattr(cls, _METHOD, None) if cls is not None else None
    return bool(fn is not None and getattr(fn, _HOOK_FLAG, False))
