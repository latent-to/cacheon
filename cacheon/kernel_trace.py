"""Name the CUDA kernels a candidate launched, without knowing anything about it.

Every other rung of the evidence ladder is a Python fact: the entry was called,
this many times, inside a capturing graph. None of them says what reached the
device. A candidate can be called, be inside the captured graph, and still be
launching nothing but stock torch kernels — or, for a bundle that branches
internally, be taking a path nobody can name from the outside.

Attribution here is by CONTAINMENT, not by symbol. The profiler is armed only
around the candidate's own invocation, and the validator launches nothing inside
that window, so every kernel the device reports came from the bundle. That holds
for a native ``.so``, a Triton JIT kernel, a plain torch call, or any mixture of
them, and it requires no knowledge of which slot is being verified, which bundle
is loaded, or what its source looks like. A symbol-matching scheme would have
needed all three.

This is an untimed audit instrument. Arming a profiler perturbs the very thing a
speed arm measures, so it runs in verification and in the engine's untimed audit
launch, and never inside a scored window. ``arm`` is the engine entry point.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

# Set by the untimed audit launch, never by a timed one — the same discipline
# ``cacheon.audit`` uses, and for the same reason: arming a profiler perturbs
# the measurement. See ``arm``.
_ENV = "CACHEON_KERNEL_TRACE"

# Distinct input signatures profiled per slot before the instrument goes quiet.
# The branch a bundle takes is a function of its inputs, so a handful of
# signatures names its paths; beyond that this would be a sampler, not an audit.
_MAX_SIGNATURES = 4

# slot -> signatures already profiled in this process.
_PROFILED: dict[str, set[str]] = {}

# A pathological candidate could launch thousands of distinct kernels. The report
# is a diagnostic, not an archive: keep the busiest ones and say how many were
# dropped rather than letting one bundle write an unbounded receipt.
_MAX_NAMES = 32


@contextmanager
def launched_kernels(enabled: bool) -> Iterator[dict]:
    """Record ``{kernel_name: launch_count}`` for the device work inside the block.

    The mapping is empty until the block exits, then filled in place. Never
    raises: an instrument that can kill a verification run is worse than no
    instrument, so every failure path yields an empty report instead.
    """

    report: dict = {}
    if not enabled:
        yield report
        return
    try:
        from torch.profiler import ProfilerActivity, profile

        profiler = profile(activities=[ProfilerActivity.CUDA])
    except Exception:  # noqa: BLE001 - profiler support is optional here
        yield report
        return

    try:
        prof = profiler.__enter__()
    except Exception:  # noqa: BLE001 - a profiler that will not start is not fatal
        # Constructing the profiler succeeds on a host with no CUDA; starting it
        # is what fails. Since ``arm`` puts this around live miner entries, an
        # unguarded start would raise inside a model forward.
        logger.exception("cacheon: kernel trace could not start")
        yield report
        return
    try:
        # The candidate's own exception must surface unchanged, so the block is
        # not guarded. Only the summary, which runs after a clean window, is
        # allowed to fail quietly.
        yield report
    finally:
        try:
            profiler.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.exception("cacheon: kernel trace teardown failed")
            prof = None
    if prof is not None:
        try:
            report.update(_summarize(prof))
        except Exception:  # noqa: BLE001
            logger.exception("cacheon: kernel trace summary failed")


def _summarize(prof: object) -> dict:
    """Reduce profiler events to device kernel names and launch counts."""

    counts: dict[str, int] = {}
    for event in prof.key_averages():  # type: ignore[attr-defined]
        # Device time is the discriminator: CPU-side operator rows carry the same
        # shape but never touched the GPU, and including them would report torch
        # dispatch names as if they were kernels.
        if getattr(event, "self_device_time_total", 0) <= 0:
            continue
        name = str(event.key)
        counts[name] = counts.get(name, 0) + int(getattr(event, "count", 1))
    if len(counts) <= _MAX_NAMES:
        return counts
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = dict(ranked[:_MAX_NAMES])
    kept["<%d more kernels not listed>" % (len(counts) - _MAX_NAMES)] = 0
    return kept


def format_kernels(counts: dict, *, indent: str = "      ") -> list[str]:
    """Render the launch table for a report; empty list when nothing was recorded.

    Names are truncated for display only — the untruncated name stays in the
    data. Truncation keeps the head because that is where provenance lives: a
    C++ template kernel names its owning namespace first (``at::native::`` for
    torch), and a Triton or hand-written kernel is a bare identifier.
    """

    if not counts:
        return []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = [f"{indent}device kernels ({len(ranked)}):"]
    lines.extend(
        f"{indent}  x{count:<4d} {_display(name)}" for name, count in ranked
    )
    return lines


def _display(name: str, limit: int = 110) -> str:
    name = name[5:] if name.startswith("void ") else name
    return name if len(name) <= limit else name[:limit] + "..."


def _signature(args: tuple, kwargs: dict) -> str:
    """A stable name for this call's input shape, derived from the tensors alone.

    Nothing here knows a slot, a model, or an argument's meaning: it reads
    ``.shape`` and ``.dtype`` off whatever was passed and ignores the rest. That
    is the whole reason it survives a new arena — a signature built from a slot's
    declared argument names would have to be rewritten for every slot added.
    """

    parts: list[str] = []
    for value in (*args, *(kwargs[k] for k in sorted(kwargs))):
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is None or dtype is None:
            continue
        parts.append("x".join(str(int(d)) for d in shape) + ":" + str(dtype))
    return " ".join(parts)[:200] or "no tensor inputs"


def _traced(slot: str, entry, receipts) -> object:
    """Wrap one entry point so its first launches at each new signature are named."""

    def call(*args, **kwargs):
        seen = _PROFILED.setdefault(slot, set())
        if len(seen) >= _MAX_SIGNATURES:
            return entry(*args, **kwargs)
        try:
            signature = _signature(args, kwargs)
        except Exception:  # noqa: BLE001 - never let the instrument fail the call
            return entry(*args, **kwargs)
        # Profiling inside a capture is the one way this instrument could break
        # an engine rather than merely fail to observe it, so capture wins.
        if signature in seen or receipts.capturing():
            return entry(*args, **kwargs)
        seen.add(signature)
        with launched_kernels(True) as counts:
            result = entry(*args, **kwargs)
        try:
            receipts.record_kernels(slot, signature, counts)
        except Exception:  # noqa: BLE001 - diagnostics never break an engine
            logger.exception("cacheon: kernel trace record failed for %s", slot)
        return result

    return call


def arm(registry: object) -> None:
    """Trace every registered entry, unless the launch forbids it.

    Off unless ``CACHEON_KERNEL_TRACE`` is set, because a timed arm must not pay
    profiler overhead — an untimed audit launch sets it, a scored one never does.

    Wrapping at the registry rather than in the dispatchers is what makes this
    generic: containment holds because only the candidate's own entry runs inside
    the window, and every present and future dispatcher is covered by one call
    instead of an edit each. It reports what reached the device, so a bundle that
    branches internally between a native kernel, a Triton kernel, and stock torch
    is distinguishable from the outside without reading its source.

    Never raises: an unarmed instrument is a lost diagnostic, a raising one is a
    dead engine.
    """

    if not os.environ.get(_ENV, "").strip():
        return
    try:
        from cacheon import receipts

        for slot in registry.slots():  # type: ignore[attr-defined]
            for impl in registry.variants(slot):  # type: ignore[attr-defined]
                if getattr(impl.entry, "_cacheon_traced", False):
                    continue
                wrapped = _traced(slot, impl.entry, receipts)
                wrapped._cacheon_traced = True  # type: ignore[attr-defined]
                # KernelImpl is a frozen dataclass: the registry owns the row, and
                # rebuilding it here would need every field this module has no
                # business knowing. Bypassing __setattr__ keeps the seam narrow.
                object.__setattr__(impl, "entry", wrapped)
    except Exception:  # noqa: BLE001 - an instrument must not break a launch
        logger.exception("cacheon: kernel trace arming failed")
