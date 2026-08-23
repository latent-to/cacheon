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
speed arm measures, so it belongs in verification and never inside a scored
window.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

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

    with profiler as prof:
        # The candidate's own exception must surface unchanged, so the block is
        # not guarded. Only the summary below, which runs after a clean window,
        # is allowed to fail quietly.
        yield report
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
