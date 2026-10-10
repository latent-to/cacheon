"""Call-site facts the node adapter reads: capture, tracing, tuning, arch, dtype."""

from __future__ import annotations

import importlib
import sys
from typing import Optional

import torch

from cacheon import receipts as _receipts


def _arch_tag(device_index: int = 0) -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm{major}{minor}"


def _dynamo_compiling() -> bool:
    """Keep registry machinery out of Dynamo-traced regions."""
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001 - older torch without torch.compiler
        return False


# (module name, detector attribute) of the pinned and the legacy piecewise-capture
# authorities. Each name is imported at most once per process: a failed import of
# the absent one cost 28 us on every dispatched call, 40 calls per eager step on
# DeepSeek-V4.1 (B300, 2026-10-09), and the scheduler thread is CPU-bound there.
_CAPTURE_DETECTORS = (
    ("sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph",
     "is_in_tc_piecewise_cuda_graph"),
    ("sglang.srt.compilation.piecewise_context_manager", "is_in_piecewise_cuda_graph"),
)
_IMPORT_ATTEMPTED: set[str] = set()


def _in_cuda_graph() -> bool:
    """Probe pinned, legacy, then direct CUDA capture authorities."""

    for name, attribute in _CAPTURE_DETECTORS:
        if name not in sys.modules and name not in _IMPORT_ATTEMPTED:
            _IMPORT_ATTEMPTED.add(name)
            try:
                importlib.import_module(name)
            except Exception:  # noqa: BLE001 - older pin, removed legacy module, or CPU-only unit environment
                pass
        detector = getattr(sys.modules.get(name), attribute, None)
        if detector is None:
            continue
        try:
            if bool(detector()):
                return True
        except Exception:  # noqa: BLE001 - continue to independent CUDA authority
            pass
    try:
        return bool(
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
    except Exception:  # noqa: BLE001 - CPU/initialization failure means eager
        return False


# Capture detection has exactly one authority and it is the one above. Receipts
# record whether a candidate was invoked inside a capture — the fact that decides
# whether the scored replays run its code or stock — and get it from here rather
# than growing a second, quietly divergent detector.
_receipts.set_graph_probe(_in_cuda_graph)


def _flashinfer_tuning() -> bool:
    """Keep candidate code out of FlashInfer tactic profiling."""

    try:
        from flashinfer.autotuner import AutoTuner

        return bool(AutoTuner.get().is_tuning_mode)
    except Exception:  # noqa: BLE001 - an absent tuner is ordinary stock behavior
        return False


def _dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }.get(dtype, str(dtype).replace("torch.", ""))
