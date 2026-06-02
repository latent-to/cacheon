"""Wire the Optima dispatcher into SGLang's tensor-parallel all-reduce — the comms waist.

A TP model spends 30–40%+ of decode in the cross-GPU all-reduce
(``GroupCoordinator.all_reduce`` in ``sglang.srt.distributed.parallel_state`` — the
chokepoint every TP reduce funnels through, regardless of which backend, custom/NCCL,
actually runs). This seam patches that one method, so a miner can submit a
lower-latency or compute-overlapped reduce while the validator keeps the output buffer,
the process group, the model, and the sampler. The reduce stays mid-network — there is
no final output to substitute.

This is a COLLECTIVE slot: the kernel is handed the process group (a wider capability
than the op/block "fill a tensor" contract), so it is verified DISTRIBUTED
(optima.verify_collective) and the end-to-end gate is mandatory. See
docs/SLOT_CONTRACT.md.
"""

from __future__ import annotations

from optima.dispatch import make_allreduce_dispatcher
from optima.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_optima_allreduce_patched"
_MODULE = "sglang.srt.distributed.parallel_state"


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Patch ``GroupCoordinator.all_reduce``. No-ops until parallel_state is imported."""
    import sys

    mod = sys.modules.get(_MODULE)
    GroupCoordinator = getattr(mod, "GroupCoordinator", None) if mod is not None else None
    if GroupCoordinator is None:
        return

    if getattr(GroupCoordinator, _PATCH_FLAG, False):
        return

    orig = GroupCoordinator.all_reduce
    GroupCoordinator.all_reduce = make_allreduce_dispatcher(orig, registry=registry)
    GroupCoordinator._optima_orig_all_reduce = orig  # type: ignore[attr-defined]
    setattr(GroupCoordinator, _PATCH_FLAG, True)


def uninstall() -> None:
    import sys

    if _MODULE not in sys.modules:
        return
    from sglang.srt.distributed.parallel_state import GroupCoordinator

    if not getattr(GroupCoordinator, _PATCH_FLAG, False):
        return
    GroupCoordinator.all_reduce = GroupCoordinator._optima_orig_all_reduce  # type: ignore[attr-defined]
    delattr(GroupCoordinator, "_optima_orig_all_reduce")
    setattr(GroupCoordinator, _PATCH_FLAG, False)


def is_installed() -> bool:
    import sys

    mod = sys.modules.get(_MODULE)
    GroupCoordinator = getattr(mod, "GroupCoordinator", None) if mod is not None else None
    if GroupCoordinator is None:
        return False
    return bool(getattr(GroupCoordinator, _PATCH_FLAG, False))
