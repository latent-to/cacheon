"""Bind registered collectives to SGLang's public and opaque runtime bodies."""

from __future__ import annotations

from cacheon.dispatch import make_allreduce_dispatcher
from cacheon.dispatch_collective import (
    make_all_gather_dispatcher,
    make_allreduce_inplace_dispatcher,
    make_allreduce_outplace_dispatcher,
    make_reduce_scatter_dispatcher,
)
from cacheon.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_cacheon_allreduce_patched"
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
    orig_gather = GroupCoordinator.all_gather_into_tensor
    orig_scatter = GroupCoordinator.reduce_scatter_tensor
    orig_gather_runtime = GroupCoordinator._all_gather_into_tensor
    orig_scatter_runtime = GroupCoordinator._reduce_scatter_tensor
    orig_ar_inplace = GroupCoordinator._all_reduce_in_place
    orig_ar_outplace = GroupCoordinator._all_reduce_out_place
    GroupCoordinator.all_reduce = make_allreduce_dispatcher(orig, registry=registry)
    GroupCoordinator.all_gather_into_tensor = make_all_gather_dispatcher(
        orig_gather, registry=registry
    )
    GroupCoordinator.reduce_scatter_tensor = make_reduce_scatter_dispatcher(
        orig_scatter, registry=registry
    )
    GroupCoordinator._all_gather_into_tensor = make_all_gather_dispatcher(
        orig_gather_runtime, registry=registry
    )
    GroupCoordinator._reduce_scatter_tensor = make_reduce_scatter_dispatcher(
        orig_scatter_runtime, registry=registry
    )
    GroupCoordinator._all_reduce_in_place = make_allreduce_inplace_dispatcher(
        orig_ar_inplace, registry=registry
    )
    GroupCoordinator._all_reduce_out_place = make_allreduce_outplace_dispatcher(
        orig_ar_outplace, registry=registry
    )
    GroupCoordinator._cacheon_orig_all_reduce = orig  # type: ignore[attr-defined]
    GroupCoordinator._cacheon_orig_all_gather_into_tensor = orig_gather
    GroupCoordinator._cacheon_orig_reduce_scatter_tensor = orig_scatter
    GroupCoordinator._cacheon_orig_all_gather_runtime = orig_gather_runtime
    GroupCoordinator._cacheon_orig_reduce_scatter_runtime = orig_scatter_runtime
    GroupCoordinator._cacheon_orig_all_reduce_in_place = orig_ar_inplace
    GroupCoordinator._cacheon_orig_all_reduce_out_place = orig_ar_outplace
    setattr(GroupCoordinator, _PATCH_FLAG, True)


def uninstall() -> None:
    import sys

    if _MODULE not in sys.modules:
        return
    from sglang.srt.distributed.parallel_state import GroupCoordinator

    if not getattr(GroupCoordinator, _PATCH_FLAG, False):
        return
    GroupCoordinator.all_reduce = GroupCoordinator._cacheon_orig_all_reduce  # type: ignore[attr-defined]
    GroupCoordinator.all_gather_into_tensor = (
        GroupCoordinator._cacheon_orig_all_gather_into_tensor
    )
    GroupCoordinator.reduce_scatter_tensor = (
        GroupCoordinator._cacheon_orig_reduce_scatter_tensor
    )
    GroupCoordinator._all_gather_into_tensor = (
        GroupCoordinator._cacheon_orig_all_gather_runtime
    )
    GroupCoordinator._reduce_scatter_tensor = (
        GroupCoordinator._cacheon_orig_reduce_scatter_runtime
    )
    GroupCoordinator._all_reduce_in_place = (
        GroupCoordinator._cacheon_orig_all_reduce_in_place
    )
    GroupCoordinator._all_reduce_out_place = (
        GroupCoordinator._cacheon_orig_all_reduce_out_place
    )
    delattr(GroupCoordinator, "_cacheon_orig_all_reduce")
    delattr(GroupCoordinator, "_cacheon_orig_all_gather_into_tensor")
    delattr(GroupCoordinator, "_cacheon_orig_reduce_scatter_tensor")
    delattr(GroupCoordinator, "_cacheon_orig_all_gather_runtime")
    delattr(GroupCoordinator, "_cacheon_orig_reduce_scatter_runtime")
    delattr(GroupCoordinator, "_cacheon_orig_all_reduce_in_place")
    delattr(GroupCoordinator, "_cacheon_orig_all_reduce_out_place")
    setattr(GroupCoordinator, _PATCH_FLAG, False)


def is_installed() -> bool:
    import sys

    mod = sys.modules.get(_MODULE)
    GroupCoordinator = getattr(mod, "GroupCoordinator", None) if mod is not None else None
    if GroupCoordinator is None:
        return False
    return bool(getattr(GroupCoordinator, _PATCH_FLAG, False))
