"""Live SGLang adapters for caller-owned gather/scatter tensor exchanges."""

from __future__ import annotations

import logging
import os
from functools import wraps

import torch

from cacheon.dispatch import (
    _allreduce_group_role,
    _audit,
    _collective_call_descriptor,
    _dynamo_compiling,
    _in_cuda_graph,
    _process_group_size,
    _receipts,
)
from cacheon.registry import REGISTRY, KernelRegistry

logger = logging.getLogger("cacheon.collective")
_LOGGED: set[str] = set()


def _active() -> bool:
    return os.environ.get("CACHEON_COLLECTIVE_SEAM") == "1"


def _selection(coordinator, input_, output, slot, registry):
    if not (
        _active()
        and torch.is_tensor(input_)
        and torch.is_tensor(output)
        and input_.dim() == output.dim() == 2
        and input_.is_floating_point()
        and input_.dtype == output.dtype
        and input_.device == output.device
        and input_.is_contiguous()
        and output.is_contiguous()
    ):
        return None
    group = getattr(coordinator, "device_group", None)
    size = _process_group_size(group)
    if (
        size is None
        or size <= 1
        or _allreduce_group_role(coordinator, group) is None
        or getattr(coordinator, "world_size", size) != size
    ):
        return None
    if (
        slot == "collective.all_gather_into_tensor"
        and (
            output.shape[0] != input_.shape[0] * size
            or output.shape[1:] != input_.shape[1:]
        )
    ) or (
        slot == "collective.reduce_scatter_tensor"
        and (
            input_.shape[0] != output.shape[0] * size
            or input_.shape[1:] != output.shape[1:]
        )
    ):
        return None
    descriptor = _collective_call_descriptor(output, group_size=size)
    impl = registry.select(slot, descriptor).impl
    return None if impl is None else (impl, group, size, descriptor)


def _complete(slot: str) -> None:
    if slot not in _LOGGED:
        _LOGGED.add(slot)
        logger.warning("cacheon: %s seam ACTIVE", slot)
    _receipts.completed(slot)


def make_all_gather_dispatcher(
    baseline,
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "collective.all_gather_into_tensor",
):
    @wraps(baseline)
    def dispatched(self, output, input_):
        if _dynamo_compiling():
            return baseline(self, output, input_)
        selected = _selection(self, input_, output, slot, registry)
        if selected is None:
            return baseline(self, output, input_)
        impl, group, _, descriptor = selected
        if registry.select(slot, descriptor).impl is not impl:
            raise RuntimeError("all-gather selection changed before commit")
        audit = not _in_cuda_graph() and _audit.sampled()
        audit_input = input_.clone() if audit else None
        _receipts.invoke(slot, impl.entry, input_, output, group)
        if audit:
            def stock_reference():
                expected = torch.empty_like(output)
                baseline(self, expected, audit_input)
                return expected

            _audit.run(slot, (output,), stock_reference)
        _complete(slot)
        return None

    return dispatched


def make_reduce_scatter_dispatcher(
    baseline,
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "collective.reduce_scatter_tensor",
):
    @wraps(baseline)
    def dispatched(self, output, input_):
        if _dynamo_compiling():
            return baseline(self, output, input_)
        selected = _selection(self, input_, output, slot, registry)
        if selected is None:
            return baseline(self, output, input_)
        impl, group, _, descriptor = selected
        if registry.select(slot, descriptor).impl is not impl:
            raise RuntimeError("reduce-scatter selection changed before commit")
        audit = not _in_cuda_graph() and _audit.sampled()
        audit_input = input_.clone() if audit else None
        _receipts.invoke(slot, impl.entry, input_, output, group)
        if audit:
            def stock_reference():
                expected = torch.empty_like(output)
                baseline(self, expected, audit_input)
                return expected

            _audit.run(slot, (output,), stock_reference)
        _complete(slot)
        return None

    return dispatched


def make_allreduce_inplace_dispatcher(
    baseline,
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "collective.all_reduce",
):
    """Intercept the runtime body called by SGLang's opaque Dynamo custom op."""

    @wraps(baseline)
    def dispatched(self, input_):
        output = torch.empty_like(input_)
        selected = _selection(self, input_, output, slot, registry)
        if selected is None:
            return baseline(self, input_)
        impl, group, _, descriptor = selected
        if registry.select(slot, descriptor).impl is not impl:
            raise RuntimeError("in-place all-reduce selection changed before commit")
        _receipts.invoke(slot, impl.entry, input_, output, group)
        input_.copy_(output)
        _complete(slot)
        return None

    return dispatched


def make_allreduce_outplace_dispatcher(
    baseline,
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "collective.all_reduce",
):
    """Intercept the out-of-place runtime body behind SGLang's custom op."""

    @wraps(baseline)
    def dispatched(self, input_, method):
        output = torch.empty_like(input_)
        selected = _selection(self, input_, output, slot, registry)
        if selected is None:
            return baseline(self, input_, method)
        impl, group, _, descriptor = selected
        if registry.select(slot, descriptor).impl is not impl:
            raise RuntimeError("out-of-place all-reduce selection changed before commit")
        _receipts.invoke(slot, impl.entry, input_, output, group)
        _complete(slot)
        return output

    return dispatched
