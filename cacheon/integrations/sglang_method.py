"""Serve every data-bound ``Class.method`` seam row through one call body.

A row in ``cacheon.seams`` names a stock method and maps the slot's ABI tensor
names onto that method's parameters. Everything else comes from the slot contract
the offline verifier already uses: output allocation, the entry call shape
(``SlotSpec.invoke_entry``) and the audit grader. A kernel location that is a
whole method is therefore a row and a slot, never another adapter, and every such
location shares this one audited body.

``install()`` must run before the engine constructs the model: ``BaseFusedOp``
memoizes a bound ``forward_*`` per instance, so an instance built earlier keeps
the stock method.
"""

from __future__ import annotations

import inspect
import sys
from typing import Callable

import torch

from cacheon import audit as _audit
from cacheon import receipts as _receipts
from cacheon import seams
from cacheon.dispatch import (
    _allocate_live_outputs,
    _dynamo_compiling,
    _elementwise_descriptor,
    _in_cuda_graph,
    _validate_live_outputs,
)
from cacheon.registry import REGISTRY, KernelRegistry
from cacheon.slots import get_slot

_STOCK = "_cacheon_stock_methods"


def make_method_dispatcher(
    baseline: Callable[..., object],
    adapter: seams.SeamAdapter,
    *,
    registry: KernelRegistry = REGISTRY,
) -> Callable[..., object]:
    """Build the replacement for one stock method named by a data-bound row."""

    (slot,) = adapter.slots
    signature = inspect.signature(baseline)

    def dispatched(self, *args, **kwargs):
        # A Dynamo-traced region bakes pure stock, and a running candidate that
        # calls back into the engine gets stock rather than selecting itself.
        if (
            _dynamo_compiling()
            or _receipts.is_invoking()
            or not (registry.active and registry.variants(slot))
        ):
            return baseline(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        inputs = {name: bound.arguments[param] for name, param in adapter.inputs}
        if not all(torch.is_tensor(v) and v.dim() for v in inputs.values()):
            return baseline(self, *args, **kwargs)
        x = inputs[adapter.inputs[0][0]]
        in_graph = _in_cuda_graph()
        impl = registry.select(
            slot,
            _elementwise_descriptor(x, last_dim=int(x.shape[-1])).with_updates(
                graph_mode="cuda_graph" if in_graph else "eager"
            ),
        ).impl
        if impl is None:
            return baseline(self, *args, **kwargs)

        # Stock answers first, on the real input addresses, and its outputs are
        # snapshotted. The inputs are then restored, so a stock method that works
        # in place and a candidate that scribbles on its inputs are both graded
        # against the same pristine call without a per-row audit order.
        expected = None
        if not in_graph and _audit.sampled():
            pristine = [value.clone() for value in inputs.values()]
            expected = _audit.capture_reference(
                slot, lambda: baseline(self, *args, **kwargs)
            )
            for value, saved in zip(inputs.values(), pristine):
                value.copy_(saved)

        contract, allocation, tensor_inputs, input_bindings = _allocate_live_outputs(
            slot, inputs, like=x
        )
        get_slot(slot).invoke_entry(
            lambda *call: _receipts.invoke(slot, impl.entry, *call),
            inputs,
            allocation.outputs,
            None,
        )
        _validate_live_outputs(
            contract, allocation, tensor_inputs, input_bindings, like=x
        )
        if expected is not None:
            _audit.record(slot, allocation.outputs, expected, scaled=True)
        _receipts.completed(slot)
        outputs = allocation.outputs
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    return dispatched


def _rows() -> tuple[seams.SeamAdapter, ...]:
    return tuple(row for row in seams.SEAM_ADAPTERS if row.inputs)


def _stock_class(row: seams.SeamAdapter):
    module = sys.modules.get(row.target_module)
    return getattr(module, row.chokepoint.partition(".")[0], None)


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Patch every data-bound row whose stock class is already defined.

    A module that is not imported yet, or is still mid-import, is skipped: the
    bootstrap post-import hook calls this again once it finishes loading.
    """

    for row in _rows():
        cls = _stock_class(row)
        if cls is None:
            continue
        stock = cls.__dict__.get(_STOCK)
        if stock is None:
            stock = {}
            setattr(cls, _STOCK, stock)
        for method in (row.chokepoint.partition(".")[2], *row.also):
            if method in stock:
                continue
            stock[method] = getattr(cls, method)
            setattr(
                cls,
                method,
                make_method_dispatcher(stock[method], row, registry=registry),
            )


def uninstall() -> None:
    for row in _rows():
        cls = _stock_class(row)
        for method, original in (getattr(cls, _STOCK, None) or {}).items():
            setattr(cls, method, original)
        if cls is not None and _STOCK in cls.__dict__:
            delattr(cls, _STOCK)


def is_installed(name: str) -> bool:
    """Whether the named row's chokepoint currently routes through this adapter."""

    for row in _rows():
        cls = _stock_class(row)
        if row.name == name and cls is not None:
            return row.chokepoint.partition(".")[2] in cls.__dict__.get(_STOCK, {})
    return False
