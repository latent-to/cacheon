"""Bind the MSA prefill dispatcher at its source and by-value consumer."""

from __future__ import annotations

from cacheon.dispatch import make_msa_prefill_dispatcher
from cacheon.integrations._by_value_function import ByValueFunctionPatch
from cacheon.registry import REGISTRY, KernelRegistry

_SOURCE_MODULE = (
    "sglang.srt.layers.attention.minimax_sparse_ops.prefill.flash_with_topk_idx"
)
_CONSUMER_MODULE = "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse"
_FUNC = "flash_prefill_with_topk_index"
_PATCH = ByValueFunctionPatch(
    _SOURCE_MODULE, _CONSUMER_MODULE, _FUNC, "cacheon_msa_prefill"
)


def install(registry: KernelRegistry = REGISTRY) -> None:
    _PATCH.install(
        lambda original, source: make_msa_prefill_dispatcher(
            original, source, registry=registry
        )
    )


def uninstall() -> None:
    _PATCH.uninstall()


def is_installed() -> bool:
    return _PATCH.is_installed()
