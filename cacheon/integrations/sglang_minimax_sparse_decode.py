"""Bind MiniMax sparse-attend and decode-score seams to the pinned runtime."""

from __future__ import annotations

from cacheon import audit as _audit
from cacheon.integrations._by_value_function import ByValueFunctionPatch
from cacheon.minimax_sparse_decode_dispatch import (
    make_msa_block_score_kernel,
    make_minimax_sparse_decode_dispatcher,
)
from cacheon.registry import REGISTRY, KernelRegistry

_SOURCE = "sglang.srt.layers.attention.minimax_sparse_ops.decode.topk_sparse"
_CONSUMER = "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse"
_FUNCTION = "flash_decode_with_gqa_share_sparse"
_PATCH = ByValueFunctionPatch(_SOURCE, _CONSUMER, _FUNCTION, "cacheon_sparse_decode")
_PATCHED = _PATCH.patched
_BACKEND = "sglang.srt.layers.attention.minimax_sparse_backend"
_BACKEND_CLASS = "MiniMaxSparseAttnBackend"
_BACKEND_ORIGINAL = "_cacheon_orig_decode_audit_init"
_SCORE_SOURCE = (
    "sglang.srt.layers.attention.minimax_sparse_ops.decode.flash_with_topk_idx"
)
_SCORE_FUNCTION = "_decode_score_kernel"
_SCORE_PATCH = ByValueFunctionPatch(
    _SCORE_SOURCE, None, _SCORE_FUNCTION, "cacheon_msa_score", False
)


def _install_audit_mode() -> None:
    import sys

    module = sys.modules.get(_BACKEND)
    cls = getattr(module, _BACKEND_CLASS, None) if module is not None else None
    if cls is None or getattr(cls, _PATCHED, False):
        return
    original = cls.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if _audit.enabled():
            self._use_msa_decode = self._msa_owns_decode = False

    setattr(cls, _BACKEND_ORIGINAL, original)
    cls.__init__ = initialize
    setattr(cls, _PATCHED, True)


def install(registry: KernelRegistry = REGISTRY) -> None:
    _install_audit_mode()
    _SCORE_PATCH.install(
        lambda original, _: make_msa_block_score_kernel(original, registry=registry)
    )
    _PATCH.install(
        lambda original, _source: make_minimax_sparse_decode_dispatcher(
            original, registry=registry
        )
    )


def uninstall() -> None:
    import sys

    _SCORE_PATCH.uninstall()
    module = sys.modules.get(_BACKEND)
    cls = getattr(module, _BACKEND_CLASS, None) if module is not None else None
    if cls is not None and getattr(cls, _PATCHED, False):
        cls.__init__ = getattr(cls, _BACKEND_ORIGINAL)
        delattr(cls, _BACKEND_ORIGINAL)
        setattr(cls, _PATCHED, False)
    _PATCH.uninstall()


def is_installed() -> bool:
    import sys

    score = sys.modules.get(_SCORE_SOURCE)
    if not _PATCH.is_installed() or (
        score is not None and not _SCORE_PATCH.is_installed()
    ):
        return False
    module = sys.modules.get(_BACKEND)
    cls = getattr(module, _BACKEND_CLASS, None) if module is not None else None
    return cls is None or (
        getattr(cls, _PATCHED, False) and hasattr(cls, _BACKEND_ORIGINAL)
    )


__all__ = ["install", "is_installed", "uninstall"]
