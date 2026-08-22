"""Wire the Cacheon dispatcher into SGLang's RMSNorm seams.

Same approach as the SiluAndMul seam: the norm classes are ``MultiPlatformOp``s, so
we replace their class methods ``forward_cuda`` / ``forward_native`` (before the model
is built) with a dispatcher that routes to a miner kernel when one is registered and
eligible, else falls back to the captured baseline.

Two classes, one slot. ``RMSNorm`` scales by ``weight``; ``GemmaRMSNorm`` scales by
``1 + weight`` and is a sibling class, not a subclass — patching only ``RMSNorm``
leaves the Gemma callsites cold. That is not hypothetical: MiniMax-M3 sets
``use_gemma_norm=True``, so every layer norm, the final norm, and the per-head q/k
norms are ``GemmaRMSNorm``, and a candidate for ``norm.rmsnorm`` could not execute at
all in that arena. The slot contract stays one plain ``x_normed * weight``; the Gemma
binding folds the ``+ 1`` into the scale it hands the candidate.

``GemmaRMSNorm.forward_with_allreduce_fusion`` is deliberately NOT patched here: that
path belongs to ``collective.ar_residual_rmsnorm`` and its own seam owns it.
"""

from __future__ import annotations

from cacheon.dispatch import make_rmsnorm_dispatcher
from cacheon.registry import REGISTRY, KernelRegistry

_MODULE = "sglang.srt.layers.layernorm"
_PATCH_FLAG = "_cacheon_patched"
# class name -> whether its scale is (1 + weight) rather than weight.
_CLASSES = (("RMSNorm", False), ("GemmaRMSNorm", True))


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Patch forward_cuda/native on every norm class. No-op until layernorm imports."""
    import sys

    mod = sys.modules.get(_MODULE)
    if mod is None:
        return

    for name, gemma_fold in _CLASSES:
        cls = getattr(mod, name, None)
        # An absent class is owned by the compat canary, which asserts every
        # chokepoint in the seam table against the pinned runtime.
        if cls is None or getattr(cls, _PATCH_FLAG, False):
            continue
        orig_cuda = cls.forward_cuda
        orig_native = cls.forward_native
        cls.forward_cuda = make_rmsnorm_dispatcher(
            orig_cuda, registry=registry, gemma_fold=gemma_fold)
        cls.forward_native = make_rmsnorm_dispatcher(
            orig_native, registry=registry, gemma_fold=gemma_fold)
        cls._cacheon_orig_cuda = orig_cuda
        cls._cacheon_orig_native = orig_native
        setattr(cls, _PATCH_FLAG, True)


def uninstall() -> None:
    import sys

    mod = sys.modules.get(_MODULE)
    if mod is None:
        return
    for name, _gemma_fold in _CLASSES:
        cls = getattr(mod, name, None)
        if cls is None or not getattr(cls, _PATCH_FLAG, False):
            continue
        cls.forward_cuda = cls._cacheon_orig_cuda
        cls.forward_native = cls._cacheon_orig_native
        del cls._cacheon_orig_cuda
        del cls._cacheon_orig_native
        setattr(cls, _PATCH_FLAG, False)
