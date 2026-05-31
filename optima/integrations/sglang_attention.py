"""Wire the Optima dispatcher into SGLang's attention layer.

Unlike SiluAndMul / RMSNorm (single pure ops), attention is a *block*: the miner
kernel computes scaled-dot-product attention over q, k, v. We patch the single
chokepoint every attention call funnels through — ``RadixAttention.forward`` in
``sglang.srt.layers.radix_attention`` — so the seam is backend-agnostic (it works
whether the configured backend is FlashInfer / FA3 / FlashMLA / Triton / ...).

This is the concrete answer to "let miners change the attention backend while the
validator keeps ONE pinned, unmodified sglang": we never fork or reconfigure
sglang; the miner's kernel is injected at runtime, exactly like the silu/norm
seams. The validator keeps the backend's metadata, the CUDA-graph hooks, and the
KV-cache write; the miner kernel only reads q/k/v and fills the output, which feeds
the residual stream + downstream layers + sampler (all stock) — so there is still
no final output to substitute.

See ``optima/dispatch.make_attention_dispatcher`` for the scope of what's wired
today (the self-contained case, gated behind ``OPTIMA_ATTENTION_SEAM=1``) versus
the paged-decode / MLA-latent GPU integration point.
"""

from __future__ import annotations

from optima.dispatch import make_attention_dispatcher
from optima.registry import REGISTRY, KernelRegistry

_PATCH_FLAG = "_optima_attn_patched"


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Patch ``RadixAttention.forward``. No-ops until radix_attention is imported."""
    import sys

    mod = sys.modules.get("sglang.srt.layers.radix_attention")
    RadixAttention = getattr(mod, "RadixAttention", None) if mod is not None else None
    if RadixAttention is None:
        return

    if getattr(RadixAttention, _PATCH_FLAG, False):
        return

    orig_forward = RadixAttention.forward
    RadixAttention.forward = make_attention_dispatcher(orig_forward, registry=registry)
    RadixAttention._optima_orig_forward = orig_forward  # type: ignore[attr-defined]
    setattr(RadixAttention, _PATCH_FLAG, True)


def uninstall() -> None:
    import sys

    if "sglang.srt.layers.radix_attention" not in sys.modules:
        return
    from sglang.srt.layers.radix_attention import RadixAttention

    if not getattr(RadixAttention, _PATCH_FLAG, False):
        return
    RadixAttention.forward = RadixAttention._optima_orig_forward  # type: ignore[attr-defined]
    delattr(RadixAttention, "_optima_orig_forward")
    setattr(RadixAttention, _PATCH_FLAG, False)
