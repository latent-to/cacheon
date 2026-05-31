"""The validator-owned dispatcher.

This is the only place a miner kernel is actually invoked during serving. It is
written so that the *validator* owns everything risky around the call:

  * output allocation (shape/dtype/device/stride) — never the miner,
  * eligibility gating via the registry,
  * a fallback to the trusted baseline on ineligibility or error,
  * a single, auditable call into the miner ``entry(*inputs, out)``.

The miner's ``entry`` therefore only ever sees already-allocated tensors and is
expected to fill ``out``. That is the smallest host surface we can give a
Triton/CuteDSL submission while still letting it own the actual computation.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from optima.registry import REGISTRY, KernelRegistry


def _arch_tag(device_index: int = 0) -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm{major}{minor}"


def make_silu_and_mul_dispatcher(
    baseline_forward: Callable[[object, torch.Tensor], torch.Tensor],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "activation.silu_and_mul",
) -> Callable[[object, torch.Tensor], torch.Tensor]:
    """Build a replacement for ``SiluAndMul.forward_*``.

    ``baseline_forward`` is the captured original (used for fallback). The
    returned function has the same ``(self, x)`` signature.
    """

    def dispatched(self: object, x: torch.Tensor) -> torch.Tensor:
        last_dim = x.shape[-1]
        impl = registry.lookup(
            slot,
            dtype_name=_dtype_name(x.dtype),
            last_dim=last_dim,
            arch=_arch_tag(x.device.index or 0) if x.is_cuda else None,
        )
        if impl is None:
            return baseline_forward(self, x)

        d = last_dim // 2
        out = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
        try:
            impl.entry(x, out)
        except Exception:
            if registry.strict:
                raise
            # Quality/throughput already protect us; a crashing kernel just loses.
            return baseline_forward(self, x)
        return out

    return dispatched


def make_rmsnorm_dispatcher(
    baseline_forward: Callable[..., object],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "norm.rmsnorm",
) -> Callable[..., object]:
    """Build a replacement for ``RMSNorm.forward_cuda`` / ``forward_native``.

    sglang's RMSNorm has two modes: plain (``residual is None`` -> return normed)
    and fused add+norm (``residual`` given -> return ``(normed, x+residual)``).
    The validator owns the residual add; the miner kernel only ever computes the
    pure rmsnorm: ``entry(x, weight, out, eps)``. Unusual paths fall back to the
    trusted baseline.
    """

    def dispatched(self, x, residual=None, post_residual_addition=None):
        # Rare / fp32 paths -> trusted baseline (keeps the contract simple & safe).
        if post_residual_addition is not None or getattr(self, "fp32_residual", False):
            return baseline_forward(self, x, residual, post_residual_addition)

        impl = registry.lookup(
            slot,
            dtype_name=_dtype_name(x.dtype),
            last_dim=x.shape[-1],
            arch=_arch_tag(x.device.index or 0) if x.is_cuda else None,
        )
        if impl is None:
            return baseline_forward(self, x, residual, post_residual_addition)

        eps = float(self.variance_epsilon)
        weight = self.weight.data
        try:
            if residual is None:
                out = torch.empty_like(x)
                impl.entry(x, weight, out, eps)
                return out
            new_residual = x + residual  # validator owns the add
            out = torch.empty_like(new_residual)
            impl.entry(new_residual, weight, out, eps)
            return out, new_residual
        except Exception:
            if registry.strict:
                raise
            return baseline_forward(self, x, residual, post_residual_addition)

    return dispatched


def make_attention_dispatcher(
    baseline_forward: Callable[..., object],
    *,
    registry: KernelRegistry = REGISTRY,
    slot: str = "attention.decode",
) -> Callable[..., object]:
    """Build a replacement for ``RadixAttention.forward`` — the single chokepoint
    every attention call funnels through (so it is backend-agnostic).

    Attention is a *block* slot. At **decode** (one query token per request) we
    extract the request's paged KV out of ``forward_batch`` and hand the miner kernel
    a dense ``(q, k, v, seq_lens)`` view; the validator keeps the backend metadata,
    owns the KV-cache **write** (``set_kv_buffer``), and only ever lets the miner
    *read* q/k/v. The kernel output feeds the residual stream -> downstream layers ->
    sampler (all stock), so there is no final output to substitute — the same
    property that makes the op slots safe.

    SCOPE: this routes **decode** attention through the ``attention.decode`` slot, and
    only when ``OPTIMA_ATTENTION_SEAM=1`` (opt-in until paged-direct lands). It is a
    *gather* MVP — it pulls the paged KV into a dense padded tensor so the miner writes
    an ordinary attention kernel, but the gather is variable-shape, hence
    **eager-only** (a per-step ``max_len`` is not CUDA-graph-capturable). The
    production rung is a paged-direct contract (the miner consumes the page table +
    pool buffers, graph-safe). Prefill / MLA / cross-attention / windowed paths fall
    back to the trusted backend. Conservative by construction: when in doubt, trust
    the baseline.
    """

    def dispatched(self, q, k, v, forward_batch, save_kv_cache: bool = True, **kwargs):
        if _attention_seam_active():
            try:
                if forward_batch.forward_mode.is_decode() and _decode_supported(self, k, v, kwargs):
                    impl = registry.lookup(
                        slot,
                        dtype_name=_dtype_name(q.dtype),
                        last_dim=getattr(self, "qk_head_dim", q.shape[-1]),
                        arch=_arch_tag(q.device.index or 0) if q.is_cuda else None,
                    )
                    if impl is not None:
                        return _run_decode_kernel(self, q, k, v, forward_batch, save_kv_cache, impl)
            except Exception:
                if registry.strict:
                    raise
                # any mismatch with this sglang's internals -> trust the baseline
        return baseline_forward(self, q, k, v, forward_batch, save_kv_cache, **kwargs)

    return dispatched


def _attention_seam_active() -> bool:
    # Opt-in until the paged-direct (graph-safe) contract lands; keeps the attention
    # seam from engaging in production before it is validated end-to-end.
    import os

    return os.environ.get("OPTIMA_ATTENTION_SEAM") == "1"


def _decode_supported(self, k, v, kwargs) -> bool:
    # The gather MVP supports standard MHA decode only: real k/v, uniform head dim,
    # no MLA-rope-split / cross-attention / sliding window. Anything else -> baseline.
    if k is None or v is None or "k_rope" in kwargs:
        return False
    if getattr(self, "is_cross_attention", False):
        return False
    if getattr(self, "sliding_window_size", -1) not in (-1, 0):
        return False
    return getattr(self, "qk_head_dim", None) == getattr(self, "v_head_dim", None)


def _run_decode_kernel(self, q, k, v, forward_batch, save_kv_cache, impl):
    """Extract this decode step's paged KV, gather it dense, run the miner kernel.

    Mirrors what a stock backend does: store the new token's k/v at ``out_cache_loc``
    (validator-owned write), then gather each request's context via
    ``req_to_token[req_pool_idx, :seq_len]`` and let the miner compute attention.
    """
    Hq = self.tp_q_head_num
    Hkv = self.tp_k_head_num
    D = self.qk_head_dim
    pool = forward_batch.token_to_kv_pool

    # Validator owns the KV-cache write (miner only reads). Store BEFORE gathering so
    # the gathered context includes the current token.
    if save_kv_cache and k is not None:
        pool.set_kv_buffer(self, forward_batch.out_cache_loc,
                           k.view(-1, Hkv, D), v.view(-1, Hkv, D))

    seq_lens = forward_batch.seq_lens
    B = seq_lens.shape[0]
    max_len = int(seq_lens.max().item())  # variable shape -> eager only (see docstring)
    req_to_token = forward_batch.req_to_token_pool.req_to_token
    slots = req_to_token[forward_batch.req_pool_indices][:, :max_len].long()  # (B, max_len)

    k_cache = pool.get_key_buffer(self.layer_id)   # (pool_size, Hkv, D)
    v_cache = pool.get_value_buffer(self.layer_id)
    k_pad = k_cache[slots]                          # (B, max_len, Hkv, D)
    v_pad = v_cache[slots]
    q3 = q.view(B, Hq, D)

    out = torch.empty((B, Hq, D), dtype=q.dtype, device=q.device)
    impl.entry(q3, k_pad, v_pad, seq_lens, float(self.scaling), out)  # miner fills out
    return out.reshape(B, Hq * D)


def _dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }.get(dtype, str(dtype).replace("torch.", ""))
