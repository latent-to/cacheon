"""Passthrough policy — uncompressed FP16 KV cache with standard attention.

Used as the baseline in every evaluation round. Both baseline and miner
run through the same monkey-patched harness so relative comparisons are fair.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .policy import KVCachePolicy, CacheConfig, AttentionOutput


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query head count (GQA)."""
    if n_rep == 1:
        return x
    bsz, num_kv_heads, slen, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, slen, head_dim)
    return x.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)


class PassthroughPolicy(KVCachePolicy):

    def setup(self, config: CacheConfig) -> None:
        self.config = config
        self.num_kv_groups = config.num_heads // config.num_kv_heads
        self.k_cache: list[torch.Tensor | None] = [None] * config.num_layers
        self.v_cache: list[torch.Tensor | None] = [None] * config.num_layers

    def write(self, keys, values, layer_idx, positions):
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = keys
            self.v_cache[layer_idx] = values
        else:
            self.k_cache[layer_idx] = torch.cat(
                [self.k_cache[layer_idx], keys], dim=2
            )
            self.v_cache[layer_idx] = torch.cat(
                [self.v_cache[layer_idx], values], dim=2
            )

    def attend(self, query, layer_idx, attention_mask=None, **kwargs):
        k = _repeat_kv(self.k_cache[layer_idx], self.num_kv_groups)
        v = _repeat_kv(self.v_cache[layer_idx], self.num_kv_groups)

        # Prefill (q_len > 1): standard lower-triangular causal mask.
        # Decode  (q_len == 1): attend to all cached positions, no mask.
        # SDPA dispatches to Flash Attention on CUDA, which tiles the
        # computation in O(N) memory — no O(N²) attention matrix allocated.
        is_causal = query.shape[2] > 1
        output = F.scaled_dot_product_attention(
            query, k, v, is_causal=is_causal,
        )

        return AttentionOutput(output=output, attention_weights=None)

    def memory_bytes(self) -> int:
        total = 0
        for k, v in zip(self.k_cache, self.v_cache):
            if k is not None:
                total += k.nelement() * k.element_size()
            if v is not None:
                total += v.nelement() * v.element_size()
        return total

    def get_config(self) -> dict:
        return {"name": "passthrough", "description": "Uncompressed FP16 KV cache"}
