"""Passthrough baseline — uncompressed FP16 KV cache.

This is the identity policy: same math as the unpatched model, just wired
through the KVCachePolicy interface. It should score ~0 on every axis.
"""

import torch
import torch.nn.functional as F

from inference_engine.policy import KVCachePolicy, CacheConfig, AttentionOutput


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

        # SDPA dispatches to Flash Attention on CUDA — O(N) memory, no OOM at 32K.
        is_causal = query.shape[2] > 1
        output = F.scaled_dot_product_attention(query, k, v, is_causal=is_causal)

        return AttentionOutput(output=output, attention_weights=None)
