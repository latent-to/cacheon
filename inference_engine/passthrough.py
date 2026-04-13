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

        scale = self.config.head_dim ** -0.5
        attn_weights = torch.matmul(query, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            # Use the HF-provided 4D mask [batch, 1, q_len, kv_len] directly.
            # It already encodes causal masking and sliding window constraints.
            causal_mask = attention_mask[:, :, :, : k.shape[2]]
            attn_weights = attn_weights + causal_mask
        else:
            # Fallback: hand-rolled causal mask for use outside the harness
            # (e.g. unit tests that call attend() directly).
            q_len = query.shape[2]
            kv_len = k.shape[2]
            if q_len > 1:
                mask = torch.triu(
                    torch.full(
                        (q_len, kv_len),
                        float("-inf"),
                        device=query.device,
                        dtype=query.dtype,
                    ),
                    diagonal=kv_len - q_len + 1,
                )
                attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        output = torch.matmul(attn_weights, v)

        return AttentionOutput(output=output, attention_weights=attn_weights)

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
