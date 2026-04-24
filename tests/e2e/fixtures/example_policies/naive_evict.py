"""Naive eviction — keep only the most recent N tokens.

This is deliberately simple (no importance tracking). With a small budget
it will likely fail the KL gate on long-context prompts, which is useful
for testing the scoring pipeline's DQ path.
"""

import torch
import torch.nn.functional as F

from inference_engine.policy import KVCachePolicy, CacheConfig, AttentionOutput


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, num_kv_heads, slen, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, slen, head_dim)
    return x.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)


class NaiveEvictPolicy(KVCachePolicy):
    def __init__(self, budget: int = 1024):
        self.budget = budget

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

        # Evict oldest tokens if over budget
        if self.k_cache[layer_idx].shape[2] > self.budget:
            self.k_cache[layer_idx] = self.k_cache[layer_idx][:, :, -self.budget :, :]
            self.v_cache[layer_idx] = self.v_cache[layer_idx][:, :, -self.budget :, :]

    def attend(self, query, layer_idx, attention_mask=None, **kwargs):
        k = _repeat_kv(self.k_cache[layer_idx], self.num_kv_groups)
        v = _repeat_kv(self.v_cache[layer_idx], self.num_kv_groups)

        scale = self.config.head_dim ** -0.5
        attn_weights = torch.matmul(query, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : k.shape[2]]
            attn_weights = attn_weights + causal_mask
        else:
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
