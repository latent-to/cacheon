"""INT8 quantization with SDPA attention — long-context safe.

Same quantization as int8.py (per-row symmetric INT8 scales), but uses
F.scaled_dot_product_attention instead of manual matmul. This avoids
allocating the O(N²) attention matrix and works at 32K+ context on H100.

Expected: ~2x KV-cache memory reduction, passes KL gate, scores > 0.
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


class Int8SdpaPolicy(KVCachePolicy):
    def setup(self, config: CacheConfig) -> None:
        self.config = config
        self.num_kv_groups = config.num_heads // config.num_kv_heads
        self.k_cache: list[list[torch.Tensor]] = [[] for _ in range(config.num_layers)]
        self.v_cache: list[list[torch.Tensor]] = [[] for _ in range(config.num_layers)]
        self.k_scales: list[list[torch.Tensor]] = [[] for _ in range(config.num_layers)]
        self.v_scales: list[list[torch.Tensor]] = [[] for _ in range(config.num_layers)]

    def write(self, keys, values, layer_idx, positions):
        k_scale = keys.abs().amax(dim=-1, keepdim=True) / 127
        v_scale = values.abs().amax(dim=-1, keepdim=True) / 127
        k_int8 = (keys / k_scale.clamp(min=1e-8)).round().to(torch.int8)
        v_int8 = (values / v_scale.clamp(min=1e-8)).round().to(torch.int8)
        self.k_cache[layer_idx].append(k_int8)
        self.v_cache[layer_idx].append(v_int8)
        self.k_scales[layer_idx].append(k_scale)
        self.v_scales[layer_idx].append(v_scale)

    def attend(self, query, layer_idx, attention_mask=None, **kwargs):
        K_int = torch.cat(self.k_cache[layer_idx], dim=2).float()
        V_int = torch.cat(self.v_cache[layer_idx], dim=2).float()
        K_scale = torch.cat(self.k_scales[layer_idx], dim=2)
        V_scale = torch.cat(self.v_scales[layer_idx], dim=2)

        K = (K_int * K_scale).to(query.dtype)
        V = (V_int * V_scale).to(query.dtype)

        k = _repeat_kv(K, self.num_kv_groups)
        v = _repeat_kv(V, self.num_kv_groups)

        is_causal = query.shape[2] > 1
        output = F.scaled_dot_product_attention(
            query, k, v, is_causal=is_causal,
        )

        return AttentionOutput(output=output, attention_weights=None)

    def memory_bytes(self) -> int:
        total = 0
        for layer_idx in range(len(self.k_cache)):
            for k in self.k_cache[layer_idx]:
                total += k.nelement() * k.element_size()
            for v in self.v_cache[layer_idx]:
                total += v.nelement() * v.element_size()
            for s in self.k_scales[layer_idx]:
                total += s.nelement() * s.element_size()
            for s in self.v_scales[layer_idx]:
                total += s.nelement() * s.element_size()
        return total

    def get_config(self) -> dict:
        return {"name": "int8_sdpa", "description": "INT8 quantization with SDPA attention"}
