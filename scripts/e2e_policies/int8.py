"""INT8 quantization — per-row symmetric scales.

Dequantizes to FP16 inside attend(). Expected ~2x memory reduction,
neutral latency.
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


class Int8Policy(KVCachePolicy):
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
