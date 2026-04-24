"""KVCachePolicy interface — every miner submission implements this class."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CacheConfig:
    num_layers: int
    num_heads: int        # query heads (28 for Qwen2.5-7B)
    num_kv_heads: int     # KV heads (4 for Qwen2.5-7B) — fewer due to GQA
    head_dim: int         # 128 for Qwen2.5-7B
    max_seq_len: int
    dtype: torch.dtype


@dataclass
class AttentionOutput:
    output: torch.Tensor                        # [batch, heads, seq_len, head_dim]
    attention_weights: torch.Tensor | None = None  # optional, for validation


class KVCachePolicy:
    """Base class for all KV cache policies.

    The miner owns the cache lifecycle: how K/V are stored (write) and
    how attention is computed against them (attend). Memory is measured
    by the harness via CUDA allocator delta, not self-reported.
    """

    def setup(self, config: CacheConfig) -> None:
        """Called once per sequence before prefill. Initialize internal state."""
        raise NotImplementedError

    def write(
        self,
        keys: torch.Tensor,       # [batch, num_kv_heads, seq_len, head_dim]
        values: torch.Tensor,      # [batch, num_kv_heads, seq_len, head_dim]
        layer_idx: int,
        positions: torch.Tensor,   # [seq_len] — token positions in the sequence
    ) -> None:
        """Store K/V entries. Compress, quantize, evict — miner's choice.

        Keys and values arrive post-RoPE and post-projection. During prefill,
        seq_len is the full prompt length. During decode, seq_len is 1.
        """
        raise NotImplementedError

    def attend(
        self,
        query: torch.Tensor,       # [batch, num_heads, seq_len, head_dim]
        layer_idx: int,
        **kwargs,
    ) -> AttentionOutput:
        """Compute attention against the stored cache.

        Scoring, softmax, aggregation — all in one call.
        Fuse or decompress as you see fit.

        kwargs may include `attention_mask` — a 4D float tensor `[batch, 1, q_len, kv_len]`
        with 0/-inf values (HF causal mask, already incorporates sliding window if applicable).
        Miners may use or ignore it.
        """
        raise NotImplementedError

    def get_config(self) -> dict:
        """Human-readable description of the policy."""
        raise NotImplementedError
