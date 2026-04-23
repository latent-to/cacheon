"""Unit tests for inference_engine — no GPU, no model download required.

Tests policy math with small hand-crafted tensors.
Run with: pytest tests/test_inference_engine.py -v
"""

import math

import pytest
import torch

from inference_engine.passthrough import PassthroughPolicy, _repeat_kv
from inference_engine.policy import AttentionOutput, CacheConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def small_config(num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8, seq_len=16):
    return CacheConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=seq_len,
        dtype=torch.float32,
    )


def rand_kv(batch=1, num_kv_heads=2, seq_len=5, head_dim=8):
    return (
        torch.randn(batch, num_kv_heads, seq_len, head_dim),
        torch.randn(batch, num_kv_heads, seq_len, head_dim),
    )


# ---------------------------------------------------------------------------
# _repeat_kv
# ---------------------------------------------------------------------------

class TestRepeatKV:
    def test_no_repeat(self):
        x = torch.randn(1, 2, 5, 8)
        assert _repeat_kv(x, 1) is x

    def test_repeat_2x(self):
        x = torch.randn(1, 2, 5, 8)
        out = _repeat_kv(x, 2)
        assert out.shape == (1, 4, 5, 8)
        # First kv head repeated: out[:, 0] == out[:, 1]
        assert torch.equal(out[:, 0], out[:, 1])
        # Second kv head repeated: out[:, 2] == out[:, 3]
        assert torch.equal(out[:, 2], out[:, 3])

    def test_values_preserved(self):
        x = torch.randn(1, 2, 5, 8)
        out = _repeat_kv(x, 3)
        assert out.shape == (1, 6, 5, 8)
        assert torch.allclose(out[:, 0], x[:, 0])
        assert torch.allclose(out[:, 3], x[:, 1])


# ---------------------------------------------------------------------------
# PassthroughPolicy.write
# ---------------------------------------------------------------------------

class TestPassthroughWrite:
    def test_first_write_stores_tensor(self):
        p = PassthroughPolicy()
        p.setup(small_config())
        k, v = rand_kv(seq_len=5)
        p.write(k, v, layer_idx=0, positions=torch.arange(5))
        assert torch.equal(p.k_cache[0], k)
        assert torch.equal(p.v_cache[0], v)

    def test_second_write_appends(self):
        p = PassthroughPolicy()
        p.setup(small_config())
        k1, v1 = rand_kv(seq_len=5)
        k2, v2 = rand_kv(seq_len=1)
        p.write(k1, v1, layer_idx=0, positions=torch.arange(5))
        p.write(k2, v2, layer_idx=0, positions=torch.tensor([5]))
        assert p.k_cache[0].shape[2] == 6
        assert torch.equal(p.k_cache[0][:, :, :5, :], k1)
        assert torch.equal(p.k_cache[0][:, :, 5:, :], k2)

    def test_layers_are_independent(self):
        p = PassthroughPolicy()
        p.setup(small_config(num_layers=2))
        k0, v0 = rand_kv(seq_len=3)
        k1, v1 = rand_kv(seq_len=7)
        p.write(k0, v0, layer_idx=0, positions=torch.arange(3))
        p.write(k1, v1, layer_idx=1, positions=torch.arange(7))
        assert p.k_cache[0].shape[2] == 3
        assert p.k_cache[1].shape[2] == 7


# ---------------------------------------------------------------------------
# PassthroughPolicy.attend
# ---------------------------------------------------------------------------

class TestPassthroughAttend:
    def test_output_shape_prefill(self):
        """attend() output matches [batch, num_heads, q_len, head_dim] during prefill."""
        cfg = small_config(num_heads=4, num_kv_heads=2, head_dim=8)
        p = PassthroughPolicy()
        p.setup(cfg)
        k, v = rand_kv(num_kv_heads=2, seq_len=5)
        p.write(k, v, layer_idx=0, positions=torch.arange(5))
        q = torch.randn(1, 4, 5, 8)
        out = p.attend(q, layer_idx=0)
        assert out.output.shape == (1, 4, 5, 8)

    def test_output_shape_decode(self):
        """attend() output matches [batch, num_heads, 1, head_dim] during decode."""
        cfg = small_config(num_heads=4, num_kv_heads=2, head_dim=8)
        p = PassthroughPolicy()
        p.setup(cfg)
        k, v = rand_kv(num_kv_heads=2, seq_len=5)
        p.write(k, v, layer_idx=0, positions=torch.arange(5))
        k2, v2 = rand_kv(num_kv_heads=2, seq_len=1)
        p.write(k2, v2, layer_idx=0, positions=torch.tensor([5]))
        q = torch.randn(1, 4, 1, 8)
        out = p.attend(q, layer_idx=0)
        assert out.output.shape == (1, 4, 1, 8)

    def test_attention_weights_none_with_sdpa(self):
        """SDPA does not return attention weights; field should be None."""
        cfg = small_config(num_heads=4, num_kv_heads=2, head_dim=8)
        p = PassthroughPolicy()
        p.setup(cfg)
        k, v = rand_kv(num_kv_heads=2, seq_len=5)
        p.write(k, v, layer_idx=0, positions=torch.arange(5))
        q = torch.randn(1, 4, 5, 8)
        out = p.attend(q, layer_idx=0)
        assert out.attention_weights is None

    def test_causal_mask_prefill(self):
        """Position i must not attend to positions j > i during prefill.

        V is identity so output[i] == softmax-weights[i]; checking the
        output upper-triangle is equivalent to checking the weight matrix.
        """
        cfg = small_config(num_heads=1, num_kv_heads=1, head_dim=4)
        p = PassthroughPolicy()
        p.setup(cfg)
        seq_len = 4

        k = torch.zeros(1, 1, seq_len, 4)
        v = torch.eye(seq_len).unsqueeze(0).unsqueeze(0)
        for i in range(seq_len):
            k[0, 0, i, i] = 10.0

        p.write(k, v, layer_idx=0, positions=torch.arange(seq_len))
        q = k.clone()
        out = p.attend(q, layer_idx=0)
        # V is identity → output ≈ attention weights
        weights = out.output[0, 0]   # [seq_len, seq_len]

        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights[i, j].item() < 1e-4, (
                    f"Position {i} attended to future position {j}: {weights[i, j].item()}"
                )

    def test_decode_no_causal_mask(self):
        """During decode (q_len=1), new token should see the full cache.

        V is identity (head_dim == seq_len == 5) so output == weights;
        all elements should be positive (all positions visible).
        """
        cfg = small_config(num_heads=1, num_kv_heads=1, head_dim=5)
        p = PassthroughPolicy()
        p.setup(cfg)
        seq_len = 5
        k = torch.randn(1, 1, seq_len, 5)
        v = torch.eye(seq_len).unsqueeze(0).unsqueeze(0)
        p.write(k, v, layer_idx=0, positions=torch.arange(seq_len))

        q = torch.randn(1, 1, 1, 5)
        out = p.attend(q, layer_idx=0)
        weights = out.output[0, 0, 0]  # [seq_len]
        assert (weights > 1e-6).all(), "Decode token cannot see all cache positions"


# ---------------------------------------------------------------------------
# PassthroughPolicy.memory_bytes
# ---------------------------------------------------------------------------

class TestMemoryBytes:
    def test_empty_before_write(self):
        p = PassthroughPolicy()
        p.setup(small_config())
        assert p.memory_bytes() == 0

    def test_correct_after_write(self):
        p = PassthroughPolicy()
        p.setup(small_config())
        k, v = rand_kv(batch=1, num_kv_heads=2, seq_len=10, head_dim=8)
        p.write(k, v, layer_idx=0, positions=torch.arange(10))

        expected = k.nelement() * k.element_size() + v.nelement() * v.element_size()
        assert p.memory_bytes() == expected

    def test_grows_with_decode_writes(self):
        p = PassthroughPolicy()
        p.setup(small_config())
        k, v = rand_kv(seq_len=5)
        p.write(k, v, layer_idx=0, positions=torch.arange(5))
        before = p.memory_bytes()

        k2, v2 = rand_kv(seq_len=1)
        p.write(k2, v2, layer_idx=0, positions=torch.tensor([5]))
        after = p.memory_bytes()

        assert after > before


# ---------------------------------------------------------------------------
# setup resets state
# ---------------------------------------------------------------------------

class TestSetupReset:
    def test_setup_clears_cache(self):
        p = PassthroughPolicy()
        cfg = small_config()
        p.setup(cfg)
        k, v = rand_kv(seq_len=5)
        p.write(k, v, layer_idx=0, positions=torch.arange(5))
        assert p.memory_bytes() > 0

        # Reset for a new sequence
        p.setup(cfg)
        assert p.memory_bytes() == 0
        assert all(c is None for c in p.k_cache)


# ---------------------------------------------------------------------------
# GQA: 7-to-1 repeat ratio (matches Qwen2.5-7B's 28Q / 4KV topology)
# ---------------------------------------------------------------------------

class TestGQARepeat:
    """PassthroughPolicy with num_kv_groups=7 (as in 28Q/4KV Qwen2.5-7B)."""

    def _make_policy(self):
        cfg = CacheConfig(
            num_layers=1, num_heads=28, num_kv_heads=4,
            head_dim=8, max_seq_len=64, dtype=torch.float32,
        )
        p = PassthroughPolicy()
        p.setup(cfg)
        return p

    def test_repeat_kv_7x(self):
        x = torch.randn(1, 4, 5, 8)
        out = _repeat_kv(x, 7)
        assert out.shape == (1, 28, 5, 8)
        # Each KV head block of 7 Q heads should be identical
        for kv in range(4):
            for rep in range(7):
                assert torch.equal(out[:, kv * 7 + rep], x[:, kv])

    def test_attend_output_shape_28q_4kv(self):
        p = self._make_policy()
        k = torch.randn(1, 4, 6, 8)
        v = torch.randn(1, 4, 6, 8)
        p.write(k, v, layer_idx=0, positions=torch.arange(6))
        q = torch.randn(1, 28, 6, 8)
        out = p.attend(q, layer_idx=0)
        assert out.output.shape == (1, 28, 6, 8)

    def test_attend_decode_shape_28q_4kv(self):
        p = self._make_policy()
        k = torch.randn(1, 4, 10, 8)
        v = torch.randn(1, 4, 10, 8)
        p.write(k, v, layer_idx=0, positions=torch.arange(10))
        q = torch.randn(1, 28, 1, 8)
        out = p.attend(q, layer_idx=0)
        assert out.output.shape == (1, 28, 1, 8)
