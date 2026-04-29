"""Architecture stub tests — validates Qwen2.5-7B GQA topology with random weights.

No model download required: we instantiate a randomly-initialised Qwen2 model
with the same attention topology as 7B (28 Q-heads / 4 KV-heads) but a tiny
hidden size so it runs fast on CPU.

This catches shape bugs that only surface at the real 7B head counts without
incurring a 14 GB download.

Run with:
    pytest tests/test_harness_arch.py -v -m arch
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from inference_engine.harness import _make_patched_forward
from inference_engine.passthrough import PassthroughPolicy
from inference_engine.policy import CacheConfig


# ---------------------------------------------------------------------------
# Shared constants — match Qwen2.5-7B attention topology exactly
# ---------------------------------------------------------------------------

NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 32  # tiny (real 7B uses 128) — just validates shapes
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM  # 896


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubAttn(nn.Module):
    """Minimal attention module with 7B-topology projections."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)


@pytest.fixture(scope="module")
def stub_policy():
    cfg = CacheConfig(
        num_layers=1,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=256,
        dtype=torch.float32,
    )
    p = PassthroughPolicy()
    p.setup(cfg)
    return p


# ---------------------------------------------------------------------------
# _make_patched_forward: shape tests with 7B GQA topology
# ---------------------------------------------------------------------------


@pytest.mark.arch
class TestPatchedForward7BTopology:
    def _run(self, seq_len, policy):
        attn = _StubAttn()
        fwd = _make_patched_forward(
            attn,
            policy,
            layer_idx=0,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
        )
        hidden = torch.randn(1, seq_len, HIDDEN_SIZE)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        # cos/sin shape that Qwen2RotaryEmbedding produces: [batch, seq_len, head_dim]
        cos = torch.randn(1, seq_len, HEAD_DIM)
        sin = torch.randn(1, seq_len, HEAD_DIM)
        return fwd(hidden, position_ids=position_ids, position_embeddings=(cos, sin))

    def test_prefill_output_shape(self, stub_policy):
        out, cache = self._run(seq_len=8, policy=stub_policy)
        assert out.shape == (1, 8, HIDDEN_SIZE)
        assert cache is None

    def test_decode_output_shape(self, stub_policy):
        # Re-setup policy so cache is fresh
        cfg = CacheConfig(
            num_layers=1,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            max_seq_len=256,
            dtype=torch.float32,
        )
        stub_policy.setup(cfg)

        # Prefill first
        self._run(seq_len=5, policy=stub_policy)
        # Then decode one token at a time
        attn = _StubAttn()
        fwd = _make_patched_forward(
            attn,
            stub_policy,
            layer_idx=0,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
        )
        hidden = torch.randn(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[5]])
        cos = torch.randn(1, 1, HEAD_DIM)
        sin = torch.randn(1, 1, HEAD_DIM)
        out, cache = fwd(
            hidden, position_ids=position_ids, position_embeddings=(cos, sin)
        )
        assert out.shape == (1, 1, HIDDEN_SIZE)
        assert cache is None

    def test_kv_cache_grows_after_decode(self, stub_policy):
        cfg = CacheConfig(
            num_layers=1,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            max_seq_len=256,
            dtype=torch.float32,
        )
        stub_policy.setup(cfg)
        self._run(seq_len=4, policy=stub_policy)
        after_prefill = stub_policy.k_cache[0].shape[2]

        attn = _StubAttn()
        fwd = _make_patched_forward(
            attn,
            stub_policy,
            layer_idx=0,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
        )
        hidden = torch.randn(1, 1, HIDDEN_SIZE)
        cos = torch.randn(1, 1, HEAD_DIM)
        sin = torch.randn(1, 1, HEAD_DIM)
        fwd(hidden, position_ids=torch.tensor([[4]]), position_embeddings=(cos, sin))

        assert stub_policy.k_cache[0].shape[2] == after_prefill + 1


# ---------------------------------------------------------------------------
# Harness patch / unpatch with a stub full model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stub_harness():
    """Harness initialised with a randomly-weighted Qwen2 model (7B topology, tiny dims)."""
    from transformers import Qwen2Config, AutoModelForCausalLM
    from inference_engine.harness import Harness

    cfg = Qwen2Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=2048,
        num_hidden_layers=2,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=256,
        vocab_size=512,
    )
    model = AutoModelForCausalLM.from_config(cfg).eval()

    # Tokenizer stub: returns the same 5-token prompt for any input
    _fixed_ids = torch.randint(0, 512, (1, 5))
    tok = MagicMock()
    tok.return_value = {"input_ids": _fixed_ids.clone()}
    tok.__call__ = lambda self, *a, **kw: {"input_ids": _fixed_ids.clone()}
    tok.eos_token_id = 2

    with (
        patch(
            "inference_engine.harness.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ),
        patch(
            "inference_engine.harness.AutoTokenizer.from_pretrained", return_value=tok
        ),
    ):
        h = Harness(model_name="stub-7b-arch", device="cpu", dtype=torch.float32)

    return h


@pytest.mark.arch
class TestHarnessPatchUnpatch:
    def test_patch_replaces_all_forwards(self, stub_harness):
        originals = [
            layer.self_attn.forward for layer in stub_harness.model.model.layers
        ]
        policy = PassthroughPolicy()
        stub_harness._patch_attention(policy)
        try:
            for i, layer in enumerate(stub_harness.model.model.layers):
                assert layer.self_attn.forward is not originals[i], (
                    f"Layer {i} forward was not replaced"
                )
        finally:
            stub_harness._unpatch_attention()

    def test_unpatch_restores_all_forwards(self, stub_harness):
        originals = [
            layer.self_attn.forward for layer in stub_harness.model.model.layers
        ]
        policy = PassthroughPolicy()
        stub_harness._patch_attention(policy)
        stub_harness._unpatch_attention()
        for i, layer in enumerate(stub_harness.model.model.layers):
            assert layer.self_attn.forward is originals[i], (
                f"Layer {i} forward was not restored after unpatch"
            )

    def test_cache_config_matches_7b_topology(self, stub_harness):
        cfg = stub_harness._cache_config
        assert cfg.num_heads == NUM_HEADS
        assert cfg.num_kv_heads == NUM_KV_HEADS
        assert cfg.head_dim == HEAD_DIM


# ---------------------------------------------------------------------------
# score_policy_on_sequence (teacher-forced logits)
# ---------------------------------------------------------------------------


@pytest.mark.arch
class TestScorePolicyOnSequence:
    def test_returns_one_tensor_per_prompt(self, stub_harness):
        policy = PassthroughPolicy()
        ref_ids = [[10, 20, 30], [40, 50]]
        result = stub_harness.score_policy_on_sequence(
            policy,
            ["hello", "world"],
            ref_ids,
        )
        assert len(result) == 2

    def test_tensor_shape_matches_gen_len(self, stub_harness):
        policy = PassthroughPolicy()
        gen_ids = [42, 99, 7, 3]
        result = stub_harness.score_policy_on_sequence(
            policy,
            ["test prompt"],
            [gen_ids],
        )
        assert len(result) == 1
        assert result[0].shape[0] == len(gen_ids)
        assert result[0].shape[1] == stub_harness.model.config.vocab_size

    def test_deterministic_on_same_input(self, stub_harness):
        """Two calls with the same tokens should produce identical logits
        (no autoregressive drift on CPU)."""
        gen_ids = [10, 20, 30]
        r1 = stub_harness.score_policy_on_sequence(
            PassthroughPolicy(),
            ["x"],
            [gen_ids],
        )
        r2 = stub_harness.score_policy_on_sequence(
            PassthroughPolicy(),
            ["x"],
            [gen_ids],
        )
        assert torch.allclose(r1[0], r2[0], atol=1e-5)
