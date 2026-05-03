"""Integration tests for the Harness with Qwen2.5-0.5B-Instruct.

Requires the 0.5B model (~1 GB download, cached after first run).
Validates monkey-patch mechanics and passthrough correctness end-to-end.

Run with:
    pytest tests/test_harness_integration.py -v -m integration
"""

import pytest
import torch

SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "What is 2 + 2?",
    "Name the capital of Japan.",
    "What color is the sky?",
]

MAX_NEW_TOKENS = 8   # short enough for CPU CI, long enough to catch divergence


@pytest.fixture(scope="module")
def harness():
    from inference_engine.harness import Harness
    if torch.cuda.is_available():
        return Harness(model_name=SMOKE_MODEL, device="cuda", dtype=torch.float16)
    return Harness(model_name=SMOKE_MODEL, device="cpu", dtype=torch.float32)


@pytest.mark.integration
def test_monkey_patch_calls_all_layers(harness):
    """write() and attend() must be called once per layer per forward step."""
    from inference_engine.passthrough import PassthroughPolicy

    call_log = []

    class TracePolicy(PassthroughPolicy):
        def write(self, keys, values, layer_idx, positions):
            call_log.append(("write", layer_idx))
            super().write(keys, values, layer_idx, positions)

        def attend(self, query, layer_idx, **kwargs):
            call_log.append(("attend", layer_idx))
            return super().attend(query, layer_idx, **kwargs)

    policy = TracePolicy()
    policy.setup(harness._cache_config)
    harness.run(policy, [PROMPTS[0]], max_new_tokens=3)

    num_layers = harness._cache_config.num_layers
    writes = [e for e in call_log if e[0] == "write"]
    attends = [e for e in call_log if e[0] == "attend"]

    assert len(writes) >= num_layers, (
        f"write() called {len(writes)} times, expected >= {num_layers} (one per layer per step)"
    )
    assert len(attends) >= num_layers, (
        f"attend() called {len(attends)} times, expected >= {num_layers}"
    )
    # Every layer index must appear at least once
    write_layers = {e[1] for e in writes}
    assert write_layers == set(range(num_layers)), (
        f"Not all layers were written: missing {set(range(num_layers)) - write_layers}"
    )


@pytest.mark.integration
def test_passthrough_matches_reference(harness):
    """PassthroughPolicy must produce identical tokens to the unpatched model."""
    ok = harness.verify(PROMPTS, max_new_tokens=MAX_NEW_TOKENS)
    assert ok, "Passthrough output diverged from unpatched reference"


@pytest.mark.integration
def test_run_result_shapes(harness):
    """RunResult fields have the expected shapes and types."""
    from inference_engine.passthrough import PassthroughPolicy

    result = harness.run(PassthroughPolicy(), PROMPTS, max_new_tokens=MAX_NEW_TOKENS)

    assert len(result.output_texts) == len(PROMPTS)
    assert len(result.output_ids) == len(PROMPTS)
    assert len(result.all_logits) == len(PROMPTS)

    vocab_size = harness.model.config.vocab_size
    for logits in result.all_logits:
        assert logits.ndim == 2                  # [num_generated, vocab_size]
        assert logits.shape[-1] == vocab_size

    assert result.latency_s > 0
    if harness.device.type == "cuda":
        assert result.policy_memory_bytes > 0
