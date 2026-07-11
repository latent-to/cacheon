"""Block-hash prompt seeding — determinism (consensus) + rotation (anti-prebake)."""

from __future__ import annotations

from optima.eval.prompts import (
    derive_seed,
    sample_prompt_batches,
    sample_prompts,
    sample_prompts_for_block,
)


def test_derive_seed_is_deterministic():
    bh = "0xabc123"
    assert derive_seed(bh) == derive_seed(bh)
    assert isinstance(derive_seed(bh), int)


def test_same_block_yields_same_prompts():
    # cross-validator consensus: identical prompts at the same block
    bh = "0xdeadbeef"
    assert sample_prompts_for_block(bh, 8) == sample_prompts_for_block(bh, 8)


def test_different_block_yields_different_prompts():
    # anti-prebake: the set rotates per block
    assert derive_seed("0xaaaa") != derive_seed("0xbbbb")
    assert sample_prompts_for_block("0xaaaa", 8) != sample_prompts_for_block("0xbbbb", 8)


def test_version_reshuffles_at_same_block():
    bh = "0xsameblock"
    assert derive_seed(bh, version=1) != derive_seed(bh, version=2)
    assert (sample_prompts_for_block(bh, 8, version=1)
            != sample_prompts_for_block(bh, 8, version=2))


def test_for_block_matches_manual_seed():
    bh = "0x99"
    assert sample_prompts_for_block(bh, 5) == sample_prompts(5, derive_seed(bh))


def test_short_prompts_remain_unique_above_corpus_size():
    prompts = sample_prompts(256, 9)
    assert len(prompts) == len(set(prompts)) == 256
    assert all(prompt.startswith("[case ") for prompt in prompts)


def test_warmup_and_timed_batches_never_repeat_concrete_prompts():
    batches = sample_prompt_batches(6, 64, 123)
    flattened = [prompt for batch in batches for prompt in batch]
    assert len(flattened) == len(set(flattened))
    assert batches == sample_prompt_batches(6, 64, 123)
    assert batches != sample_prompt_batches(6, 64, 124)


def test_long_batches_are_prefix_disjoint_across_iterations():
    batches = sample_prompt_batches(4, 3, 77, input_len=256)
    headers = [prompt.split("]", 1)[0] for batch in batches for prompt in batch]
    assert len(headers) == len(set(headers))
