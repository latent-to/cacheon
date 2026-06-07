"""Block-hash prompt seeding — determinism (consensus) + rotation (anti-prebake)."""

from __future__ import annotations

from optima.eval.prompts import (
    derive_seed,
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
