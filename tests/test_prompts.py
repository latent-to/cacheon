"""Unit tests for validator.prompts -- no network, no HuggingFace download."""

from __future__ import annotations

import random

import pytest

from validator.prompts import (
    CHARS_PER_TOKEN,
    MAX_CONTEXT_CHARS,
    MAX_OUTPUT_TOKENS,
    MAX_SAMPLE_ATTEMPTS,
    MIN_ALPHA_RATIO,
    MIN_CONTEXT_CHARS,
    OVERHEAD_TOKENS,
    PROMPT_ENGINE_VERSION,
    TEMPLATES,
    _is_valid_passage,
    _sample_passage,
    derive_seed,
    max_passage_chars,
    sample_prompts,
)

pytestmark = pytest.mark.unit


def _make_rows(n: int = 5, length: int = 20_000) -> list[str]:
    """Produce fake PG19-like rows of English-ish prose."""
    rng = random.Random(42)
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumped",
        "over",
        "lazy",
        "dog",
        "and",
        "then",
        "it",
        "ran",
        "across",
        "fields",
        "of",
        "green",
        "while",
        "sun",
        "shone",
        "brightly",
        "above",
        "distant",
        "hills",
    ]
    rows: list[str] = []
    for _ in range(n):
        text_words = [rng.choice(words) for _ in range(length // 5)]
        rows.append(" ".join(text_words))
    return rows


FAKE_ROWS = _make_rows()


# --------------------------------------------------------------------------- #
# derive_seed
# --------------------------------------------------------------------------- #


class TestDeriveSeed:
    def test_deterministic(self):
        assert derive_seed("0xabc") == derive_seed("0xabc")

    def test_different_hashes(self):
        assert derive_seed("0xabc") != derive_seed("0xdef")

    def test_returns_int(self):
        assert isinstance(derive_seed("0xabc"), int)


# --------------------------------------------------------------------------- #
# _is_valid_passage
# --------------------------------------------------------------------------- #


class TestIsValidPassage:
    def test_valid_english(self):
        text = "a" * MIN_CONTEXT_CHARS
        assert _is_valid_passage(text) is True

    def test_too_short(self):
        assert _is_valid_passage("hello") is False

    def test_too_much_whitespace(self):
        text = " " * (MIN_CONTEXT_CHARS + 1)
        assert _is_valid_passage(text) is False

    def test_low_alpha_ratio(self):
        text = "1234567890!@#$%^&*()" * 1000
        assert _is_valid_passage(text) is False

    def test_empty(self):
        assert _is_valid_passage("") is False


# --------------------------------------------------------------------------- #
# _sample_passage
# --------------------------------------------------------------------------- #


class TestSamplePassage:
    def test_returns_valid_passage(self):
        rng = random.Random(123)
        passage = _sample_passage(rng, FAKE_ROWS)
        assert len(passage) >= MIN_CONTEXT_CHARS
        assert len(passage) <= MAX_CONTEXT_CHARS
        assert _is_valid_passage(passage)

    def test_respects_max_chars(self):
        cap = 18_000
        rng = random.Random(123)
        passage = _sample_passage(rng, FAKE_ROWS, max_chars=cap)
        assert len(passage) <= cap

    def test_raises_on_all_junk(self):
        junk_rows = ["x" * 10 for _ in range(5)]
        rng = random.Random(0)
        with pytest.raises(RuntimeError, match="Could not sample"):
            _sample_passage(rng, junk_rows)

    def test_deterministic(self):
        p1 = _sample_passage(random.Random(99), FAKE_ROWS)
        p2 = _sample_passage(random.Random(99), FAKE_ROWS)
        assert p1 == p2


# --------------------------------------------------------------------------- #
# sample_prompts
# --------------------------------------------------------------------------- #


class TestSamplePrompts:
    def test_returns_correct_count(self):
        prompts = sample_prompts("0xblock1", n=5, _rows=FAKE_ROWS)
        assert len(prompts) == 5

    def test_deterministic(self):
        p1 = sample_prompts("0xblock1", n=3, _rows=FAKE_ROWS)
        p2 = sample_prompts("0xblock1", n=3, _rows=FAKE_ROWS)
        for a, b in zip(p1, p2):
            assert a.messages[0].content == b.messages[0].content

    def test_different_block_hash_different_prompts(self):
        p1 = sample_prompts("0xblock1", n=3, _rows=FAKE_ROWS)
        p2 = sample_prompts("0xblock2", n=3, _rows=FAKE_ROWS)
        contents1 = [p.messages[0].content for p in p1]
        contents2 = [p.messages[0].content for p in p2]
        assert contents1 != contents2

    def test_prompt_has_chat_message(self):
        prompts = sample_prompts("0xtest", n=1, _rows=FAKE_ROWS)
        assert len(prompts[0].messages) == 1
        assert prompts[0].messages[0].role == "user"
        assert len(prompts[0].messages[0].content) > 0

    def test_max_tokens_default(self):
        prompts = sample_prompts("0xtest", n=1, _rows=FAKE_ROWS)
        assert prompts[0].max_tokens == 256

    def test_template_applied(self):
        prompts = sample_prompts("0xtest", n=10, _rows=FAKE_ROWS)
        for p in prompts:
            content = p.messages[0].content
            has_template = any(
                content.startswith(t.split("{context}")[0]) for t in TEMPLATES
            )
            assert has_template, f"No template prefix found in: {content[:80]}..."


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


class TestMaxPassageChars:
    def test_none_returns_fallback(self):
        assert max_passage_chars(None) == MAX_CONTEXT_CHARS

    def test_32k_context(self):
        result = max_passage_chars(32_768)
        expected = int((32_768 - MAX_OUTPUT_TOKENS - OVERHEAD_TOKENS) * CHARS_PER_TOKEN)
        assert result == expected
        assert result < MAX_CONTEXT_CHARS

    def test_65k_context(self):
        result = max_passage_chars(65_536)
        expected = int((65_536 - MAX_OUTPUT_TOKENS - OVERHEAD_TOKENS) * CHARS_PER_TOKEN)
        assert result == expected

    def test_never_below_min(self):
        result = max_passage_chars(600)
        assert result >= MIN_CONTEXT_CHARS


class TestConstants:
    def test_prompt_engine_version_positive(self):
        assert PROMPT_ENGINE_VERSION >= 1

    def test_min_context_chars(self):
        assert MIN_CONTEXT_CHARS == 16_000

    def test_max_context_chars(self):
        assert MAX_CONTEXT_CHARS == 131_072

    def test_max_sample_attempts(self):
        assert MAX_SAMPLE_ATTEMPTS == 1_000

    def test_min_alpha_ratio(self):
        assert MIN_ALPHA_RATIO == 0.5

    def test_template_count(self):
        assert len(TEMPLATES) >= 15

    def test_all_templates_have_placeholder(self):
        for t in TEMPLATES:
            assert "{context}" in t, f"Template missing {{context}}: {t[:50]}"
