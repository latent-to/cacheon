"""Tests for inference_engine.prompts — deterministic PG19 sampling."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from inference_engine.prompts import (
    _seed_from_hash,
    _truncate_at_boundary,
    sample_prompts,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fake dataset — avoids downloading PG19 in CI
# ---------------------------------------------------------------------------

_LONG_TEXT = "A" * 200_000
_SHORT_TEXT = "Hi"
_FAKE_PG19 = (
    [{"text": f"Book {i} passage text that is long enough to pass min_chars easily."} for i in range(95)]
    + [{"text": _SHORT_TEXT}]  # idx 95 — below min_chars
    + [{"text": ""}]           # idx 96 — empty
    + [{"text": _LONG_TEXT}]   # idx 97 — over max_chars
    + [{"text": f"Filler book {i} with enough characters."} for i in range(98, 100)]
)


class _FakeDataset:
    """Minimal stand-in for a HuggingFace Dataset."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


@pytest.fixture(autouse=True)
def _mock_load_dataset():
    with patch(
        "inference_engine.prompts.load_dataset",
        return_value=_FakeDataset(_FAKE_PG19),
    ):
        yield


# ---------------------------------------------------------------------------
# _seed_from_hash
# ---------------------------------------------------------------------------


class TestSeedFromHash:
    def test_deterministic(self):
        assert _seed_from_hash("aabbcc") == _seed_from_hash("aabbcc")

    def test_different_hashes_produce_different_seeds(self):
        assert _seed_from_hash("aabbcc") != _seed_from_hash("ddeeff")

    def test_returns_int(self):
        assert isinstance(_seed_from_hash("abc123"), int)


# ---------------------------------------------------------------------------
# _truncate_at_boundary
# ---------------------------------------------------------------------------


class TestTruncateAtBoundary:
    def test_short_text_unchanged(self):
        assert _truncate_at_boundary("Hello world.", 1000) == "Hello world."

    def test_cuts_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _truncate_at_boundary(text, 35)
        assert result.endswith(".")
        assert len(result) <= 35

    def test_cuts_at_word_boundary_when_no_sentence(self):
        text = "word " * 100
        result = _truncate_at_boundary(text, 30)
        assert len(result) <= 30
        assert not result.endswith(" ")

    def test_hard_cut_at_max_chars(self):
        text = "x" * 500
        result = _truncate_at_boundary(text, 100)
        assert len(result) == 100


# ---------------------------------------------------------------------------
# sample_prompts
# ---------------------------------------------------------------------------


class TestSamplePrompts:
    def test_deterministic_same_hash(self):
        a = sample_prompts("block_hash_1", n=5, min_chars=10)
        b = sample_prompts("block_hash_1", n=5, min_chars=10)
        assert a == b

    def test_different_hash_different_prompts(self):
        a = sample_prompts("block_hash_1", n=5, min_chars=10)
        b = sample_prompts("block_hash_2", n=5, min_chars=10)
        assert a != b

    def test_returns_requested_count(self):
        assert len(sample_prompts("hash_a", n=3, min_chars=10)) == 3
        assert len(sample_prompts("hash_b", n=7, min_chars=10)) == 7

    def test_default_n_is_10(self):
        result = sample_prompts("hash_c", min_chars=10)
        assert len(result) == 10

    def test_all_strings_non_empty(self):
        for text in sample_prompts("hash_d", n=10, min_chars=10):
            assert isinstance(text, str)
            assert len(text) > 0

    def test_no_duplicates_within_sample(self):
        result = sample_prompts("hash_e", n=20, min_chars=10)
        assert len(result) == len(set(result))

    def test_skips_short_passages(self):
        results = sample_prompts("hash_f", n=90, min_chars=10)
        assert _SHORT_TEXT not in results
        assert "" not in results

    def test_truncates_long_passages(self):
        results = sample_prompts("hash_g", n=90, max_chars=500, min_chars=10)
        for text in results:
            assert len(text) <= 500

    def test_min_chars_respected(self):
        results = sample_prompts("hash_h", n=90, min_chars=100)
        for text in results:
            assert len(text) >= 100
