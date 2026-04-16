"""Tests for inference_engine.prompts — deterministic PG19 sampling."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from inference_engine.prompts import _seed_from_hash, sample_prompts

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fake dataset — avoids downloading PG19 in CI
# ---------------------------------------------------------------------------

_FAKE_PG19 = [{"text": f"Book {i} passage text."} for i in range(100)]


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
# sample_prompts
# ---------------------------------------------------------------------------


class TestSamplePrompts:
    def test_deterministic_same_hash(self):
        a = sample_prompts("block_hash_1", n=5)
        b = sample_prompts("block_hash_1", n=5)
        assert a == b

    def test_different_hash_different_prompts(self):
        a = sample_prompts("block_hash_1", n=5)
        b = sample_prompts("block_hash_2", n=5)
        assert a != b

    def test_returns_requested_count(self):
        assert len(sample_prompts("hash_a", n=3)) == 3
        assert len(sample_prompts("hash_b", n=7)) == 7

    def test_default_n_is_10(self):
        result = sample_prompts("hash_c")
        assert len(result) == 10

    def test_all_strings_non_empty(self):
        for text in sample_prompts("hash_d", n=10):
            assert isinstance(text, str)
            assert len(text) > 0

    def test_no_duplicates_within_sample(self):
        result = sample_prompts("hash_e", n=20)
        assert len(result) == len(set(result))

    def test_n_clamped_to_dataset_size(self):
        result = sample_prompts("hash_f", n=999)
        assert len(result) == len(_FAKE_PG19)
