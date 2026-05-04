"""Unit tests for validator.baseline -- cache I/O, no Docker."""

from __future__ import annotations

import json

import pytest

from validator.baseline import (
    BaselineCache,
    BaselinePromptResult,
    derive_cache_key,
    load_cached_baseline,
    save_baseline_cache,
)

pytestmark = pytest.mark.unit


def _sample_result(**overrides) -> BaselinePromptResult:
    defaults = dict(
        tokens=["Hello", " world"],
        top_logprobs=[
            [{"token": "Hello", "logprob": -0.01}],
            [{"token": " world", "logprob": -0.02}],
        ],
        ttft_s=0.045,
        throughput_tps=120.5,
        output_tokens=2,
    )
    defaults.update(overrides)
    return BaselinePromptResult(**defaults)


# --------------------------------------------------------------------------- #
# BaselinePromptResult
# --------------------------------------------------------------------------- #


class TestBaselinePromptResult:
    def test_round_trip(self):
        r = _sample_result()
        restored = BaselinePromptResult.from_dict(r.to_dict())
        assert restored.tokens == r.tokens
        assert restored.ttft_s == r.ttft_s
        assert restored.throughput_tps == r.throughput_tps
        assert restored.output_tokens == r.output_tokens


# --------------------------------------------------------------------------- #
# BaselineCache
# --------------------------------------------------------------------------- #


class TestBaselineCache:
    def test_round_trip(self):
        cache = BaselineCache(
            cache_key="abc123",
            results=[_sample_result(), _sample_result(ttft_s=0.05)],
        )
        restored = BaselineCache.from_dict(cache.to_dict())
        assert restored.cache_key == cache.cache_key
        assert len(restored.results) == 2
        assert restored.results[1].ttft_s == 0.05


# --------------------------------------------------------------------------- #
# derive_cache_key
# --------------------------------------------------------------------------- #


class TestDeriveCacheKey:
    def test_deterministic(self):
        k1 = derive_cache_key("0xabc123")
        k2 = derive_cache_key("0xabc123")
        assert k1 == k2

    def test_different_hashes_differ(self):
        k1 = derive_cache_key("0xabc123")
        k2 = derive_cache_key("0xdef456")
        assert k1 != k2

    def test_different_baseline_digest_differs(self):
        k1 = derive_cache_key("0xabc123", "sha256:" + "a" * 64)
        k2 = derive_cache_key("0xabc123", "sha256:" + "b" * 64)
        assert k1 != k2

    def test_same_hash_and_digest_deterministic(self):
        d = "sha256:" + "c" * 64
        assert derive_cache_key("0xabc", d) == derive_cache_key("0xabc", d)

    def test_length_is_16(self):
        assert len(derive_cache_key("anything")) == 16

    def test_hex_chars_only(self):
        key = derive_cache_key("test")
        assert all(c in "0123456789abcdef" for c in key)


# --------------------------------------------------------------------------- #
# Disk cache
# --------------------------------------------------------------------------- #


class TestDiskCache:
    def test_save_and_load(self, tmp_path):
        cache = BaselineCache(
            cache_key="key1",
            results=[_sample_result(), _sample_result(throughput_tps=200.0)],
        )
        save_baseline_cache(tmp_path, "key1", cache)
        loaded = load_cached_baseline(tmp_path, "key1")
        assert loaded is not None
        assert loaded.cache_key == "key1"
        assert len(loaded.results) == 2
        assert loaded.results[1].throughput_tps == 200.0

    def test_missing_cache_returns_none(self, tmp_path):
        assert load_cached_baseline(tmp_path, "nonexistent") is None

    def test_corrupt_json_returns_none(self, tmp_path):
        path = tmp_path / "baseline_bad.json"
        path.write_text("{not valid json!!")
        assert load_cached_baseline(tmp_path, "bad") is None

    def test_empty_file_returns_none(self, tmp_path):
        path = tmp_path / "baseline_empty.json"
        path.write_text("")
        assert load_cached_baseline(tmp_path, "empty") is None

    def test_wrong_schema_returns_none(self, tmp_path):
        path = tmp_path / "baseline_wrong.json"
        path.write_text(json.dumps({"foo": "bar"}))
        assert load_cached_baseline(tmp_path, "wrong") is None

    def test_no_temp_files_left(self, tmp_path):
        cache = BaselineCache(cache_key="k", results=[_sample_result()])
        save_baseline_cache(tmp_path, "k", cache)
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "baseline_k.json"

    def test_creates_cache_dir(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        cache = BaselineCache(cache_key="k", results=[_sample_result()])
        save_baseline_cache(nested, "k", cache)
        assert (nested / "baseline_k.json").exists()
