"""Unit tests for the baseline-cache manifest in `scripts.pod_eval`.

The cached file is supposed to be a memo of "this baseline was produced
under exactly these settings". On load we recompute the manifest and
require an exact match — otherwise we silently recompute. These tests
exercise that contract without loading torch models or talking to a GPU.

Why these matter:
- Operators tune `n_prompts`, `max_new_tokens`, `dtype` between runs.
  Reusing a baseline produced under a different setting would mis-score
  challengers without raising.
- The manifest is the single source of truth — the on-disk filename
  alone (the cache key) is allowed to be coarse.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from scripts.pod_eval import (
    BASELINE_CACHE_VERSION,
    _build_baseline_manifest,
    _hash_prompts,
    _manifest_mismatch_reason,
    _save_baseline,
    _try_load_baseline,
)
from validator.eval_schema import SCHEMA_VERSION

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@dataclass
class _FakeBaseline:
    """Minimal stand-in for `RunResult` — enough fields for `_save_baseline`."""

    all_logits: list
    output_ids: list[list[int]]
    output_texts: list[str]
    latency_s: float
    peak_memory_bytes: int
    policy_memory_bytes: int


def _baseline_fixture(n: int = 2) -> _FakeBaseline:
    return _FakeBaseline(
        all_logits=[torch.zeros(4, 8) for _ in range(n)],
        output_ids=[[1, 2, 3, 4] for _ in range(n)],
        output_texts=[f"text {i}" for i in range(n)],
        latency_s=1.5,
        peak_memory_bytes=2**30,
        policy_memory_bytes=2**20,
    )


def _manifest(**overrides) -> dict:
    base = dict(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        n_prompts=10,
        max_new_tokens=256,
        dtype_name="float16",
        prompts=["alpha", "beta", "gamma"],
    )
    base.update(overrides)
    return _build_baseline_manifest(**base)


# --------------------------------------------------------------------------- #
# _hash_prompts
# --------------------------------------------------------------------------- #


class TestHashPrompts:
    def test_same_prompts_same_hash(self):
        assert _hash_prompts(["a", "b"]) == _hash_prompts(["a", "b"])

    def test_order_matters(self):
        assert _hash_prompts(["a", "b"]) != _hash_prompts(["b", "a"])

    def test_distinguishes_concat_collisions(self):
        # ['ab', 'c'] and ['a', 'bc'] would collide under naive joining;
        # we include length prefixes to avoid that.
        assert _hash_prompts(["ab", "c"]) != _hash_prompts(["a", "bc"])

    def test_handles_unicode(self):
        h = _hash_prompts(["héllo", "世界"])
        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex


# --------------------------------------------------------------------------- #
# _build_baseline_manifest
# --------------------------------------------------------------------------- #


class TestBuildManifest:
    def test_includes_required_fields(self):
        m = _manifest()
        assert m["cache_version"] == BASELINE_CACHE_VERSION
        assert m["schema_version"] == SCHEMA_VERSION
        assert m["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
        assert m["n_prompts"] == 10
        assert m["max_new_tokens"] == 256
        assert m["dtype_name"] == "float16"
        assert m["prompt_hash"] == _hash_prompts(["alpha", "beta", "gamma"])

    def test_prompt_hash_changes_when_prompts_change(self):
        a = _manifest(prompts=["x"])
        b = _manifest(prompts=["y"])
        assert a["prompt_hash"] != b["prompt_hash"]


# --------------------------------------------------------------------------- #
# _manifest_mismatch_reason — pure logic
# --------------------------------------------------------------------------- #


class TestManifestMismatchReason:
    def test_identical_manifests_match(self):
        m = _manifest()
        assert _manifest_mismatch_reason({"manifest": m}, m) is None

    @pytest.mark.parametrize(
        "field,override",
        [
            ("model_name", {"model_name": "other/model"}),
            ("n_prompts", {"n_prompts": 11}),
            ("max_new_tokens", {"max_new_tokens": 257}),
            ("dtype_name", {"dtype_name": "bfloat16"}),
            ("prompt_hash", {"prompts": ["different"]}),
        ],
    )
    def test_each_field_is_required_to_match(self, field, override):
        saved = _manifest()
        expected = _manifest(**override)
        reason = _manifest_mismatch_reason({"manifest": saved}, expected)
        assert reason is not None
        assert field in reason

    def test_payload_without_manifest_key_is_rejected(self):
        # Pre-manifest cache files (or any other torch.load output) must
        # be treated as unusable rather than reused.
        assert _manifest_mismatch_reason({"all_logits": []}, _manifest())

    def test_payload_with_non_dict_manifest_is_rejected(self):
        assert _manifest_mismatch_reason({"manifest": "nope"}, _manifest())

    def test_non_dict_payload_is_rejected(self):
        assert _manifest_mismatch_reason("garbage", _manifest())

    def test_cache_version_mismatch_blocks_reuse(self):
        saved = _manifest()
        bumped = dict(saved)
        bumped["cache_version"] = saved["cache_version"] + 1
        reason = _manifest_mismatch_reason({"manifest": saved}, bumped)
        assert reason is not None
        assert "cache_version" in reason


# --------------------------------------------------------------------------- #
# _save_baseline + _try_load_baseline round-trip (uses torch.save/load)
# --------------------------------------------------------------------------- #


class TestSaveLoadRoundTrip:
    def test_round_trip_with_matching_manifest_returns_payload(self, tmp_path):
        manifest = _manifest()
        _save_baseline(
            tmp_path, "key1", _baseline_fixture(), manifest=manifest,
        )
        loaded = _try_load_baseline(
            tmp_path, "key1", expected_manifest=manifest,
        )
        assert loaded is not None
        assert loaded["manifest"] == manifest
        assert loaded["latency_s"] == pytest.approx(1.5)
        assert loaded["output_texts"] == ["text 0", "text 1"]

    def test_missing_file_returns_none(self, tmp_path):
        assert _try_load_baseline(
            tmp_path, "never-saved", expected_manifest=_manifest(),
        ) is None

    def test_changing_max_new_tokens_blocks_reuse(self, tmp_path):
        saved_manifest = _manifest(max_new_tokens=256)
        _save_baseline(
            tmp_path, "key2", _baseline_fixture(), manifest=saved_manifest,
        )

        new_manifest = _manifest(max_new_tokens=512)
        assert _try_load_baseline(
            tmp_path, "key2", expected_manifest=new_manifest,
        ) is None

    def test_changing_n_prompts_blocks_reuse(self, tmp_path):
        saved_manifest = _manifest(n_prompts=10)
        _save_baseline(
            tmp_path, "key3", _baseline_fixture(), manifest=saved_manifest,
        )
        new_manifest = _manifest(n_prompts=8)
        assert _try_load_baseline(
            tmp_path, "key3", expected_manifest=new_manifest,
        ) is None

    def test_changing_dtype_blocks_reuse(self, tmp_path):
        saved = _manifest(dtype_name="float16")
        _save_baseline(
            tmp_path, "key4", _baseline_fixture(), manifest=saved,
        )
        new = _manifest(dtype_name="bfloat16")
        assert _try_load_baseline(
            tmp_path, "key4", expected_manifest=new,
        ) is None

    def test_changing_prompts_blocks_reuse(self, tmp_path):
        # Same block_hash → same cache key → without the prompt_hash check
        # this would be a false hit if `sample_prompts` changes shape.
        saved = _manifest(prompts=["a", "b", "c"])
        _save_baseline(
            tmp_path, "key5", _baseline_fixture(), manifest=saved,
        )
        new = _manifest(prompts=["a", "b", "d"])
        assert _try_load_baseline(
            tmp_path, "key5", expected_manifest=new,
        ) is None

    def test_legacy_payload_without_manifest_is_rejected(self, tmp_path):
        # Simulate an older cache file written before the manifest
        # existed: `_try_load_baseline` must treat it as a miss rather
        # than rehydrate it as a baseline.
        path = tmp_path / "baseline-key6.pt"
        torch.save(
            {
                "all_logits": [torch.zeros(2, 4)],
                "output_ids": [[1, 2]],
                "output_texts": ["hi"],
                "latency_s": 1.0,
                "peak_memory_bytes": 1,
                "policy_memory_bytes": 1,
            },
            path,
        )
        assert _try_load_baseline(
            tmp_path, "key6", expected_manifest=_manifest(),
        ) is None

    def test_unreadable_cache_file_returns_none(self, tmp_path):
        path = tmp_path / "baseline-key7.pt"
        path.write_bytes(b"not a torch file")
        assert _try_load_baseline(
            tmp_path, "key7", expected_manifest=_manifest(),
        ) is None
