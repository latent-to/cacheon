"""Unit tests for validator.eval_schema — pure JSON codec, no torch."""

from __future__ import annotations

import json

import pytest

from validator.eval_schema import (
    BaselineMetrics,
    ChallengerJob,
    ChallengerResult,
    EvaluationJob,
    EvaluationResult,
    JOB_FILE_NAME,
    RESULTS_FILE_NAME,
    SCHEMA_VERSION,
    read_job,
    read_results,
    write_job,
    write_results,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #


def _make_challenger_job(uid: int = 1, hotkey: str = "hk_alice") -> ChallengerJob:
    return ChallengerJob(
        uid=uid,
        hotkey=hotkey,
        commit_block=100 + uid,
        model=f"hf-user/policy-{uid}",
        revision=f"sha{uid}",
        policy_path=f"/tmp/policies/{hotkey}/policy.py",
    )


def _make_job(n_challengers: int = 2) -> EvaluationJob:
    return EvaluationJob(
        schema_version=SCHEMA_VERSION,
        job_id="block-1000-abcd1234",
        current_block=1000,
        block_hash="0xdeadbeef",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=256,
        n_prompts=10,
        baseline_cache_dir="/workspace/baseline-cache",
        baseline_cache_key="Qwen_Qwen2.5-7B-Instruct-deadbeef",
        challengers=[_make_challenger_job(i) for i in range(1, n_challengers + 1)],
    )


def _make_challenger_result(uid: int = 1) -> ChallengerResult:
    return ChallengerResult(
        uid=uid,
        hotkey=f"hk{uid}",
        commit_block=100 + uid,
        model=f"hf-user/policy-{uid}",
        revision=f"sha{uid}",
        score=0.25,
        kl_divergence=0.015,
        memory_reduction=0.3,
        latency_improvement=0.1,
        disqualified=False,
        disqualify_reason=None,
    )


def _make_result(n_challengers: int = 2) -> EvaluationResult:
    return EvaluationResult(
        schema_version=SCHEMA_VERSION,
        job_id="block-1000-abcd1234",
        current_block=1000,
        block_hash="0xdeadbeef",
        baseline=BaselineMetrics(
            latency_s=42.5,
            peak_memory_bytes=20 * 1024**3,
            cached=False,
        ),
        challenger_results=[
            _make_challenger_result(i) for i in range(1, n_challengers + 1)
        ],
    )


# --------------------------------------------------------------------------- #
# Job roundtrip
# --------------------------------------------------------------------------- #


class TestEvaluationJobRoundtrip:
    def test_dict_roundtrip_preserves_all_fields(self):
        job = _make_job(n_challengers=3)
        restored = EvaluationJob.from_dict(job.to_dict())
        assert restored == job

    def test_disk_roundtrip(self, tmp_path):
        job = _make_job(n_challengers=2)
        path = tmp_path / JOB_FILE_NAME
        write_job(job, path)
        assert read_job(path) == job

    def test_challengers_preserve_order(self):
        job = _make_job(n_challengers=5)
        restored = EvaluationJob.from_dict(job.to_dict())
        assert [c.uid for c in restored.challengers] == [1, 2, 3, 4, 5]

    def test_null_block_hash_preserved(self):
        job = _make_job()
        job_with_none = EvaluationJob(
            **{**job.__dict__, "block_hash": None}
        )
        restored = EvaluationJob.from_dict(job_with_none.to_dict())
        assert restored.block_hash is None

    def test_empty_challengers(self):
        job = _make_job(n_challengers=0)
        restored = EvaluationJob.from_dict(job.to_dict())
        assert restored.challengers == []

    def test_newer_schema_version_rejected(self):
        job = _make_job()
        data = job.to_dict()
        data["schema_version"] = SCHEMA_VERSION + 5
        with pytest.raises(ValueError, match="newer than"):
            EvaluationJob.from_dict(data)

    def test_json_is_sorted_for_stable_diffs(self, tmp_path):
        """CI should get stable diffs on regenerated job files."""
        job = _make_job()
        path = tmp_path / "job.json"
        write_job(job, path)
        text = path.read_text()
        reserialized = json.dumps(json.loads(text), indent=2, sort_keys=True)
        assert text == reserialized


# --------------------------------------------------------------------------- #
# Result roundtrip
# --------------------------------------------------------------------------- #


class TestEvaluationResultRoundtrip:
    def test_dict_roundtrip_preserves_all_fields(self):
        r = _make_result(n_challengers=3)
        restored = EvaluationResult.from_dict(r.to_dict())
        assert restored == r

    def test_disk_roundtrip(self, tmp_path):
        r = _make_result()
        path = tmp_path / RESULTS_FILE_NAME
        write_results(r, path)
        assert read_results(path) == r

    def test_dq_result_roundtrip(self):
        dq = ChallengerResult(
            uid=7,
            hotkey="hk7",
            commit_block=107,
            model="hf/bad",
            revision="sha7",
            score=0.0,
            kl_divergence=float("inf"),
            memory_reduction=0.0,
            latency_improvement=0.0,
            disqualified=True,
            disqualify_reason="KL divergence 0.9 exceeds threshold 0.1",
        )
        restored = ChallengerResult.from_dict(dq.to_dict())
        assert restored == dq
        # inf survives JSON roundtrip (json module allows it)
        serialized = json.dumps(dq.to_dict())
        assert "Infinity" in serialized

    def test_newer_schema_version_rejected(self):
        r = _make_result()
        data = r.to_dict()
        data["schema_version"] = SCHEMA_VERSION + 5
        with pytest.raises(ValueError, match="newer than"):
            EvaluationResult.from_dict(data)

    def test_cached_baseline_flag_survives_roundtrip(self):
        r = _make_result()
        cached = EvaluationResult(
            **{**r.__dict__, "baseline": BaselineMetrics(
                latency_s=r.baseline.latency_s,
                peak_memory_bytes=r.baseline.peak_memory_bytes,
                cached=True,
            )},
        )
        restored = EvaluationResult.from_dict(cached.to_dict())
        assert restored.baseline.cached is True
