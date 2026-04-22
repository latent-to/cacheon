"""Unit tests for validator.eval_local — uses a fake `job_runner` so we
exercise the schema plumbing without a subprocess, torch, or GPU.

`pod_eval.py` itself is tested under `integration` (requires Qwen); here
we test that the CPU side:
  - materializes a valid EvaluationJob on disk,
  - hands it to the runner,
  - maps the runner's EvaluationResult back into EvaluationRecord(s),
  - handles partial / empty results gracefully.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from validator.chain import CommitmentRecord
from validator.eval_local import _baseline_cache_key, make_local_eval_fn
from validator.eval_schema import (
    BaselineMetrics,
    ChallengerResult,
    EvaluationResult,
    SCHEMA_VERSION,
    read_job,
    write_results,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_commitment(uid: int = 1, commit_block: int = 100) -> CommitmentRecord:
    return CommitmentRecord(
        uid=uid,
        hotkey=f"hk{uid}",
        commit_block=commit_block,
        repo=f"hf/policy-{uid}",
        revision=f"{uid:040x}",
        raw="{}",
    )


def _touch_policy(tmp_path: Path, com: CommitmentRecord) -> Path:
    path = tmp_path / "policies" / com.hotkey / "policy.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# fake policy for testing\n")
    return path


def _fake_runner_factory(result_fn):
    """Build a JobRunner that reads the job, calls `result_fn(job)` to
    produce an EvaluationResult, and writes it to the expected path."""

    def _runner(ctx):
        job = read_job(ctx.job_path)
        result = result_fn(job)
        write_results(result, ctx.results_path)

    return _runner


def _passthrough_result(job) -> EvaluationResult:
    """Return one ChallengerResult per challenger with a nontrivial score —
    simulates a happy-path pod run."""
    return EvaluationResult(
        schema_version=SCHEMA_VERSION,
        job_id=job.job_id,
        current_block=job.current_block,
        block_hash=job.block_hash,
        baseline=BaselineMetrics(latency_s=10.0, peak_memory_bytes=2**30, cached=False),
        challenger_results=[
            ChallengerResult(
                uid=c.uid,
                hotkey=c.hotkey,
                commit_block=c.commit_block,
                repo=c.repo,
                revision=c.revision,
                score=0.1 * c.uid,
                kl_divergence=0.02,
                memory_reduction=0.2,
                latency_improvement=0.05,
                disqualified=False,
                disqualify_reason=None,
                source_hash=c.source_hash,
            )
            for c in job.challengers
        ],
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestEmptyChallengers:
    def test_no_challengers_returns_empty(self, tmp_path):
        ran = []

        def runner(ctx):
            ran.append(ctx)

        eval_fn = make_local_eval_fn(
            policy_source_fn=lambda com: tmp_path / "unused.py",
            work_dir=tmp_path / "work",
            baseline_cache_dir=tmp_path / "baseline",
            job_runner=runner,
        )
        result = eval_fn([], current_block=500, block_hash="0x1")
        assert result == []
        assert ran == []  # never launched


class TestJobConstruction:
    def test_job_is_written_with_expected_fields(self, tmp_path):
        captured = {}

        def runner(ctx):
            captured["job"] = read_job(ctx.job_path)
            write_results(_passthrough_result(captured["job"]), ctx.results_path)

        coms = [_make_commitment(1, 100), _make_commitment(2, 200)]
        for c in coms:
            _touch_policy(tmp_path, c)

        eval_fn = make_local_eval_fn(
            policy_source_fn=lambda com: _touch_policy(tmp_path, com),
            work_dir=tmp_path / "work",
            baseline_cache_dir=tmp_path / "baseline",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            max_new_tokens=128,
            n_prompts=7,
            job_runner=runner,
        )
        eval_fn(coms, current_block=1234, block_hash="0xabc")

        job = captured["job"]
        assert job.schema_version == SCHEMA_VERSION
        assert job.current_block == 1234
        assert job.block_hash == "0xabc"
        assert job.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert job.max_new_tokens == 128
        assert job.n_prompts == 7
        assert [c.uid for c in job.challengers] == [1, 2]
        assert all(Path(c.policy_path).exists() for c in job.challengers)
        assert "Qwen_Qwen2.5-7B-Instruct" in job.baseline_cache_key

    def test_baseline_cache_key_depends_on_block_hash(self, tmp_path):
        seen_keys = []

        def runner(ctx):
            job = read_job(ctx.job_path)
            seen_keys.append(job.baseline_cache_key)
            write_results(_passthrough_result(job), ctx.results_path)

        com = _make_commitment(1)
        _touch_policy(tmp_path, com)

        eval_fn = make_local_eval_fn(
            policy_source_fn=lambda c: _touch_policy(tmp_path, c),
            work_dir=tmp_path / "work",
            baseline_cache_dir=tmp_path / "baseline",
            job_runner=runner,
        )
        eval_fn([com], current_block=1, block_hash="0xAAA")
        eval_fn([com], current_block=2, block_hash="0xBBB")

        assert len(seen_keys) == 2
        assert seen_keys[0] != seen_keys[1]

    def test_baseline_cache_key_preserves_leading_hex_zeros(self):
        """Regression: `.lstrip('0x')` treats the prefix as a character
        set, so "0x000abc" and "0xabc" would collapse onto the same key,
        reusing a stale baseline against different prompts."""
        model = "Qwen/Qwen2.5-7B-Instruct"
        k_leading_zeros = _baseline_cache_key(model, "0x000abc")
        k_no_zeros = _baseline_cache_key(model, "0xabc")
        k_single_zero = _baseline_cache_key(model, "0x0abc")
        assert k_leading_zeros != k_no_zeros
        assert k_leading_zeros != k_single_zero
        assert k_single_zero != k_no_zeros

    def test_baseline_cache_key_all_zeros_not_collapsed_to_nohash(self):
        """All-zero hash is a valid (if cosmically unlucky) block hash —
        don't collapse it onto the block_hash=None sentinel."""
        model = "Qwen/Qwen2.5-7B-Instruct"
        k_zeros = _baseline_cache_key(model, "0x00000000000000000000")
        k_none = _baseline_cache_key(model, None)
        assert k_zeros != k_none

    def test_baseline_cache_key_handles_missing_0x_prefix(self):
        model = "Qwen/Qwen2.5-7B-Instruct"
        with_prefix = _baseline_cache_key(model, "0xdeadbeef")
        without_prefix = _baseline_cache_key(model, "deadbeef")
        assert with_prefix == without_prefix

    def test_missing_policy_path_raises(self, tmp_path):
        def runner(ctx):
            pytest.fail("runner should not be called if policy.py is missing")

        eval_fn = make_local_eval_fn(
            policy_source_fn=lambda c: tmp_path / "nope.py",
            work_dir=tmp_path / "work",
            baseline_cache_dir=tmp_path / "baseline",
            job_runner=runner,
        )
        with pytest.raises(FileNotFoundError):
            eval_fn([_make_commitment(1)], current_block=1, block_hash="0x1")


class TestResultMapping:
    def test_maps_challenger_results_to_evaluation_records(self, tmp_path):
        coms = [_make_commitment(1, 100), _make_commitment(2, 200)]
        for c in coms:
            _touch_policy(tmp_path, c)

        eval_fn = make_local_eval_fn(
            policy_source_fn=lambda com: _touch_policy(tmp_path, com),
            work_dir=tmp_path / "work",
            baseline_cache_dir=tmp_path / "baseline",
            job_runner=_fake_runner_factory(_passthrough_result),
        )
        records = eval_fn(coms, current_block=555, block_hash="0xaaa")
        assert len(records) == 2
        assert sorted(r.uid for r in records) == [1, 2]
        for r, com in zip(records, coms):
            assert r.hotkey == com.hotkey
            assert r.commit_block == com.commit_block
            assert r.repo == com.repo
            assert r.revision == com.revision
            assert r.evaluation_block == 555
            assert r.disqualified is False
            assert r.score == pytest.approx(0.1 * com.uid)
            # source_hash propagated CPU → job → pod → record round-trip
            assert r.source_hash
            assert len(r.source_hash) == 64  # sha256 hex

    def test_dq_result_flows_through_as_disqualified(self, tmp_path):
        com = _make_commitment(1)
        _touch_policy(tmp_path, com)

        def dq_result(job) -> EvaluationResult:
            base = _passthrough_result(job)
            cr = base.challenger_results[0]
            dq = ChallengerResult(
                **{**cr.__dict__, "score": 0.0, "disqualified": True,
                   "disqualify_reason": "KL too high"},
            )
            return EvaluationResult(
                **{**base.__dict__, "challenger_results": [dq]},
            )

        eval_fn = make_local_eval_fn(
            policy_source_fn=lambda c: _touch_policy(tmp_path, c),
            work_dir=tmp_path / "work",
            baseline_cache_dir=tmp_path / "baseline",
            job_runner=_fake_runner_factory(dq_result),
        )
        records = eval_fn([com], current_block=1, block_hash="0x1")
        assert len(records) == 1
        assert records[0].disqualified is True
        assert records[0].disqualify_reason == "KL too high"
        assert records[0].score == 0.0

    def test_missing_challenger_in_results_is_warned_not_fatal(
        self, tmp_path, caplog,
    ):
        coms = [_make_commitment(1), _make_commitment(2)]
        for c in coms:
            _touch_policy(tmp_path, c)

        def partial(job) -> EvaluationResult:
            full = _passthrough_result(job)
            return EvaluationResult(
                **{**full.__dict__,
                   "challenger_results": full.challenger_results[:1]},
            )

        eval_fn = make_local_eval_fn(
            policy_source_fn=lambda c: _touch_policy(tmp_path, c),
            work_dir=tmp_path / "work",
            baseline_cache_dir=tmp_path / "baseline",
            job_runner=_fake_runner_factory(partial),
        )
        with caplog.at_level("WARNING"):
            records = eval_fn(coms, current_block=1, block_hash="0x1")

        assert len(records) == 1
        assert records[0].uid == 1
        assert any("missing challenger" in rec.message for rec in caplog.records)


class TestSubprocessDefaultRunner:
    """The default runner shells out — we don't exercise it end-to-end here
    (that's integration turf), but we verify surfaces that matter for debugging."""

    def test_runner_failure_propagates_as_runtime_error(self, tmp_path):
        com = _make_commitment(1)
        _touch_policy(tmp_path, com)

        def broken_runner(ctx):
            raise RuntimeError("pod_eval exited 137")

        eval_fn = make_local_eval_fn(
            policy_source_fn=lambda c: _touch_policy(tmp_path, c),
            work_dir=tmp_path / "work",
            baseline_cache_dir=tmp_path / "baseline",
            job_runner=broken_runner,
        )
        with pytest.raises(RuntimeError, match="137"):
            eval_fn([com], current_block=1, block_hash="0x1")

    def test_runner_may_raise_timeout_error(self, tmp_path):
        com = _make_commitment(1)
        _touch_policy(tmp_path, com)

        def slow_runner(ctx):
            raise TimeoutError("pod_eval exceeded 600s timeout")

        eval_fn = make_local_eval_fn(
            policy_source_fn=lambda c: _touch_policy(tmp_path, c),
            work_dir=tmp_path / "work",
            baseline_cache_dir=tmp_path / "baseline",
            job_runner=slow_runner,
        )
        with pytest.raises(TimeoutError):
            eval_fn([com], current_block=1, block_hash="0x1")
