"""Small shared constructors for settlement/report tests.

Production B300 arenas remain explicitly uncalibrated. Tests that exercise the
positive settlement path use a replaced, test-local provisional policy with the same
shape; they never mutate the consensus registry.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from optima.arenas import TeacherForcedQualityPolicy
from optima.eval.external_quality import (
    PromptClusterEvidence,
    RolloutQualitySummary,
    TeacherForcedBatchEvidence,
    TeacherForcedExternalQualityEvidence,
    score_teacher_forced_quality,
)
from optima.eval.scoring import score_speedup


def calibrated_test_arena(arena):
    policy = TeacherForcedQualityPolicy(
        protocol="controller-posthoc-teacher-forced-v2",
        calibration_state="provisional-rtx",
        clusters_per_batch=8,
        nll_clip=100.0,
        tail_nll_threshold=10.0,
        familywise_z=2.0,
        stock_mean_nll_envelope=0.1,
        stock_worst_nll_envelope=0.1,
        stock_tail_rate_envelope=0.05,
        stock_topk_kl_envelope=0.05,
        stock_argmax_rate_envelope=0.05,
        stock_coverage_envelope=0.05,
        mean_nll_delta=0.1,
        worst_nll_delta=0.1,
        tail_rate_delta=0.05,
        topk_kl_delta=0.05,
        argmax_rate_delta=0.05,
        coverage_delta=0.05,
        require_hidden_tasks=True,
        stock_hidden_score_envelope=0.05,
        hidden_score_delta=0.05,
        hidden_score_floor=0.8,
    )
    return replace(
        arena,
        fidelity=replace(arena.fidelity, teacher_forced_policy=policy),
    )


def patch_qualification_registry(monkeypatch, arena) -> None:
    lookup = lambda name: arena if name == arena.name else None
    monkeypatch.setattr("optima.eval.qualification.get_arena", lookup)
    monkeypatch.setattr("optima.chain.validator_loop.get_arena", lookup)


def teacher_evidence(arena, *, quality: bool = True):
    tokens = arena.workload.max_new_tokens

    def summary(nll):
        return RolloutQualitySummary(
            token_count=tokens,
            target_nll_sum=nll * tokens,
            target_nll_max=nll,
            target_tail_count=0,
            topk_positions=tokens,
            topk_mean_kl=0.001,
            topk_max_kl=0.001,
            topk_p99_kl=0.001,
            topk_argmax_disagreements=0,
            topk_mean_coverage_dev=0.001,
            hidden_score=1.0,
            hidden_total=1,
        )

    cluster = PromptClusterEvidence(
        baseline=summary(1.0),
        candidate=summary(1.0 if quality else 2.0),
        stock_control=summary(1.0),
        exact_token_matches=tokens if quality else 0,
        exact_token_total=tokens,
    )
    batch = TeacherForcedBatchEvidence((cluster,) * 8)
    return TeacherForcedExternalQualityEvidence(
        protocol="controller-posthoc-teacher-forced-v2",
        sealed_rollout_sha256="sha256:" + "4" * 64,
        raw_evidence_sha256="sha256:" + "5" * 64,
        raw_evidence_size=1,
        raw_artifact_published=True,
        hidden_tasks_present=True,
        timed_batches=(batch,) * arena.scoring.timed_iters,
        warmup_batches=(batch,) * arena.scoring.warmup_iters,
    )


def evaluation_report(
    arena,
    *,
    candidate_rate: float = 108.0,
    quality: bool = True,
    malformed_baseline: bool = False,
):
    count = arena.scoring.timed_iters
    baseline_rate = 100.0
    baseline_samples = [baseline_rate] * count
    if malformed_baseline:
        baseline_samples = baseline_samples[:1]
    baseline = SimpleNamespace(
        tok_per_s=baseline_rate,
        tok_per_s_samples=baseline_samples,
        conditioning_tok_per_s=baseline_rate,
        spread=(baseline_rate, baseline_rate, 0.0),
    )
    candidate = SimpleNamespace(
        tok_per_s=candidate_rate,
        tok_per_s_samples=[candidate_rate] * count,
        conditioning_tok_per_s=candidate_rate,
        spread=(candidate_rate, candidate_rate, 0.0),
    )
    speed = score_speedup(
        [baseline_rate, baseline_rate],
        candidate_rate,
        min_margin=arena.scoring.speedup_margin,
        k=arena.scoring.score_k,
        max_noise=arena.scoring.max_noise,
    )
    evidence = teacher_evidence(arena, quality=quality)
    quality_verdict = score_teacher_forced_quality(evidence, arena=arena)
    quality_ok = quality_verdict.passed
    positions = (
        arena.scoring.timed_iters
        * arena.workload.num_prompts
        * arena.workload.max_new_tokens
    )
    kl = SimpleNamespace(
        num_positions=positions,
        mean_kl=0.002 if quality else 1.0,
        max_kl=0.004 if quality else 1.0,
        p99_kl=0.003 if quality else 1.0,
        argmax_disagreements=0,
        mean_coverage_dev=0.001,
        dropped_positions=0,
    )
    return SimpleNamespace(
        baseline=baseline,
        candidate=candidate,
        baseline2=baseline,
        passed_quality=quality_ok,
        passed_timed_quality=quality_verdict.timed_decision == "PASS",
        passed_warmup_quality=quality_verdict.warmup_decision == "PASS",
        quality_decision=quality_verdict.decision,
        timed_quality_decision=quality_verdict.timed_decision,
        warmup_quality_decision=quality_verdict.warmup_decision,
        passed_speedup=speed.passed_speedup,
        confident=speed.confident,
        speedup=speed.speedup,
        noise=speed.noise,
        required_speedup=speed.required,
        score=speed.speedup if quality_ok and speed.passed_speedup else 0.0,
        kl=kl,
        control_kl=kl,
        external_quality_evidence=evidence,
        external_quality_desc=quality_verdict.detail,
        token_match=1.0 if quality else 0.0,
        fidelity_mode="audit",
        audit_desc="diagnostic only",
    )
