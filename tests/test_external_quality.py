from dataclasses import replace
import hashlib

import pytest

from optima.arenas import (
    MINIMAX_M3_B300_TP4_LONGPREFILL_V1,
    TeacherForcedQualityPolicy,
)
from optima.eval.external_quality import (
    ExternalQualityError,
    PromptClusterEvidence,
    QUALITY_FAIL,
    QUALITY_NO_DECISION,
    QUALITY_PASS,
    RolloutQualitySummary,
    TeacherForcedBatchEvidence,
    TeacherForcedExternalQualityEvidence,
    TeacherForcedPromptTrace,
    TeacherForcedTrace,
    build_teacher_forced_evidence,
    publish_raw_quality_artifact,
    reopen_raw_quality_artifact,
    score_teacher_forced_quality,
    seal_posthoc_reference_plan,
    summarize_rollout,
)


def _policy(**changes):
    base = TeacherForcedQualityPolicy(
        protocol="controller-posthoc-teacher-forced-v2",
        calibration_state="provisional-rtx",
        clusters_per_batch=8,
        nll_clip=100.0,
        tail_nll_threshold=10.0,
        familywise_z=2.0,
        stock_mean_nll_envelope=0.10,
        stock_worst_nll_envelope=0.10,
        stock_tail_rate_envelope=0.05,
        stock_topk_kl_envelope=0.05,
        stock_argmax_rate_envelope=0.05,
        stock_coverage_envelope=0.05,
        mean_nll_delta=0.10,
        worst_nll_delta=0.10,
        tail_rate_delta=0.05,
        topk_kl_delta=0.05,
        argmax_rate_delta=0.05,
        coverage_delta=0.05,
        require_hidden_tasks=True,
        stock_hidden_score_envelope=0.05,
        hidden_score_delta=0.05,
        hidden_score_floor=0.80,
    )
    return replace(base, **changes)


def _arena(policy=None):
    arena = MINIMAX_M3_B300_TP4_LONGPREFILL_V1
    return replace(
        arena,
        fidelity=replace(
            arena.fidelity,
            teacher_forced_policy=policy or _policy(),
        ),
    )


def _summary(
    *, nll=1.0, worst=None, tail=0, topk=0.01, hidden=1.0, tokens=64
):
    return RolloutQualitySummary(
        token_count=tokens,
        target_nll_sum=nll * tokens,
        target_nll_max=nll if worst is None else worst,
        target_tail_count=tail,
        topk_positions=tokens,
        topk_mean_kl=topk,
        topk_max_kl=topk,
        topk_p99_kl=topk,
        topk_argmax_disagreements=0,
        topk_mean_coverage_dev=0.01,
        hidden_score=hidden,
        hidden_total=1,
    )


def _evidence(*, candidate_nll=1.0, control_nll=1.0, exact=0):
    cluster = PromptClusterEvidence(
        baseline=_summary(nll=1.0),
        candidate=_summary(nll=candidate_nll),
        stock_control=_summary(nll=control_nll),
        exact_token_matches=exact,
        exact_token_total=64,
    )
    batch = TeacherForcedBatchEvidence((cluster,) * 8)
    return TeacherForcedExternalQualityEvidence(
        protocol="controller-posthoc-teacher-forced-v2",
        sealed_rollout_sha256="sha256:" + "1" * 64,
        raw_evidence_sha256="sha256:" + "2" * 64,
        raw_evidence_size=1,
        raw_artifact_published=True,
        hidden_tasks_present=True,
        timed_batches=(batch,) * 3,
        warmup_batches=(batch,) * 3,
    )


def test_exact_trajectory_match_is_diagnostic_not_crown_authority():
    verdict = score_teacher_forced_quality(_evidence(exact=0), arena=_arena())
    assert verdict.decision == QUALITY_PASS
    assert "non-authoritative" in verdict.detail


def test_teacher_forced_noninferiority_has_three_way_verdicts():
    arena = _arena()
    assert score_teacher_forced_quality(
        _evidence(candidate_nll=1.5), arena=arena
    ).decision == QUALITY_FAIL
    assert score_teacher_forced_quality(
        _evidence(control_nll=1.5), arena=arena
    ).decision == QUALITY_NO_DECISION


def test_hidden_task_and_sparse_late_corruption_cannot_hide_under_mean_nll():
    arena = _arena()
    bland = PromptClusterEvidence(
        _summary(), _summary(hidden=0.0), _summary(), 64, 64
    )
    sparse = PromptClusterEvidence(
        _summary(), _summary(worst=20.0, tail=1), _summary(), 63, 64
    )
    for cluster in (bland, sparse):
        batch = TeacherForcedBatchEvidence((cluster,) * 8)
        evidence = TeacherForcedExternalQualityEvidence(
            protocol="controller-posthoc-teacher-forced-v2",
            sealed_rollout_sha256="sha256:" + "1" * 64,
            raw_evidence_sha256="sha256:" + "2" * 64,
            raw_evidence_size=1,
            raw_artifact_published=True,
            hidden_tasks_present=True,
            timed_batches=(batch,) * 3,
            warmup_batches=(batch,) * 3,
        )
        assert score_teacher_forced_quality(evidence, arena=arena).decision == QUALITY_FAIL


def test_warmup_and_timed_batches_cannot_subsidize_each_other():
    arena = _arena()
    good = _evidence()
    bad_cluster = PromptClusterEvidence(
        _summary(), _summary(nll=2.0), _summary(), 64, 64
    )
    bad = TeacherForcedBatchEvidence((bad_cluster,) * 8)
    evidence = replace(good, warmup_batches=(bad,) * 3)
    verdict = score_teacher_forced_quality(evidence, arena=arena)
    assert verdict.timed_decision == QUALITY_PASS
    assert verdict.warmup_decision == QUALITY_FAIL
    assert verdict.decision == QUALITY_FAIL


def test_phase_level_statistics_do_not_repeat_an_all_or_nothing_null_per_batch():
    arena = _arena(_policy(mean_nll_delta=0.01, worst_nll_delta=0.01))

    def batch(candidate_nll):
        cluster = PromptClusterEvidence(
            _summary(), _summary(nll=candidate_nll), _summary(), 64, 64
        )
        return TeacherForcedBatchEvidence((cluster,) * 8)

    evidence = replace(
        _evidence(),
        timed_batches=(batch(1.02), batch(0.98), batch(1.0)),
    )
    verdict = score_teacher_forced_quality(evidence, arena=arena)
    assert verdict.timed_decision == QUALITY_PASS


def test_uncalibrated_hardware_explicitly_refuses_crown():
    arena = _arena(_policy(calibration_state="uncalibrated"))
    verdict = score_teacher_forced_quality(_evidence(), arena=arena)
    assert verdict.decision == QUALITY_NO_DECISION
    assert "uncalibrated" in verdict.detail


def test_missing_hidden_judge_is_only_valid_in_explicit_uncalibrated_state():
    cluster = PromptClusterEvidence(
        _summary(hidden=0.0),
        _summary(hidden=0.0),
        _summary(hidden=0.0),
        64,
        64,
    )
    # Hidden score zero with total one is still "present" in _summary; replace the
    # summaries with an explicit no-judge denominator.
    cluster = replace(
        cluster,
        baseline=replace(cluster.baseline, hidden_total=0),
        candidate=replace(cluster.candidate, hidden_total=0),
        stock_control=replace(cluster.stock_control, hidden_total=0),
    )
    batch = TeacherForcedBatchEvidence((cluster,) * 8)
    evidence = replace(
        _evidence(),
        hidden_tasks_present=False,
        timed_batches=(batch,) * 3,
        warmup_batches=(batch,) * 3,
    )
    assert score_teacher_forced_quality(
        evidence, arena=_arena(_policy(calibration_state="uncalibrated"))
    ).decision == QUALITY_NO_DECISION
    with pytest.raises(ExternalQualityError, match="calibrated policy"):
        evidence.validated(_arena())


def test_prompt_cluster_evidence_is_exact_and_bounded():
    arena = _arena()
    raw = _evidence().to_dict()
    raw["timed_batches"][0]["clusters"][0]["candidate"]["token_count"] = 63
    parsed = TeacherForcedExternalQualityEvidence.from_dict(raw)
    with pytest.raises(ExternalQualityError, match="fixed rollout"):
        parsed.validated(arena)


def test_forged_candidate_topk_is_regraded_against_trusted_prefix_context():
    trusted = tuple(
        ((-0.1, 1, None), (-3.0, 2, None)) for _ in range(4)
    )
    forged = tuple(
        ((-0.1, 2, None), (-3.0, 1, None)) for _ in range(4)
    )
    summary = summarize_rollout(
        forged,
        TeacherForcedTrace(tuple([-0.1] * 4), trusted),
        tail_nll_threshold=10.0,
        nll_clip=100.0,
        topk_num=2,
    )
    assert summary.topk_argmax_disagreements == 4
    assert summary.topk_mean_kl > 1.0


def test_seal_covers_all_bc_outputs_before_secret_cluster_selection():
    topk = tuple(((-0.1, 1, None), (-2.4, 2, None)) for _ in range(2))
    run = ((1, 1), topk)
    prompts = [[f"p{index}" for index in range(3)]]
    plan = seal_posthoc_reference_plan(
        prompts,
        baseline_batches=[[run, run, run]],
        candidate_batches=[[run, run, run]],
        warmup_iters=0,
        clusters_per_batch=2,
        expected_tokens=2,
        topk_num=2,
        selection_secret=b"s" * 32,
    )
    changed = seal_posthoc_reference_plan(
        prompts,
        baseline_batches=[[run, run, run]],
        candidate_batches=[[run, (((2, 1)), topk), run]],
        warmup_iters=0,
        clusters_per_batch=2,
        expected_tokens=2,
        topk_num=2,
        selection_secret=b"s" * 32,
    )
    assert plan.sealed_rollout_sha256 != changed.sealed_rollout_sha256


def test_raw_teacher_frames_are_content_addressed_and_reopenable(tmp_path):
    raw = b'{"raw":"teacher"}'
    evidence = replace(
        _evidence(),
        raw_evidence_sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
        raw_evidence_size=len(raw),
        raw_artifact_published=False,
        raw_evidence_bytes=raw,
    )
    with pytest.raises(ExternalQualityError, match="not content-addressed"):
        evidence.validated(_arena())
    published = publish_raw_quality_artifact(tmp_path, evidence)
    assert published.raw_artifact_published
    assert reopen_raw_quality_artifact(tmp_path, published) == raw


def test_raw_digest_binds_stock_control_ids_and_reported_topk():
    policy = _policy(
        clusters_per_batch=2,
        require_hidden_tasks=False,
        hidden_score_floor=0.0,
    )
    base = _arena(policy)
    arena = replace(
        base,
        workload=replace(
            base.workload, num_prompts=3, max_new_tokens=2, top_logprobs=2
        ),
        scoring=replace(
            base.scoring, timed_iters=2, warmup_iters=2, conditioning_iters=2
        ),
    )
    topk = ((-0.1, 1, None), (-3.0, 2, None))
    run = ((1, 1), (topk, topk))
    prompts = [[f"b{batch}-p{index}" for index in range(3)] for batch in range(4)]
    batches = [[run, run, run] for _ in range(4)]
    plan = seal_posthoc_reference_plan(
        prompts,
        baseline_batches=batches,
        candidate_batches=batches,
        warmup_iters=2,
        clusters_per_batch=2,
        expected_tokens=2,
        topk_num=2,
        selection_secret=b"r" * 32,
    )
    trace = TeacherForcedTrace((-0.1, -0.1), (topk, topk))
    traces = {
        (batch.phase, batch.batch_index): tuple(
            TeacherForcedPromptTrace(
                2, f"{index + 1:x}" * 64, trace, trace, trace
            )
            for index, _prompt in enumerate(batch.prompts)
        )
        for batch in (*plan.warmup_batches, *plan.timed_batches)
    }
    first = build_teacher_forced_evidence(
        plan,
        stock_control_batches=batches,
        warmup_iters=2,
        traces=traces,
        arena=arena,
    )
    changed_batches = [list(batch) for batch in batches]
    selected = plan.timed_batches[0].prompts[0].prompt_index
    changed_topk = ((-0.1, 2, None), (-3.0, 1, None))
    changed_batches[2][selected] = ((2, 1), (changed_topk, topk))
    second = build_teacher_forced_evidence(
        plan,
        stock_control_batches=changed_batches,
        warmup_iters=2,
        traces=traces,
        arena=arena,
    )
    assert first.raw_evidence_sha256 != second.raw_evidence_sha256
