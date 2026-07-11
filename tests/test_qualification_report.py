from __future__ import annotations

import json
import hashlib
import os
import stat
from dataclasses import replace
from types import SimpleNamespace

import pytest

from optima.cli import _strictest_kl_threshold, main
from optima.competition import ResolvedCompetition
from optima.arenas import (
    MINIMAX_M3_B300_TP4_DECODE_V1,
    TeacherForcedQualityPolicy,
    derive_prompt_seed,
)
from optima.eval.qualification import (
    QualificationReport,
    QualificationReportError,
)
from optima.eval.scoring import score_speedup
from optima.eval.external_quality import (
    PromptClusterEvidence,
    RolloutQualitySummary,
    TeacherForcedBatchEvidence,
    TeacherForcedExternalQualityEvidence,
    score_teacher_forced_quality,
)
from optima.bundle_hash import content_hash
from optima.commit_reveal import make_chain_scope
from optima.eval.host_attestation import host_attestation_context


_TEST_POLICY = TeacherForcedQualityPolicy(
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
ARENA = replace(
    MINIMAX_M3_B300_TP4_DECODE_V1,
    fidelity=replace(
        MINIMAX_M3_B300_TP4_DECODE_V1.fidelity,
        teacher_forced_policy=_TEST_POLICY,
    ),
)
BUNDLE_HASH = "a" * 64
HOST_ATTESTATION_SHA256 = "sha256:" + "3" * 64
CHAIN_SCOPE = make_chain_scope(genesis_hash="0x" + "9" * 64, netuid=120)
VALIDATOR_HOTKEY = "validator-hotkey"
EVALUATION_ID = "8" * 64
MINER_HOTKEY = "miner-hotkey"
SEED_BLOCK = 100
SEED_ROUND_ID = 1
SEED_BLOCK_HASH = "0x" + "2" * 64
PROMPT_SEED = derive_prompt_seed(
    ARENA,
    bundle_hash=BUNDLE_HASH,
    round_id=SEED_ROUND_ID,
    block_hash=SEED_BLOCK_HASH,
)


EXPECTED_TOKENS = (
    ARENA.scoring.timed_iters
    * ARENA.workload.num_prompts
    * ARENA.workload.max_new_tokens
)
WARMUP_TOKENS = (
    ARENA.scoring.warmup_iters
    * ARENA.workload.num_prompts
    * ARENA.workload.max_new_tokens
)
BATCH_TOKENS = ARENA.workload.num_prompts * ARENA.workload.max_new_tokens


@pytest.fixture(autouse=True)
def _registered_test_arena(monkeypatch):
    lookup = lambda name: ARENA if name == ARENA.name else None
    monkeypatch.setattr("optima.eval.qualification.get_arena", lookup)
    monkeypatch.setattr("optima.arenas.get_arena", lookup)


def _kl(*, mean: float = 0.002, argmax: int = 0, positions=EXPECTED_TOKENS):
    return SimpleNamespace(
        num_positions=positions,
        mean_kl=mean,
        max_kl=max(mean, 0.004),
        p99_kl=max(mean, 0.003),
        argmax_disagreements=argmax,
        mean_coverage_dev=0.001,
        dropped_positions=0,
    )


def _summary(*, nll: float = 1.0):
    tokens = ARENA.workload.max_new_tokens
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


def _teacher_evidence(*, quality: bool):
    cluster = PromptClusterEvidence(
        baseline=_summary(),
        candidate=_summary(nll=1.0 if quality else 2.0),
        stock_control=_summary(),
        exact_token_matches=ARENA.workload.max_new_tokens if quality else 0,
        exact_token_total=ARENA.workload.max_new_tokens,
    )
    batch = TeacherForcedBatchEvidence((cluster,) * 8)
    return TeacherForcedExternalQualityEvidence(
        protocol="controller-posthoc-teacher-forced-v2",
        sealed_rollout_sha256="sha256:" + "4" * 64,
        raw_evidence_sha256="sha256:" + "5" * 64,
        raw_evidence_size=1,
        raw_artifact_published=True,
        hidden_tasks_present=True,
        timed_batches=(batch,) * ARENA.scoring.timed_iters,
        warmup_batches=(batch,) * ARENA.scoring.warmup_iters,
    )


def _eval_report(
    *,
    quality: bool = True,
    speed: bool = True,
    candidate_conditioning_rate: float | None = None,
):
    baseline_rate = 100.0
    candidate_timed_rate = 108.0 if speed else 100.0
    candidate_conditioning_rate = (
        candidate_timed_rate
        if candidate_conditioning_rate is None
        else candidate_conditioning_rate
    )
    candidate_rate = min(candidate_timed_rate, candidate_conditioning_rate)
    sample_count = ARENA.scoring.timed_iters
    baseline = SimpleNamespace(
        tok_per_s=baseline_rate,
        tok_per_s_samples=[baseline_rate] * sample_count,
        conditioning_tok_per_s=baseline_rate,
        spread=(baseline_rate, baseline_rate, 0.0),
    )
    candidate = SimpleNamespace(
        tok_per_s=candidate_rate,
        tok_per_s_samples=[candidate_timed_rate] * sample_count,
        conditioning_tok_per_s=candidate_conditioning_rate,
        spread=(candidate_timed_rate, candidate_timed_rate, 0.0),
    )
    candidate_kl = _kl(mean=0.002 if quality else 1.0)
    control_kl = _kl(mean=0.001)
    external_quality_evidence = _teacher_evidence(quality=quality)
    quality_verdict = score_teacher_forced_quality(
        external_quality_evidence, arena=ARENA
    )
    speed_verdict = score_speedup(
        [baseline_rate, baseline_rate], candidate_rate,
        min_margin=ARENA.scoring.speedup_margin,
        k=ARENA.scoring.score_k,
        max_noise=ARENA.scoring.max_noise,
    )
    timed_quality = quality_verdict.timed_decision == "PASS"
    warmup_quality = quality_verdict.warmup_decision == "PASS"
    quality_ok = timed_quality and warmup_quality
    crownable = quality_ok and speed_verdict.passed_speedup
    return SimpleNamespace(
        baseline=baseline,
        candidate=candidate,
        baseline2=baseline,
        passed_quality=quality_ok,
        passed_timed_quality=timed_quality,
        passed_warmup_quality=warmup_quality,
        passed_speedup=speed_verdict.passed_speedup,
        confident=speed_verdict.confident,
        speedup=speed_verdict.speedup,
        noise=speed_verdict.noise,
        required_speedup=speed_verdict.required,
        score=speed_verdict.speedup if crownable else 0.0,
        kl=candidate_kl,
        control_kl=control_kl,
        token_matches=EXPECTED_TOKENS if quality else 0,
        token_total=EXPECTED_TOKENS,
        token_match=1.0 if quality else 0.0,
        stock_token_matches=EXPECTED_TOKENS,
        stock_token_total=EXPECTED_TOKENS,
        warmup_kl=_kl(
            mean=0.002 if quality else 1.0, positions=WARMUP_TOKENS
        ),
        warmup_control_kl=_kl(mean=0.001, positions=WARMUP_TOKENS),
        warmup_token_matches=WARMUP_TOKENS if quality else 0,
        warmup_token_total=WARMUP_TOKENS,
        warmup_stock_token_matches=WARMUP_TOKENS,
        warmup_stock_token_total=WARMUP_TOKENS,
        external_quality_evidence=external_quality_evidence,
        quality_decision=quality_verdict.decision,
        timed_quality_decision=quality_verdict.timed_decision,
        warmup_quality_decision=quality_verdict.warmup_decision,
        external_quality_desc=quality_verdict.detail,
        fidelity_mode="audit",
        audit_desc="diagnostic audit FAIL: candidate-owned receipt",
    )


def _competition(
    target: str = "attention.decode",
    mode: str = "slot",
    members: tuple[str, ...] = ("attention.decode",),
) -> ResolvedCompetition:
    return ResolvedCompetition(
        target=target,
        mode=mode,
        members=members,
        crownable=True,
    )


def _prepared(report=None, *, competition=None) -> QualificationReport:
    return QualificationReport.prepare_evidence(
        _eval_report() if report is None else report,
        competition=_competition() if competition is None else competition,
        arena=ARENA,
        bundle_hash=BUNDLE_HASH,
        prompt_seed=PROMPT_SEED,
        seed_round_id=SEED_ROUND_ID,
        seed_block=SEED_BLOCK,
        seed_block_hash=SEED_BLOCK_HASH,
        chain_scope=CHAIN_SCOPE,
        validator_hotkey=VALIDATOR_HOTKEY,
        evaluation_id=EVALUATION_ID,
        miner_hotkey=MINER_HOTKEY,
        settlement_round_id=SEED_ROUND_ID,
        evaluation_block=SEED_BLOCK,
    )


def _qualification(report=None, *, competition=None) -> QualificationReport:
    return _prepared(report, competition=competition).bind_host_attestation(
        HOST_ATTESTATION_SHA256
    )


def _rehash_raw(raw: dict) -> None:
    evidence = dict(raw)
    del evidence["host_attestation_sha256"]
    del evidence["qualification_evidence_sha256"]
    encoded = json.dumps(
        evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    raw["qualification_evidence_sha256"] = (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )


def test_qualification_report_roundtrip_distinguishes_quality_and_crownability(tmp_path):
    path = tmp_path / "report.json"
    quality_only = _qualification(_eval_report(speed=False))
    quality_only.write(path)

    parsed = QualificationReport.read(path)
    assert parsed.completed and parsed.passed_quality
    assert not parsed.passed_speedup and not parsed.crownable
    assert parsed.score == 0.0

    crowned = _qualification()
    assert crowned.crownable and crowned.score == 1.08
    assert crowned.target == "attention.decode"
    assert crowned.mode == "slot"
    assert crowned.member_slots == ("attention.decode",)


def test_final_warmup_conditioning_rate_caps_crownable_throughput():
    report = _eval_report(candidate_conditioning_rate=90.0)
    qualification = _qualification(report)

    assert report.candidate.tok_per_s == 90.0
    assert qualification.throughput_evidence.candidate_samples == (108.0,) * 3
    assert (
        qualification.throughput_evidence.candidate_conditioning_tok_per_s
        == 90.0
    )
    assert qualification.speedup == 0.9
    assert not qualification.passed_speedup
    assert not qualification.crownable and qualification.score == 0.0


def test_qualification_recomputes_teacher_forced_quality_for_every_mode():
    report = _eval_report()
    valid = _qualification(report)
    assert valid.crownable and valid.member_slots == ("attention.decode",)

    tampered = valid.to_dict()
    tampered["external_quality_evidence"]["timed_batches"][0]["clusters"][0][
        "candidate"
    ]["target_nll_sum"] = 512.0
    _rehash_raw(tampered)
    with pytest.raises(
        QualificationReportError,
        match="passed_timed_quality|passed_quality|quality_evidence",
    ):
        QualificationReport.from_dict(tampered)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"completed": False}, "completed"),
        ({"crownable": False, "score": 0.0}, "crownable headline"),
        ({"passed_speedup": False, "crownable": False, "score": 0.0},
         "passed_speedup headline"),
        ({"confident": False}, "confident headline"),
        ({"teacher_forced_mean_nll": float("nan")}, "finite number"),
        ({"target": ""}, "non-empty"),
        ({"mode": "unknown"}, "slot.*atomic"),
        ({"member_slots": []}, "sole member"),
        ({"member_slots": ["attention.decode", "attention.decode"]}, "duplicates"),
        ({"arena_fingerprint": "0" * 64}, "arena_fingerprint disagrees"),
        ({"referee_tree_digest": "sha256:" + "1" * 64},
         "referee_tree_digest disagrees"),
        ({"bundle_hash": "A" * 64}, "bundle_hash"),
        ({"host_attestation_sha256": "sha256:" + "A" * 64},
         "host_attestation_sha256"),
        ({"prompt_seed": 0}, "prompt_seed"),
        ({"prompt_engine_version": "other"}, "prompt_engine_version disagrees"),
    ],
)
def test_qualification_report_rejects_inconsistent_values(change, match):
    raw = _qualification().to_dict()
    raw.update(change)
    with pytest.raises(QualificationReportError, match=match):
        QualificationReport.from_dict(raw)


def _valid_raw_report() -> dict:
    return _qualification().to_dict()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda raw: raw.update(speedup=9.0), "speedup headline"),
        (lambda raw: raw["throughput_evidence"].update(
            candidate_samples=[108.0, 1.0, 1.0]), "speedup headline"),
        (lambda raw: raw["throughput_evidence"].update(
            candidate_conditioning_tok_per_s=1.0), "speedup headline"),
        (lambda raw: raw["throughput_evidence"].update(
            candidate_conditioning_tok_per_s=109.0), "first timed sample"),
        (lambda raw: raw.update(passed_quality=False), "passed_quality headline"),
        (lambda raw: raw["external_quality_evidence"]["timed_batches"][0][
            "clusters"
        ][0]["candidate"].update(target_nll_sum=2560.0, target_nll_max=10.0),
         "passed_timed_quality headline"),
        (lambda raw: raw["external_quality_evidence"]["warmup_batches"][0][
            "clusters"
        ][0]["candidate"].update(target_nll_sum=2560.0, target_nll_max=10.0),
         "passed_warmup_quality headline"),
        (lambda raw: raw["throughput_evidence"].update(
            candidate_samples=[108.0]), "timed_iters"),
        (lambda raw: raw["external_quality_evidence"]["timed_batches"][0][
            "clusters"
        ].pop(), "prompt clusters"),
        (lambda raw: raw["external_quality_evidence"]["timed_batches"][0][
            "clusters"
        ][0]["stock_control"].update(token_count=1), "fixed rollout"),
    ],
)
def test_qualification_recomputes_headlines_from_bounded_raw_evidence(mutate, match):
    raw = _valid_raw_report()
    mutate(raw)
    with pytest.raises(QualificationReportError, match=match):
        QualificationReport.from_dict(raw)


def test_audit_evidence_is_diagnostic_not_quality_authority():
    raw = _valid_raw_report()
    raw["audit_evidence"] = "diagnostic audit FAIL: deliberately forged local receipt"
    _rehash_raw(raw)
    parsed = QualificationReport.from_dict(raw)
    assert parsed.passed_quality and parsed.crownable


def test_prepared_evidence_hash_is_non_circular_and_host_binding_is_stable():
    prepared = _prepared()
    evidence = prepared.evidence_dict()
    evidence_bytes = prepared.evidence_bytes()

    assert "host_attestation_sha256" not in evidence
    assert "qualification_evidence_sha256" not in evidence
    assert prepared.qualification_evidence_sha256 == (
        "sha256:" + hashlib.sha256(evidence_bytes).hexdigest()
    )
    with pytest.raises(QualificationReportError, match="placeholder"):
        prepared.to_dict()

    bound = prepared.bind_host_attestation(HOST_ATTESTATION_SHA256)
    assert bound.evidence_bytes() == evidence_bytes
    assert (
        bound.qualification_evidence_sha256
        == prepared.qualification_evidence_sha256
    )
    reconstituted = QualificationReport.from_evidence_dict(
        evidence,
        qualification_evidence_sha256=prepared.qualification_evidence_sha256,
    )
    assert reconstituted.evidence_bytes() == evidence_bytes


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("chain_scope", "wrong:sha256:" + "1" * 64, "chain_scope"),
        ("validator_hotkey", "", "validator_hotkey"),
        ("evaluation_id", "A" * 64, "evaluation_id"),
        ("miner_hotkey", "", "miner_hotkey"),
        ("settlement_round_id", 2, "settlement round"),
        ("evaluation_block", 99, "settlement round|predate"),
        (
            "qualification_evidence_sha256",
            "sha256:" + "1" * 64,
            "qualification_evidence_sha256",
        ),
    ),
)
def test_qualification_binding_fields_fail_closed(field, value, match):
    raw = _valid_raw_report()
    raw[field] = value
    with pytest.raises(QualificationReportError, match=match):
        QualificationReport.from_dict(raw)


def test_one_canonical_host_context_matches_chain_report_and_champion():
    report = QualificationReport.from_dict(_valid_raw_report())
    expected = host_attestation_context(
        ARENA,
        bundle_hash=BUNDLE_HASH,
        prompt_seed=PROMPT_SEED,
        seed_round_id=SEED_ROUND_ID,
        seed_block=SEED_BLOCK,
        seed_block_hash=SEED_BLOCK_HASH,
        chain_scope=CHAIN_SCOPE,
        validator_hotkey=VALIDATOR_HOTKEY,
        evaluation_id=EVALUATION_ID,
        miner_hotkey=report.miner_hotkey,
        settlement_round_id=report.settlement_round_id,
        evaluation_block=report.evaluation_block,
        target=report.target,
        mode=report.mode,
        member_slots=report.member_slots,
        score=report.score,
        passed_quality=report.passed_quality,
        passed_timed_quality=report.passed_timed_quality,
        passed_warmup_quality=report.passed_warmup_quality,
        passed_speedup=report.passed_speedup,
        confident=report.confident,
        crownable=report.crownable,
        quality_evidence=report.quality_evidence,
        qualification_evidence_sha256=(
            report.qualification_evidence_sha256
        ),
    )
    assert report.attestation_context() == expected


def test_qualification_report_write_fsyncs_file_and_directory(tmp_path, monkeypatch):
    calls = []
    real_fsync = os.fsync

    def observed(fd):
        calls.append(os.fstat(fd).st_mode)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", observed)
    QualificationReport.from_dict(_valid_raw_report()).write(tmp_path / "report.json")
    assert len(calls) == 2
    assert stat.S_ISREG(calls[0]) and stat.S_ISDIR(calls[1])


def test_qualification_report_read_refuses_links_and_duplicate_keys(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_valid_raw_report()))
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(report)
    with pytest.raises(QualificationReportError, match="cannot read report"):
        QualificationReport.read(symlink)

    hardlink = tmp_path / "hardlink.json"
    os.link(report, hardlink)
    with pytest.raises(QualificationReportError, match="bounded regular file"):
        QualificationReport.read(report)

    hardlink.unlink()
    payload = json.dumps(_valid_raw_report())
    report.write_text('{"schema_version":8,' + payload[1:])
    with pytest.raises(QualificationReportError, match="duplicate JSON key"):
        QualificationReport.read(report)


def test_cli_rejects_zero_margin_before_loading_model_or_bundle(capsys):
    rc = main([
        "evaluate", "does-not-exist", "--model", "unused",
        "--speedup-margin", "0",
    ])
    assert rc == 2
    assert "speedup-margin > 0" in capsys.readouterr().out

    rc = main([
        "evaluate", "does-not-exist", "--model", "unused",
        "--speedup-margin", "nan",
    ])
    assert rc == 2


def test_scoring_outputs_require_registered_arena_and_post_commit_seed(capsys):
    rc = main([
        "evaluate", "does-not-exist", "--model", "unused",
        "--report", "unused.json",
    ])
    assert rc == 2
    assert "requires an explicit registered --arena" in capsys.readouterr().out

    rc = main([
        "evaluate", "does-not-exist", "--arena", ARENA.name,
        "--report", "unused.json",
    ])
    assert rc == 2
    assert "post-commit --prompt-seed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "override",
    [
        ["--no-isolate"],
        ["--allow-unsafe-no-isolation"],
        ["--disable-cuda-graph"],
        ["--conditioning-iters", "3"],
        ["--no-ignore-eos"],
        ["--candidate-attention-backend", "flashinfer"],
        ["--framework-mode"],
        ["--model", "other-model"],
    ],
)
def test_registered_arena_rejects_score_affecting_cli_overrides(capsys, override):
    rc = main([
        "evaluate", "does-not-exist", "--arena", ARENA.name,
        *override,
    ])
    assert rc == 2
    assert "profile-authoritative" in capsys.readouterr().out


def test_cli_evaluate_atomically_emits_typed_quality_only_report(
    monkeypatch, capsys, tmp_path
):
    # Report serialization is isolated here from the execution-class boundary,
    # which has dedicated adversarial/chain tests.
    monkeypatch.setattr(
        "optima.device_component.component_crown_rejection", lambda _manifest: None
    )
    monkeypatch.setattr(
        type(ARENA), "verify_model_receipt", lambda self, model_path=None: None
    )
    monkeypatch.setattr(type(ARENA), "verify_referee_source", lambda self: None)
    monkeypatch.setattr(type(ARENA), "verify_runtime_packages", lambda self: None)
    monkeypatch.setattr(
        "optima.cli._registered_oci_launcher",
        lambda *args, **kwargs: SimpleNamespace(attestation_receipts=[]),
    )
    monkeypatch.setattr(
        "optima.cli._publish_direct_host_attestation",
        lambda *args, **kwargs: SimpleNamespace(
            sha256=HOST_ATTESTATION_SHA256
        ),
    )
    bundle = tmp_path / "bundle"
    (bundle / "kernels").mkdir(parents=True)
    (bundle / "manifest.toml").write_text(
        'bundle_id = "report-test"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        "[[ops]]\n"
        'slot = "activation.silu_and_mul"\n'
        'source = "kernels/k.py"\n'
        'entry = "k"\n'
        'dtypes = ["float32"]\n'
    )
    (bundle / "kernels" / "k.py").write_text(
        "def k(x, out):\n    out.copy_(x)\n"
    )
    bundle_seed = derive_prompt_seed(
        ARENA,
        bundle_hash=content_hash(bundle),
        round_id=SEED_ROUND_ID,
        block_hash=SEED_BLOCK_HASH,
    )

    report = _eval_report(speed=False)
    monkeypatch.setattr(
        "optima.eval.throughput_kl.evaluate",
        lambda cfg, path, **kwargs: report,
    )

    report_path = tmp_path / "qualification.json"
    rc = main([
        "evaluate", str(bundle), "--arena", ARENA.name,
        "--prompt-seed", str(bundle_seed),
        "--seed-round-id", str(SEED_ROUND_ID),
        "--seed-block", str(SEED_BLOCK),
        "--seed-block-hash", SEED_BLOCK_HASH,
        "--chain-scope", CHAIN_SCOPE,
        "--validator-hotkey-address", VALIDATOR_HOTKEY,
        "--evaluation-id", EVALUATION_ID,
        "--miner-hotkey-address", MINER_HOTKEY,
        "--settlement-round-id", str(SEED_ROUND_ID),
        "--evaluation-block", str(SEED_BLOCK),
        "--report", str(report_path),
    ])
    assert rc == 0
    parsed = QualificationReport.read(report_path)
    assert parsed.passed_quality and not parsed.crownable and parsed.score == 0.0
    assert parsed.target == "activation.silu_and_mul"
    assert parsed.mode == "slot"
    assert parsed.member_slots == ("activation.silu_and_mul",)
    assert parsed.arena_name == ARENA.name
    assert parsed.arena_bracket == ARENA.bracket
    assert parsed.prompt_seed == bundle_seed
    assert parsed.quality_evidence.startswith("timed[phase:PASS:mean_nll")
    assert "non-authoritative" in parsed.quality_evidence
    assert not list(tmp_path.glob(".qualification.json.tmp.*"))
    assert "crownable=NO" in capsys.readouterr().out


def test_cli_atomic_report_uses_validator_canonical_member_order(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "optima.device_component.component_crown_rejection", lambda _manifest: None
    )
    monkeypatch.setattr(
        type(ARENA), "verify_model_receipt", lambda self, model_path=None: None
    )
    monkeypatch.setattr(type(ARENA), "verify_referee_source", lambda self: None)
    monkeypatch.setattr(type(ARENA), "verify_runtime_packages", lambda self: None)
    monkeypatch.setattr(
        "optima.cli._registered_oci_launcher",
        lambda *args, **kwargs: SimpleNamespace(attestation_receipts=[]),
    )
    monkeypatch.setattr(
        "optima.cli._publish_direct_host_attestation",
        lambda *args, **kwargs: SimpleNamespace(
            sha256=HOST_ATTESTATION_SHA256
        ),
    )
    bundle = tmp_path / "atomic"
    bundle.mkdir()
    # Deliberately reverse the manifest op order. The report must use the
    # validator-owned order in competition.ATOMIC_TARGETS.
    (bundle / "manifest.toml").write_text(
        'bundle_id = "atomic-report"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        "[competition]\n"
        'target = "collective.moe_epilogue.v1"\n'
        'mode = "atomic"\n\n'
        "[[ops]]\n"
        'slot = "collective.moe_finalize_ar_rmsnorm"\n'
        'source = "kernel.py"\n'
        'entry = "deep"\n\n'
        "[[ops]]\n"
        'slot = "collective.ar_residual_rmsnorm"\n'
        'source = "kernel.py"\n'
        'entry = "shallow"\n'
    )
    (bundle / "kernel.py").write_text(
        "def deep(*args):\n    return None\n\n"
        "def shallow(*args):\n    return None\n"
    )
    bundle_seed = derive_prompt_seed(
        ARENA,
        bundle_hash=content_hash(bundle),
        round_id=SEED_ROUND_ID,
        block_hash=SEED_BLOCK_HASH,
    )
    report = _eval_report()
    monkeypatch.setattr(
        "optima.eval.throughput_kl.evaluate",
        lambda cfg, path, **kwargs: report,
    )

    report_path = tmp_path / "atomic-report.json"
    assert main([
        "evaluate",
        str(bundle),
        "--arena",
        ARENA.name,
        "--prompt-seed",
        str(bundle_seed),
        "--seed-round-id", str(SEED_ROUND_ID),
        "--seed-block", str(SEED_BLOCK),
        "--seed-block-hash", SEED_BLOCK_HASH,
        "--chain-scope", CHAIN_SCOPE,
        "--validator-hotkey-address", VALIDATOR_HOTKEY,
        "--evaluation-id", EVALUATION_ID,
        "--miner-hotkey-address", MINER_HOTKEY,
        "--settlement-round-id", str(SEED_ROUND_ID),
        "--evaluation-block", str(SEED_BLOCK),
        "--report",
        str(report_path),
    ]) == 0

    parsed = QualificationReport.read(report_path)
    assert parsed.target == "collective.moe_epilogue.v1"
    assert parsed.mode == "atomic"
    assert parsed.member_slots == (
        "collective.ar_residual_rmsnorm",
        "collective.moe_finalize_ar_rmsnorm",
    )


def test_chain_cli_rejects_zero_margin_only_for_real_eval(capsys):
    rc = main([
        "chain-validate", "--netuid", "1", "--network", "unused",
        "--arena", ARENA.name,
        "--eval-cmd", "true", "--margin", "0",
    ])
    assert rc == 2
    assert "immutable arena settlement policy" in capsys.readouterr().out


def test_bench_explicitly_rejects_ledger_scoring_before_gpu_work(capsys, tmp_path):
    rc = main([
        "bench", "does-not-exist", "--model", "unused",
        "--ledger", str(tmp_path / "ledger.json"), "--hotkey", "miner",
    ])
    assert rc == 2
    assert "cannot write a crownable ledger score" in capsys.readouterr().out


def test_atomic_target_uses_strictest_member_kl_threshold(monkeypatch):
    thresholds = {"slot.loose": 3e-2, "slot.strict": 5e-3}
    monkeypatch.setattr(
        "optima.slots.get_slot",
        lambda name: SimpleNamespace(kl_threshold=thresholds[name]),
    )

    assert _strictest_kl_threshold(
        ("slot.loose", "slot.strict"),
        advisory=False,
        fallback=1e-2,
    ) == 5e-3
    assert _strictest_kl_threshold(
        ("slot.loose", "slot.strict"),
        advisory=True,
        fallback=1e-2,
    ) is None
