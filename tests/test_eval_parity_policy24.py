"""Contracts for speed policy 24, host-clocked ITL, ORQ4 feedback T, and light A/T configs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cacheon.eval.agent_slice import load_slice_manifest
from cacheon.eval.api_revenue import ApiPriceTable, effective_revenue_summary, turn_list_value
from cacheon.eval.b300_qualification_commission import _light_teacher_config
from cacheon.eval.goodput_runtime import GoodputPolicy
from cacheon.eval.multiturn_replay import reconstruct_turn_inputs
from cacheon.eval.oci_session_protocol import (
    BatchEvidence,
    BatchRequest,
    EngineSessionConfig,
    PromptEvidence,
    SessionProtocolError,
)
from cacheon.eval.phase_latency import HostTokenClock, token_boundary
from cacheon.eval.reference_protocol import (
    ROLE_TOKEN_REQUEST_MAGIC,
    TOKEN_REQUEST_MAGIC,
    ReferencePromptInput,
    ReferenceRequest,
    ReferenceRoleInput,
    decode_reference_request,
    encode_reference_request,
)
from cacheon.eval.resident_audit_authority import _audit_engine_config
from cacheon.eval.resident_speed_policy import ResidentSpeedPolicy
from cacheon.eval.service_capacity import LoadRead, ServiceContract, TurnRecord, reference_grade
from cacheon.eval.speed_verdict import SpeedStageDecision


def _clock(times):
    it = iter(times)
    return lambda: next(it)


def _batch_req(n_out: int = 6):
    return BatchRequest("a" * 32, "b" * 64, "c" * 32, "d" * 32, 0, ("hi",), n_out, 0, 1.0, 2, True)


def _batch_ev(output_ids: tuple[int, ...]):
    return BatchEvidence((PromptEvidence(output_ids, ((),) * len(output_ids), 2),))


def test_host_token_clock_verifies_step_tokens_and_penalizes_dropped_steps():
    req = _batch_req(6)
    ev = _batch_ev((10, 11, 12, 13, 14, 15))
    frame = lambda *a, **kw: json.loads(token_boundary(req, *a, **kw)[8:])
    # Valid stream: first @ 1.0s, step(3, tok=12) @ 1.2s, step(5, tok=14) @ 1.5s, last @ 1.6s -> max_gap = 0.3s
    clock = HostTokenClock(req, _clock([1.0, 1.2, 1.5, 1.6]), 0.5)
    clock.observe(frame(0, "first", 10))
    clock.observe(frame(0, "step", 12, completion_tokens=3))
    clock.observe(frame(0, "step", 14, completion_tokens=5))
    clock.observe(frame(0, "last", 15))
    assert clock.finish(ev, completed=1.7)[0] == pytest.approx((0.5, 1.1))
    assert clock.max_gaps(ev) == pytest.approx((0.3,))

    # Suppressing intermediate steps worsens max_gap to the entire decode span (0.6s vs 0.3s).
    dropped = HostTokenClock(req, _clock([1.0, 1.6]), 0.5)
    dropped.observe(frame(0, "first", 10))
    dropped.observe(frame(0, "last", 15))
    dropped.finish(ev, completed=1.7)
    assert dropped.max_gaps(ev) == pytest.approx((0.6,))

    # Fabricated step token_id fails protocol verification.
    bad = HostTokenClock(req, _clock([1.0, 1.2, 1.6]), 0.5)
    bad.observe(frame(0, "first", 10))
    bad.observe(frame(0, "step", 999, completion_tokens=3))
    bad.observe(frame(0, "last", 15))
    with pytest.raises(SessionProtocolError, match="token boundaries disagree"):
        bad.finish(ev, completed=1.7)


def test_api_revenue_ideal_cache_billing_and_policy24_gates():
    prices = ApiPriceTable(input_per_m=0.15, cached_input_per_m=0.015, output_per_m=0.50)
    contract = ServiceContract(ttft_bound_s=2.0, decode_floor_tps=100.0, attainment=0.80, max_itl_s=0.5)

    # Evicting prefixes (cached_tokens=0) never inflates list value above validator-sealed ideal_cached_tokens.
    base = 1_000_000_000
    r_hit = TurnRecord("s0", "main", 0, base, base, base + 200_000_000, base + 1_200_000_000, 10_000, 150, "ok", 9_000, 100_000_000, 9_000)
    r_evict = TurnRecord("s0", "main", 0, base, base, base + 200_000_000, base + 1_200_000_000, 10_000, 150, "ok", 0, 100_000_000, 9_000)
    assert turn_list_value(r_hit, prices) == turn_list_value(r_evict, prices)

    def make_read(arm: str, boot_idx: int, *, end_ns: int, itl_ns: int):
        b0 = (boot_idx + 1) * 100_000_000_000
        recs = []
        for i in range(4):
            recs.append(TurnRecord(f"s{i}", "main", 0, b0, b0, b0 + 200_000_000, b0 + 1_200_000_000, 10_000, 200, "ok", 0, 100_000_000, 0))
            t1 = b0 + 2_000_000_000
            recs.append(TurnRecord(f"s{i}", "main", 1, t1, t1, t1 + 200_000_000, t1 + end_ns, 10_000, 200, "ok", 9_000, itl_ns, 9_000))
        return LoadRead(arm, 1, "lane0", 4, tuple(recs))

    work = {4: {f"s{i}": (2, 0) for i in range(4)}}
    stock = tuple(make_read("incumbent", b, end_ns=1_700_000_000, itl_ns=100_000_000) for b in (0, 1))

    # Candidate with streaming freeze (max_itl_ns = 1.2s > 0.5s) fails service_itl_not_met.
    frozen = tuple(make_read("candidate", b, end_ns=1_500_000_000, itl_ns=1_200_000_000) for b in (2, 3))
    v_itl = reference_grade(frozen, stock, contract, work, window_noise=0.01, boot_noise=0.01, error_rate=0.05,
                            boots=2, attainment_tolerance=0.1, attainment_margin=0.05,
                            prices=prices, itl_bound_s=0.5, decode_p10_floors={4: 90.0})
    assert v_itl.decision is SpeedStageDecision.FAIL and "service_itl_not_met" in v_itl.detail

    # Candidate below decode_p10 floor (200 tok / 2.6s = 76.9 tok/s < 90) fails service_tail_floor_not_met.
    slow_tail = tuple(make_read("candidate", b, end_ns=2_800_000_000, itl_ns=100_000_000) for b in (4, 5))
    v_tail = reference_grade(slow_tail, stock, contract, work, window_noise=0.01, boot_noise=0.01, error_rate=0.05,
                             boots=2, attainment_tolerance=0.1, attainment_margin=0.05,
                             prices=prices, itl_bound_s=0.5, decode_p10_floors={4: 90.0})
    assert v_tail.decision is SpeedStageDecision.FAIL and "service_tail_floor_not_met" in v_tail.detail

    # Zero effective revenue returns inf cost (and FAIL verdict) rather than raising.
    zero_turn = TurnRecord("s0", "main", 0, base, base, base + 5_000_000_000, base + 15_000_000_000, 10_000, 200, "ok", 0, 5_000_000_000, 9_000)
    cost, usd_hr = effective_revenue_summary([zero_turn], 15.0, prices=prices, contract=contract, gpus=2)
    assert cost == float("inf") and usd_hr == 0.0

    # Historical Policy 21 omits new Policy 24 optional wire keys.
    legacy = GoodputPolicy(ServiceContract(2.0, 100.0, 0.8), 1.0, 0.01, 0.1, 0.05, error_rate=0.05, boot_noise=0.01)
    assert not ({"api_prices", "itl_bound_s", "decode_p10_floors"} & set(legacy.to_dict()))
    p24 = ResidentSpeedPolicy(
        max_stage_seconds=3600, min_margin=0.0, noise_multiplier=1.0, max_noise=0.02,
        calibration_digest="a" * 64, calibration_context_digest="b" * 64, version=24,
        max_qualification_seconds=7200, min_windows=0, max_window_scatter=0.0,
        max_conditioning_slowdown=0.0, prefill_min_margin=0.0, prefill_credit_weight=0.0,
        goodput=GoodputPolicy(contract, 1.0, 0.01, 0.1, 0.05, error_rate=0.125, boot_noise=0.01,
                              attainment_pooled=1.0, decode_floor=1.0,
                              api_prices=prices, itl_bound_s=0.5, decode_p10_floors=((4, 90.0),)),
    )
    assert ResidentSpeedPolicy.from_dict(p24.to_dict()) == p24


def test_orq4_per_role_input_ids_and_cross_process_feedback_reconstruction(tmp_path: Path):
    # Shared prompt uses ORQ3; diverged per-role feedback prompts use ORQ4 and round-trip losslessly.
    shared_roles = tuple(ReferenceRoleInput((1, 2, 3), ((), (), ())) for _ in range(3))
    r3 = ReferenceRequest("a" * 32, "b" * 64, "c" * 64, "d" * 32, "e" * 32, 0, 3, 0,
                          (ReferencePromptInput("f" * 64, "", shared_roles, (10, 20)),))
    b3 = encode_reference_request(r3)
    assert b3[:4] == TOKEN_REQUEST_MAGIC and decode_reference_request(b3) == r3

    diverged_roles = (
        ReferenceRoleInput((1, 2), ((), ()), (10, 20, 100, 101, 99)),
        ReferenceRoleInput((3, 4), ((), ()), (10, 20, 200, 201, 99)),
        ReferenceRoleInput((5, 6), ((), ()), (10, 20, 300, 301, 99)),
    )
    r4 = ReferenceRequest("a" * 32, "b" * 64, "c" * 64, "d" * 32, "e" * 32, 0, 2, 0,
                          (ReferencePromptInput("f" * 64, "", diverged_roles, (10, 20, 100, 101, 99)),))
    b4 = encode_reference_request(r4)
    assert b4[:4] == ROLE_TOKEN_REQUEST_MAGIC and decode_reference_request(b4) == r4

    # Cross-process deterministic reconstruction of feedback turn inputs from sealed slice + retained outputs.
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    trace = {"id": "sess0", "turns": [
        {"prompt_ids": [10, 20], "out_len": 2, "asst_len": 2, "think_time": 0.0},
        {"prompt_ids": [10, 20, 77, 78, 99], "delta_ids": [99], "out_len": 2, "asst_len": 2, "think_time": 5.0},
    ]}
    body = json.dumps(trace, separators=(",", ":")).encode()
    (sdir / "000_sess0.json").write_bytes(body)
    dig = hashlib.sha256(b"000_sess0.json" + body).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataset": "test", "revision": "v1", "loader": "multi_turn_tokens", "rules": {"seed": 1}, "rejected": {},
        "dir": "sessions", "files": 1, "order": "sorted filename = sealed order", "sha256_of_named_files": dig,
        "sessions": [{"id": "sess0", "n_main_total": 2, "start": 0, "k": 2, "groups": 0, "inner": 0,
                      "cold_prefix": 2, "peak": 5, "new_input": 3, "reused_input": 4, "output": 4,
                      "inner_input": 0, "inner_output": 0, "recorded_span_s": 5.0}],
    }))
    manifest = load_slice_manifest(tmp_path / "manifest.json")
    rebuilt = reconstruct_turn_inputs(manifest, 1, {("sess0", 0): (500, 501), ("sess0", 1): (600, 601)}, feedback=True)
    assert rebuilt[("sess0", 0)] == (10, 20)
    assert rebuilt[("sess0", 1)] == (10, 20, 500, 501, 99)


def test_light_teacher_and_compact_audit_configs():
    charged = EngineSessionConfig(
        model_path="/cacheon/input/model", dtype="bfloat16", deterministic=False, attention_backend="fa4",
        disable_cuda_graph=False, mem_fraction_static=0.93, log_level="error", max_running_requests=64,
        tp_size=2, moe_runner_backend="auto", disable_custom_all_reduce=False,
        engine_kwargs={"enable_hierarchical_cache": True, "hicache_ratio": 2, "speculative_algorithm": "DSPARK",
                       "speculative_num_steps": 3, "swa_full_tokens_ratio": 0.05},
    )
    light_t = _light_teacher_config(charged)
    assert light_t.disable_cuda_graph is True
    assert "enable_hierarchical_cache" not in light_t.engine_kwargs
    assert "hicache_ratio" not in light_t.engine_kwargs
    assert "speculative_algorithm" not in light_t.engine_kwargs
    assert light_t.engine_kwargs["swa_full_tokens_ratio"] == 0.05

    audit_a = _audit_engine_config(charged)
    assert audit_a.disable_cuda_graph is True
    assert audit_a.engine_kwargs["enable_hierarchical_cache"] is True
    assert "hicache_ratio" not in audit_a.engine_kwargs
    assert audit_a.engine_kwargs["hicache_size"] == 8


def test_audit_first_block_stratified_coverage(monkeypatch):
    from cacheon import audit

    slots = [f"model.layers.{i}.attn.attn" for i in range(44)]
    try:
        for seed in ("101", "202", "303"):
            monkeypatch.setenv("CACHEON_SLOT_AUDIT", "0.25")
            monkeypatch.setenv("CACHEON_SLOT_AUDIT_SEED", seed)
            traces = []
            for _rank in (0, 1):
                audit._state.update({"rate": None, "rng": None})
                audit._first_block.clear()
                hits = {s: 0 for s in slots}
                seq = []
                for _step in range(8):
                    for s in slots:
                        d = audit.sampled(s)
                        seq.append(d)
                        hits[s] += int(d)
                assert all(c >= 1 for c in hits.values()), f"missed slot under seed={seed}"
                traces.append(seq)
            assert traces[0] == traces[1]
    finally:
        audit._state.update({"rate": None, "rng": None})
        audit._first_block.clear()


def test_arrival_open_stagger_weaves_warm_turns_before_later_cold_openings():
    import asyncio
    from cacheon.eval.agent_replay import _Arrival

    async def _run():
        now = [1000.0]
        arr = _Arrival(4, lambda: now[0], limit=2, open_stagger_s=10.0)
        order = []

        async def sess(idx: int):
            await arr.hold(f"s{idx}-0", (idx, 0, 1.0))
            order.append((idx, 0))
            await asyncio.sleep(0)
            now[0] += 2.0
            arr.done(f"s{idx}-0")
            await arr.hold(f"s{idx}-1", (idx, 1, None))
            order.append((idx, 1))
            arr.done(f"s{idx}-1")

        await asyncio.gather(*(sess(i) for i in range(4)))
        return order

    seq = asyncio.run(_run())
    # Sessions 0 and 1 open first; session 0's turn 1 (due ~1003s) must beat session 2's cold opening (due 1020s).
    assert seq.index((0, 1)) < seq.index((2, 0))


def test_two_sided_mean_nll_blocks_greedy_sharpening_under_t1():
    from cacheon.eval.calibration import (
        CalibrationContext,
        CalibrationControl,
        CalibrationManifest,
        MetricCalibration,
        SpeedCalibration,
    )
    from cacheon.eval.reference_quality import (
        HiddenTaskEvidence,
        PromptQualityEvidence,
        ReferenceQualityEvidence,
        RolloutQualityEvidence,
        TeacherNLLEvidence,
        score_reference_quality,
    )

    one_sided = MetricCalibration("mean_nll", "lower", "0.15", "0.08")
    assert "two_sided" not in one_sided.to_dict()
    two_sided = MetricCalibration("mean_nll", "lower", "0.15", "0.08", two_sided=True)
    assert MetricCalibration.from_dict(two_sided.to_dict()) == two_sided

    d = "a" * 64
    ctx = CalibrationContext(d, d, d, d, d, d, d, d, d, d)
    ctrls = (
        CalibrationControl("negative", "1" * 64, d, "FAIL"),
        CalibrationControl("positive", "2" * 64, d, "PASS"),
        CalibrationControl("stock", "3" * 64, d, "PASS"),
    )
    manifest = CalibrationManifest(
        context=ctx, algorithm_id="teacher-familywise-upper-bound-v1", status="frozen",
        speed=SpeedCalibration("0.01", "2", "0.02"), quality_metrics=(two_sided,),
        familywise_z="2", raw_evidence_digest=d, seed_digests=("1" * 64, "2" * 64, "3" * 64), controls=ctrls,
    )

    def _ev(cand_nll: str) -> ReferenceQualityEvidence:
        def _roll(nll: str) -> RolloutQualityEvidence:
            return RolloutQualityEvidence(
                TeacherNLLEvidence(10, f"{float(nll) * 10:g}", nll, 0, "0"),
                None, HiddenTaskEvidence("0", 0),
            )
        prompts = tuple(
            PromptQualityEvidence(f"{i:064x}", _roll("0.45"), _roll(cand_nll), _roll("0.45"), 5, 10)
            for i in range(1, 5)
        )
        return ReferenceQualityEvidence(
            reference_manifest_digest=d, calibration_digest=manifest.digest,
            raw_evidence_digest=d, prompts=prompts, hidden_tasks_present=False,
        )

    assert score_reference_quality(_ev("0.44"), calibration=manifest, expected_context=ctx).decision == "PASS"
    bad_up = score_reference_quality(_ev("0.7"), calibration=manifest, expected_context=ctx)
    assert bad_up.decision == "FAIL" and "mean_nll" in bad_up.failed_metrics
    greedy_cheat = score_reference_quality(_ev("0.12"), calibration=manifest, expected_context=ctx)
    assert greedy_cheat.decision == "FAIL" and "mean_nll.lower_bound" in greedy_cheat.failed_metrics
