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

    # Candidate breaching TTFT p95 (> 2.0s) fails service_ttft_not_met.
    slow_ttft = tuple(
        LoadRead(
            "candidate", 1, "lane0", 4,
            tuple(
                TurnRecord(f"s{i}", "main", ord_, (b + 1) * 100_000_000_000 + ord_ * 5_000_000_000,
                           (b + 1) * 100_000_000_000 + ord_ * 5_000_000_000,
                           (b + 1) * 100_000_000_000 + ord_ * 5_000_000_000 + 2_500_000_000,
                           (b + 1) * 100_000_000_000 + ord_ * 5_000_000_000 + 3_500_000_000,
                           10_000, 200, "ok", 0, 100_000_000, 0 if ord_ == 0 else 9_000)
                for i in range(4) for ord_ in (0, 1)
            ),
        )
        for b in (6, 7)
    )
    v_ttft = reference_grade(slow_ttft, stock, contract, work, window_noise=0.01, boot_noise=0.01, error_rate=0.05,
                             boots=2, attainment_tolerance=0.1, attainment_margin=0.05,
                             prices=prices, itl_bound_s=0.5, decode_p10_floors={4: 90.0})
    assert v_ttft.decision is SpeedStageDecision.FAIL and "service_ttft_not_met" in v_ttft.detail

    # Zero completed revenue returns inf cost (and FAIL verdict) rather than raising.
    err_turn = TurnRecord("s0", "main", 0, base, base, base + 5_000_000_000, base + 15_000_000_000, 10_000, 200, "error", 0, 5_000_000_000, 9_000)
    cost, usd_hr = effective_revenue_summary([err_turn], 15.0, prices=prices, contract=contract, gpus=2)
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

    async def _run(speed_scale: float):
        now = [1000.0]
        arr = _Arrival(4, lambda: now[0], limit=2, open_stagger_s=10.0)
        order = []

        async def sess(idx: int):
            await arr.hold(f"s{idx}-0", (idx, 0, 1.0, 2.0))
            order.append((idx, 0))
            await asyncio.sleep(0)
            now[0] += 2.0 * speed_scale
            arr.done(f"s{idx}-0")
            await arr.hold(f"s{idx}-1", (idx, 1, None, 2.0))
            order.append((idx, 1))
            arr.done(f"s{idx}-1")

        await asyncio.gather(*(sess(i) for i in range(4)))
        return order

    slow = asyncio.run(_run(3.0))
    fast = asyncio.run(_run(1.0))
    # Pure trace-derived virtual timeline: slow and fast engines see identical admission order.
    assert slow == fast
    assert slow.index((0, 1)) < slow.index((2, 0))


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


def test_audit5_banned_sglang_imports_and_post_load_integrity(monkeypatch):
    import sys
    from types import ModuleType
    from cacheon.sandbox import scan_source
    from cacheon.seam import _snapshot_sealed_attrs, _verify_sealed_attrs

    for bad in (
        "import sglang.srt.speculative.eagle_worker",
        "from sglang.srt.managers import scheduler",
        "from sglang.srt.sampling.sampling_params import SamplingParams",
        "from sglang.srt.layers.sampler import Sampler",
        "from sglang.srt.server_args import ServerArgs",
        "from sglang.srt.model_executor.model_runner import ModelRunner",
        "from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner",
    ):
        assert not scan_source(bad).ok, f"expected violation for {bad}"

    for ok in (
        "from sglang.kernels.ops.moe import inkling2_route",
        "from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method",
        "from sglang.srt.model_executor.forward_context import get_attn_backend",
        "from sglang.srt.models.inkling_common.inkling2_moe import Inkling2MoE",
    ):
        assert scan_source(ok).ok, f"unexpected violation for {ok}"

    fake = ModuleType("sglang.srt.managers.scheduler")

    class _FakeSched:
        def __init__(self): ...
        def flush_cache(self): ...

    fake.Scheduler = _FakeSched
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", fake)
    snap = _snapshot_sealed_attrs()
    _verify_sealed_attrs(snap)
    _FakeSched.flush_cache = lambda self: None
    with pytest.raises(RuntimeError, match="candidate mutated non-slot engine object"):
        _verify_sealed_attrs(snap)

    from types import SimpleNamespace
    from cacheon.integrations.sglang_engine import _receipted

    def cheat_adder(req):
        req.sampling_params.temperature = 0.0

    wrapped = _receipted("sglang.srt.managers.schedule_policy.PrefillAdder", cheat_adder)
    req = SimpleNamespace(sampling_params=SimpleNamespace(temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, ignore_eos=True))
    with pytest.raises(PermissionError, match="mutated request sampling_params"):
        wrapped(req)


def test_policy24_seal_quality_noise_and_render_operator_rewrites(tmp_path: Path):
    import importlib.util
    from cacheon.eval.goodput_runtime import GoodputReadSet
    from cacheon.eval.tokenizer_identity import SealedTokenizerDecoder

    assert SealedTokenizerDecoder is not None
    spec = importlib.util.spec_from_file_location("seal_qc", Path("scripts/arena/seal_quality_controls.py"))
    seal_qc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seal_qc)

    prices = ApiPriceTable(0.15, 0.015, 0.50)
    contract = ServiceContract(100.0, 2.0, 0.8, max_itl_s=0.5)
    policy = GoodputPolicy(contract, 1.0, 0.01, 1.0, 0.0, error_rate=0.125, boot_noise=0.01,
                           attainment_pooled=1.0, decode_floor=1.0, decode_floors=((4, 80.0),),
                           steady_span=0.5, cold_span=1.0, api_prices=prices, itl_bound_s=2.0,
                           decode_p10_floors=((4, 60.0),))

    def _read(arm: str, idx: int, dur_ns: int):
        b0 = (idx + 1) * 100_000_000_000
        recs = tuple(
            TurnRecord(f"s{i}", "main", ord_, b0 + ord_ * 5_000_000_000, b0 + ord_ * 5_000_000_000,
                       b0 + ord_ * 5_000_000_000 + 200_000_000, b0 + ord_ * 5_000_000_000 + dur_ns,
                       1000, 100, "ok", 0, 100_000_000, 0 if ord_ == 0 else 800)
            for i in range(4) for ord_ in (0, 1)
        )
        return LoadRead(arm, 1, "lane0", 4, recs)

    reads = GoodputReadSet(
        (_read("incumbent", 0, 1_000_000_000), _read("incumbent", 1, 1_020_000_000)),
        (_read("candidate", 2, 900_000_000), _read("candidate", 3, 910_000_000)),
        tuple((f"s{i}", 2, 0) for i in range(4)), window_limit=1, boots=2, reference_boots=2,
    )
    witness_path = tmp_path / "witness.json"
    witness_path.write_text(json.dumps({
        "speed_witness": {"resident_policy": {"goodput": policy.to_dict()}, "goodput": reads.to_dict()}
    }))
    sha, sd = seal_qc._stock_speed_noise(witness_path)
    assert len(sha) == 64 and sd > 0.0

    ro_spec = importlib.util.spec_from_file_location("render_op", Path("scripts/arena/render_operator.py"))
    ro = importlib.util.module_from_spec(ro_spec)
    ro_spec.loader.exec_module(ro)
    op_path = tmp_path / "operator.json"
    op_path.write_text(json.dumps({
        "schema": "cacheon-arena-operator-v1", "arena_id": "inkling2mini-h200-node-v1", "slug": "inkling2mini",
        "model_profile_key": "Inkling-2-Mini", "tensor_parallel_size": 2, "replay_load": [6, 12],
        "decode_floors": {"6": "180", "12": "80"}, "decode_p10_floors": {"6": "130", "12": "60"},
        "load_sessions": {"6": 18, "12": 115}, "flush_each_load": True, "feedback_turns": True,
        "pause_order": True, "quality_every_load": True, "bar_futility": True, "cold_span": True,
        "steady_span": "0.5", "load_margin": "0.03", "replay_windows": 1, "replay_warm_windows": 0,
        "speed_policy_version": 24, "temperature": 1, "itl_bound_s": "2",
        "api_prices": {"input_per_m": "0.15", "cached_input_per_m": "0.015", "output_per_m": "0.5"},
        "service_contract": {"decode_floor_tps": "100", "ttft_bound_s": "2", "attainment": "0.8", "max_itl_s": "0.5"},
        "tokens_per_prompt": 1024, "tokenizer_asset": "tokenizer-inkling2mini", "model_in_ram": False,
        "owner": "inkling2mini-node", "audit_control_delta_digests": ["a" * 64],
        "node": {"name": "8-GPU H200 pod", "architecture": "sm90", "topology_class": "nvlink-sxm5",
                 "scheduler_starts": [32, 20, 3, 39, 87, 54, 77, 68],
                 "helper_starts": [42, 41, 43, 37, 78, 80, 82, 86],
                 "client_cpus": {"0,1": [2, 21], "2,3": [47, 40], "4,5": [85, 81], "6,7": [88, 89]}},
    }))
    loaded = ro.load_operator(op_path)
    assert loaded["speed_policy_version"] == 24 and loaded["temperature"] == 1


def test_reference_request_preserves_per_role_input_ids_orq4():
    from types import SimpleNamespace
    from cacheon.eval.qualification_runner import _reference_request
    from cacheon.eval.reference_protocol import (
        ROLE_TOKEN_REQUEST_MAGIC,
        ReferencePromptInput,
        ReferenceRoleInput,
        decode_reference_request,
        encode_reference_request,
    )

    d = "b" * 64
    p_dig = "c" * 64
    roles = (
        ReferenceRoleInput((10, 11), ((), ()), (1, 2, 3)),
        ReferenceRoleInput((12, 13), ((), ()), (1, 2, 4)),
        ReferenceRoleInput((14, 15), ((), ()), (1, 2, 5)),
    )
    ref_in = ReferencePromptInput(p_dig, "", roles, (1, 2, 3))
    frames = [{"output_ids": list(r.output_ids), "top_logprobs": [[], []]} for r in roles]

    # Build minimal duck-typed lifecycle/authority/selection accepted by _reference_request
    import cacheon.eval.qualification_runner as qr

    orig = qr._selected_frames
    try:
        qr._selected_frames = lambda _lc, _sel, _prompts: ((p_dig, (1, 2, 3), tuple(frames)),)
        lifecycle = SimpleNamespace(crossover=SimpleNamespace(reference_inputs=(ref_in,)))
        authority = SimpleNamespace(
            selected_delta_digest=d,
            profile=SimpleNamespace(reference=SimpleNamespace(pristine_launch_digest=d), topk_width=0),
        )
        selection = SimpleNamespace(selected_prompt_digests=(p_dig,))
        req = _reference_request(
            lifecycle, authority, selection,
            session_id="1" * 32, plan_digest=d, request_id="2" * 32, nonce="3" * 32, index=0,
        )
    finally:
        qr._selected_frames = orig

    assert req.request_magic == ROLE_TOKEN_REQUEST_MAGIC
    assert tuple(r.input_ids for r in req.prompts[0].roles) == ((1, 2, 3), (1, 2, 4), (1, 2, 5))
    decoded = decode_reference_request(encode_reference_request(req))
    assert decoded == req


def test_ore3_teacher_entropy_protocol_and_martingale_scoring():
    import hashlib
    from cacheon.eval.calibration import (
        CalibrationContext, CalibrationControl, CalibrationManifest, MetricCalibration, SpeedCalibration,
    )
    from cacheon.eval.reference_protocol import (
        ENTROPY_EVIDENCE_MAGIC, ReferenceEvidence, ReferencePromptEvidence, ReferencePromptInput,
        ReferenceRequest, ReferenceRoleEvidence, ReferenceRoleInput, ReferenceTokenEvidence,
        decode_reference_evidence, encode_reference_evidence, request_sha256,
    )
    from cacheon.eval.reference_quality import (
        HiddenTaskEvidence, PromptQualityEvidence, ReferenceQualityEvidence, RolloutQualityEvidence,
        TeacherNLLEvidence, score_reference_quality,
    )

    roles = (
        ReferenceRoleInput((10, 11), ((), ()), (1, 2, 3)),
        ReferenceRoleInput((12, 13), ((), ()), (1, 2, 4)),
        ReferenceRoleInput((14, 15), ((), ()), (1, 2, 5)),
    )
    p_dig = "c" * 64
    req = ReferenceRequest("1" * 32, "a" * 64, "b" * 64, "2" * 32, "3" * 32, 0, 2, 0,
                           (ReferencePromptInput(p_dig, "", roles, (1, 2, 3)),))
    assert req.evidence_magic == ENTROPY_EVIDENCE_MAGIC
    ev_roles = tuple(
        ReferenceRoleEvidence((
            ReferenceTokenEvidence(-0.5, 10, (), 0.52),
            ReferenceTokenEvidence(-0.4, 11, (), 0.41),
        ))
        for _ in range(3)
    )
    p_sha = hashlib.sha256(req.prompts[0].input_bytes).hexdigest()
    ev = ReferenceEvidence("1" * 32, "a" * 64, "b" * 64, request_sha256(req), "2" * 32, "3" * 32, 0, 100,
                           (ReferencePromptEvidence(p_dig, 3, p_sha, ev_roles),))
    wire = encode_reference_evidence(ev, req)
    assert wire[:4] == ENTROPY_EVIDENCE_MAGIC
    assert decode_reference_evidence(wire, req) == ev

    # Martingale entropy_residual tests candidate D = mean_nll - mean_H directly against 0,
    # invariant to baseline branching (here baseline mean_nll=0.90 vs candidate mean_nll=0.30).
    d = "a" * 64
    ctx = CalibrationContext(d, d, d, d, d, d, d, d, d, d)
    ctrls = (
        CalibrationControl("negative", "1" * 64, d, "FAIL"),
        CalibrationControl("positive", "2" * 64, d, "PASS"),
        CalibrationControl("stock", "3" * 64, d, "PASS"),
    )
    ent_metric = MetricCalibration("entropy_residual", "lower", "0.05", "0.05", two_sided=True)
    manifest = CalibrationManifest(
        context=ctx, algorithm_id="teacher-familywise-upper-bound-v1", status="frozen",
        speed=SpeedCalibration("0.01", "2", "0.02"), quality_metrics=(ent_metric,),
        familywise_z="2", raw_evidence_digest=d, seed_digests=("1" * 64, "2" * 64, "3" * 64), controls=ctrls,
    )

    def _ev(cand_nll: str, cand_ent: str) -> ReferenceQualityEvidence:
        def _roll(nll: str, ent: str) -> RolloutQualityEvidence:
            return RolloutQualityEvidence(
                TeacherNLLEvidence(10, f"{float(nll) * 10:g}", nll, 0, "0", f"{float(ent) * 10:g}"),
                None, HiddenTaskEvidence("0", 0),
            )
        prompts = tuple(
            PromptQualityEvidence(f"{i:064x}", _roll("0.9", "0.9"), _roll(cand_nll, cand_ent), _roll("0.5", "0.5"), 5, 10)
            for i in range(1, 5)
        )
        return ReferenceQualityEvidence(
            reference_manifest_digest=d, calibration_digest=manifest.digest,
            raw_evidence_digest=d, prompts=prompts, hidden_tasks_present=False,
        )

    assert score_reference_quality(_ev("0.3", "0.3"), calibration=manifest, expected_context=ctx).decision == "PASS"
    sharp = score_reference_quality(_ev("0.2", "0.3"), calibration=manifest, expected_context=ctx)
    assert sharp.decision == "FAIL" and "entropy_residual.lower_bound" in sharp.failed_metrics
    corrupt = score_reference_quality(_ev("0.42", "0.3"), calibration=manifest, expected_context=ctx)
    assert corrupt.decision == "FAIL" and "entropy_residual" in corrupt.failed_metrics


def test_policy24_single_engine_crossover_plan_accepts_version_24(tmp_path: Path):
    """Policy 24 shares Policy 23's single-engine lane topology (not PD pair); regression for == 23 check."""
    from cacheon.eval.agent_replay import AgentReplayPlan
    from cacheon.eval.api_revenue import ApiPriceTable
    from cacheon.eval.crossover_runtime import ResidentArmPlan, ResidentCrossoverPlan, ResidentSpeedPolicy
    from cacheon.eval.engine_launch import (
        EngineLaunchSpec, LogicalHardwareSpec, NativeBuildSpec, PhysicalHardwareBinding, TrustedLaunchBinding,
        native_compiler_policy_digest, native_patcher_digest, native_toolchain_digest,
    )
    from cacheon.eval.goodput_runtime import GoodputPolicy
    from cacheon.eval.oci_backend import RuntimePreflightReceipt, expected_runtime_preflight
    from cacheon.eval.oci_outer_session import SessionExecutionPlan
    from cacheon.eval.oci_session_protocol import CONTAINER_MODEL_PATH, EngineSessionConfig
    from cacheon.eval.service_capacity import ServiceContract
    from cacheon.stack_identity import canonical_digest

    sdir = tmp_path / "sessions"; sdir.mkdir()
    body = json.dumps({"id": "s0", "turns": [{"prompt_ids": [1, 2], "out_len": 4, "asst_len": 4, "think_time": 0.0}]}, separators=(",", ":")).encode()
    (sdir / "000_s0.json").write_bytes(body)
    manifest_p = tmp_path / "manifest.json"
    manifest_p.write_text(json.dumps({
        "dataset": "t", "revision": "v1", "loader": "multi_turn_tokens", "rules": {"seed": 1}, "rejected": {},
        "dir": "sessions", "files": 1, "order": "sorted filename = sealed order",
        "sha256_of_named_files": hashlib.sha256(b"000_s0.json" + body).hexdigest(),
        "sessions": [{"id": "s0", "n_main_total": 1, "start": 0, "k": 1, "groups": 0, "inner": 0,
                      "cold_prefix": 2, "peak": 6, "new_input": 2, "reused_input": 0, "output": 4,
                      "inner_input": 0, "inner_output": 0, "recorded_span_s": 0.0}],
    }))
    tok_dir = tmp_path / "tok"; tok_dir.mkdir()
    aiperf = tmp_path / "aiperf"; aiperf.write_text("#!/bin/sh\n"); aiperf.chmod(0o755)
    tree_dir = tmp_path / "tree"; tree_dir.mkdir()
    d = "a" * 64
    contract = ServiceContract(100.0, 2.0, 0.8)
    replay = AgentReplayPlan(manifest_p, (1,), aiperf, tok_dir, tmp_path / "rep", contract, "incumbent", 1, "p")
    cfg = EngineSessionConfig(CONTAINER_MODEL_PATH, "bfloat16", False, "fa4", False, 0.85, "error", 64, 2, "auto", False)
    hw = LogicalHardwareSpec(2, "sm90", "nvlink-sxm5", d, 2, 2, 1, d)
    phw = PhysicalHardwareBinding(("0", "1"), "sm90", "nvlink-sxm5", d, 2, 2, 1, d)
    from cacheon.compat import PINNED_SGLANG
    from cacheon.eval.runtime_preflight import HOST_RECEIPT_SCHEMA
    pf = RuntimePreflightReceipt(
        HOST_RECEIPT_SCHEMA, "img@sha256:" + d, d, d, ("img@sha256:" + d,),
        "linux/amd64", d, "/usr/bin/docker", 65534, 65534, PINNED_SGLANG, "cacheon", "0.1.0", d,
        10, 1024, "CPython", "/usr/bin/python3.12", "3.12.0", "cp312", "linux", "x86_64",
        (("sglang", PINNED_SGLANG),), "libcudart.so", "", "void", d,
    )
    nb = NativeBuildSpec(
        d, pf.image_digest, pf.platform_digest, pf.worker_distribution_digest,
        native_toolchain_digest(image_digest=pf.image_digest, platform_digest=pf.platform_digest),
        native_patcher_digest(worker_distribution_digest=pf.worker_distribution_digest),
        native_compiler_policy_digest(
            image_digest=pf.image_digest, worker_distribution_digest=pf.worker_distribution_digest,
            dependency_policy_digest=d, target_architecture="sm90",
        ),
        "sm90", d,
    )
    launch = EngineLaunchSpec(d, d, d, d, d, pf.image_digest, pf.platform_digest, d, d, d, d, d, d, cfg.digest, d, d, nb.digest, hw)
    binding = TrustedLaunchBinding(tree_dir, d, nb, pf, phw)
    splan = SessionExecutionPlan(
        launch_digest=launch.digest, expected_engine_config_digest=cfg.digest, engine_config=cfg,
        expected_preflight=expected_runtime_preflight(launch, pf), prompt_batches=(("hi",), ("hi",)),
        warmup_count=1, conditioning_count=1, max_new_tokens=4, top_logprobs_num=0, temperature=1.0,
        expected_prompt_tokens=1, measure_phase_latency=True, replay=replay,
        batch_max_new_tokens=(4, 4), batch_expected_prompt_tokens=(1, 1),
    )
    arm = ResidentArmPlan(launch, binding, splan, d, d, d)
    gp = GoodputPolicy(
        contract, 1.0, 0.01, 1.0, 0.0, error_rate=0.125, boot_noise=0.01, attainment_pooled=1.0,
        decode_floor=1.0, decode_floors=((1, 100.0),), api_prices=ApiPriceTable(0.075, 0.015, 0.30),
    )
    rsp = ResidentSpeedPolicy(3600, 0.0, 2.0, 0.02, d, d, version=24, goodput=gp)
    plan = ResidentCrossoverPlan(canonical_digest("test", {"c": 1}), arm, arm, rsp)
    assert plan.policy.version == 24
