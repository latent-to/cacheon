"""Op-correctness test that needs torch but not a GPU.

Runs the full slot -> sandbox-load -> verify_entry path against the pure-torch
example bundle on CPU. Skipped automatically where torch is unavailable (e.g. the
dev laptop); runs on the VM.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from cacheon.capabilities import CallDescriptor  # noqa: E402
from cacheon.minimax_sparse_prefill_slot import INPUT_NAMES  # noqa: E402
from cacheon.registry import Eligibility, eligibility_from_metadata  # noqa: E402
from cacheon.sandbox import load_entry  # noqa: E402
from cacheon.slots import get_slot  # noqa: E402
from cacheon.tensor_spec import OutputSpec, TensorSpec  # noqa: E402
from cacheon.verify import format_verify, verify_entry  # noqa: E402
from cacheon.verification_outcomes import (  # noqa: E402
    GraphPhaseOutcome,
    PhaseDisposition,
    VerificationCaseDescriptor,
    VerificationCaseKind,
)

from pathlib import Path  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent  # cwd-independent
TORCH_BUNDLE = str(_REPO / "examples/miner_silu_torch/kernels/silu_and_mul.py")
BROKEN_TORCH_BUNDLE = str(_REPO / "examples/miner_silu_broken_torch/kernels/silu_and_mul.py")


class _FakeGraphBackend:
    """CPU model of capture/replay orchestration (not CUDA semantics themselves)."""

    def __init__(self):
        self.phase = "eager"
        self.replay_index = -1

    def warmup(self, fn):
        self.phase = "warmup"
        fn()
        self.phase = "eager"

    def capture(self, fn):
        self.phase = "capture"
        fn()
        self.phase = "eager"
        return fn

    def replay(self, graph):
        self.replay_index += 1
        self.phase = "replay"
        graph()
        self.phase = "eager"

    def synchronize(self):
        pass


def _faithful_silu(x, out):
    d = x.shape[-1] // 2
    out.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])


def _faithful_rmsnorm(x, weight, out, eps):
    x32 = x.float()
    normalized = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    out.copy_((normalized * weight).to(x.dtype))


def test_torch_silu_passes_correctness_cpu():
    entry = load_entry(TORCH_BUNDLE, "silu_and_mul")
    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(slot, entry, dtype=torch.float32, device="cpu", seed=0)
    assert result.passed, "\n".join(
        f"{r.shape}: max_abs={r.max_abs_err} {r.detail}" for r in result.shape_results
    )


def test_broken_torch_example_bundle_fails_cpu():
    # The committed adversarial bundle the miner guide's no-GPU walkthrough runs
    # (drops the SiLU). If this ever passes verify, the walkthrough demo is broken.
    entry = load_entry(BROKEN_TORCH_BUNDLE, "silu_and_mul")
    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(slot, entry, dtype=torch.float32, device="cpu", seed=0)
    assert not result.passed


def test_wrong_kernel_fails_correctness_cpu():
    # A deliberately broken "kernel": forgets the multiply, just copies silu(gate).
    def broken(x, out):
        d = x.shape[-1] // 2
        out.copy_(torch.nn.functional.silu(x[..., :d]))

    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(slot, broken, dtype=torch.float32, device="cpu", seed=0)
    assert not result.passed
    assert all(
        row.phase_outcome.eager is PhaseDisposition.CANDIDATE_FAILED
        and row.phase_outcome.observation_complete
        and row.phase_outcome.failure_is_candidate_attributable
        for row in result.shape_results
    )


def test_rmsnorm_catalog_exercises_exact_6144_hidden_domain():
    slot = get_slot("norm.rmsnorm")
    shapes = [shape for shape in slot.shapes if shape.get("hidden") == 6144]
    eligibility = eligibility_from_metadata(
        {
            "graph_safe": False,
            "capabilities": {"dtype": "float32", "last_dim": 6144},
        },
        ("float32",),
        (),
    )

    result = verify_entry(
        slot,
        _faithful_rmsnorm,
        dtype=torch.float32,
        device="cpu",
        shapes=shapes,
        eligibility=eligibility,
        graph_safe=False,
    )

    assert shapes == [{"num_tokens": 64, "hidden": 6144}]
    assert result.passed, format_verify(result)
    assert result.num_applicable == 1


def test_cpu_verify_reports_graph_proof_not_obtained():
    # Op slots execute under capture in serving, so graph proof is required by
    # default.  CPU verify remains a useful numerical PASS but must not claim that
    # capture/replay was verified.
    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(
        slot, _faithful_silu, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}],
    )
    assert result.passed
    assert result.graph_required
    assert not result.graph_verified
    assert not result.fully_verified
    assert result.shape_results[0].graph_replays == 0
    outcome = result.shape_results[0].phase_outcome
    assert outcome.eager_passed and not outcome.capture_succeeded
    assert outcome.capture is PhaseDisposition.INFRASTRUCTURE_FAILED
    assert not outcome.observation_complete
    assert not outcome.failure_is_candidate_attributable
    assert format_verify(result).startswith("[NUMERICAL_PASS]")


def test_graph_replay_orchestration_passes_all_replays_with_cpu_backend():
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()
    result = verify_entry(
        slot, _faithful_silu, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}], graph_safe=True,
        graph_replays=3, _graph_backend=backend,
    )
    assert result.passed
    assert result.graph_verified
    assert result.shape_results[0].graph_replays == 3
    outcome = result.shape_results[0].phase_outcome
    assert outcome.eager_passed and outcome.capture_succeeded
    assert outcome.replay_passed and outcome.replay_count == 3
    assert outcome.observation_complete
    assert not outcome.failure_is_candidate_attributable


def test_typed_not_applicable_outcome_never_claims_candidate_execution():
    result = verify_entry(
        get_slot("activation.silu_and_mul"),
        _faithful_silu,
        dtype=torch.float32,
        device="cpu",
        shapes=[{"num_tokens": 2, "d": 8}],
        graph_safe=False,
        eligibility=Eligibility(min_num_tokens=100),
    )

    assert not result.passed and result.num_not_applicable == 1
    row = result.shape_results[0]
    assert not row.applicable
    assert row.phase_outcome.eager is PhaseDisposition.NOT_RUN
    assert row.phase_outcome.observation_complete
    assert not row.phase_outcome.failure_is_candidate_attributable
    assert row.case_descriptor is not None


def test_phase_outcome_rejects_causal_and_attribution_contradictions():
    with pytest.raises(ValueError, match="capture/replay cannot precede"):
        GraphPhaseOutcome(
            PhaseDisposition.CANDIDATE_FAILED,
            PhaseDisposition.PASSED,
            PhaseDisposition.NOT_RUN,
            0,
            True,
        )
    with pytest.raises(ValueError, match="infrastructure failure cannot be a complete"):
        GraphPhaseOutcome(
            PhaseDisposition.PASSED,
            PhaseDisposition.INFRASTRUCTURE_FAILED,
            PhaseDisposition.NOT_RUN,
            0,
            True,
        )


def test_candidate_capture_exception_is_typed_without_parsing_detail():
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()

    def fails_during_capture(x, out):
        if backend.phase == "capture":
            raise RuntimeError("candidate capture path")
        _faithful_silu(x, out)

    result = verify_entry(
        slot,
        fails_during_capture,
        dtype=torch.float32,
        device="cpu",
        shapes=[{"num_tokens": 2, "d": 8}],
        graph_safe=True,
        graph_replays=3,
        _graph_backend=backend,
    )

    outcome = result.shape_results[0].phase_outcome
    assert not result.passed
    assert outcome.eager_passed
    assert outcome.capture is PhaseDisposition.CANDIDATE_FAILED
    assert outcome.replay is PhaseDisposition.NOT_RUN
    assert outcome.observation_complete
    assert outcome.failure_is_candidate_attributable


def test_candidate_first_replay_exception_is_typed_at_zero_completed_replays():
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()

    def fails_during_first_replay(x, out):
        if backend.phase == "replay" and backend.replay_index == 0:
            raise RuntimeError("candidate replay path")
        _faithful_silu(x, out)

    result = verify_entry(
        slot,
        fails_during_first_replay,
        dtype=torch.float32,
        device="cpu",
        shapes=[{"num_tokens": 2, "d": 8}],
        graph_safe=True,
        graph_replays=3,
        _graph_backend=backend,
    )

    outcome = result.shape_results[0].phase_outcome
    assert not result.passed
    assert outcome.capture_succeeded
    assert outcome.replay is PhaseDisposition.CANDIDATE_FAILED
    assert outcome.replay_count == 0
    assert outcome.observation_complete
    assert outcome.failure_is_candidate_attributable


@pytest.mark.parametrize(
    "slot_name", ("moe.fused_experts", "moe.fused_experts_reduce")
)
def test_moe_topk_one_generators_emit_raw_varying_weights(slot_name):
    slot = get_slot(slot_name)
    shape = {
        "num_tokens": 4,
        "num_experts": 3,
        "hidden": 8,
        "inter": 4,
        "topk": 1,
    }
    first = slot.make_inputs(
        **shape, dtype=torch.float32, device="cpu", seed=17
    )
    second = slot.make_inputs(
        **shape, dtype=torch.float32, device="cpu", seed=18
    )

    assert slot.graph_dynamic_inputs == ("x", "topk_ids", "topk_weights")
    assert first["topk_weights"].dtype == torch.float32
    assert torch.all(first["topk_weights"] > 0)
    assert torch.all(first["topk_weights"] < 1)
    assert not torch.equal(first["topk_weights"], second["topk_weights"])
    assert not torch.equal(
        first["topk_weights"], torch.ones_like(first["topk_weights"])
    )


def test_moe_topk_one_graph_replay_verifies_dynamic_routing_weights():
    slot = get_slot("moe.fused_experts")
    backend = _FakeGraphBackend()
    shape = {
        "num_tokens": 4,
        "num_experts": 3,
        "hidden": 8,
        "inter": 4,
        "topk": 1,
    }

    def prepare(w13, w2):
        return w13, w2

    def entry(x, topk_ids, topk_weights, prepared, out):
        w13, w2 = prepared
        expected = slot.invoke_reference(
            {
                "x": x,
                "w13": w13,
                "w2": w2,
                "topk_ids": topk_ids,
                "topk_weights": topk_weights,
            }
        )[0]
        out.copy_(expected.to(out.dtype))

    result = verify_entry(
        slot,
        entry,
        prepare=prepare,
        dtype=torch.float32,
        device="cpu",
        seed=29,
        shapes=[shape],
        graph_safe=True,
        graph_replays=3,
        _graph_backend=backend,
    )

    assert result.passed, format_verify(result)
    assert result.graph_verified
    assert result.shape_results[0].graph_replays == 3


def test_capture_only_wrong_branch_fails_graph_verification():
    # Models the real attack: eager/audit returns the reference, while a branch on
    # is_current_stream_capturing freezes a wrong kernel into the timed graph.
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()

    def capture_branch(x, out):
        if backend.phase in {"capture", "replay"}:
            out.zero_()
        else:
            _faithful_silu(x, out)

    result = verify_entry(
        slot, capture_branch, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}], graph_safe=True,
        graph_replays=3, _graph_backend=backend,
    )
    assert not result.passed
    assert not result.graph_verified
    assert result.shape_results[0].graph_replays == 1
    assert "cuda graph replay[0]" in result.shape_results[0].detail


def test_output_poison_catches_graph_that_does_not_write():
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()

    def capture_noop(x, out):
        if backend.phase not in {"capture", "replay"}:
            _faithful_silu(x, out)

    result = verify_entry(
        slot, capture_noop, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}], graph_safe=True,
        graph_replays=3, _graph_backend=backend,
    )
    assert not result.passed
    assert "actual has non-finite values" in result.shape_results[0].detail


def test_fresh_graph_inputs_catch_cached_correct_output():
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()
    cached = None

    def cached_capture(x, out):
        nonlocal cached
        if backend.phase == "capture":
            d = x.shape[-1] // 2
            cached = (
                torch.nn.functional.silu(x[..., :d]) * x[..., d:]
            ).clone()
            out.copy_(cached)
        elif backend.phase == "replay":
            out.copy_(cached)
        else:
            _faithful_silu(x, out)

    result = verify_entry(
        slot, cached_capture, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}], graph_safe=True,
        graph_replays=3, _graph_backend=backend,
    )

    assert not result.passed
    assert "cuda graph replay[0]" in result.shape_results[0].detail


def test_candidate_may_not_mutate_validator_inputs():
    slot = get_slot("activation.silu_and_mul")

    def mutating(x, out):
        _faithful_silu(x, out)
        x.zero_()

    result = verify_entry(
        slot, mutating, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}], graph_safe=False,
    )

    assert not result.passed
    assert "input 'x' was mutated" in result.shape_results[0].detail


def test_candidate_may_not_rebind_validator_inputs_to_equal_storage():
    slot = get_slot("activation.silu_and_mul")

    def rebinding(x, out):
        replacement = x.detach().clone()
        x.set_(replacement)
        _faithful_silu(x, out)

    result = verify_entry(
        slot,
        rebinding,
        dtype=torch.float32,
        device="cpu",
        seed=0,
        shapes=[{"num_tokens": 2, "d": 8}],
        graph_safe=False,
    )

    assert not result.passed
    assert "validator-owned storage/tensor binding" in result.shape_results[0].detail


def test_candidate_may_not_replace_validator_output_storage():
    slot = get_slot("activation.silu_and_mul")

    def replacing(x, out):
        replacement = torch.empty_like(out)
        out.set_(replacement)
        _faithful_silu(x, out)

    result = verify_entry(
        slot,
        replacing,
        dtype=torch.float32,
        device="cpu",
        seed=0,
        shapes=[{"num_tokens": 2, "d": 8}],
        graph_safe=False,
    )

    assert not result.passed
    assert "validator-owned storage" in result.shape_results[0].detail


def test_candidate_may_not_change_validator_output_strides():
    slot = get_slot("activation.silu_and_mul")

    def restriding(x, out):
        out.as_strided_(out.shape, (1, out.shape[0]))
        _faithful_silu(x, out)

    result = verify_entry(
        slot,
        restriding,
        dtype=torch.float32,
        device="cpu",
        seed=0,
        shapes=[{"num_tokens": 2, "d": 8}],
        graph_safe=False,
    )

    assert not result.passed
    assert "validator-owned storage/tensor binding" in result.shape_results[0].detail


def test_one_loaded_entry_cannot_cache_first_captured_shape():
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()
    captured_tokens = None

    def first_shape_only(x, out):
        nonlocal captured_tokens
        if backend.phase == "capture" and captured_tokens is None:
            captured_tokens = x.shape[0]
        if backend.phase in {"capture", "replay"} and x.shape[0] != captured_tokens:
            out.zero_()
        else:
            _faithful_silu(x, out)

    result = verify_entry(
        slot, first_shape_only, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}, {"num_tokens": 5, "d": 8}],
        graph_safe=True, graph_replays=3, _graph_backend=backend,
    )

    assert not result.passed
    assert result.shape_results[0].passed
    assert not result.shape_results[1].passed


def test_topk_graph_poison_catches_partial_selection_write():
    # MSA serves eagerly; force graph orchestration here solely to prove that a
    # candidate cannot leave poisoned int32 padding behind on replay.
    slot = get_slot("attention.msa_prefill_block_score")
    backend = _FakeGraphBackend()

    def partial_during_capture(*args):
        *values, out = args
        inputs = dict(zip(INPUT_NAMES, values, strict=True))
        expected = slot.invoke_reference(inputs)[0]
        if backend.phase in {"capture", "replay"}:
            valid = expected >= 0
            out[valid] = expected[valid]
        else:
            out.copy_(expected)

    result = verify_entry(
        slot,
        partial_during_capture,
        dtype=torch.bfloat16,
        device="cpu",
        seed=0,
        shapes=[slot.shapes[1]],
        graph_safe=True,
        graph_replays=3,
        _graph_backend=backend,
    )
    assert not result.passed
    assert result.shape_results[0].graph_replays == 1
    assert "padding" in result.shape_results[0].detail


def test_later_graph_replay_corruption_is_not_hidden_by_first_replay():
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()

    def stateful_capture(x, out):
        if backend.phase == "replay" and backend.replay_index == 1:
            out.zero_()
        else:
            _faithful_silu(x, out)

    result = verify_entry(
        slot, stateful_capture, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}], graph_safe=True,
        graph_replays=3, _graph_backend=backend,
    )
    assert not result.passed
    assert result.shape_results[0].graph_replays == 2
    assert "cuda graph replay[1]" in result.shape_results[0].detail
    outcome = result.shape_results[0].phase_outcome
    assert outcome.replay is PhaseDisposition.CANDIDATE_FAILED
    assert outcome.replay_count == 2
    assert outcome.observation_complete
    assert outcome.failure_is_candidate_attributable


def test_explicit_non_graph_safe_skips_graph_gate():
    slot = get_slot("activation.silu_and_mul")
    backend = _FakeGraphBackend()
    result = verify_entry(
        slot, _faithful_silu, dtype=torch.float32, device="cpu", seed=0,
        shapes=[{"num_tokens": 2, "d": 8}], graph_safe=False,
        _graph_backend=backend,
    )
    assert result.passed
    assert not result.graph_required
    assert not result.graph_verified
    assert backend.replay_index == -1


def test_ordinary_case_descriptor_binds_full_context_and_digest():
    result = verify_entry(
        get_slot("activation.silu_and_mul"),
        _faithful_silu,
        dtype=torch.float32,
        device="cpu",
        architecture="sm999",
        tp_size=2,
        world_size=2,
        variant_name="variant-a",
        shapes=[{"num_tokens": 2, "d": 8}],
        graph_safe=True,
        graph_replays=3,
        _graph_backend=_FakeGraphBackend(),
    )
    case = result.shape_results[0].case_descriptor
    assert case is not None
    assert case.slot_id == "activation.silu_and_mul"
    assert case.variant_id == "variant-a"
    assert case.case_kind is VerificationCaseKind.ORDINARY_SINGLE
    call = dict(case.calls[0])
    assert {
        "dtype": call["dtype"],
        "architecture": call["architecture"],
        "tp_size": call["tp_size"],
        "world_size": call["world_size"],
        "graph_mode": call["graph_mode"],
    } == {
        "dtype": "float32",
        "architecture": "sm999",
        "tp_size": 2,
        "world_size": 2,
        "graph_mode": "cuda_graph",
    }

    for field, value in (
        ("dtype", "bfloat16"),
        ("architecture", "sm998"),
        ("tp_size", 4),
        ("world_size", 4),
        ("graph_mode", "eager"),
    ):
        changed = dict(call)
        changed[field] = value
        rebound = VerificationCaseDescriptor(
            slot_id=case.slot_id,
            variant_id=case.variant_id,
            case_kind=case.case_kind,
            calls=(CallDescriptor(changed),),
        )
        assert rebound.digest != case.digest

    missing = dict(call)
    missing.pop("architecture")
    with pytest.raises(ValueError, match="missing sealed execution context"):
        VerificationCaseDescriptor(
            slot_id=case.slot_id,
            variant_id=case.variant_id,
            case_kind=case.case_kind,
            calls=(CallDescriptor(missing),),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_silu_real_cuda_graph_capture_and_replay():
    slot = get_slot("activation.silu_and_mul")
    result = verify_entry(
        slot,
        _faithful_silu,
        dtype=torch.float32,
        device="cuda",
        seed=17,
        shapes=[{"num_tokens": 8, "d": 128}],
        graph_safe=True,
        graph_replays=3,
    )
    assert result.passed, result.shape_results
    assert result.graph_verified
    assert result.shape_results[0].graph_replays == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_typed_strided_fp32_output_real_cuda_graph_capture_and_replay():
    base = get_slot("activation.silu_and_mul")
    slot = replace(
        base,
        name="test.activation.typed_strided_fp32",
        output_spec=lambda inputs: OutputSpec(
            outputs=(
                TensorSpec(
                    shape=(*inputs["x"].shape[:-1], inputs["x"].shape[-1] // 2),
                    dtype=torch.float32,
                    stride_policy="strided",
                    stride_padding=11,
                    name="typed_scores",
                ),
            )
        ),
        invoke_reference=lambda inputs: [
            (
                torch.nn.functional.silu(inputs["x"][..., : inputs["x"].shape[-1] // 2])
                * inputs["x"][..., inputs["x"].shape[-1] // 2 :]
            ).float()
        ],
    )

    def typed_entry(x, out):
        assert out.dtype == torch.float32
        assert not out.is_contiguous() and out.stride(-1) == 1
        d = x.shape[-1] // 2
        out.copy_((torch.nn.functional.silu(x[..., :d]) * x[..., d:]).float())

    result = verify_entry(
        slot,
        typed_entry,
        dtype=torch.bfloat16,
        device="cuda",
        seed=19,
        shapes=[{"num_tokens": 8, "d": 128}],
        graph_safe=True,
        graph_replays=3,
    )
    assert result.passed, result.shape_results
    assert result.graph_verified
    assert result.shape_results[0].graph_replays == 3


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA GPUs",
)
def test_real_cuda_graph_uses_output_device_not_process_current_device():
    original = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        slot = get_slot("activation.silu_and_mul")
        result = verify_entry(
            slot,
            _faithful_silu,
            dtype=torch.float32,
            device="cuda:1",
            seed=23,
            shapes=[{"num_tokens": 8, "d": 128}],
            graph_safe=True,
            graph_replays=3,
        )
        assert result.passed, result.shape_results
        assert result.graph_verified
    finally:
        torch.cuda.set_device(original)
