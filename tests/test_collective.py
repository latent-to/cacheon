"""Distributed verification of the collective.all_reduce slot (CPU / gloo, 2 ranks).

Spawns 2 gloo ranks, runs the example all-reduce, and checks each rank's output equals
the trusted fp32 cross-rank sum. No GPU needed; torch-only (skipped where torch absent).
gloo has no bf16, so verify_collective uses fp32 on the CPU path.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from cacheon.slots import get_slot  # noqa: E402
from cacheon.capabilities import CallDescriptor  # noqa: E402
from cacheon.registry import Eligibility  # noqa: E402
from cacheon.verify_collective import (  # noqa: E402
    _MAX_VERDICT_BYTES,
    _RankVerdict,
    _VERDICT_VERSION,
    _direct_aot_collective_callables,
    _init_rank_process_group,
    _rank_barrier,
    _read_rank_verdict,
    _regular_identity,
    _write_rank_verdict,
    CollectiveVerdictError,
    verify_collective,
)
from cacheon.verification_outcomes import (  # noqa: E402
    GraphPhaseOutcome,
    PhaseDisposition,
    VerificationCaseDescriptor,
    VerificationCaseKind,
    conservative_phase_aggregate,
)

ALLREDUCE_BUNDLE = "examples/miner_allreduce_torch/kernels/all_reduce.py"
DP_EXCHANGE_BUNDLE = "examples/miner_dp_attention_exchange_torch/kernels/exchange.py"
SMALL_SHAPES = [{"num_tokens": 2, "hidden": 8}]


def _verify(source=ALLREDUCE_BUNDLE, **kwargs):
    options = dict(
        world_size=2, backend="gloo", device="cpu", shapes=SMALL_SHAPES,
    )
    options.update(kwargs)
    return verify_collective(
        get_slot("collective.all_reduce"), str(source), "all_reduce", **options
    )


def test_collective_kind_discriminator():
    assert get_slot("collective.all_reduce").kind == "collective"


def test_cuda_process_group_and_barrier_bind_the_local_rank_device():
    cuda = Mock()
    device_factory = Mock(return_value="cuda-device-3")
    fake_torch = Mock(cuda=cuda, device=device_factory)
    fake_dist = Mock()

    _init_rank_process_group(
        fake_torch,
        fake_dist,
        backend="nccl",
        init_method="file:///tmp/pg",
        rank=3,
        world_size=4,
        device="cuda",
    )
    _rank_barrier(fake_dist, rank=3, device="cuda")

    cuda.set_device.assert_called_once_with(3)
    device_factory.assert_called_once_with("cuda:3")
    fake_dist.init_process_group.assert_called_once_with(
        backend="nccl",
        init_method="file:///tmp/pg",
        rank=3,
        world_size=4,
        device_id="cuda-device-3",
    )
    fake_dist.barrier.assert_called_once_with(device_ids=[3])


def test_cpu_process_group_and_barrier_keep_gloo_call_signature():
    fake_torch = Mock()
    fake_dist = Mock()

    _init_rank_process_group(
        fake_torch,
        fake_dist,
        backend="gloo",
        init_method="file:///tmp/pg",
        rank=1,
        world_size=2,
        device="cpu",
    )
    _rank_barrier(fake_dist, rank=1, device="cpu")

    fake_torch.cuda.set_device.assert_not_called()
    fake_torch.device.assert_not_called()
    fake_dist.init_process_group.assert_called_once_with(
        backend="gloo",
        init_method="file:///tmp/pg",
        rank=1,
        world_size=2,
    )
    fake_dist.barrier.assert_called_once_with()


def test_collective_direct_aot_propagates_only_validator_prepare_boundary():
    class DirectEntry:
        def __call__(self, *_args):
            return None

        def prepare(self, w13, w2):
            return ("validator-prepared", w13, w2)

    slot = get_slot("moe.fused_experts_reduce")
    direct_entry = DirectEntry()
    entry, prepare = _direct_aot_collective_callables(
        slot, direct_entry, prepare_name=None
    )

    assert entry is direct_entry
    assert callable(prepare)
    assert prepare("w13", "w2") == ("validator-prepared", "w13", "w2")
    with pytest.raises(RuntimeError, match="validator-generated.*prepare boundary"):
        _direct_aot_collective_callables(
            slot, lambda *_args: None, prepare_name=None
        )
    with pytest.raises(ValueError, match="candidate Python prepare"):
        _direct_aot_collective_callables(
            slot, direct_entry, prepare_name="prepare"
        )


def test_allreduce_faithful_passes_gloo_cpu():
    slot = get_slot("collective.all_reduce")
    res = verify_collective(slot, ALLREDUCE_BUNDLE, "all_reduce",
                            world_size=2, backend="gloo", device="cpu", seed=0)
    assert res.passed, "\n".join(f"{r.shape}: {r.detail}" for r in res.shape_results)
    # The all-reduce seam serves from the captured decode graph, so capture is
    # required and a gloo/CPU run cannot produce it. Only the deliberately
    # unsynchronized temporal burst is eager by construction.
    assert all(row.phase_outcome in (
        GraphPhaseOutcome.capture_infrastructure_failed(),
        GraphPhaseOutcome.eager_only_passed(),
    ) for row in res.shape_results)
    assert all(row.case_descriptor is not None for row in res.shape_results)


@pytest.mark.parametrize(
    "slot_name,entry_name",
    (
        ("collective.all_gather_into_tensor", "all_gather_into_tensor"),
        ("collective.reduce_scatter_tensor", "reduce_scatter_tensor"),
    ),
)
def test_dp_exchange_faithful_passes_gloo_cpu(monkeypatch, slot_name, entry_name):
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    result = verify_collective(
        get_slot(slot_name),
        DP_EXCHANGE_BUNDLE,
        entry_name,
        world_size=2,
        backend="gloo",
        device="cpu",
        shapes=SMALL_SHAPES,
        seed=0,
    )
    assert result.passed, "\n".join(
        f"{row.shape}: {row.detail}" for row in result.shape_results
    )


def test_collective_cpu_verify_does_not_claim_graph_proof():
    res = _verify()
    assert res.passed
    assert res.graph_required
    assert not res.graph_verified
    assert not res.fully_verified
    assert all(result.graph_replays == 0 for result in res.shape_results)
    assert all(
        result.phase_outcome.capture is PhaseDisposition.INFRASTRUCTURE_FAILED
        and not result.phase_outcome.observation_complete
        and not result.phase_outcome.failure_is_candidate_attributable
        for result in res.shape_results
    )


def test_non_reducing_kernel_fails_gloo_cpu(tmp_path):
    # A "reduce" that returns the LOCAL partial (forgets to sum across ranks) must fail:
    # out = x_rank != sum_r(x_r). Distributed verify is what catches this — a single-rank
    # check never would.
    broken = tmp_path / "broken_allreduce.py"
    broken.write_text("def all_reduce(x, out, group=None):\n    out.copy_(x)  # BUG: no cross-rank sum\n")
    slot = get_slot("collective.all_reduce")
    res = verify_collective(slot, str(broken), "all_reduce",
                            world_size=2, backend="gloo", device="cpu", seed=0)
    assert not res.passed


def test_candidate_cannot_poison_its_own_trusted_reference(tmp_path):
    poisoned = tmp_path / "poison_reference.py"
    poisoned.write_text(
        "def all_reduce(x, out, group=None):\n"
        "    x.zero_()\n"
        "    out.zero_()\n"
    )

    result = _verify(poisoned)

    assert not result.passed
    assert result.shape_results[0].detail


def test_candidate_cannot_rebind_collective_input_to_equal_storage(tmp_path):
    rebinding = tmp_path / "rebind_input.py"
    rebinding.write_text(
        "import torch.distributed as dist\n"
        "def all_reduce(x, out, group=None):\n"
        "    replacement = x.detach().clone()\n"
        "    x.set_(replacement)\n"
        "    out.copy_(x)\n"
        "    dist.all_reduce(out, group=group)\n"
    )

    result = _verify(rebinding)

    assert not result.passed
    assert "validator-owned storage/tensor binding" in result.shape_results[0].detail


def test_candidate_cannot_replace_collective_output_storage(tmp_path):
    replacing = tmp_path / "replace_output.py"
    replacing.write_text(
        "import torch\n"
        "def all_reduce(x, out, group=None):\n"
        "    replacement = torch.empty_like(out)\n"
        "    out.set_(replacement)\n"
        "    out.copy_(x)\n"
    )

    result = _verify(replacing)

    assert not result.passed
    assert "validator-owned storage" in result.shape_results[0].detail


def test_candidate_cannot_change_collective_output_strides(tmp_path):
    restriding = tmp_path / "restride_output.py"
    restriding.write_text(
        "import torch.distributed as dist\n"
        "def all_reduce(x, out, group=None):\n"
        "    expected = x.clone()\n"
        "    dist.all_reduce(expected, group=group)\n"
        "    out.as_strided_(out.shape, (1, out.shape[0]))\n"
        "    out.copy_(expected)\n"
    )

    result = _verify(restriding)

    assert not result.passed
    assert "validator-owned storage/tensor binding" in result.shape_results[0].detail


def test_collective_dtype_and_topology_are_truthful():
    result = _verify(dtype_name="float64")
    assert result.passed, result.shape_results[0].detail
    assert result.dtype == "float64"
    assert all(row.dtype == "float64" for row in result.shape_results)
    case = result.shape_results[0].case_descriptor
    assert case is not None
    assert case.case_kind is VerificationCaseKind.COLLECTIVE_SINGLE
    call = dict(case.calls[0])
    assert call["dtype"] == "float64"
    assert call["architecture"] == "cpu"
    assert call["tp_size"] == 2 and call["world_size"] == 2
    assert call["graph_mode"] == "eager"

    with pytest.raises(ValueError, match="world_size >= 2"):
        _verify(world_size=1)
    with pytest.raises(ValueError, match="tp_size must equal"):
        _verify(tp_size=4)
    with pytest.raises(ValueError, match="floating torch dtype"):
        _verify(dtype_name="int32")


def test_collective_temporal_descriptor_commits_explicit_sequence_order():
    result = _verify(
        shapes=[
            {"num_tokens": 2, "hidden": 8},
            {"num_tokens": 5, "hidden": 8},
        ]
    )
    temporal = [
        row.case_descriptor
        for row in result.shape_results
        if row.case_descriptor is not None
        and row.case_descriptor.case_kind
        is VerificationCaseKind.COLLECTIVE_TEMPORAL_EAGER
    ]
    assert len(temporal) == 1
    case = temporal[0]
    assert len(case.calls) > 2
    observed_order = tuple(dict(call)["num_tokens"] for call in case.calls)
    assert observed_order[:3] == (5, 2, 5)
    rotated_calls = case.calls[1:] + case.calls[:1]
    rebound = VerificationCaseDescriptor(
        case.slot_id,
        case.variant_id,
        case.case_kind,
        rotated_calls,
    )
    assert rebound.digest != case.digest
    mismatched = list(case.calls)
    changed = dict(mismatched[1])
    changed["architecture"] = "different-architecture"
    mismatched[1] = CallDescriptor(changed)
    with pytest.raises(ValueError, match="disagree on sealed execution context"):
        VerificationCaseDescriptor(
            case.slot_id,
            case.variant_id,
            case.case_kind,
            tuple(mismatched),
        )


def test_collective_eligibility_routes_off_context_to_na_before_import(tmp_path):
    marker = tmp_path / "imported"
    source = tmp_path / "must_not_import.py"
    source.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported')\n"
        "def all_reduce(x, out, group=None):\n"
        "    raise AssertionError('off-domain entry ran')\n"
    )
    result = _verify(
        source,
        eligibility=Eligibility(
            architectures=frozenset({"sm103"}), min_num_tokens=8,
        ),
        bundle_path=str(tmp_path / "missing_bundle"),
    )

    assert result.context_inapplicable
    assert not result.passed
    assert result.num_applicable == 0
    assert result.num_not_applicable == 1
    assert not marker.exists()


def test_collective_eligibility_runs_only_matching_shapes():
    result = _verify(
        shapes=[
            {"num_tokens": 1, "hidden": 8},
            {"num_tokens": 8, "hidden": 8},
        ],
        eligibility=Eligibility(max_num_tokens=2),
    )

    assert result.passed
    assert result.coverage_sufficient
    assert result.num_applicable == 1
    assert result.num_not_applicable == 1


def test_collective_graph_replay_and_timeout_arguments_fail_closed():
    with pytest.raises(ValueError, match="at least two"):
        _verify(graph_replays=1)
    for timeout in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite positive"):
            _verify(timeout_s=timeout)


def test_collective_verify_watchdog_terminates_hung_ranks(tmp_path):
    hanging = tmp_path / "hanging_allreduce.py"
    hanging.write_text(
        "def all_reduce(x, out, group=None):\n"
        "    while True:\n"
        "        pass\n"
    )
    started = time.monotonic()
    result = _verify(hanging, timeout_s=1.0)

    assert not result.passed
    assert "timed out" in result.shape_results[0].detail
    outcome = result.shape_results[0].phase_outcome
    assert outcome.eager is PhaseDisposition.INFRASTRUCTURE_FAILED
    assert not outcome.observation_complete
    assert not outcome.failure_is_candidate_attributable
    assert time.monotonic() - started < 20


def test_collective_verify_rejects_abrupt_rank_exit(tmp_path):
    abrupt = tmp_path / "abrupt.py"
    abrupt.write_text(
        "import os\n"
        "def all_reduce(x, out, group=None):\n"
        "    os._exit(7)\n"
    )
    result = _verify(abrupt, timeout_s=10.0)

    assert not result.passed
    assert result.shape_results[0].detail


def test_collective_valid_json_cannot_hide_nonzero_worker_exit(tmp_path):
    source = tmp_path / "exit_after_verdict.py"
    source.write_text(
        "import os\n"
        "import torch.distributed as dist\n"
        "import cacheon.verify_collective as verifier\n"
        "original_write = verifier._write_rank_verdict\n"
        "def exit_after_write(*args, **kwargs):\n"
        "    original_write(*args, **kwargs)\n"
        "    os._exit(7)\n"
        "verifier._write_rank_verdict = exit_after_write\n"
        "def all_reduce(x, out, group=None):\n"
        "    out.copy_(x)\n"
        "    dist.all_reduce(out, group=group)\n"
    )
    result = _verify(source, timeout_s=10.0)

    assert not result.passed
    assert "worker" in result.shape_results[0].detail


def _valid_verdict(
    *, rank=0, world_size=2, phase_outcome=None, version=_VERDICT_VERSION
):
    outcome = phase_outcome or GraphPhaseOutcome.graph_passed(3)
    return _RankVerdict(
        version=version,
        rank=rank,
        world_size=world_size,
        passed=True,
        score=1.0,
        max_abs=0.0,
        detail="",
        metric="ratio",
        error=None,
        graph_replays=outcome.replay_count,
        phase_outcome=outcome,
    )


def _precreated(path: Path):
    path.touch(mode=0o600, exist_ok=False)
    return _regular_identity(path)


def test_collective_rank_verdict_round_trip(tmp_path):
    path = tmp_path / "rank0.json"
    identity = _precreated(path)
    expected = _valid_verdict()
    _write_rank_verdict(path, expected, identity)

    assert _read_rank_verdict(
        path,
        expected_rank=0,
        expected_world_size=2,
        expected_identity=identity,
    ) == expected


def test_collective_rank_verdict_rejects_legacy_and_mixed_wire(tmp_path):
    new_path = tmp_path / "rank0.json"
    old_path = tmp_path / "rank1.json"
    new_identity = _precreated(new_path)
    old_identity = _precreated(old_path)
    _write_rank_verdict(new_path, _valid_verdict(rank=0), new_identity)
    legacy = _valid_verdict(rank=1, version=1)
    _write_rank_verdict(old_path, legacy, old_identity)

    assert _read_rank_verdict(
        new_path,
        expected_rank=0,
        expected_world_size=2,
        expected_identity=new_identity,
    ).version == _VERDICT_VERSION
    with pytest.raises(CollectiveVerdictError):
        _read_rank_verdict(
            old_path,
            expected_rank=1,
            expected_world_size=2,
            expected_identity=old_identity,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda phase: {**phase, "capture": "UNKNOWN"},
        lambda phase: {**phase, "replay_count": 2},
        lambda phase: {**phase, "replay": "CANDIDATE_FAILED"},
        lambda phase: {key: value for key, value in phase.items() if key != "eager"},
        lambda phase: {**phase, "observation_complete": 1},
    ],
)
def test_collective_rank_verdict_rejects_malformed_typed_phase(tmp_path, mutate):
    path = tmp_path / "rank0.json"
    identity = _precreated(path)
    _write_rank_verdict(path, _valid_verdict(), identity)
    value = json.loads(path.read_text())
    value["phase_outcome"] = mutate(value["phase_outcome"])
    path.write_text(json.dumps(value, separators=(",", ":")))

    with pytest.raises(CollectiveVerdictError):
        _read_rank_verdict(
            path,
            expected_rank=0,
            expected_world_size=2,
            expected_identity=identity,
        )


def test_collective_phase_aggregation_is_causal_and_conservative():
    missing = conservative_phase_aggregate(
        (GraphPhaseOutcome.graph_passed(3),), expected_count=2
    )
    assert missing.eager is PhaseDisposition.INFRASTRUCTURE_FAILED
    assert not missing.observation_complete
    assert not missing.failure_is_candidate_attributable

    replay_disagreement = conservative_phase_aggregate(
        (
            GraphPhaseOutcome.graph_passed(3),
            GraphPhaseOutcome.replay_candidate_failed(2),
        ),
        expected_count=2,
    )
    assert replay_disagreement.eager_passed
    assert replay_disagreement.capture_succeeded
    assert replay_disagreement.replay is PhaseDisposition.CANDIDATE_FAILED
    assert replay_disagreement.replay_count == 2
    assert replay_disagreement.failure_is_candidate_attributable

    eager_precedes_replay = conservative_phase_aggregate(
        (
            GraphPhaseOutcome.eager_candidate_failed(),
            GraphPhaseOutcome.replay_candidate_failed(1),
        ),
        expected_count=2,
    )
    assert eager_precedes_replay.eager is PhaseDisposition.CANDIDATE_FAILED
    assert eager_precedes_replay.replay is PhaseDisposition.NOT_RUN

    count_disagreement = conservative_phase_aggregate(
        (
            GraphPhaseOutcome.graph_passed(2),
            GraphPhaseOutcome.graph_passed(3),
        ),
        expected_count=2,
    )
    assert count_disagreement.replay is PhaseDisposition.INFRASTRUCTURE_FAILED
    assert count_disagreement.replay_count == 2
    assert not count_disagreement.failure_is_candidate_attributable

    mixed_infrastructure = conservative_phase_aggregate(
        (
            GraphPhaseOutcome.replay_candidate_failed(1),
            GraphPhaseOutcome.replay_infrastructure_failed(1),
        ),
        expected_count=2,
    )
    assert mixed_infrastructure.replay is PhaseDisposition.INFRASTRUCTURE_FAILED
    assert not mixed_infrastructure.observation_complete
    assert not mixed_infrastructure.failure_is_candidate_attributable


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.replace('"rank":0', '"rank":0,"rank":0'),
        lambda value: value.replace('"score":1.0', '"score":NaN'),
        lambda value: value[:-1] + ',"extra":1}',
        lambda value: value.replace('"rank":0', '"rank":true'),
    ],
)
def test_collective_rank_verdict_rejects_malformed_json(tmp_path, mutation):
    path = tmp_path / "rank0.json"
    identity = _precreated(path)
    payload = json.dumps(
        {
            "version": _VERDICT_VERSION,
            "rank": 0,
            "world_size": 2,
            "passed": True,
            "score": 1.0,
            "max_abs": 0.0,
            "detail": "",
            "metric": "ratio",
            "error": None,
            "graph_replays": 3,
            "phase_outcome": {
                "eager": "PASSED",
                "capture": "PASSED",
                "replay": "PASSED",
                "replay_count": 3,
                "observation_complete": True,
            },
        },
        separators=(",", ":"),
    )
    path.write_text(mutation(payload))

    with pytest.raises(CollectiveVerdictError):
        _read_rank_verdict(
            path,
            expected_rank=0,
            expected_world_size=2,
            expected_identity=identity,
        )


def test_collective_rank_verdict_rejects_rank_spoof_and_oversize(tmp_path):
    path = tmp_path / "rank0.json"
    identity = _precreated(path)
    _write_rank_verdict(path, _valid_verdict(rank=1), identity)
    with pytest.raises(CollectiveVerdictError, match="identity mismatch"):
        _read_rank_verdict(
            path,
            expected_rank=0,
            expected_world_size=2,
            expected_identity=identity,
        )

    path.write_bytes(b"x" * (_MAX_VERDICT_BYTES + 1))
    with pytest.raises(CollectiveVerdictError, match="size"):
        _read_rank_verdict(
            path,
            expected_rank=0,
            expected_world_size=2,
            expected_identity=identity,
        )


def test_collective_rank_verdict_rejects_replaced_inode_and_symlink(tmp_path):
    path = tmp_path / "rank0.json"
    identity = _precreated(path)
    original_fd = os.open(path, os.O_RDONLY)
    try:
        path.unlink()
        path.write_text("{}")
        with pytest.raises(CollectiveVerdictError, match="inode changed"):
            _read_rank_verdict(
                path,
                expected_rank=0,
                expected_world_size=2,
                expected_identity=identity,
            )
    finally:
        os.close(original_fd)

    path.unlink()
    target = tmp_path / "target"
    target.write_text("{}")
    path.symlink_to(target)
    with pytest.raises(CollectiveVerdictError, match="regular file"):
        _read_rank_verdict(
            path,
            expected_rank=0,
            expected_world_size=2,
            expected_identity=identity,
        )


def test_collective_rank_parser_never_executes_pickle_payload(tmp_path):
    marker = tmp_path / "pickle_executed"

    class Payload:
        def __reduce__(self):
            return Path.write_text, (marker, "executed")

    path = tmp_path / "rank0.json"
    identity = _precreated(path)
    path.write_bytes(pickle.dumps(Payload()))
    with pytest.raises(CollectiveVerdictError):
        _read_rank_verdict(
            path,
            expected_rank=0,
            expected_world_size=2,
            expected_identity=identity,
        )
    assert not marker.exists()


def test_verify_entry_rejects_collective():
    # Collective slots must be verified distributed, not via the single-process verify_entry.
    from cacheon.verify import verify_entry

    slot = get_slot("collective.all_reduce")
    with pytest.raises(ValueError, match="collective"):
        verify_entry(slot, lambda *a, **k: None, device="cpu")
