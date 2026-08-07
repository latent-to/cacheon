"""CPU contracts for the prepared B300 focused-graph worker driver."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from cacheon.eval import b300_prepared_graph_probe as probe
from cacheon.eval.b300_prepared_graph_probe import (
    PreparedGraphProbeError,
    PreparedGraphProbeIncompleteError,
    PreparedGraphProbePolicy,
    PreparedGraphProbeRequest,
    PreparedGraphVariantAuthority,
    execute_prepared_graph_probe,
)
from cacheon.eval.b300_qualification_graph_provider import (
    B300QualificationGraphArtifact,
    B300QualificationGraphBinding,
)
from cacheon.manifest import AOTExport, Manifest, OpEntry, load_manifest
from cacheon.verification_outcomes import (
    GraphPhaseOutcome,
    ShapeResult,
    VerificationCaseDescriptor,
    VerificationCaseKind,
    VerifyResult,
)
from tests.test_b300_qualification_graph_provider import _profile
from tests.test_marginal_runtime import FUSED, SILU


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _policy(profile, **changes) -> PreparedGraphProbePolicy:
    hardware = profile.prepared.launch.hardware
    values = {
        "verification_policy_digest": _h("prepared graph policy"),
        "expected_graph_replays": 3,
        "dtype_name": "bfloat16",
        "architecture": hardware.architecture,
        "tp_size": hardware.tp_size,
        "world_size": hardware.tp_size,
        "graph_mode": "cuda_graph",
        "model_profile_key": "MiniMax-M3",
        "seed": 17,
        "jitter_seed": 29,
        "collective_timeout_s": 41,
    }
    values.update(changes)
    return PreparedGraphProbePolicy(**values)


def _request(profile, **policy_changes) -> PreparedGraphProbeRequest:
    binding = B300QualificationGraphBinding.derive(
        profile.candidate, profile.prepared
    )
    return PreparedGraphProbeRequest.derive(
        binding, profile.candidate, profile.prepared, _policy(profile, **policy_changes)
    )


@pytest.fixture
def singleton(tmp_path: Path):
    source = tmp_path / "singleton-source"
    shutil.copytree(
        SILU,
        source,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return _profile(tmp_path / "singleton", source, "prepared-singleton")


@pytest.fixture
def atomic(tmp_path: Path):
    source = tmp_path / "atomic-source"
    shutil.copytree(
        FUSED,
        source,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return _profile(tmp_path / "atomic", source, "prepared-atomic")


def _descriptor(
    policy: PreparedGraphProbePolicy,
    slot: str,
    variant: str,
    kind: VerificationCaseKind,
    *,
    ordinal: int = 0,
) -> VerificationCaseDescriptor:
    graph_mode = (
        "eager"
        if kind is VerificationCaseKind.COLLECTIVE_TEMPORAL_EAGER
        else policy.graph_mode
    )
    call = {
        "architecture": policy.architecture,
        "case": ordinal,
        "dtype": policy.dtype_name,
        "graph_mode": graph_mode,
        "tp_size": policy.tp_size,
        "world_size": policy.world_size,
    }
    calls = (call, {**call, "case": ordinal + 1}) if kind in {
        VerificationCaseKind.COLLECTIVE_TEMPORAL_EAGER,
        VerificationCaseKind.COLLECTIVE_GRAPH_SEQUENCE,
    } else (call,)
    return VerificationCaseDescriptor.from_call_dicts(
        slot_id=slot,
        variant_id=variant,
        case_kind=kind,
        calls=calls,
    )


def _shape(
    policy: PreparedGraphProbePolicy,
    slot: str,
    variant: str,
    outcome: GraphPhaseOutcome,
    *,
    kind: VerificationCaseKind,
    ordinal: int = 0,
    applicable: bool = True,
) -> ShapeResult:
    passed = not applicable or (
        outcome.eager_passed and outcome.capture_succeeded and outcome.replay_passed
    )
    if kind is VerificationCaseKind.COLLECTIVE_TEMPORAL_EAGER:
        passed = outcome == GraphPhaseOutcome.eager_only_passed()
    return ShapeResult(
        shape={"ordinal": ordinal},
        dtype=policy.dtype_name,
        passed=passed,
        max_abs_err=0.0,
        max_rel_err=0.0,
        graph_replays=outcome.replay_count,
        applicable=applicable,
        phase_outcome=outcome,
        case_descriptor=_descriptor(policy, slot, variant, kind, ordinal=ordinal),
    )


def _result(
    policy: PreparedGraphProbePolicy,
    slot: str,
    variant: str,
    *,
    collective: bool = False,
    outcome: GraphPhaseOutcome | None = None,
    not_applicable: bool = False,
    temporal_outcome: GraphPhaseOutcome | None = None,
) -> VerifyResult:
    if not_applicable:
        rows = [
            _shape(
                policy,
                slot,
                variant,
                GraphPhaseOutcome.not_applicable(),
                kind=(
                    VerificationCaseKind.COLLECTIVE_SINGLE
                    if collective
                    else VerificationCaseKind.ORDINARY_SINGLE
                ),
                applicable=False,
            )
        ]
        passed, graph_verified = False, False
    else:
        outcome = outcome or GraphPhaseOutcome.graph_passed(
            policy.expected_graph_replays
        )
        rows = [
            _shape(
                policy,
                slot,
                variant,
                outcome,
                kind=(
                    VerificationCaseKind.COLLECTIVE_SINGLE
                    if collective
                    else VerificationCaseKind.ORDINARY_SINGLE
                ),
            )
        ]
        if collective:
            rows.extend(
                (
                    _shape(
                        policy,
                        slot,
                        variant,
                        temporal_outcome or GraphPhaseOutcome.eager_only_passed(),
                        kind=VerificationCaseKind.COLLECTIVE_TEMPORAL_EAGER,
                        ordinal=10,
                    ),
                    _shape(
                        policy,
                        slot,
                        variant,
                        outcome,
                        kind=VerificationCaseKind.COLLECTIVE_GRAPH_SEQUENCE,
                        ordinal=20,
                    ),
                )
            )
        passed = all(row.passed for row in rows)
        graph_verified = passed
    return VerifyResult(
        slot=slot,
        dtype=policy.dtype_name,
        passed=passed,
        shape_results=rows,
        graph_required=True,
        graph_verified=graph_verified,
        coverage_required=1,
        context_inapplicable=not_applicable,
        domain_coverage_complete=True,
    )


def test_request_is_canonical_path_free_restart_stable_and_drift_closed(singleton) -> None:
    request = _request(singleton)
    payload = request.canonical_bytes
    reopened = PreparedGraphProbeRequest.from_canonical_bytes(payload)

    assert reopened == request
    assert reopened.canonical_bytes == payload
    assert singleton.prepared.binding.tree.root.as_posix().encode() not in payload
    assert request.digest == reopened.digest

    value = json.loads(payload)
    value["policy"]["seed"] += 1
    with pytest.raises(PreparedGraphProbeError, match="policy digest"):
        PreparedGraphProbeRequest.from_canonical_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        )
    with pytest.raises(PreparedGraphProbeError, match="canonical"):
        PreparedGraphProbeRequest.from_canonical_bytes(payload + b"\n")


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"expected_graph_replays": 1}, "replays"),
        ({"graph_mode": "eager"}, "cuda_graph"),
        ({"architecture": "sm999"}, "hardware"),
        ({"tp_size": 2, "world_size": 2}, "hardware"),
        ({"tp_size": 1, "world_size": 2}, "world_size"),
        ({"collective_timeout_s": 0}, "timeout"),
    ],
)
def test_policy_rejects_wrong_replays_architecture_tp_world_graph_mode_or_timeout(
    singleton, changes, message
) -> None:
    if changes.get("expected_graph_replays") == 1 or changes.get("graph_mode") == "eager" or changes.get("tp_size") == 1:
        with pytest.raises(PreparedGraphProbeError, match=message):
            _policy(singleton, **changes)
    else:
        with pytest.raises(PreparedGraphProbeError, match=message):
            _request(singleton, **changes)


def test_request_rejects_missing_extra_duplicate_and_reordered_variant_authority(singleton) -> None:
    request = _request(singleton)
    only = request.target_variants[0]
    extra = PreparedGraphVariantAuthority(only.slot_id, "second")
    with pytest.raises(PreparedGraphProbeError, match="target.*authority"):
        replace(request, target_variants=())
    with pytest.raises(PreparedGraphProbeError, match="canonical"):
        replace(request, target_variants=(only, only))
    with pytest.raises(PreparedGraphProbeError, match="canonical"):
        replace(request, target_variants=(extra, only))

    manifest = load_manifest(singleton.prepared.binding.tree.root)
    missing = replace(manifest, ops=())
    with pytest.raises(PreparedGraphProbeError, match="missing target member"):
        probe._target_variant_authority(request.binding, missing)
    duplicate = replace(manifest, ops=manifest.ops + (manifest.ops[0],))
    with pytest.raises(PreparedGraphProbeError, match="repeats"):
        probe._target_variant_authority(request.binding, duplicate)
    second_op = replace(manifest.ops[0], variant="z")
    reordered = replace(manifest, ops=(second_op, manifest.ops[0]))
    with pytest.raises(PreparedGraphProbeError, match="reorders"):
        probe._target_variant_authority(request.binding, reordered)

    extra_request = replace(request, target_variants=(only, extra))
    with pytest.raises(PreparedGraphProbeError, match="differs from request"):
        execute_prepared_graph_probe(
            extra_request, singleton.prepared.binding.tree.root
        )


def test_singleton_execution_routes_exact_ordinary_kwargs_and_returns_restart_bytes(
    singleton, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(singleton)
    calls = []

    def ordinary(slot, source, entry, **kwargs):
        calls.append((slot, source, entry, kwargs))
        return _result(request.policy, slot, kwargs["variant_name"])

    monkeypatch.setattr(probe, "_VERIFY_ENTRY_FROM_SOURCE", ordinary)
    artifact = execute_prepared_graph_probe(
        request, singleton.prepared.binding.tree.root
    )
    assert type(artifact) is B300QualificationGraphArtifact
    assert artifact.binding == request.binding
    assert artifact.verification_policy_digest == request.policy.verification_policy_digest
    assert artifact.canonical_bytes == B300QualificationGraphArtifact.from_canonical_bytes(
        artifact.canonical_bytes
    ).canonical_bytes
    assert len(calls) == 1
    slot, source, _entry, kwargs = calls[0]
    op = load_manifest(singleton.prepared.binding.tree.root).ops[0]
    assert slot == op.slot
    assert source == str(singleton.prepared.binding.tree.root / op.source)
    assert kwargs == {
        "prepare_name": op.prepare,
        "dtype_name": request.policy.dtype_name,
        "device": "cuda",
        "seed": request.policy.seed,
        "jitter_seed": request.policy.jitter_seed,
        "model_key": request.policy.model_profile_key,
        "override_point": op.override_point,
        "graph_safe": True,
        "graph_replays": request.policy.expected_graph_replays,
        "eligibility_metadata": json.loads(
            (singleton.prepared.binding.tree.root / op.metadata).read_text()
        ),
        "manifest_dtypes": op.dtypes,
        "manifest_architectures": op.architectures,
        "tp_size": request.policy.tp_size,
        "world_size": request.policy.world_size,
        "bundle_path": str(singleton.prepared.binding.tree.root),
        "variant_name": op.variant,
    }


def test_atomic_execution_routes_both_collectives_and_includes_graph_sequence_only(
    atomic, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(atomic)
    calls = []

    def collective(slot, source, entry, **kwargs):
        calls.append((slot, source, entry, kwargs))
        return _result(
            request.policy,
            slot.name,
            kwargs["variant_name"],
            collective=True,
        )

    monkeypatch.setattr(probe, "_VERIFY_COLLECTIVE", collective)
    artifact = execute_prepared_graph_probe(request, atomic.prepared.binding.tree.root)

    assert tuple(row.slot_id for row in artifact.variants) == request.binding.target_members
    assert len(calls) == 2
    for (slot, source, _entry, kwargs), record in zip(calls, artifact.variants):
        assert slot.kind == "collective"
        assert Path(source).is_file()
        assert kwargs["backend"] == "nccl"
        assert kwargs["device"] == "cuda"
        assert kwargs["world_size"] == request.policy.world_size
        assert kwargs["tp_size"] == request.policy.tp_size
        assert kwargs["timeout_s"] == float(request.policy.collective_timeout_s)
        assert kwargs["graph_replays"] == request.policy.expected_graph_replays
        assert kwargs["bundle_path"] == str(atomic.prepared.binding.tree.root)
        assert type(kwargs["eligibility"]).__name__ == "Eligibility"
        # The temporal-eager row is a gate, while single + sequence are evidence.
        assert len(record.shapes) == 2
        kinds = {
            row.case_descriptor.case_kind
            for row in _result(
                request.policy, record.slot_id, record.variant_id, collective=True
            ).shape_results
            if row.case_descriptor.digest
            in {shape.descriptor_digest for shape in record.shapes}
        }
        assert kinds == {
            VerificationCaseKind.COLLECTIVE_SINGLE,
            VerificationCaseKind.COLLECTIVE_GRAPH_SEQUENCE,
        }


def test_no_target_literal_or_shape_detail_parsing_in_production_source() -> None:
    source = Path(probe.__file__).read_text()
    assert "collective.ar_residual_rmsnorm" not in source
    assert "activation.silu_and_mul" not in source
    assert "row.detail" not in source
    assert "row.shape" not in source


def test_not_applicable_becomes_complete_nonapplicable(singleton, monkeypatch) -> None:
    request = _request(singleton)
    monkeypatch.setattr(
        probe,
        "_VERIFY_ENTRY_FROM_SOURCE",
        lambda slot, _source, _entry, **kwargs: _result(
            request.policy, slot, kwargs["variant_name"], not_applicable=True
        ),
    )
    artifact = execute_prepared_graph_probe(request, singleton.prepared.binding.tree.root)
    shape = artifact.variants[0].shapes[0]
    assert not shape.applicable
    assert shape.observation_complete
    assert not shape.failure_is_candidate_attributable
    assert shape.replay_count == 0


def test_collective_not_applicable_needs_no_synthetic_sequence_rows(atomic, monkeypatch) -> None:
    request = _request(atomic)
    monkeypatch.setattr(
        probe,
        "_VERIFY_COLLECTIVE",
        lambda slot, _source, _entry, **kwargs: _result(
            request.policy,
            slot.name,
            kwargs["variant_name"],
            collective=True,
            not_applicable=True,
        ),
    )
    artifact = execute_prepared_graph_probe(request, atomic.prepared.binding.tree.root)
    assert all(not shape.applicable for row in artifact.variants for shape in row.shapes)


@pytest.mark.parametrize(
    "outcome",
    [
        GraphPhaseOutcome.eager_candidate_failed(),
        GraphPhaseOutcome.capture_candidate_failed(),
        GraphPhaseOutcome.replay_candidate_failed(0),
        GraphPhaseOutcome.replay_candidate_failed(2),
    ],
)
def test_complete_candidate_eager_capture_or_replay_failure_is_structured_evidence(
    singleton, monkeypatch, outcome
) -> None:
    request = _request(singleton)
    monkeypatch.setattr(
        probe,
        "_VERIFY_ENTRY_FROM_SOURCE",
        lambda slot, _source, _entry, **kwargs: _result(
            request.policy, slot, kwargs["variant_name"], outcome=outcome
        ),
    )
    artifact = execute_prepared_graph_probe(request, singleton.prepared.binding.tree.root)
    shape = artifact.variants[0].shapes[0]
    assert shape.failure_is_candidate_attributable
    assert shape.observation_complete


@pytest.mark.parametrize(
    "outcome",
    [
        GraphPhaseOutcome.infrastructure_before_eager(),
        GraphPhaseOutcome.capture_infrastructure_failed(),
        GraphPhaseOutcome.replay_infrastructure_failed(1),
        GraphPhaseOutcome.unobserved(),
    ],
)
def test_infrastructure_or_incomplete_shape_aborts_publication(
    singleton, monkeypatch, outcome
) -> None:
    request = _request(singleton)
    monkeypatch.setattr(
        probe,
        "_VERIFY_ENTRY_FROM_SOURCE",
        lambda slot, _source, _entry, **kwargs: _result(
            request.policy, slot, kwargs["variant_name"], outcome=outcome
        ),
    )
    with pytest.raises(PreparedGraphProbeIncompleteError) as raised:
        execute_prepared_graph_probe(request, singleton.prepared.binding.tree.root)
    assert raised.value.decision == "HOLD"


def test_collective_temporal_failure_is_hold_not_graph_evidence(atomic, monkeypatch) -> None:
    request = _request(atomic)
    monkeypatch.setattr(
        probe,
        "_VERIFY_COLLECTIVE",
        lambda slot, _source, _entry, **kwargs: _result(
            request.policy,
            slot.name,
            kwargs["variant_name"],
            collective=True,
            temporal_outcome=GraphPhaseOutcome.eager_candidate_failed(),
        ),
    )
    with pytest.raises(PreparedGraphProbeIncompleteError, match="temporal-eager"):
        execute_prepared_graph_probe(request, atomic.prepared.binding.tree.root)


def test_collective_missing_graph_sequence_is_hold(atomic, monkeypatch) -> None:
    request = _request(atomic)

    def without_sequence(slot, _source, _entry, **kwargs):
        result = _result(
            request.policy, slot.name, kwargs["variant_name"], collective=True
        )
        result.shape_results = [
            row
            for row in result.shape_results
            if row.case_descriptor.case_kind
            is not VerificationCaseKind.COLLECTIVE_GRAPH_SEQUENCE
        ]
        return result

    monkeypatch.setattr(probe, "_VERIFY_COLLECTIVE", without_sequence)
    with pytest.raises(PreparedGraphProbeIncompleteError, match="graph-sequence"):
        execute_prepared_graph_probe(request, atomic.prepared.binding.tree.root)


@pytest.mark.parametrize(
    "field, value",
    [
        ("dtype", "float16"),
        ("architecture", "sm999"),
        ("tp_size", 9),
        ("world_size", 9),
        ("graph_mode", "eager"),
    ],
)
def test_typed_descriptor_execution_context_must_match_policy(
    singleton, monkeypatch, field, value
) -> None:
    request = _request(singleton)

    def drifted(slot, _source, _entry, **kwargs):
        result = _result(request.policy, slot, kwargs["variant_name"])
        row = result.shape_results[0]
        descriptor = row.case_descriptor
        call = dict(descriptor.calls[0])
        call[field] = value
        row.case_descriptor = VerificationCaseDescriptor.from_call_dicts(
            slot_id=descriptor.slot_id,
            variant_id=descriptor.variant_id,
            case_kind=descriptor.case_kind,
            calls=(call,),
        )
        return result

    monkeypatch.setattr(probe, "_VERIFY_ENTRY_FROM_SOURCE", drifted)
    with pytest.raises(PreparedGraphProbeError, match="execution context|graph mode"):
        execute_prepared_graph_probe(request, singleton.prepared.binding.tree.root)


def test_verifier_exception_without_typed_outcome_is_hold(singleton, monkeypatch) -> None:
    request = _request(singleton)

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("diagnostic text is not evidence")

    monkeypatch.setattr(probe, "_VERIFY_ENTRY_FROM_SOURCE", unavailable)
    with pytest.raises(PreparedGraphProbeIncompleteError, match="typed evidence") as raised:
        execute_prepared_graph_probe(request, singleton.prepared.binding.tree.root)
    assert raised.value.decision == "HOLD"


def test_wrong_descriptor_replay_count_tree_and_binding_fail_closed(singleton, monkeypatch) -> None:
    request = _request(singleton)

    def wrong_replay(slot, _source, _entry, **kwargs):
        result = _result(request.policy, slot, kwargs["variant_name"])
        result.shape_results[0].graph_replays = 2
        return result

    monkeypatch.setattr(probe, "_VERIFY_ENTRY_FROM_SOURCE", wrong_replay)
    with pytest.raises(PreparedGraphProbeError, match="graph_replays"):
        execute_prepared_graph_probe(request, singleton.prepared.binding.tree.root)

    foreign = tmp_path = singleton.prepared.binding.tree.root.parent / "foreign"
    shutil.copytree(singleton.prepared.binding.tree.root, foreign)
    foreign_source = foreign / load_manifest(foreign).ops[0].source
    foreign_source.chmod(0o644)
    foreign_source.write_text("def changed(): pass\n")
    foreign_source.chmod(0o444)
    with pytest.raises(PreparedGraphProbeError, match="tree"):
        execute_prepared_graph_probe(request, foreign)

    crossed = json.loads(request.canonical_bytes)
    crossed["binding"]["candidate_binding_digest"] = _h("crossed")
    with pytest.raises(PreparedGraphProbeError, match="binding.*digest"):
        PreparedGraphProbeRequest.from_canonical_bytes(
            json.dumps(crossed, sort_keys=True, separators=(",", ":")).encode()
        )


def test_execute_rejects_symlink_root(singleton, tmp_path: Path) -> None:
    request = _request(singleton)
    link = tmp_path / "runtime-link"
    link.symlink_to(singleton.prepared.binding.tree.root, target_is_directory=True)
    with pytest.raises(PreparedGraphProbeError, match="nonsymlink"):
        execute_prepared_graph_probe(request, link)


def test_internal_ordinary_seam_preserves_prepare_override_metadata_and_bundle_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    (root / "kernel.py").write_text("def entry(): pass\n")
    (root / "metadata.json").write_text(
        json.dumps({"graph_safe": True, "dtypes": ["bfloat16"]})
    )
    op = OpEntry(
        "activation.silu_and_mul",
        "kernel.py",
        "entry",
        ("bfloat16",),
        (),
        "metadata.json",
        variant="prepared",
        prepare="prepare",
        base_kernel="base",
        override_point="epilogue",
    )
    policy = PreparedGraphProbePolicy(
        _h("policy"), 3, "bfloat16", "sm103", 1, 1, "cuda_graph",
        "MiniMax-M3", 1, 2, 3,
    )
    seen = {}

    def verifier(slot, source, entry, **kwargs):
        seen.update(slot=slot, source=source, entry=entry, **kwargs)
        return _result(policy, slot, kwargs["variant_name"])

    monkeypatch.setattr(probe, "_VERIFY_ENTRY_FROM_SOURCE", verifier)
    probe._execute_variant(root, op, policy)
    assert seen["prepare_name"] == "prepare"
    assert seen["override_point"] == "epilogue"
    assert seen["eligibility_metadata"] == {
        "graph_safe": True,
        "dtypes": ["bfloat16"],
    }
    assert seen["bundle_path"] == str(root)
    assert seen["variant_name"] == "prepared"


def test_direct_aot_row_stays_on_existing_bundle_aware_validator_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    (root / "kernel.py").write_text("def factory(): pass\n")
    direct = OpEntry(
        "activation.silu_and_mul",
        "kernel.py",
        "factory",
        ("bfloat16",),
        ("sm103",),
        None,
        variant="direct",
        aot_exports=(AOTExport("cute_cubin", "run", "factory", (), (), (), ()),),
    )
    policy = PreparedGraphProbePolicy(
        _h("direct policy"), 3, "bfloat16", "sm103", 1, 1,
        "cuda_graph", "MiniMax-M3", 1, 2, 3,
    )
    seen = {}

    def validator(slot, source, entry, **kwargs):
        seen.update(slot=slot, source=source, entry=entry, **kwargs)
        return _result(policy, slot, kwargs["variant_name"])

    monkeypatch.setattr(probe, "_VERIFY_ENTRY_FROM_SOURCE", validator)
    probe._execute_variant(root, direct, policy)
    assert direct.aot_exports
    assert seen["bundle_path"] == str(root)
    assert seen["variant_name"] == "direct"
    assert seen["prepare_name"] is None
    assert seen["override_point"] is None


def test_multi_variant_singleton_executes_in_sorted_authority_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    shutil.copytree(
        SILU,
        source,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    metadata = source / "metadata"
    (metadata / "small.json").write_text(
        json.dumps({"dtypes": ["bfloat16"], "graph_safe": True, "max_num_tokens": 32})
    )
    (metadata / "large.json").write_text(
        json.dumps({"dtypes": ["bfloat16"], "graph_safe": True, "min_num_tokens": 33})
    )
    (source / "manifest.toml").write_text(
        """bundle_id = "multi-variant-singleton"
abi_version = "cacheon-op-abi-v0"

[[ops]]
slot = "activation.silu_and_mul"
variant = "large"
source = "kernels/silu_and_mul.py"
entry = "silu_and_mul"
dtypes = ["bfloat16"]
metadata = "metadata/large.json"

[[ops]]
slot = "activation.silu_and_mul"
variant = "small"
source = "kernels/silu_and_mul.py"
entry = "silu_and_mul"
dtypes = ["bfloat16"]
metadata = "metadata/small.json"
"""
    )
    profile = _profile(tmp_path / "multi", source, "prepared-multi")
    request = _request(profile)
    calls = []
    monkeypatch.setattr(
        probe,
        "_VERIFY_ENTRY_FROM_SOURCE",
        lambda slot, _source, _entry, **kwargs: (
            calls.append(kwargs["variant_name"])
            or _result(request.policy, slot, kwargs["variant_name"])
        ),
    )
    artifact = execute_prepared_graph_probe(request, profile.prepared.binding.tree.root)
    assert calls == ["large", "small"]
    assert tuple(row.variant_id for row in artifact.variants) == ("large", "small")
