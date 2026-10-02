"""CPU-only contracts for the closed B300 arena provider."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import pytest

from cacheon.arena_service import (
    ArenaCandidateBinding,
    ArenaCapacityPolicy,
    ArenaQualificationRequest,
    ArenaQualificationWork,
    ArenaService,
    ArenaServiceManifest,
    Workload,
    WorkloadCell,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain.publication import publish_worker_bundle
from cacheon.eval.b300_arena_provider import (
    B300ArenaProviderError,
    B300ArenaServiceProvider,
    B300DeclaredAuthorities,
    B300DeclaredQualificationAuthorities,
    B300DeploymentAuthorities,
    B300QualificationLanePair,
    B300QualificationLanePolicy,
    b300_arena_provider_digest,
)
from cacheon.eval.device_state import DeviceStatePolicy
from cacheon.eval.oci_backend import (
    OCIBackendConfig,
    OCIEngineExecutor,
)
from cacheon.eval.oci_prebuild import OCIPrebuildConfig
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationPlanFactory,
    QualificationReservation,
)
from tests.support.b300 import StubHiddenJudge as _Judge, arena_runtime as _runtime, gpu as _gpu, prebuild_policy as _prebuild_policy, runtime_policy as _runtime_policy, sha as _h


SLOT = "activation.silu_and_mul"


@pytest.fixture
def executor_factory(tmp_path: Path):
    executors: list[OCIEngineExecutor] = []
    sequence = 0

    def create(role: str, lane: str = "A") -> OCIEngineExecutor:
        nonlocal sequence
        sequence += 1
        normalized_role = (
            "candidate" if role == "candidate" else "resident_baseline"
        )
        if lane not in {"A", "B"}:
            raise AssertionError("fixture lane must be A or B")
        first_gpu = 0 if lane == "A" else 4
        runtime = _runtime_policy()
        policy = _prebuild_policy(runtime)
        root = tmp_path / f"executor-{sequence}-{role}"
        config = OCIBackendConfig(
            OCIPrebuildConfig(
                docker_binary="/usr/bin/docker",
                recovery_root=root / "recovery",
                publication_root=root / "publications",
                seccomp_profile=root / "seccomp.json",
                executor_id=f"qualification-{normalized_role}",
                policy=policy,
            ),
            runtime,
        )
        executor = OCIEngineExecutor(
            config,
            DeviceStatePolicy(
                expected_gpus=tuple(
                    _gpu(index) for index in range(first_gpu, first_gpu + 4)
                ),
                required_consecutive_idle_samples=2,
                poll_interval_s=0.05,
                ready_poll_interval_s=0.05,
                drain_timeout_s=2.0,
                maximum_samples=8,
            ),
        )
        executors.append(executor)
        return executor

    yield create
    for executor in executors:
        executor.manager.close()


class _FactoryBuilder:
    def __init__(self, *, reverse: bool = False, fail: bool = False) -> None:
        self.reverse = reverse
        self.fail = fail
        self.calls = []

    def __call__(self, request, state):
        self.calls.append((request, state))
        if self.fail:
            raise OSError("private authority store unavailable")
        reservations = tuple(row.reservation for row in request.candidates)
        if self.reverse:
            reservations = tuple(reversed(reservations))
        manifest = QualificationAuthorityManifest(
            "registered",
            _h("qualification-authority"),
            _h("qualification-source"),
            _h("selection-commitment"),
            _h("selection-secret-reference"),
            tuple(row.selected_delta_digest for row in reservations),
            reservations,
        )
        return QualificationPlanFactory(
            manifest,
            lambda _reference: b"s" * 32,
            lambda _secret: None,
        )


def _authorities(executor_factory, *, builder: _FactoryBuilder | None = None):
    factory_builder = builder or _FactoryBuilder()
    policy_digest = _h("qualification-policy")
    candidate_executor = executor_factory("candidate", "A")
    baseline_executor = executor_factory("resident_baseline", "B")
    lane_pair = B300QualificationLanePair(
        B300QualificationLanePolicy.from_device_policy(
            "A", candidate_executor.device_policy
        ),
        B300QualificationLanePolicy.from_device_policy(
            "B", baseline_executor.device_policy
        ),
    )
    authorities = B300DeploymentAuthorities(
        runtime_identity=_runtime(),
        qualification_policy_digest=policy_digest,
        qualification_builder_digest=_h("qualification-builder"),
        qualification_factory_builder=factory_builder,
        executor=candidate_executor,
        resident_baseline_executor=baseline_executor,
        entropy_provider_digest=_h("entropy-provider"),
        entropy_provider=lambda *_args: None,
        hidden_judge=_Judge(),
        deadline_policy_digest=_h("deadline-policy"),
        deadline_provider=lambda _request, _state: time.monotonic() + 600.0,
        qualification_lane_pair=lane_pair,
        qualification_stage="primary",
    )
    return authorities, factory_builder


def _manifest(
    authorities: B300DeploymentAuthorities,
    **changes,
) -> ArenaServiceManifest:
    workload = Workload(
        _h("prompt-corpus"),
        "sealed-prompt-seeds-v1",
        (WorkloadCell("s8", 8192, 1024, 64, 8),),
    )
    values = {
        "runtime": authorities.runtime_identity,
        "workload": workload,
        "capacity": ArenaCapacityPolicy(32, 100, 8, 4),
        "qualification_policy_digest": authorities.qualification_policy_digest,
        "provider_digest": b300_arena_provider_digest(authorities),
    }
    values.update(changes)
    return ArenaServiceManifest(**values)


def _bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    kernels = source / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "entry.py").write_text("def run(x, out):\n    return None\n")
    (kernels / "native.cu").write_text(
        'extern "C" __global__ void cacheon_fixture() {}\n'
    )
    (source / "rebuild.json").write_text('{"steps": []}\n')
    (source / "manifest.toml").write_text(
        "\n".join(
            (
                "bundle_id = 'b300-provider-fixture'",
                "abi_version = 'cacheon-op-abi-v0'",
                "[[ops]]",
                f"slot = '{SLOT}'",
                "source = 'kernels/entry.py'",
                "entry = 'run'",
                "dtypes = ['bfloat16']",
                "cuda_sources = ['kernels/native.cu']",
            )
        )
        + "\n"
    )
    for path in sorted(source.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    source.chmod(0o700)
    return source


def _binding(tmp_path: Path, index: int = 0) -> ArenaCandidateBinding:
    source = _bundle(tmp_path)
    publication = publish_worker_bundle(
        source,
        tmp_path / "publications",
        content_hash(source),
    )
    reservation = QualificationReservation(
        _h(f"reservation-{index}"),
        publication.digest,
        SLOT,
        _h(f"delta-{index}"),
        index,
        f"miner-{index}",
        20,
        index,
        0,
        (SLOT,),
    )
    return ArenaCandidateBinding(reservation, publication, 1)


def test_qualification_preserves_exact_request_order_and_real_authorities(
    tmp_path: Path, executor_factory
) -> None:
    authorities, builder = _authorities(executor_factory)
    manifest = _manifest(authorities)
    service = ArenaService(manifest, B300ArenaServiceProvider(manifest, authorities))
    first = _binding(tmp_path / "first", 0)
    second = _binding(tmp_path / "second", 1)

    work = service.plan_qualification((first, second), state={"attempt": 1})

    assert type(work) is ArenaQualificationWork
    assert type(work.factory) is QualificationPlanFactory
    assert work.factory.manifest.reservations == (
        first.reservation,
        second.reservation,
    )
    assert type(work.executor) is OCIEngineExecutor
    assert type(work.resident_baseline_executor) is OCIEngineExecutor
    assert work.executor is authorities.executor
    assert work.resident_baseline_executor is authorities.resident_baseline_executor
    assert work.entropy_provider is authorities.entropy_provider
    assert work.hidden_judge is authorities.hidden_judge
    assert builder.calls[0][0].candidates == (first, second)
    assert builder.calls[0][1] == {"attempt": 1}


def test_reordered_factory_is_refused(
    tmp_path: Path, executor_factory
) -> None:
    authorities, _builder = _authorities(
        executor_factory,
        builder=_FactoryBuilder(reverse=True),
    )
    manifest = _manifest(authorities)
    service = ArenaService(manifest, B300ArenaServiceProvider(manifest, authorities))
    first = _binding(tmp_path / "first", 0)
    second = _binding(tmp_path / "second", 1)

    with pytest.raises(B300ArenaProviderError, match="cohort order"):
        service.plan_qualification((first, second))


def test_runtime_model_topology_and_policy_must_match_manifest(
    executor_factory,
) -> None:
    authorities, _builder = _authorities(executor_factory)
    runtime_mismatch = dataclasses.replace(
        authorities.runtime_identity,
        model_content_digest=_h("another-model"),
    )
    with pytest.raises(B300ArenaProviderError, match="runtime, model, topology"):
        B300ArenaServiceProvider(
            _manifest(authorities, runtime=runtime_mismatch),
            authorities,
        )
    with pytest.raises(B300ArenaProviderError, match="qualification policy"):
        B300ArenaServiceProvider(
            _manifest(
                authorities,
                qualification_policy_digest=_h("another-policy"),
            ),
            authorities,
        )


def test_provider_digest_binds_builder_runtime_and_qualification_policy(
    executor_factory,
) -> None:
    authorities, _builder = _authorities(executor_factory)
    original = b300_arena_provider_digest(authorities)
    changed_builder = dataclasses.replace(
        authorities,
        qualification_builder_digest=_h("changed-qualification-builder"),
    )
    changed_policy = dataclasses.replace(
        authorities,
        qualification_policy_digest=_h("changed-qualification-policy"),
    )
    changed_runtime = dataclasses.replace(
        authorities,
        runtime_identity=dataclasses.replace(
            authorities.runtime_identity,
            topology_digest=_h("changed-topology"),
        ),
    )

    assert len(
        {
            original,
            b300_arena_provider_digest(changed_builder),
            b300_arena_provider_digest(changed_policy),
            b300_arena_provider_digest(changed_runtime),
        }
    ) == 4


def test_provider_identity_is_stable_across_exact_primary_reproduction_swap(
    executor_factory,
) -> None:
    primary, _builder = _authorities(executor_factory)
    reproduction = dataclasses.replace(
        primary,
        executor=executor_factory("candidate", "B"),
        resident_baseline_executor=executor_factory("resident_baseline", "A"),
        qualification_stage="reproduction",
    )

    assert primary.qualification_orientation.stage == "primary"
    assert primary.qualification_orientation.candidate.lane_id == "A"
    assert primary.qualification_orientation.resident_baseline.lane_id == "B"
    assert reproduction.qualification_orientation.stage == "reproduction"
    assert reproduction.qualification_orientation.candidate.lane_id == "B"
    assert reproduction.qualification_orientation.resident_baseline.lane_id == "A"
    assert primary.qualification == reproduction.qualification
    assert primary.qualification_lane_pair.digest == (
        reproduction.qualification_lane_pair.digest
    )
    assert primary.qualification_lane_pair.role_swap_digest == (
        reproduction.qualification_lane_pair.role_swap_digest
    )
    assert b300_arena_provider_digest(primary) == b300_arena_provider_digest(
        reproduction
    )


def test_lane_pair_rejects_overlap_and_full_authority_rejects_orientation_drift(
    executor_factory,
) -> None:
    primary, _builder = _authorities(executor_factory)
    with pytest.raises(B300ArenaProviderError, match="overlapping"):
        B300QualificationLanePair(
            primary.qualification_lane_pair.lane_a,
            dataclasses.replace(
                primary.qualification_lane_pair.lane_a,
                lane_id="B",
            ),
        )

    with pytest.raises(B300ArenaProviderError, match="selected physical TP4 lane"):
        dataclasses.replace(
            primary,
            executor=executor_factory("candidate", "B"),
        )
    with pytest.raises(B300ArenaProviderError, match="selected physical TP4 lane"):
        dataclasses.replace(
            primary,
            qualification_stage="reproduction",
        )

    primary.executor.device_policy = executor_factory(
        "candidate", "B"
    ).device_policy
    with pytest.raises(B300ArenaProviderError, match="selected physical TP4 lane"):
        b300_arena_provider_digest(primary)


def test_declared_and_full_authorities_share_exact_provider_identity(
    tmp_path: Path, executor_factory
) -> None:
    full, builder = _authorities(executor_factory)
    declared = full.qualification
    assert type(declared) is B300DeclaredQualificationAuthorities
    sealed = B300DeclaredAuthorities(full.runtime_identity, declared)
    manifest = _manifest(full)

    assert b300_arena_provider_digest(sealed) == b300_arena_provider_digest(full)
    assert manifest.provider_digest == b300_arena_provider_digest(sealed)

    # The sealed declaration identifies the service but grants no capability.
    request = ArenaQualificationRequest(
        manifest.digest,
        manifest.qualification_policy_digest,
        (_binding(tmp_path / "candidate"),),
    )
    with pytest.raises(B300ArenaProviderError, match="declared-only provider"):
        B300ArenaServiceProvider(manifest, sealed).build_qualification(request)
    assert builder.calls == []


def test_closed_provider_refuses_qualification(
    tmp_path: Path, executor_factory
) -> None:
    authorities, builder = _authorities(executor_factory)
    manifest = _manifest(authorities)
    provider = B300ArenaServiceProvider(manifest, authorities)
    service = ArenaService(manifest, provider)
    provider.close()

    with pytest.raises(B300ArenaProviderError, match="provider is closed"):
        service.plan_qualification((_binding(tmp_path / "candidate"),))
    assert builder.calls == []


def test_factory_exception_stays_a_provider_error(
    tmp_path: Path, executor_factory
) -> None:
    authorities, _builder = _authorities(
        executor_factory,
        builder=_FactoryBuilder(fail=True),
    )
    manifest = _manifest(authorities)
    service = ArenaService(manifest, B300ArenaServiceProvider(manifest, authorities))
    candidate = _binding(tmp_path / "candidate")

    with pytest.raises(B300ArenaProviderError, match="factory construction"):
        service.plan_qualification((candidate,))
