"""CPU-only contracts for registered-target B300 qualification composition."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

import cacheon.eval.b300_qualification_deployment as deployment
import tests.test_b300_sealed_qualification_commission as authority_fixtures
from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaCandidateBinding,
    ArenaCapacityPolicy,
    ArenaQualificationRequest,
    ArenaRuntimeIdentity,
    ArenaScreenReceipt,
    ArenaServiceManifest,
    NonCrownScreenPolicy,
    PromotionDecision,
    ScreenGrade,
    ScreenStagePolicy,
    ScreenStageResult,
    ServingShape,
    WorkloadMixture,
    WorkloadRegime,
)
from cacheon.bundle_hash import content_hash
from cacheon.chain.publication import publish_worker_bundle
from cacheon.eval.b300_arena_provider import (
    B300DeclaredQualificationAuthorities,
    B300QualificationLanePair,
    B300QualificationLanePolicy,
    B300ResidentScreenFactory,
    B300ScreenDeploymentAuthorities,
    B300ScreenStageHandler,
    b300_arena_provider_digest,
    b300_executor_role_policy_digest,
)
from cacheon.eval.b300_registered_qualification_inputs import (
    registered_b300_member_contract_projection,
    registered_b300_profile_resolver_digest,
)
from cacheon.eval.b300_qualification_graph_store_io import (
    B300QualificationGraphEvidenceHold,
)
from cacheon.eval.device_state import DeviceStatePolicy, GPUConfiguration
from cacheon.eval.oci_backend import (
    OCIBackendConfig,
    OCIEngineExecutor,
    OCIRuntimeResourcePolicy,
)
from cacheon.eval.oci_prebuild import OCIPrebuildConfig, OCIPrebuildPolicy
from cacheon.eval.qualification_intake import QualificationReservation
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.stack_manifest import EvaluationStackManifest, ProposalContributionRef
from cacheon.target_catalog import default_target_catalog


TARGET = "activation.silu_and_mul"


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _runtime() -> ArenaRuntimeIdentity:
    return ArenaRuntimeIdentity(
        arena_id="production-b300-tp4",
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base-engine"),
        validator_overlay_digest=_h("validator-overlay"),
        worker_distribution_digest=_h("worker-distribution"),
        model_revision_digest=_h("model-revision"),
        model_manifest_digest=_h("model-manifest"),
        model_content_digest=_h("model-content"),
        target_architecture="sm103",
        topology_class="nvlink-domain",
        topology_digest=_h("topology"),
        gpu_count=4,
        tensor_parallel_size=4,
    )


def _runtime_policy() -> OCIRuntimeResourcePolicy:
    return OCIRuntimeResourcePolicy(
        uid=max(1, os.getuid()),
        gid=max(1, os.getgid()),
        cpu_millis=8_000,
        memory_bytes=32 << 30,
        pids_limit=4_096,
        nofile_limit=65_536,
        cache_bytes=4 << 30,
        cache_inodes=100_000,
        tmpfs_bytes=1 << 30,
        shm_bytes=8 << 30,
        init_timeout_seconds=120.0,
        batch_timeout_seconds=60.0,
        container_python="/usr/local/bin/python3",
    )


def _prebuild_policy(runtime: OCIRuntimeResourcePolicy) -> OCIPrebuildPolicy:
    return OCIPrebuildPolicy(
        uid=runtime.uid,
        gid=runtime.gid,
        cpu_millis=8_000,
        memory_bytes=32 << 30,
        pids_limit=4_096,
        tmpfs_bytes=1 << 30,
        stage_bytes=16 << 30,
        stage_inodes=100_000,
        timeout_seconds=7_200.0,
        native_compile_timeout_seconds=6_000,
        container_python=runtime.container_python,
        build_path=("/usr/local/cuda/bin", "/usr/local/bin", "/usr/bin", "/bin"),
        build_tmpdir="/tmp",
        pinned_build_roots=("/usr/include", "/usr/lib", "/usr/local/cuda"),
        runtime_policy_digest=runtime.digest,
    )


def _gpu(index: int) -> GPUConfiguration:
    return GPUConfiguration(
        physical_id=index,
        uuid=f"GPU-00000000-{index:04x}-0000-0000-{index:012x}",
        pci_bus_id=f"00000000:{index + 1:02x}:00.0",
        name="NVIDIA B300 SXM6 AC",
        memory_total_mib=288_000,
        driver_version="600.10.01",
        power_limit_mw=1_000_000,
        compute_mode="Default",
        persistence_mode="Enabled",
        application_graphics_clock_mhz=None,
        application_memory_clock_mhz=None,
        max_graphics_clock_mhz=2_500,
        max_memory_clock_mhz=5_000,
    )


@pytest.fixture
def executor_factory(tmp_path: Path):
    executors: list[OCIEngineExecutor] = []
    sequence = 0

    def create(role: str, lane: str) -> OCIEngineExecutor:
        nonlocal sequence
        sequence += 1
        offset = 0 if lane == "A" else 4
        runtime = _runtime_policy()
        root = tmp_path / f"executor-{sequence}-{role}-{lane}"
        config = OCIBackendConfig(
            OCIPrebuildConfig(
                docker_binary="/usr/bin/docker",
                recovery_root=root / "recovery",
                publication_root=root / "publications",
                seccomp_profile=root / "seccomp.json",
                # This role identity remains stable when its physical lane swaps.
                executor_id=role,
                policy=_prebuild_policy(runtime),
            ),
            runtime,
        )
        executor = OCIEngineExecutor(
            config,
            DeviceStatePolicy(
                expected_gpus=tuple(_gpu(index) for index in range(offset, offset + 4)),
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


resident_pair_authority = authority_fixtures.resident_pair_authority
_Judge = authority_fixtures._Judge


def _incumbent(runtime: ArenaRuntimeIdentity, arena_digest: str):
    catalog = default_target_catalog()
    return catalog, EvaluationStackManifest(
        runtime_digest=runtime.runtime_digest,
        base_engine_digest=runtime.base_engine_digest,
        arena_digest=arena_digest,
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={},
    )


def _profiles(catalog, builder_source: str, resolvers=None):
    by_target = {} if resolvers is None else resolvers
    return tuple(
        deployment.B300RegisteredProfileAuthority(
            target.target_id,
            target.target_spec_digest,
            registered_b300_profile_resolver_digest(
                target,
                builder_source_digest=builder_source,
            ),
            by_target.get(target.target_id, lambda _candidate, _prepared: object()),
        )
        for target in registered_b300_member_contract_projection(catalog)
    )


def _construction(tmp_path: Path, runtime: ArenaRuntimeIdentity):
    catalog, incumbent = _incumbent(runtime, _h("arena"))
    builder_source = _h("builder-source")
    evidence_root = tmp_path / "evidence"
    count_quality = authority_fixtures._resident_count_quality(catalog, evidence_root)
    return deployment.B300QualificationConstructionAuthority(
        catalog=catalog,
        profiles=_profiles(catalog, builder_source),
        incumbent_stack=incumbent,
        incumbent_tree_digest=_h("incumbent-tree"),
        pristine_stack=incumbent,
        pristine_tree_digest=_h("pristine-tree"),
        evidence_root=evidence_root,
        evidence_policy_digest=_h("evidence-policy"),
        builder_source_digest=builder_source,
        selection_store_digest=_h("selection-store"),
        resident_count_quality_builder_digest=_h("resident-count-quality-builder"),
        resident_count_quality=count_quality,
        secret_loader=lambda _reference: b"s" * 32,
        plan_builder=lambda _cohort, _secret: object(),
        entropy_provider_digest=_h("entropy-provider"),
        entropy_provider=lambda *_args: None,
        hidden_judge=_Judge(),
        deadline_policy_digest=_h("deadline-policy"),
        deadline_provider=lambda _cohort: time.monotonic() + 600.0,
    )


def _bind_construction_to_manifest(
    construction: deployment.B300QualificationConstructionAuthority,
    manifest: ArenaServiceManifest,
) -> deployment.B300QualificationConstructionAuthority:
    _catalog, incumbent = _incumbent(manifest.runtime, manifest.digest)
    rebound = replace(
        construction,
        incumbent_stack=incumbent,
        pristine_stack=incumbent,
    )
    assert (
        rebound.qualification_builder_digest
        == construction.qualification_builder_digest
    )
    assert (
        rebound.qualification_policy_digest
        == construction.qualification_policy_digest
    )
    return rebound


def test_pristine_reference_stays_empty_after_incumbent_advances(
    tmp_path: Path,
) -> None:
    construction = _construction(tmp_path, _runtime())
    proposal = ProposalContributionRef(
        TARGET,
        construction.catalog.target_spec_digest(TARGET),
        _h("settled-artifact"),
        _h("settled-payload"),
        _h("settled-attribution"),
    )
    incumbent = EvaluationStackManifest(
        runtime_digest=construction.incumbent_stack.runtime_digest,
        base_engine_digest=construction.incumbent_stack.base_engine_digest,
        arena_digest=construction.incumbent_stack.arena_digest,
        catalog_snapshot=construction.catalog.snapshot(),
        catalog_digest=construction.catalog.digest,
        entries={TARGET: proposal},
    )

    advanced = replace(
        construction,
        incumbent_stack=incumbent,
        incumbent_tree_digest=_h("advanced-incumbent-tree"),
    )
    assert advanced.pristine_stack.entries == {}
    assert advanced.incumbent_stack.entries == {TARGET: proposal}
    assert advanced.digest != construction.digest

    with pytest.raises(
        deployment.B300QualificationDeploymentError,
        match="pristine T",
    ):
        replace(advanced, pristine_stack=incumbent)


def _handlers() -> tuple[B300ScreenStageHandler, ...]:
    def run(_manifest, stage, _candidate):
        return ScreenStageResult(stage.stage, ScreenGrade.PASS, _h(stage.stage), 1)

    return tuple(
        B300ScreenStageHandler(
            stage,
            _h(f"{stage}-handler"),
            () if stage == "static" else (f"{stage}-resource",),
            run,
        )
        for stage in SCREEN_STAGES[:-1]
    )


def _lane_pair(
    lane_a_executor: OCIEngineExecutor,
    lane_b_executor: OCIEngineExecutor,
) -> B300QualificationLanePair:
    return B300QualificationLanePair(
        B300QualificationLanePolicy.from_device_policy(
            "A", lane_a_executor.device_policy
        ),
        B300QualificationLanePolicy.from_device_policy(
            "B", lane_b_executor.device_policy
        ),
    )


def _screen_authorities(
    construction: deployment.B300QualificationConstructionAuthority,
    candidate_executor: OCIEngineExecutor,
    baseline_executor: OCIEngineExecutor,
    lane_pair: B300QualificationLanePair,
) -> B300ScreenDeploymentAuthorities:
    declared = B300DeclaredQualificationAuthorities(
        qualification_policy_digest=construction.qualification_policy_digest,
        qualification_builder_digest=construction.qualification_builder_digest,
        candidate_executor_policy_digest=b300_executor_role_policy_digest(
            candidate_executor.config,
            role="candidate",
        ),
        resident_baseline_executor_policy_digest=b300_executor_role_policy_digest(
            baseline_executor.config,
            role="resident_baseline",
        ),
        lane_pair=lane_pair,
        entropy_provider_digest=construction.entropy_provider_digest,
        hidden_judge_binding_digest=construction.hidden_judge.binding.digest,
        deadline_policy_digest=construction.deadline_policy_digest,
    )
    return B300ScreenDeploymentAuthorities(
        runtime_identity=_runtime(),
        screen_handlers=_handlers(),
        resident_screen_factory=B300ResidentScreenFactory(
            _h("resident-screen-factory"),
            ("resident-screen-resource",),
            lambda: (_ for _ in ()).throw(
                AssertionError("composition must not open a screen lifetime")
            ),
        ),
        qualification=declared,
    )


def _manifest(authorities: B300ScreenDeploymentAuthorities) -> ArenaServiceManifest:
    return ArenaServiceManifest(
        runtime=authorities.runtime_identity,
        workload=WorkloadMixture(
            _h("prompt-corpus"),
            "sealed-prompt-seeds-v1",
            (
                WorkloadRegime(
                    "decode",
                    "decode",
                    500_000,
                    (ServingShape(256, 32, 32, 4),),
                ),
                WorkloadRegime(
                    "long-prefill",
                    "long_prefill",
                    500_000,
                    (ServingShape(8192, 4, 1, 4),),
                ),
            ),
        ),
        capacity=ArenaCapacityPolicy(32, 64, 1, 4, 4, 2, 2, 3),
        screens=NonCrownScreenPolicy(
            tuple(ScreenStagePolicy(stage, 30_000) for stage in SCREEN_STAGES)
        ),
        qualification_policy_digest=(
            authorities.qualification.qualification_policy_digest
        ),
        provider_digest=b300_arena_provider_digest(authorities),
    )


def _bundle(tmp_path: Path, index: int) -> ArenaCandidateBinding:
    source = tmp_path / f"source-{index}"
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
                f"bundle_id = 'qualification-{index}'",
                "abi_version = 'cacheon-op-abi-v0'",
                "[[ops]]",
                f"slot = '{TARGET}'",
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
    publication = publish_worker_bundle(
        source,
        tmp_path / "publications",
        content_hash(source),
    )
    catalog = default_target_catalog()
    reservation = QualificationReservation(
        _h(f"reservation-{index}"),
        publication.digest,
        TARGET,
        _h(f"delta-{index}"),
        index,
        f"miner-{index}",
        20,
        index,
        0,
        catalog.require(TARGET).members,
    )
    return ArenaCandidateBinding(reservation, publication, 1)


def _receipt(service_digest: str, candidate: ArenaCandidateBinding):
    return ArenaScreenReceipt(
        service_digest,
        candidate.digest,
        candidate.screen_attempt,
        tuple(
            ScreenStageResult(stage, ScreenGrade.PASS, _h(f"{stage}-pass"), 1)
            for stage in SCREEN_STAGES
        ),
        PromotionDecision.PROMOTE,
    )


@pytest.mark.parametrize(
    ("stage", "candidate_lane", "baseline_lane"),
    (("primary", "A", "B"), ("reproduction", "B", "A")),
)
def test_composition_preserves_one_service_across_exact_role_swap(
    tmp_path: Path,
    executor_factory,
    resident_pair_authority,
    stage: str,
    candidate_lane: str,
    baseline_lane: str,
) -> None:
    construction = _construction(tmp_path, _runtime())
    primary_candidate = executor_factory("candidate", "A")
    primary_baseline = executor_factory("resident-baseline", "B")
    lane_pair = _lane_pair(primary_candidate, primary_baseline)
    screen = _screen_authorities(
        construction,
        primary_candidate,
        primary_baseline,
        lane_pair,
    )
    manifest = _manifest(screen)
    construction = _bind_construction_to_manifest(construction, manifest)
    candidate = (
        primary_candidate
        if candidate_lane == "A"
        else executor_factory("candidate", "B")
    )
    baseline = (
        primary_baseline
        if baseline_lane == "B"
        else executor_factory("resident-baseline", "A")
    )

    result = deployment.compose_b300_qualification_deployment(
        manifest=manifest,
        screen_authorities=screen,
        construction=construction,
        candidate_executor=candidate,
        resident_baseline_executor=baseline,
        resident_pair_factory=resident_pair_authority(manifest.digest),
        screen_lane=stage,
    )

    assert result.manifest is manifest
    assert result.screen_lane == stage
    assert result.authorities.qualification == screen.qualification
    assert result.authorities.qualification_stage == stage
    assert (
        result.authorities.qualification_orientation.candidate.lane_id
        == candidate_lane
    )
    assert (
        result.authorities.qualification_orientation.resident_baseline.lane_id
        == baseline_lane
    )
    assert b300_arena_provider_digest(result.authorities) == manifest.provider_digest


def test_composition_refuses_overlap_wrong_orientation_and_manifest_drift(
    tmp_path: Path,
    executor_factory,
    resident_pair_authority,
) -> None:
    construction = _construction(tmp_path, _runtime())
    candidate_a = executor_factory("candidate", "A")
    baseline_b = executor_factory("resident-baseline", "B")
    lane_pair = _lane_pair(candidate_a, baseline_b)
    screen = _screen_authorities(
        construction,
        candidate_a,
        baseline_b,
        lane_pair,
    )
    manifest = _manifest(screen)
    construction = _bind_construction_to_manifest(construction, manifest)

    with pytest.raises(
        deployment.B300QualificationDeploymentError,
        match="orientation.*selected physical TP4 lane|overlap",
    ):
        deployment.compose_b300_qualification_deployment(
            manifest=manifest,
            screen_authorities=screen,
            construction=construction,
            candidate_executor=executor_factory("candidate", "B"),
            resident_baseline_executor=baseline_b,
            resident_pair_factory=resident_pair_authority(manifest.digest),
            screen_lane="primary",
        )

    drifted = replace(manifest, qualification_policy_digest=_h("drifted-policy"))
    with pytest.raises(
        deployment.B300QualificationDeploymentError,
        match="predeclare",
    ):
        deployment.compose_b300_qualification_deployment(
            manifest=drifted,
            screen_authorities=screen,
            construction=construction,
            candidate_executor=candidate_a,
            resident_baseline_executor=baseline_b,
            resident_pair_factory=resident_pair_authority(manifest.digest),
            screen_lane="primary",
        )


def test_path_free_cohort_is_one_candidate_and_rejects_control_state(
    tmp_path: Path,
    executor_factory,
) -> None:
    construction = _construction(tmp_path, _runtime())
    candidate_executor = executor_factory("candidate", "A")
    baseline_executor = executor_factory("resident-baseline", "B")
    first = _bundle(tmp_path / "first", 0)
    second = _bundle(tmp_path / "second", 1)
    service = _h("service")
    request = ArenaQualificationRequest(
        service,
        construction.qualification_policy_digest,
        (first,),
        (_receipt(service, first),),
    )
    cohort = deployment.B300QualificationCohort(request, "primary")

    assert cohort.candidate is first
    assert str(first.publication.root) not in cohort.digest
    with pytest.raises(
        deployment.B300QualificationDeploymentError,
        match="one submitted bundle",
    ):
        deployment.B300QualificationCohort(
            ArenaQualificationRequest(
                service,
                construction.qualification_policy_digest,
                (first, second),
                (_receipt(service, first), _receipt(service, second)),
            ),
            "primary",
        )
    factory_builder = deployment._factory_builder(
        construction,
        candidate_executor,
        baseline_executor,
        screen_lane="primary",
    )
    with pytest.raises(
        deployment.B300QualificationDeploymentError,
        match="forbidden control state",
    ):
        factory_builder(request, {"candidate_path": "/tmp/untrusted"})


def test_registered_target_and_canonical_evidence_root_are_fail_closed(
    tmp_path: Path,
) -> None:
    construction = _construction(tmp_path, _runtime())

    assert construction.profile_for("attention.decode").target_id == "attention.decode"
    assert (
        construction.profile_for("collective.moe_epilogue.v1").target_id
        == "collective.moe_epilogue.v1"
    )
    with pytest.raises(deployment.B300QualificationDeploymentError, match="unsupported"):
        construction.profile_for("unknown.registered.target")
    with pytest.raises(
        deployment.B300QualificationDeploymentError,
        match="exactly cover",
    ):
        replace(construction, profiles=construction.profiles[:-1])
    with pytest.raises(
        deployment.B300QualificationDeploymentError,
        match="absolute authority",
    ):
        replace(construction, evidence_root=Path("relative-evidence"))


def _registered_fixtures():
    import importlib.util
    import sys

    path = Path(__file__).with_name("test_b300_registered_qualification.py")
    specification = importlib.util.spec_from_file_location(
        "cacheon_registered_qualification_fixtures_for_deployment", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _executor_mirror(arm):
    from types import SimpleNamespace

    return SimpleNamespace(
        device_policy=SimpleNamespace(
            physical_gpu_ids=tuple(
                int(value)
                for value in arm.binding.physical_hardware.physical_gpu_ids
            ),
            policy_sha256=arm.launch.hardware.device_policy_digest,
            configuration_sha256=arm.device_configuration_digest,
        ),
        config=SimpleNamespace(
            prebuild=SimpleNamespace(
                policy=SimpleNamespace(
                    resource_policy_digest=arm.launch.resource_policy_digest
                )
            ),
            runtime=SimpleNamespace(digest=arm.runtime_resource_policy_digest),
        ),
        manager=SimpleNamespace(namespace_digest=arm.executor_namespace_digest),
    )


class _RegisteredJudge:
    def __init__(self) -> None:
        self.binding = HiddenJudgeBinding(
            _h("hidden-corpus"),
            _h("hidden-judge"),
            _h("ordinary-hidden-task-policy"),
        )

    def __call__(self, **_kwargs):
        raise AssertionError("plan validation must not execute the judge")


def _registered_construction(harness, value, secret: bytes):
    builder_source = _h("builder-source")
    resolvers = {row.target_id: row.resolver for row in harness.factory.profiles}
    count_quality = authority_fixtures._resident_count_quality(
        harness.inputs.catalog, harness.inputs.evidence_root
    )
    return deployment.B300QualificationConstructionAuthority(
        catalog=harness.inputs.catalog,
        profiles=_profiles(harness.inputs.catalog, builder_source, resolvers),
        incumbent_stack=harness.inputs.incumbent_stack,
        incumbent_tree_digest=value.prepared.incumbent_binding.tree.tree_digest,
        pristine_stack=harness.inputs.pristine_stack,
        pristine_tree_digest=value.pristine_launch.tree_digest,
        evidence_root=harness.inputs.evidence_root,
        evidence_policy_digest=_h("evidence-policy"),
        builder_source_digest=builder_source,
        selection_store_digest=_h("selection-store"),
        resident_count_quality_builder_digest=_h("resident-count-quality-builder"),
        resident_count_quality=count_quality,
        secret_loader=lambda _reference: secret,
        plan_builder=harness.factory.plan_builder,
        entropy_provider_digest=_h("entropy-provider"),
        entropy_provider=lambda *_args: None,
        hidden_judge=_RegisteredJudge(),
        deadline_policy_digest=_h("deadline-policy"),
        deadline_provider=lambda _cohort: time.monotonic() + 600.0,
    )


def test_validate_plan_accepts_real_registered_plan_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    fixtures = _registered_fixtures()
    harness = fixtures._harness(tmp_path)
    secret = b"ordinary prefill blockscore selection"[:32]
    value = harness.factory.plan_builder(harness.cohort, secret)
    incumbent_tree = value.prepared.incumbent_binding.tree.tree_digest
    assert incumbent_tree == harness.inputs.incumbent_binding.tree.tree_digest

    construction = _registered_construction(harness, value, secret)
    resident = value.resident_speed_plan
    candidate_executor = _executor_mirror(resident.candidate)
    baseline_executor = _executor_mirror(resident.baseline)

    accepted = deployment._validate_plan(
        value,
        harness.cohort,
        secret,
        construction,
        candidate_executor,
        baseline_executor,
    )
    assert accepted is value

    with pytest.raises(
        deployment.B300QualificationDeploymentError,
        match="differs from resident-v3 deployment authority",
    ):
        deployment._validate_plan(
            value,
            harness.cohort,
            b"y" * 32,
            construction,
            candidate_executor,
            baseline_executor,
        )

    with pytest.raises(
        deployment.B300QualificationDeploymentError,
        match="candidate executor differs from the sealed resident arm",
    ):
        deployment._validate_plan(
            value,
            harness.cohort,
            secret,
            construction,
            baseline_executor,
            candidate_executor,
        )

    source = fixtures._candidate_source(tmp_path / "foreign-source")
    kernel = source / "kernels" / "msa_prefill_block_score.py"
    kernel.write_text(kernel.read_text() + "\n# foreign contribution variant\n")
    publication = fixtures.publish_worker_bundle(
        source,
        fixtures._private_directory(tmp_path / "foreign-publications"),
        fixtures.content_hash(source),
    )
    catalog = fixtures.default_target_catalog()
    inspected = fixtures.inspect_contribution(publication.root, catalog=catalog)
    foreign = fixtures.ArenaCandidateBinding(
        fixtures.QualificationReservation(
            _h("foreign-reservation"),
            publication.digest,
            fixtures.TARGET,
            inspected.selected_delta_digest,
            0,
            "miner-hotkey",
            8_775_104,
            155,
            0,
            catalog.require(fixtures.TARGET).members,
        ),
        publication,
        1,
    )
    foreign_cohort = fixtures._cohort(foreign, harness.policy.digest)
    with pytest.raises(
        deployment.B300QualificationDeploymentError,
        match="marginal arm differs from finalized intake or incumbent",
    ):
        deployment._validate_plan(
            value,
            foreign_cohort,
            secret,
            construction,
            candidate_executor,
            baseline_executor,
        )


@pytest.mark.parametrize("stage", ("primary", "reproduction"))
def test_deployment_accepts_atomic_registered_plan_on_both_retained_stages(
    tmp_path: Path,
    stage: str,
) -> None:
    fixtures = _registered_fixtures()
    harness = fixtures._harness(tmp_path, fixtures.FUSED)
    cohort = deployment.B300QualificationCohort(harness.cohort.request, stage)
    secret = b"atomic fused epilogue selection!!"[:32]
    value = harness.factory.plan_builder(cohort, secret)
    construction = _registered_construction(harness, value, secret)
    resident = value.resident_speed_plan

    accepted = deployment._validate_plan(
        value,
        cohort,
        secret,
        construction,
        _executor_mirror(resident.candidate),
        _executor_mirror(resident.baseline),
    )

    assert accepted is value
    assert cohort.candidate.reservation.target_id == "collective.moe_epilogue.v1"
    assert tuple(
        row.slot_id for row in accepted.candidates[0].graph_requirement.binding.members
    ) == cohort.candidate.reservation.target_members

    authority = accepted.candidates[0]
    for field, stale in (
        ("contract_digest", _h("stale-atomic-member-contract")),
        ("verification_profile_id", "stale.atomic.member.verify.v1"),
    ):
        members = list(authority.graph_requirement.binding.members)
        members[0] = replace(members[0], **{field: stale})
        binding = replace(
            authority.graph_requirement.binding,
            members=tuple(members),
        )
        requirement = replace(authority.graph_requirement, binding=binding)
        tampered = replace(
            authority,
            graph_requirement=requirement,
            profile=replace(
                authority.profile,
                graph_requirement_digest=requirement.digest,
            ),
        )
        with pytest.raises(
            deployment.B300QualificationDeploymentError,
            match="profile/graph authority differs",
        ):
            deployment._validate_profile_binding(
                tampered,
                cohort.candidate,
                accepted.prepared.candidates[0],
                construction,
            )


def test_factory_builder_preserves_graph_evidence_hold(tmp_path: Path) -> None:
    fixtures = _registered_fixtures()
    harness = fixtures._harness(tmp_path)
    secret = b"held graph factory selection secret"[:32]
    value = harness.factory.plan_builder(harness.cohort, secret)
    hold = B300QualificationGraphEvidenceHold("graph attempt remains armed")

    def unavailable(_cohort, _secret):
        raise hold

    construction = replace(
        _registered_construction(harness, value, secret),
        plan_builder=unavailable,
    )
    resident = value.resident_speed_plan
    builder = deployment._factory_builder(
        construction,
        _executor_mirror(resident.candidate),
        _executor_mirror(resident.baseline),
        screen_lane="primary",
    )
    receipt = replace(
        harness.cohort.receipt,
        service_digest=construction.incumbent_stack.arena_digest,
    )
    request = ArenaQualificationRequest(
        construction.incumbent_stack.arena_digest,
        construction.qualification_policy_digest,
        (harness.cohort.candidate,),
        (receipt,),
    )
    with pytest.raises(B300QualificationGraphEvidenceHold) as caught:
        builder(request, None)
    assert caught.value is hold


def test_factory_reuses_its_first_sealed_plan_without_reconstruction(
    tmp_path: Path,
) -> None:
    fixtures = _registered_fixtures()
    harness = fixtures._harness(tmp_path)
    secret = b"one sealed qualification plan secret"[:32]
    expected = harness.factory.plan_builder(harness.cohort, secret)
    construction = _registered_construction(harness, expected, secret)
    original = construction.plan_builder
    calls = 0
    built = []

    def counted(cohort, observed_secret):
        nonlocal calls
        calls += 1
        value = original(cohort, observed_secret)
        built.append(value)
        return value

    construction = replace(construction, plan_builder=counted)
    resident = expected.resident_speed_plan
    builder = deployment._factory_builder(
        construction,
        _executor_mirror(resident.candidate),
        _executor_mirror(resident.baseline),
        screen_lane="primary",
    )
    receipt = replace(
        harness.cohort.receipt,
        service_digest=construction.incumbent_stack.arena_digest,
    )
    request = ArenaQualificationRequest(
        construction.incumbent_stack.arena_digest,
        construction.qualification_policy_digest,
        (harness.cohort.candidate,),
        (receipt,),
    )
    factory = builder(request, None)
    assert calls == 1
    assert factory.build() is built[0]
    assert factory.build() is built[0]
    assert calls == 1
