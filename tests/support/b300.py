"""Device, runtime, and OCI policy fixtures shared by the B300 lane suites.

Five suites each carried their own copy of these. The copies were verbatim
except that the reference-device suite runs a different GPU on purpose, so that
is a named device profile here rather than a mode flag on one builder. Anything
a caller varies per test stays an argument.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
import time
from pathlib import Path

from cacheon.arena_service import (
    ArenaCapacityPolicy,
    ArenaRuntimeIdentity,
    ArenaServiceManifest,
    Workload,
    WorkloadCell,
)
from cacheon.eval.b300_arena_provider import (
    B300DeploymentAuthorities,
    B300QualificationLanePair,
    B300QualificationLanePolicy,
    b300_arena_provider_digest,
)
from cacheon.eval.b300_qualification_commission import B300QualificationCapabilities
from cacheon.eval.device_state import DeviceStatePolicy, GPUConfiguration
from cacheon.eval.oci_backend import (
    OCIBackendConfig,
    OCIEngineExecutor,
    OCIRuntimeResourcePolicy,
)
from cacheon.eval.oci_prebuild import OCIPrebuildConfig, OCIPrebuildPolicy
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest,
    QualificationPlanFactory,
)
from cacheon.eval.qualification_runner import HiddenJudgeBinding


# What a node arena registers: the model's nodes, and optionally the prefix cache.
NODE_TARGET_IDS = ("forward_pass",)
NODE_AND_CACHE_TARGET_IDS = ("forward_pass", "prefix_cache")


def sha(label: str) -> str:
    """A deterministic stand-in digest, named by what it stands for."""

    return hashlib.sha256(label.encode()).hexdigest()


class StubHiddenJudge:
    """Carries a judge binding; asserts if a capability check executes it."""

    binding = HiddenJudgeBinding(
        sha("hidden-corpus"), sha("hidden-judge"), sha("hidden-policy")
    )

    def __call__(self, **_kwargs):
        raise AssertionError("capability stubs must not execute the hidden judge")


class StubSourceResolver:
    def resolve_proposal(self, *_args, **_kwargs):
        raise AssertionError("capability stubs must not resolve sources")


def qualification_capabilities(**overrides: object) -> B300QualificationCapabilities:
    values: dict[str, object] = {
        "secret_loader": lambda _reference: b"s" * 32,
        "entropy_provider": lambda *_args: None,
        "hidden_judge": StubHiddenJudge(),
        "source_resolver": StubSourceResolver(),
        "source_resolver_digest": sha("source-resolver"),
        "incumbent_entries": {},
    }
    values.update(overrides)
    return B300QualificationCapabilities(**values)


def arena_runtime() -> ArenaRuntimeIdentity:
    return ArenaRuntimeIdentity(
        arena_id="production-b300-tp4",
        runtime_digest=sha("runtime"),
        base_engine_digest=sha("base-engine"),
        validator_overlay_digest=sha("validator-overlay"),
        worker_distribution_digest=sha("worker-distribution"),
        model_revision_digest=sha("model-revision"),
        model_manifest_digest=sha("model-manifest"),
        model_content_digest=sha("model-content"),
        target_architecture="sm103",
        topology_class="nvlink-domain",
        topology_digest=sha("topology"),
        gpu_count=4,
        tensor_parallel_size=4,
    )


def runtime_policy() -> OCIRuntimeResourcePolicy:
    """Qualification's runtime limits: a full sealed cell."""
    return OCIRuntimeResourcePolicy(
        uid=max(1, os.getuid()),
        gid=max(1, os.getgid()),
        cpu_millis=8_000,
        tmpfs_bytes=1 << 30,
        container_python="/usr/local/bin/python3",
        memory_bytes=32 << 30,
        pids_limit=4_096,
        nofile_limit=65_536,
        cache_bytes=4 << 30,
        cache_inodes=100_000,
        shm_bytes=8 << 30,
        init_timeout_seconds=120.0,
        batch_timeout_seconds=60.0,
    )


def prebuild_policy(runtime: OCIRuntimeResourcePolicy) -> OCIPrebuildPolicy:
    return OCIPrebuildPolicy(
        uid=runtime.uid,
        gid=runtime.gid,
        cpu_millis=8_000,
        tmpfs_bytes=1 << 30,
        container_python=runtime.container_python,
        build_path=("/usr/local/cuda/bin", "/usr/local/bin", "/usr/bin", "/bin"),
        build_tmpdir="/tmp",
        pinned_build_roots=("/usr/include", "/usr/lib", "/usr/local/cuda"),
        runtime_policy_digest=runtime.digest,
        memory_bytes=32 << 30,
        pids_limit=4_096,
        stage_bytes=16 << 30,
        stage_inodes=100_000,
        timeout_seconds=7_200.0,
        native_compile_timeout_seconds=6_000,
    )


def gpu(index: int = 0, model: str = "b300") -> GPUConfiguration:
    """One device of ``model``. The second model is not decoration.

    A generic evaluator that resolves identity from sealed inputs has to be
    exercised on more than one device profile, or "generic" is untested. The
    reference suite runs the RTX profile for exactly that reason.
    """

    if model == "h100":
        return replace(
            gpu(index), name="NVIDIA H100 80GB HBM3", memory_total_mib=81_559,
            power_limit_mw=700_000, max_graphics_clock_mhz=1_980,
            max_memory_clock_mhz=2_619,
        )
    if model == "rtx6000":
        return GPUConfiguration(
            physical_id=index,
            uuid=f"GPU-00000000-{index:04x}-0000-0000-{index:012x}",
            pci_bus_id=f"00000000:{index + 1:02x}:00.0",
            name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            memory_total_mib=98_304,
            driver_version="595.71.05",
            power_limit_mw=600_000,
            compute_mode="Default",
            persistence_mode="Enabled",
            application_graphics_clock_mhz=None,
            application_memory_clock_mhz=None,
            max_graphics_clock_mhz=2_100,
            max_memory_clock_mhz=4_000,
        )
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


def executors(tmp_path: Path):
    """Yield a builder of one executor per (role, physical lane), then close them all.

    Wrap it in a pytest fixture: ``yield from executors(tmp_path)``.
    """

    built: list[OCIEngineExecutor] = []
    sequence = 0

    def create(role: str, lane: str = "A") -> OCIEngineExecutor:
        nonlocal sequence
        sequence += 1
        normalized_role = "candidate" if role == "candidate" else "resident_baseline"
        if lane not in {"A", "B"}:
            raise AssertionError("fixture lane must be A or B")
        first_gpu = 0 if lane == "A" else 4
        runtime = runtime_policy()
        root = tmp_path / f"executor-{sequence}-{role}"
        executor = OCIEngineExecutor(
            OCIBackendConfig(
                OCIPrebuildConfig(
                    docker_binary="/usr/bin/docker",
                    recovery_root=root / "recovery",
                    publication_root=root / "publications",
                    seccomp_profile=root / "seccomp.json",
                    executor_id=f"qualification-{normalized_role}",
                    policy=prebuild_policy(runtime),
                ),
                runtime,
            ),
            DeviceStatePolicy(
                expected_gpus=tuple(gpu(index) for index in range(first_gpu, first_gpu + 4)),
                required_consecutive_idle_samples=2,
                poll_interval_s=0.05,
                ready_poll_interval_s=0.05,
                drain_timeout_s=2.0,
                maximum_samples=8,
            ),
        )
        built.append(executor)
        return executor

    yield create
    for executor in built:
        executor.manager.close()


class FactoryBuilder:
    """Records every qualification request; ``fail`` raises like a lost authority store."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, object | None]] = []

    def __call__(self, request, state):
        self.calls.append((request, state))
        if self.fail:
            raise OSError("private authority store unavailable")
        reservations = tuple(row.reservation for row in request.candidates)
        manifest = QualificationAuthorityManifest(
            "registered",
            sha("qualification-authority"),
            sha("qualification-source"),
            sha("selection-commitment"),
            sha("selection-secret-reference"),
            tuple(row.selected_delta_digest for row in reservations),
            reservations,
        )
        return QualificationPlanFactory(
            manifest, lambda _reference: b"s" * 32, lambda _secret: None
        )


def deployment_authorities(executor_factory, *, builder: FactoryBuilder | None = None):
    """A full primary-orientation deployment: candidate on lane A, resident baseline on B."""

    factory_builder = builder or FactoryBuilder()
    candidate_executor = executor_factory("candidate", "A")
    baseline_executor = executor_factory("resident_baseline", "B")
    lane_pair = B300QualificationLanePair(
        B300QualificationLanePolicy.from_device_policy("A", candidate_executor.device_policy),
        B300QualificationLanePolicy.from_device_policy("B", baseline_executor.device_policy),
    )
    authorities = B300DeploymentAuthorities(
        runtime_identity=arena_runtime(),
        qualification_policy_digest=sha("qualification-policy"),
        qualification_builder_digest=sha("qualification-builder"),
        qualification_factory_builder=factory_builder,
        executor=candidate_executor,
        resident_baseline_executor=baseline_executor,
        entropy_provider_digest=sha("entropy-provider"),
        entropy_provider=lambda *_args: None,
        hidden_judge=StubHiddenJudge(),
        deadline_policy_digest=sha("deadline-policy"),
        deadline_provider=lambda _request, _state: time.monotonic() + 600.0,
        qualification_lane_pair=lane_pair,
        qualification_stage="primary",
    )
    return authorities, factory_builder


def arena_manifest(authorities: B300DeploymentAuthorities, **changes) -> ArenaServiceManifest:
    values = {
        "runtime": authorities.runtime_identity,
        "workload": Workload(
            sha("prompt-corpus"),
            "sealed-prompt-seeds-v1",
            (WorkloadCell("s8", 8192, 1024, 64, 8),),
        ),
        "capacity": ArenaCapacityPolicy(32, 100, 8, 4),
        "qualification_policy_digest": authorities.qualification_policy_digest,
        "provider_digest": b300_arena_provider_digest(authorities),
    }
    values.update(changes)
    return ArenaServiceManifest(**values)
